/** @file 将导纳控制器输出的笛卡尔目标位姿转换为关节轨迹命令。 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <s622_dual_interfaces/msg/admittance_status.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "joint_position_limiter.hpp"

namespace s622_dual_admittance
{

/**
 * @brief 导纳目标位姿的连续执行节点。
 * @details 接收目标位姿、导纳状态和关节反馈，调用 MoveIt IK 求解关节目标，
 * 限制指令变化速度后发布 JointTrajectory。这里不生成名义路径。
 */
class CartesianPoseExecutor final : public rclcpp::Node
{
public:
  /**
   * @brief 构造节点，准备参数及 ROS 通信接口。
   * @param options ROS 节点选项，包括启动时传入的参数和名称重映射等设置。
   * @details 声明尚不存在的参数并校验配置，创建目标位姿、导纳状态和关节状态
   * 订阅，以及关节轨迹发布器和启停服务。节点初始为禁用状态。
   * 机器人模型和周期执行定时器由 initialize() 在构造完成后创建。
   */
  explicit CartesianPoseExecutor(const rclcpp::NodeOptions & options)
  : Node("cartesian_pose_executor", options)
  {
    declare_if_missing("group_name", std::string("left_arm"));
    declare_if_missing("tip_link", std::string("left_grasp_frame"));
    declare_if_missing(
      "joint_names",
      std::vector<std::string>{
        "left_j1", "left_j2", "left_j3",
        "left_j4", "left_j5", "left_j6"});
    declare_if_missing("command_rate_hz", 100.0);
    declare_if_missing("ik_timeout_sec", 0.005);
    declare_if_missing("target_timeout_sec", 0.05);
    declare_if_missing("status_timeout_sec", 0.3);
    declare_if_missing("joint_state_timeout_sec", 0.05);
    declare_if_missing("command_horizon_sec", 0.03);
    declare_if_missing("joint_velocity_limit_enabled", true);
    declare_if_missing(
      "max_joint_velocity",
      std::vector<double>{0.2, 0.2, 0.2, 0.2, 0.2, 0.2});

    load_parameters();

    target_subscription_ =
      create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose",
      rclcpp::QoS(1).reliable(),
      std::bind(
        &CartesianPoseExecutor::target_callback,
        this,
        std::placeholders::_1));

    status_subscription_ = create_subscription<s622_dual_interfaces::msg::AdmittanceStatus>(
      "admittance_status", rclcpp::QoS(1).reliable().transient_local(),
      /**
       * @brief 导纳状态回调：判断上游是否处于有效的 RUNNING 状态。
       * @param message 导纳状态消息，包含状态名称和 ROS 源时间戳。
       * @details 校验源时间戳年龄，并用单调时钟记录本次接收时间，供执行循环
       * 检查心跳间隔。状态无效时清除目标有效标记并重置限速器，暂停输出。
       */
      [this](const s622_dual_interfaces::msg::AdmittanceStatus::SharedPtr message) {
        const auto stamp = rclcpp::Time(message->header.stamp, get_clock()->get_clock_type());
        const double age = (get_clock()->now() - stamp).seconds();
        status_running_ = message->state == "RUNNING" && stamp.nanoseconds() != 0 &&
        std::isfinite(age) && age >= -0.01 && age <= status_timeout_sec_;
        last_status_time_ = std::chrono::steady_clock::now();
        if (!status_running_) {
          have_target_ = false;
          command_limiter_.reset();
        }
      });

    joint_state_subscription_ =
      create_subscription<sensor_msgs::msg::JointState>(
      "joint_states",
      rclcpp::SensorDataQoS(),
      std::bind(
        &CartesianPoseExecutor::joint_state_callback,
        this,
        std::placeholders::_1));

    command_publisher_ =
      create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "joint_trajectory",
      rclcpp::QoS(1).reliable());

    enable_service_ = create_service<std_srvs::srv::SetBool>(
      "~/enable",
      /**
       * @brief 执行器启停服务回调。
       * @param request data=true 启用执行器，data=false 禁用执行器。
       * @param response 写入操作结果和提示信息，通过服务返回给调用方。
       * @details 每次调用均清除旧目标、RUNNING 标记及限速器历史。
       * 启用后仍需收到有效目标和 RUNNING 状态，并有新鲜关节反馈，才会输出指令。
       */
      [this](
        const std_srvs::srv::SetBool::Request::SharedPtr request,
        std_srvs::srv::SetBool::Response::SharedPtr response)
      {
        enabled_ = request->data;
        command_limiter_.reset();
        have_target_ = false;
        status_running_ = false;
        response->success = true;
        response->message = enabled_ ? "Cartesian execution enabled; waiting for fresh target" :
        "Cartesian execution disabled";
      });
  }

  /**
   * @brief 加载 MoveIt 机器人模型和 IK 求解器，并启动周期执行。
   * @details 使用 robot_description 参数加载模型，确认规划组存在、关节顺序
   * 与配置一致且 IK 求解器可用，再创建 RobotState 和墙钟定时器。
   * 必须在节点由 shared_ptr 管理后调用，因为模型加载器需要 shared_from_this()。
   * @throws std::runtime_error 模型、规划组或 IK 求解器不可用，或关节顺序不一致。
   */
  void initialize()
  {
    robot_model_loader_ =
      std::make_shared<robot_model_loader::RobotModelLoader>(
      shared_from_this(), "robot_description", true);

    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_) {
      throw std::runtime_error("failed to load robot model");
    }

    joint_model_group_ = robot_model_->getJointModelGroup(group_name_);
    if (!joint_model_group_) {
      throw std::runtime_error("MoveIt group does not exist: " + group_name_);
    }
    if (joint_model_group_->getVariableNames() != joint_names_) {
      throw std::runtime_error(
              "joint_names do not match MoveIt group variable order");
    }
    if (!joint_model_group_->getSolverInstance()) {
      throw std::runtime_error("IK solver is unavailable for group: " + group_name_);
    }

    robot_state_ = std::make_unique<moveit::core::RobotState>(robot_model_);
    robot_state_->setToDefaultValues();

    const auto period =
      std::chrono::duration<double>(1.0 / command_rate_hz_);
    command_timer_ = create_wall_timer(
      period,
      std::bind(&CartesianPoseExecutor::command_callback, this));

    RCLCPP_INFO(
      get_logger(),
      "Cartesian executor ready: group=%s, tip=%s, model_frame=%s",
      group_name_.c_str(), tip_link_.c_str(),
      robot_model_->getModelFrame().c_str());
  }

private:
  /**
   * @brief 仅在参数尚未声明时，按给定类型和默认值声明参数。
   * @tparam T 参数值的类型，例如 double、std::string 或 std::vector<double>。
   * @param name 参数名称。
   * @param default_value 未通过启动配置提供该参数时使用的默认值。
   * @details main() 允许自动声明启动参数；此处避免重复声明，并保留配置覆盖值。
   */
  template<typename T>
  void declare_if_missing(const std::string & name, const T & default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter<T>(name, default_value);
    }
  }

  /**
   * @brief 读取并校验执行器配置，将参数保存到成员变量。
   * @details 加载规划组、末端、关节名称、执行频率、超时阈值、目标到达时长
   * 和每个关节的指令速度上限。频率单位为 Hz，时长单位为 s；本机器人的
   * 臂关节为转动关节，对应速度上限单位为 rad/s。
   * @throws std::invalid_argument 名称为空、数组长度不匹配，或频率、时长、
   * 速度上限包含非有限值或非正值。
   */
  void load_parameters()
  {
    group_name_ = get_parameter("group_name").as_string();
    tip_link_ = get_parameter("tip_link").as_string();
    joint_names_ = get_parameter("joint_names").as_string_array();
    command_rate_hz_ = get_parameter("command_rate_hz").as_double();
    ik_timeout_sec_ = get_parameter("ik_timeout_sec").as_double();
    target_timeout_sec_ = get_parameter("target_timeout_sec").as_double();
    status_timeout_sec_ = get_parameter("status_timeout_sec").as_double();
    joint_state_timeout_sec_ =
      get_parameter("joint_state_timeout_sec").as_double();
    command_horizon_sec_ = get_parameter("command_horizon_sec").as_double();
    joint_velocity_limit_enabled_ =
      get_parameter("joint_velocity_limit_enabled").as_bool();
    max_joint_velocity_ =
      get_parameter("max_joint_velocity").as_double_array();

    if (group_name_.empty() || tip_link_.empty()) {
      throw std::invalid_argument("group_name and tip_link must not be empty");
    }
    if (joint_names_.empty() ||
      max_joint_velocity_.size() != joint_names_.size())
    {
      throw std::invalid_argument(
              "joint_names and max_joint_velocity must have equal non-zero length");
    }
    if (!std::isfinite(command_rate_hz_) || command_rate_hz_ <= 0.0 ||
      !std::isfinite(ik_timeout_sec_) || ik_timeout_sec_ <= 0.0 ||
      !std::isfinite(target_timeout_sec_) || target_timeout_sec_ <= 0.0 ||
      !std::isfinite(status_timeout_sec_) || status_timeout_sec_ <= 0.0 ||
      !std::isfinite(joint_state_timeout_sec_) || joint_state_timeout_sec_ <= 0.0 ||
      !std::isfinite(command_horizon_sec_) || command_horizon_sec_ <= 0.0)
    {
      throw std::invalid_argument(
              "rate, timeout, and horizon parameters must be finite and positive");
    }
    for (const double velocity : max_joint_velocity_) {
      if (!std::isfinite(velocity) || velocity <= 0.0) {
        throw std::invalid_argument(
                "max_joint_velocity must contain finite positive values");
      }
    }
  }

  /**
   * @brief 检查位姿是否满足本节点的基本数值要求。
   * @param pose 待检查的位置和四元数。
   * @return 所有分量均有限且四元数模长平方大于阈值时返回 true，否则返回 false。
   * @details 此函数不执行坐标变换、四元数归一化、可达性或碰撞检查。
   */
  static bool pose_is_valid(const geometry_msgs::msg::Pose & pose)
  {
    const bool finite =
      std::isfinite(pose.position.x) &&
      std::isfinite(pose.position.y) &&
      std::isfinite(pose.position.z) &&
      std::isfinite(pose.orientation.x) &&
      std::isfinite(pose.orientation.y) &&
      std::isfinite(pose.orientation.z) &&
      std::isfinite(pose.orientation.w);
    const double quaternion_norm =
      pose.orientation.x * pose.orientation.x +
      pose.orientation.y * pose.orientation.y +
      pose.orientation.z * pose.orientation.z +
      pose.orientation.w * pose.orientation.w;
    return finite && quaternion_norm > 1.0e-12;
  }

  /**
   * @brief 接收并缓存导纳修正后的末端目标位姿。
   * @param message 带坐标系和 ROS 源时间戳的 PoseStamped 消息。
   * @details 拒绝数值无效、时间戳为零、源年龄超过 target_timeout_sec，或比
   * 当前 ROS 时间提前超过 0.01 s 的消息。有效消息覆盖旧目标，并记录单调时钟
   * 接收时间。坐标系是否匹配由 command_callback() 在实际执行前检查。
   */
  void target_callback(
    const geometry_msgs::msg::PoseStamped::SharedPtr message)
  {
    if (!pose_is_valid(message->pose)) {
      RCLCPP_WARN(get_logger(), "Ignoring invalid Cartesian target pose");
      return;
    }
    const auto stamp = rclcpp::Time(message->header.stamp, get_clock()->get_clock_type());
    const double source_age = (get_clock()->now() - stamp).seconds();
    if (stamp.nanoseconds() == 0 || !std::isfinite(source_age) ||
      source_age < -0.01 || source_age > target_timeout_sec_)
    {
      return;
    }
    latest_target_ = *message;
    have_target_ = true;
    last_target_time_ = std::chrono::steady_clock::now();
  }

  /**
   * @brief 接收关节反馈，并按本执行器配置的关节顺序整理位置。
   * @param message 关节状态消息；允许包含其他机械臂或夹爪的关节。
   * @details 检查名称和位置数组长度，按名称提取本组全部关节；缺少任一关节
   * 或位置包含非有限值时忽略整条消息。有效数据供 IK 初始状态及限速器初始化
   * 使用。此处记录本地接收时间，不使用消息头的源时间戳判断新鲜度。
   */
  void joint_state_callback(
    const sensor_msgs::msg::JointState::SharedPtr message)
  {
    if (message->name.size() != message->position.size()) {
      return;
    }

    std::vector<double> ordered_positions;
    ordered_positions.reserve(joint_names_.size());
    for (const auto & required_name : joint_names_) {
      const auto iterator = std::find(
        message->name.begin(), message->name.end(), required_name);
      if (iterator == message->name.end()) {
        return;
      }
      const std::size_t index =
        static_cast<std::size_t>(
        std::distance(message->name.begin(), iterator));
      const double position = message->position[index];
      if (!std::isfinite(position)) {
        return;
      }
      ordered_positions.push_back(position);
    }

    current_joint_positions_ = std::move(ordered_positions);
    have_joint_state_ = true;
    last_joint_state_time_ = std::chrono::steady_clock::now();
  }

  /**
   * @brief 周期执行“目标位姿 → IK → 关节指令限速 → 轨迹发布”。
   * @details 只有执行器启用、目标与关节反馈存在、导纳处于有效 RUNNING 状态
   * 时才继续。用单调时钟检查消息接收间隔，超时则重置限速器并跳过输出。
   * 目标坐标系必须与模型坐标系一致；以当前关节反馈为 IK 初始状态，求解并
   * 检查关节边界。成功后限制相邻位置指令的变化量，发布单点 JointTrajectory，
   * 要求该点在 command_horizon_sec 后到达。未通过检查时不发布本周期指令。
   * 本函数不进行名义路径规划，也未对 IK 结果执行碰撞检测。
   */
  void command_callback()
  {
    if (!enabled_ || !have_target_ || !have_joint_state_ || !status_running_) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    const double target_age =
      std::chrono::duration<double>(now - last_target_time_).count();
    const double joint_state_age =
      std::chrono::duration<double>(now - last_joint_state_time_).count();
    const double status_age =
      std::chrono::duration<double>(now - last_status_time_).count();

    if (target_age > target_timeout_sec_ ||
      joint_state_age > joint_state_timeout_sec_ ||
      status_age > status_timeout_sec_)
    {
      command_limiter_.reset();
      return;
    }
    if (latest_target_.header.frame_id != robot_model_->getModelFrame()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Target frame '%s' differs from model frame '%s'",
        latest_target_.header.frame_id.c_str(),
        robot_model_->getModelFrame().c_str());
      return;
    }

    // 使用实际关节位置作为本次 IK 的初始状态，使解从当前姿态开始求取。
    robot_state_->setJointGroupPositions(
      joint_model_group_, current_joint_positions_);
    robot_state_->update();

    if (!robot_state_->setFromIK(
        joint_model_group_, latest_target_.pose,
        tip_link_, ik_timeout_sec_))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "IK failed for group %s", group_name_.c_str());
      return;
    }
    if (!robot_state_->satisfiesBounds(joint_model_group_)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "IK solution violates joint bounds");
      return;
    }

    std::vector<double> command_positions;
    robot_state_->copyJointGroupPositions(
      joint_model_group_, command_positions);

    if (joint_velocity_limit_enabled_) {
      const double dt = 1.0 / command_rate_hz_;
      command_positions = command_limiter_.update(
        command_positions, current_joint_positions_, max_joint_velocity_, dt);
    } else {
      command_limiter_.reset();
    }

    trajectory_msgs::msg::JointTrajectory command;
    command.header.stamp = get_clock()->now();
    command.joint_names = joint_names_;

    // 这里只填位置与到达时长，由下游关节轨迹控制器进行时间插值和位置执行。
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = std::move(command_positions);
    point.time_from_start =
      rclcpp::Duration::from_seconds(command_horizon_sec_);
    command.points.push_back(std::move(point));
    command_publisher_->publish(command);
  }

  std::string group_name_;
  s622_dual_admittance::JointPositionLimiter command_limiter_;
  std::string tip_link_;
  std::vector<std::string> joint_names_;
  std::vector<double> max_joint_velocity_;
  double command_rate_hz_{100.0};
  double ik_timeout_sec_{0.005};
  double target_timeout_sec_{0.05};
  double status_timeout_sec_{0.3};
  double joint_state_timeout_sec_{0.05};
  double command_horizon_sec_{0.03};
  bool joint_velocity_limit_enabled_{true};

  bool have_target_{false};
  bool enabled_{false};
  bool have_joint_state_{false};
  bool status_running_{false};
  geometry_msgs::msg::PoseStamped latest_target_;
  std::vector<double> current_joint_positions_;
  std::chrono::steady_clock::time_point last_target_time_;
  std::chrono::steady_clock::time_point last_joint_state_time_;
  std::chrono::steady_clock::time_point last_status_time_;

  robot_model_loader::RobotModelLoaderPtr robot_model_loader_;
  moveit::core::RobotModelPtr robot_model_;
  const moveit::core::JointModelGroup * joint_model_group_{nullptr};
  std::unique_ptr<moveit::core::RobotState> robot_state_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
    target_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
    joint_state_subscription_;
  rclcpp::Subscription<s622_dual_interfaces::msg::AdmittanceStatus>::SharedPtr
    status_subscription_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr
    command_publisher_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
  rclcpp::TimerBase::SharedPtr command_timer_;
};

}  // namespace s622_dual_admittance

/**
 * @brief 程序入口：初始化 ROS、创建执行器、加载模型并处理回调。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数列表，可包含 ROS 参数和名称重映射选项。
 * @return 正常退出返回 0；initialize() 抛出标准异常时记录错误并返回 1。
 * @details 自动声明启动配置中的参数，构造节点后再调用 initialize()，最后
 * 通过 rclcpp::spin() 处理订阅、服务和定时器回调，结束时关闭 ROS。
 */
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node =
    std::make_shared<s622_dual_admittance::CartesianPoseExecutor>(options);

  try {
    node->initialize();
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      node->get_logger(),
      "Cartesian executor initialization failed: %s",
      error.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
