#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <s622_dual_interfaces/msg/admittance_status.hpp>
#include <s622_dual_interfaces/srv/configure_admittance.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "s622_dual_admittance/admittance_core.hpp"

namespace s622_dual_admittance
{

class AdmittanceNode final : public rclcpp::Node
{
public:
  AdmittanceNode()
  : Node("admittance_controller"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameters();   // 声明参数
    load_parameters();      // 加载参数

    core_ = std::make_unique<AdmittanceCore>(core_parameters_);     // 创建导纳控制器实例

    // 订阅六维力传感器话题
    wrench_subscription_ =
      create_subscription<geometry_msgs::msg::WrenchStamped>(
      "wrench",
      rclcpp::SensorDataQoS(),
      std::bind(
        &AdmittanceNode::wrench_callback,
        this,
        std::placeholders::_1));

    nominal_velocity_subscription_ =
      create_subscription<geometry_msgs::msg::TwistStamped>(
      "nominal_velocity_command",
      rclcpp::QoS(1).reliable(),
      std::bind(
        &AdmittanceNode::nominal_velocity_callback,
        this,
        std::placeholders::_1));

    // 发布目标位姿话题
    target_pose_publisher_ =
      create_publisher<geometry_msgs::msg::PoseStamped>(
      "target_pose",
      rclcpp::QoS(1).reliable());
    status_publisher_ = create_publisher<s622_dual_interfaces::msg::AdmittanceStatus>(
      "status", rclcpp::QoS(1).reliable().transient_local());
    configure_service_ = create_service<s622_dual_interfaces::srv::ConfigureAdmittance>(
      "configure", std::bind(
        &AdmittanceNode::configure_callback, this,
        std::placeholders::_1, std::placeholders::_2));

    // 创建服务端 enable_service_，用于启用或禁用导纳控制器
    enable_service_ = create_service<std_srvs::srv::SetBool>(
      "enable",
      std::bind(
        &AdmittanceNode::enable_callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    const auto period =
      rclcpp::Duration::from_seconds(1.0 / control_rate_hz_);

    // 使用节点时钟创建控制定时器。use_sim_time=true 时随 /clock 推进，
    // 避免 Gazebo 暂停或低于实时运行时被墙钟误判为力数据超时。
    control_timer_ = rclcpp::create_timer(
      this,
      get_clock(),
      period,
      std::bind(&AdmittanceNode::control_callback, this));

    // 初始化名义位姿和传感器到抓取的变换为单位变换
    nominal_pose_.setIdentity();
    grasp_from_sensor_.setIdentity();
    publish_status("DISABLED", "controller initialized");

    RCLCPP_INFO(
      get_logger(),
      "Position admittance node ready: %s -> %s, rate %.1f Hz",
      sensor_frame_.c_str(),
      grasp_frame_.c_str(),
      control_rate_hz_);
  }

private:
  void declare_parameters()
  {
    declare_parameter<std::string>("reference_frame", "world");
    declare_parameter<std::string>(
      "sensor_frame", "left_ft_sensor_link");
    declare_parameter<std::string>(
      "grasp_frame", "left_grasp_frame");

    declare_parameter<double>("control_rate_hz", 250.0);
    declare_parameter<double>("wrench_timeout_sec", 0.05);
    declare_parameter<double>("velocity_command_timeout_sec", 0.10);
    declare_parameter<double>("wrench_sign", 1.0);
    declare_parameter<bool>("limits_enabled", true);

    declare_parameter<std::vector<bool>>(
      "enabled_axes",
      {true, true, true, true, true, true});

    declare_parameter<std::vector<double>>(
      "mass",
      {3.0, 3.0, 3.0, 0.08, 0.08, 0.08});

    declare_parameter<std::vector<double>>(
      "damping",
      {83.0, 72.0, 83.0, 1.66, 1.66, 1.66});

    declare_parameter<std::vector<double>>(
      "stiffness",
      {400.0, 300.0, 400.0, 6.0, 6.0, 6.0});

    declare_parameter<std::vector<double>>(
      "deadband",
      {0.3, 0.3, 0.3, 0.02, 0.02, 0.02});

    declare_parameter<std::vector<double>>(
      "max_velocity",
      {0.01, 0.01, 0.01, 0.15, 0.15, 0.15});

    declare_parameter<std::vector<double>>(
      "max_displacement",
      {0.005, 0.015, 0.005, 0.087, 0.087, 0.087});

    declare_parameter<std::vector<double>>(
      "nominal_velocity",
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    declare_parameter<std::vector<double>>(
      "desired_wrench",
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  }

  // 加载6维向量参数,将ROS 2 的动态数组转换成程序使用的 std::array<double, 6>
  Vector6 load_vector6(const std::string & name) const
  {
    const auto values = get_parameter(name).as_double_array();

    if (values.size() != 6) {
      throw std::invalid_argument(
              name + " must contain exactly 6 values");
    }

    Vector6 result{};

    for (std::size_t i = 0; i < 6; ++i) {
      if (!std::isfinite(values[i])) {
        throw std::invalid_argument(
                name + " contains NaN or Inf");
      }

      result[i] = values[i];
    }

    return result;
  }

  // 某个方向为 false 时，AdmittanceCore 会把该方向的导纳速度和位移都保持为零
  std::array<bool, 6> load_enabled_axes() const
  {
    const auto values =
      get_parameter("enabled_axes").as_bool_array();

    if (values.size() != 6) {
      throw std::invalid_argument(
              "enabled_axes must contain exactly 6 values");
    }

    std::array<bool, 6> result{};

    for (std::size_t i = 0; i < 6; ++i) {
      result[i] = values[i];
    }

    return result;
  }

  void load_parameters()
  {
    reference_frame_ =
      get_parameter("reference_frame").as_string();
    sensor_frame_ =
      get_parameter("sensor_frame").as_string();
    grasp_frame_ =
      get_parameter("grasp_frame").as_string();

    control_rate_hz_ =
      get_parameter("control_rate_hz").as_double();
    wrench_timeout_sec_ =
      get_parameter("wrench_timeout_sec").as_double();
    velocity_command_timeout_sec_ =
      get_parameter("velocity_command_timeout_sec").as_double();
    wrench_sign_ =
      get_parameter("wrench_sign").as_double();

    if (reference_frame_.empty() ||
      sensor_frame_.empty() ||
      grasp_frame_.empty())
    {
      throw std::invalid_argument(
              "frame parameters must not be empty");
    }

    // AdmittanceCore currently accepts dt <= 0.02 s.
    if (!std::isfinite(control_rate_hz_) ||
      control_rate_hz_ < 50.0)
    {
      throw std::invalid_argument(
              "control_rate_hz must be at least 50 Hz");
    }

    if (!std::isfinite(wrench_timeout_sec_) ||
      wrench_timeout_sec_ <= 0.0)
    {
      throw std::invalid_argument(
              "wrench_timeout_sec must be positive");
    }

    if (!std::isfinite(velocity_command_timeout_sec_) ||
      velocity_command_timeout_sec_ <= 0.0)
    {
      throw std::invalid_argument(
              "velocity_command_timeout_sec must be positive");
    }

    if (wrench_sign_ != 1.0 && wrench_sign_ != -1.0) {
      throw std::invalid_argument(
              "wrench_sign must be either 1.0 or -1.0");
    }

    core_parameters_.mass = load_vector6("mass");
    core_parameters_.damping = load_vector6("damping");
    core_parameters_.stiffness = load_vector6("stiffness");
    core_parameters_.deadband = load_vector6("deadband");
    core_parameters_.max_velocity =
      load_vector6("max_velocity");
    core_parameters_.max_displacement =
      load_vector6("max_displacement");
    core_parameters_.enabled_axes =
      load_enabled_axes();
    core_parameters_.limits_enabled =
      get_parameter("limits_enabled").as_bool();

    nominal_velocity_ = load_vector6("nominal_velocity");
    desired_wrench_ = load_vector6("desired_wrench");
  }

  /*
   * @brief 服务回调函数，用于启用或禁用导纳控制器
   *
   * @param request 服务请求，包含一个布尔值，表示是否启用导纳控制器
   * @param response 服务响应，包含一个布尔值，表示操作是否成功，以及一条消息
   *
   * @details 若收到false，则停止发布目标位姿、清除 TF 就绪状态、清除已有力数据、将导纳位移和导纳速度归零。
   * 若收到true，则清空导纳核心状态、等待一帧新的力传感器数据、设置 enabled_ = true、记录控制循环起始时间。
   */
  void enable_callback(
    const std_srvs::srv::SetBool::Request::SharedPtr request,
    std_srvs::srv::SetBool::Response::SharedPtr response)
  {
    if (!request->data) {
      enabled_ = false;
      transforms_ready_ = false;
      have_wrench_ = false;
      have_velocity_command_ = false;
      core_->reset();
      publish_status("DISABLED", "controller disabled");

      response->success = true;
      response->message = "admittance disabled";
      return;
    }

    try {
      // world/reference -> grasp：记录启用瞬间的初始抓取位姿。
      const auto initial_pose =
        tf_buffer_.lookupTransform(
        reference_frame_,
        grasp_frame_,
        tf2::TimePointZero,
        tf2::durationFromSec(0.2));

      tf2::fromMsg(initial_pose.transform, nominal_pose_);

      // grasp <- sensor：用于把传感器 wrench 转到抓取坐标系。
      const auto wrench_transform =
        tf_buffer_.lookupTransform(
        grasp_frame_,
        sensor_frame_,
        tf2::TimePointZero,
        tf2::durationFromSec(0.2));

      tf2::fromMsg(
        wrench_transform.transform,
        grasp_from_sensor_);
    } catch (const tf2::TransformException & error) {
      enabled_ = false;
      transforms_ready_ = false;

      response->success = false;
      response->message =
        std::string("TF unavailable: ") + error.what();
      publish_status("TF_UNAVAILABLE", response->message);
      return;
    }

    core_->reset();
    have_wrench_ = false;
    have_velocity_command_ = false;
    transforms_ready_ = true;
    enabled_ = true;
    last_control_time_ = now_ros_time();
    enable_time_ = last_control_time_;
    publish_status("WAIT_WRENCH", "waiting for fresh source-stamped wrench");

    response->success = true;
    response->message = "admittance enabled";
  }

  void configure_callback(
    const s622_dual_interfaces::srv::ConfigureAdmittance::Request::SharedPtr request,
    s622_dual_interfaces::srv::ConfigureAdmittance::Response::SharedPtr response)
  {
    for (std::size_t index = 0; index < 6; ++index) {
      if (!std::isfinite(request->nominal_velocity[index]) ||
        !std::isfinite(request->desired_wrench[index]))
      {
        response->message = "reference contains non-finite value";
        return;
      }
    }
    if (enabled_ &&
      (request->require_velocity_command != require_velocity_command_ ||
      !std::equal(
        request->nominal_velocity.begin(), request->nominal_velocity.end(),
        nominal_velocity_.begin())))
    {
      response->message = "only desired wrench may change while enabled";
      return;
    }
    std::copy(
      request->nominal_velocity.begin(), request->nominal_velocity.end(),
      nominal_velocity_.begin());
    std::copy(
      request->desired_wrench.begin(), request->desired_wrench.end(),
      desired_wrench_.begin());
    require_velocity_command_ = request->require_velocity_command;
    bypass_admittance_ = request->bypass_admittance;
    response->success = true;
    response->message = "reference configured";
  }

  void publish_status(const std::string & state, const std::string & detail)
  {
    s622_dual_interfaces::msg::AdmittanceStatus message;
    message.header.stamp = now_ros_time();
    message.header.frame_id = grasp_frame_;
    message.state = state;
    message.detail = detail;
    status_publisher_->publish(message);
  }

  /*
   * @brief 力传感器话题回调函数
   *
   * @param message 力传感器消息，包含力和力矩数据
   *
   * @details 如果 TF 未就绪，则忽略消息。
   * 如果消息的坐标系与预期不符，则打印警告并忽略消息。
   * 否则，将力和力矩从传感器坐标系转换到抓取坐标系，并存储在 latest_wrench_ 中。
   */
  void wrench_callback(
    const geometry_msgs::msg::WrenchStamped::SharedPtr message)
  {
    if (!transforms_ready_) {
      return;
    }

    if (message->header.frame_id != sensor_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Ignoring wrench in frame '%s'; expected '%s'",
        message->header.frame_id.c_str(),
        sensor_frame_.c_str());
      return;
    }
    const auto source_time = rclcpp::Time(message->header.stamp, get_clock()->get_clock_type());
    const double source_age = (now_ros_time() - source_time).seconds();
    if (source_time.nanoseconds() == 0 || !std::isfinite(source_age) ||
      source_age < -0.01 || source_age > wrench_timeout_sec_ ||
      source_time <= enable_time_ ||
      (have_wrench_ && source_time <= last_wrench_time_))
    {
      return;
    }

    const tf2::Vector3 force_sensor(
      message->wrench.force.x,
      message->wrench.force.y,
      message->wrench.force.z);

    const tf2::Vector3 torque_sensor(
      message->wrench.torque.x,
      message->wrench.torque.y,
      message->wrench.torque.z);

    const tf2::Vector3 force_grasp =
      grasp_from_sensor_.getBasis() * force_sensor;

    // t 是抓取原点到传感器原点的向量，表达在抓取坐标系中。
    const tf2::Vector3 torque_grasp =
      grasp_from_sensor_.getBasis() * torque_sensor +
      grasp_from_sensor_.getOrigin().cross(force_grasp);

    latest_wrench_[0] = wrench_sign_ * force_grasp.x();
    latest_wrench_[1] = wrench_sign_ * force_grasp.y();
    latest_wrench_[2] = wrench_sign_ * force_grasp.z();

    latest_wrench_[3] = wrench_sign_ * torque_grasp.x();
    latest_wrench_[4] = wrench_sign_ * torque_grasp.y();
    latest_wrench_[5] = wrench_sign_ * torque_grasp.z();
    for (const double value : latest_wrench_) {
      if (!std::isfinite(value)) {
        return;
      }
    }

    have_wrench_ = true;
    last_wrench_time_ = source_time;
  }

  void nominal_velocity_callback(
    const geometry_msgs::msg::TwistStamped::SharedPtr message)
  {
    if (!enabled_ || !require_velocity_command_) {
      return;
    }
    if (message->header.frame_id != grasp_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Ignoring velocity command in frame '%s'; expected '%s'",
        message->header.frame_id.c_str(), grasp_frame_.c_str());
      return;
    }

    const auto source_time = rclcpp::Time(
      message->header.stamp, get_clock()->get_clock_type());
    const double source_age = (now_ros_time() - source_time).seconds();
    if (source_time.nanoseconds() == 0 || !std::isfinite(source_age) ||
      source_age < -0.01 || source_age > velocity_command_timeout_sec_ ||
      source_time <= enable_time_ ||
      (have_velocity_command_ && source_time <= last_velocity_command_time_))
    {
      return;
    }

    const Vector6 command{
      message->twist.linear.x,
      message->twist.linear.y,
      message->twist.linear.z,
      message->twist.angular.x,
      message->twist.angular.y,
      message->twist.angular.z};
    for (std::size_t index = 0; index < command.size(); ++index) {
      if (!std::isfinite(command[index])) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Ignoring non-finite nominal velocity command");
        return;
      }
    }

    commanded_nominal_velocity_ = command;
    have_velocity_command_ = true;
    last_velocity_command_time_ = source_time;
  }

  /*
   * @brief 将旋转向量转换为四元数
   *
   * @param rotation_vector 旋转向量，表示旋转轴和旋转角度的乘积
   * @return 对应的四元数，表示相同的旋转
   *
   * @details 如果旋转角度接近零，则返回单位四元数。
   * 否则，将旋转向量归一化为旋转轴，并使用旋转角度构造四元数。
   */
  static tf2::Quaternion quaternion_from_rotation_vector(
    const tf2::Vector3 & rotation_vector)
  {
    const double angle = rotation_vector.length();

    tf2::Quaternion quaternion;

    if (angle < 1.0e-12) {
      quaternion.setValue(0.0, 0.0, 0.0, 1.0);
      return quaternion;
    }

    quaternion.setRotation(rotation_vector / angle, angle);
    quaternion.normalize();

    return quaternion;
  }

  /*
   * @brief 位置型导纳控制的主循环，用于计算导纳控制器的输出位姿
   *
   * @details 如果导纳控制器未启用或 TF 未就绪或没有力数据，则直接返回。
   * 如果力数据超时，则禁用导纳控制器并重置状态。
   * 如果控制周期不合法，则打印警告并重置状态。
   * 否则，将名义速度积分为名义位姿增量，并调用 AdmittanceCore 更新导纳位移和速度。
   * 最后，将名义位姿和导纳偏移组合成目标位姿，并发布到 target_pose 话题。
   */
  void control_callback()
  {
    if (!enabled_ || !transforms_ready_) {
      return;
    }

    const auto now = now_ros_time();
    if (!have_wrench_) {
      if ((now - last_control_time_).seconds() > wrench_timeout_sec_) {
        enabled_ = false;
        core_->reset();
        publish_status("STALE_WRENCH", "no fresh wrench after enable");
      }
      return;
    }

    const double wrench_age =
      (now - last_wrench_time_).seconds();

    // 如果力数据超时，则禁用导纳控制器并重置状态
    if (wrench_age > wrench_timeout_sec_) {
      RCLCPP_ERROR(
        get_logger(),
        "Wrench timed out after %.4f s; admittance disabled",
        wrench_age);

      enabled_ = false;
      core_->reset();
      publish_status("STALE_WRENCH", "source-stamped wrench timed out");
      return;
    }

    if (require_velocity_command_) {
      if (!have_velocity_command_) {
        if ((now - enable_time_).seconds() > velocity_command_timeout_sec_) {
          enabled_ = false;
          core_->reset();
          publish_status("STALE_COMMAND", "no fresh velocity command after enable");
        }
        return;
      }
      const double command_age =
        (now - last_velocity_command_time_).seconds();
      if (command_age > velocity_command_timeout_sec_) {
        enabled_ = false;
        have_velocity_command_ = false;
        core_->reset();
        publish_status("STALE_COMMAND", "nominal velocity command timed out");
        return;
      }
    }

    // 计算控制周期 dt，并检查其合法性
    const double dt = (now - last_control_time_).seconds();

    last_control_time_ = now;

    if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.02) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Invalid control period %.6f s; resetting admittance state",
        dt);

      core_->reset();
      return;
    }

    // 将局部名义速度积分为局部位姿增量。
    const Vector6 & active_nominal_velocity =
      require_velocity_command_ ? commanded_nominal_velocity_ : nominal_velocity_;
    const tf2::Vector3 nominal_translation_step(
      active_nominal_velocity[0] * dt,
      active_nominal_velocity[1] * dt,
      active_nominal_velocity[2] * dt);

    const tf2::Vector3 nominal_rotation_step(
      active_nominal_velocity[3] * dt,
      active_nominal_velocity[4] * dt,
      active_nominal_velocity[5] * dt);

    const tf2::Transform nominal_increment(
      quaternion_from_rotation_vector(nominal_rotation_step),
      nominal_translation_step);

    // 右乘表示名义速度定义在抓取坐标系中。
    nominal_pose_ = nominal_pose_ * nominal_increment;

    Vector6 displacement{};
    if (bypass_admittance_) {
      core_->reset();
    } else {
      try {
        displacement = core_->update(
          latest_wrench_,
          desired_wrench_,
          dt);
      } catch (const std::exception & error) {
        RCLCPP_ERROR(
          get_logger(),
          "Admittance update failed: %s",
          error.what());

        enabled_ = false;
        core_->reset();
        publish_status("CALC_ERROR", error.what());
        return;
      }
    }

    const tf2::Vector3 position_offset(
      displacement[0],
      displacement[1],
      displacement[2]);

    const tf2::Vector3 rotation_offset(
      displacement[3],
      displacement[4],
      displacement[5]);

    const tf2::Transform admittance_offset(
      quaternion_from_rotation_vector(rotation_offset),
      position_offset);

    // 位移和旋转偏移都定义在名义抓取坐标系中。
    const tf2::Transform command_pose =
      nominal_pose_ * admittance_offset;

    geometry_msgs::msg::PoseStamped message;
    message.header.stamp = now_ros_time();
    message.header.frame_id = reference_frame_;

    tf2::toMsg(command_pose, message.pose);
    target_pose_publisher_->publish(message);
    publish_status("RUNNING", "fresh wrench and target pose active");
  }

  rclcpp::Time now_ros_time()
  {
    return get_clock()->now();
  }

  AdmittanceParameters core_parameters_;
  std::unique_ptr<AdmittanceCore> core_;

  std::string reference_frame_;
  std::string sensor_frame_;
  std::string grasp_frame_;

  double control_rate_hz_{250.0};
  double wrench_timeout_sec_{0.05};
  double velocity_command_timeout_sec_{0.10};
  double wrench_sign_{1.0};

  Vector6 nominal_velocity_{};
  Vector6 commanded_nominal_velocity_{};
  Vector6 desired_wrench_{};
  Vector6 latest_wrench_{};

  bool enabled_{false};
  bool transforms_ready_{false};
  bool have_wrench_{false};
  bool require_velocity_command_{false};
  bool bypass_admittance_{false};
  bool have_velocity_command_{false};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  tf2::Transform nominal_pose_;
  tf2::Transform grasp_from_sensor_;

  rclcpp::Time last_control_time_;
  rclcpp::Time last_wrench_time_;
  rclcpp::Time enable_time_;
  rclcpp::Time last_velocity_command_time_;

  rclcpp::Subscription<
    geometry_msgs::msg::WrenchStamped>::SharedPtr
    wrench_subscription_;
  rclcpp::Subscription<
    geometry_msgs::msg::TwistStamped>::SharedPtr
    nominal_velocity_subscription_;

  rclcpp::Publisher<
    geometry_msgs::msg::PoseStamped>::SharedPtr
    target_pose_publisher_;
  rclcpp::Publisher<s622_dual_interfaces::msg::AdmittanceStatus>::SharedPtr status_publisher_;
  rclcpp::Service<s622_dual_interfaces::srv::ConfigureAdmittance>::SharedPtr
    configure_service_;

  rclcpp::Service<
    std_srvs::srv::SetBool>::SharedPtr
    enable_service_;

  rclcpp::TimerBase::SharedPtr control_timer_;
};

}  // namespace s622_dual_admittance

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<
      s622_dual_admittance::AdmittanceNode>());

  rclcpp::shutdown();
  return 0;
}
