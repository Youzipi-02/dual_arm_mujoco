from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder

import xacro


def generate_launch_description():
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    # 1. 找到安装后的 MoveIt 配置包。
    config_share = Path(
        get_package_share_directory("s622_dual_moveit_config")
    )

    robot_xacro = (
        config_share
        / "config"
        / "s622_dual_descriptions.urdf.xacro"
    )

    initial_positions = (
        config_share / "config" / "initial_positions.yaml"
    )

    controllers_yaml = (
        config_share / "config" / "ros2_controllers.yaml"
    )

    # 2. 展开 xacro，得到包含 ros2_control 配置的完整 URDF。
    robot_urdf = xacro.process_file(
        str(robot_xacro),
        mappings={
            "initial_positions_file": str(initial_positions),
        },
    ).toxml()

    robot_description = {
        "robot_description": ParameterValue(
            robot_urdf,
            value_type=str,
        )
    }

    # 3. 发布机器人描述，并根据 /joint_states 发布 TF。
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {
                "use_sim_time": use_sim_time,
                "publish_frequency": 250.0,
            },
        ],
    )

    # 4. 启动 MuJoCo 仿真与 controller_manager。
    mujoco_control = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            str(controllers_yaml),
            {"use_sim_time": use_sim_time},
        ],
        # Humble 的 controller_manager 从这个话题获取 URDF。
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        # 关闭 MuJoCo 或控制节点退出时，结束整个 launch。
        on_exit=Shutdown(),
    )

    # 5. 等待 controller_manager 就绪，依次加载并激活控制器。
    controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "left_ft_sensor_broadcaster",
            "right_ft_sensor_broadcaster",
            "left_arm_controller",
            "right_arm_controller",
            "left_gripper_controller",
            "right_gripper_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--param-file",
            str(controllers_yaml),
        ],
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "s622_dual_descriptions",
            package_name="s622_dual_moveit_config",
        )
        .robot_description(
            file_path="config/s622_dual_descriptions.urdf.xacro",
            mappings={
                "initial_positions_file": str(
                    config_share / "config/initial_positions.yaml"
                ),
            },
        )
        .robot_description_semantic(
            file_path="config/s622_dual_descriptions.srdf"
        )
        .robot_description_kinematics(
            file_path="config/kinematics.yaml"
        )
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml"
        )
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl"],
        )
        .to_moveit_configs()
    )

    # 当前 sensors_3d.yaml 是 Kinect 示例配置。
    # 暂不加载深度相机占据地图，不影响六维力传感器。
    moveit_config.sensors_3d = {}

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,
                "allow_trajectory_execution": True,
                "publish_robot_description": True,
                "publish_robot_description_semantic": True,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="moveit_rviz",
        output="screen",
        arguments=[
            "-d",
            str(config_share / "config/moveit.rviz"),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": use_sim_time},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        robot_state_publisher,
        mujoco_control,
        controller_spawner,
        move_group,
        rviz,
    ])