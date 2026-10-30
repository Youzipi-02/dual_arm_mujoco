"""启动左右连续笛卡尔位置执行器."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """Create two namespaced Cartesian pose executors."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    package_share = FindPackageShare("s622_dual_admittance")
    left_config = PathJoinSubstitution(
        [package_share, "config", "left_cartesian_executor.yaml"]
    )
    right_config = PathJoinSubstitution(
        [package_share, "config", "right_cartesian_executor.yaml"]
    )

    moveit_config = (
        MoveItConfigsBuilder(
            "s622_dual_descriptions",
            package_name="s622_dual_moveit_config",
        )
        .to_moveit_configs()
    )
    robot_model_parameters = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.robot_description_kinematics,
    ]

    common_parameters = {
        "use_sim_time": ParameterValue(use_sim_time, value_type=bool)
    }

    left_executor = Node(
        package="s622_dual_admittance",
        executable="cartesian_pose_executor",
        namespace="left_cartesian_follower",
        name="cartesian_pose_executor",
        output="screen",
        parameters=[left_config, *robot_model_parameters, common_parameters],
        remappings=[
            ("target_pose", "/left_admittance/target_pose"),
            ("admittance_status", "/left_admittance/status"),
            ("joint_states", "/joint_states"),
            (
                "joint_trajectory",
                "/left_arm_controller/joint_trajectory",
            ),
        ],
    )

    right_executor = Node(
        package="s622_dual_admittance",
        executable="cartesian_pose_executor",
        namespace="right_cartesian_follower",
        name="cartesian_pose_executor",
        output="screen",
        parameters=[right_config, *robot_model_parameters, common_parameters],
        remappings=[
            ("target_pose", "/right_admittance/target_pose"),
            ("admittance_status", "/right_admittance/status"),
            ("joint_states", "/joint_states"),
            (
                "joint_trajectory",
                "/right_arm_controller/joint_trajectory",
            ),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            left_executor,
            right_executor,
        ]
    )
