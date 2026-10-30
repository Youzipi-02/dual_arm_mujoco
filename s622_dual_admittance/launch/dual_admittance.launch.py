"""启动左右导纳控制器."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create two namespaced admittance controller nodes."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_service = LaunchConfiguration("enable_service")

    package_share = FindPackageShare("s622_dual_admittance")

    left_admittance_config = PathJoinSubstitution(
        [package_share, "config", "left_admittance.yaml"]
    )
    right_admittance_config = PathJoinSubstitution(
        [package_share, "config", "right_admittance.yaml"]
    )

    left_admittance_node = Node(
        package="s622_dual_admittance",
        executable="admittance_node",
        namespace="left_admittance",
        name="admittance_controller",
        output="screen",
        parameters=[
            left_admittance_config,
            {
                "use_sim_time": ParameterValue(
                    use_sim_time, value_type=bool
                )
            },
        ],
        remappings=[
            ("enable", enable_service),
            (
                "wrench",
                "/left_ft_sensor_processing/wrench_filtered",
            ),
        ],
    )

    right_admittance_node = Node(
        package="s622_dual_admittance",
        executable="admittance_node",
        namespace="right_admittance",
        name="admittance_controller",
        output="screen",
        parameters=[
            right_admittance_config,
            {
                "use_sim_time": ParameterValue(
                    use_sim_time, value_type=bool
                )
            },
        ],
        remappings=[
            ("enable", enable_service),
            (
                "wrench",
                "/right_ft_sensor_processing/wrench_filtered",
            ),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "enable_service",
                default_value="enable",
                description="Low-level enable service; full bringup uses enable_controller.",
            ),
            left_admittance_node,
            right_admittance_node,
        ]
    )
