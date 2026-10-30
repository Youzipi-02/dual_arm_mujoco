"""Start the S622 planar spiral-search Action server."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the parameterized planar-search node."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    config = PathJoinSubstitution(
        [FindPackageShare("s622_dual_assembly"), "config", "planar_search.yaml"]
    )
    node = Node(
        package="s622_dual_assembly",
        executable="planar_search_node",
        name="planar_search_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            config,
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        node,
    ])
