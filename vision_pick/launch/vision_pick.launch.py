"""Launch only the pick capability; simulation and perception are composed by bringup."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('execute', default_value='true'),
        Node(package='vision_pick', executable='vision_pick', name='vision_pick', output='screen',
             parameters=[PathJoinSubstitution([FindPackageShare('s622_dual_descriptions'), 'config', 'shared_model.yaml']),
                         PathJoinSubstitution([FindPackageShare('vision_pick'), 'config', 'vision_pick.yaml']),
                         {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                          'pick.execute': ParameterValue(LaunchConfiguration('execute'), value_type=bool)}])])
