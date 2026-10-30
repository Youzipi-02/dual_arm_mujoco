"""Continuous assembly control; no generic motion execution server."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim = LaunchConfiguration('use_sim_time')

    start_mode = LaunchConfiguration('start_control_mode')

    includes = [IncludeLaunchDescription(PythonLaunchDescriptionSource(PathJoinSubstitution([
        FindPackageShare('s622_dual_admittance'), 'launch', filename])),
        launch_arguments={
            'use_sim_time': sim,
            **({'enable_service': 'enable_controller'} if filename == 'dual_admittance.launch.py' else {}),
        }.items())
        for filename in ('dual_admittance.launch.py', 'dual_cartesian_executor.launch.py')]
    
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('start_control_mode', default_value='true'),
        *includes,
        Node(package='s622_dual_assembly', executable='control_mode', name='assembly_control',
             output='screen', condition=IfCondition(start_mode),
             parameters=[{'use_sim_time': ParameterValue(sim, value_type=bool), 'followers_enabled': True}])])
