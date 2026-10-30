# Copyright 2026 dsl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch the left and right force-torque processing pipelines."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _processor(side, config_file, use_sim_time):
    """Create one side-specific processor using the shared implementation."""
    return Node(
        package="ft_sensor_processing",
        executable="wrench_processor",
        namespace=f"{side}_ft_sensor_processing",
        name="wrench_processor",
        output="screen",
        parameters=[config_file, {"use_sim_time": use_sim_time}],
        remappings=[
            ("wrench_raw", f"/{side}_ft_sensor_broadcaster/wrench"),
            ("joint_states", "/joint_states"),
        ],
    )


def generate_launch_description():
    """Create independent left and right processors in one launch file."""
    use_sim_time = LaunchConfiguration("use_sim_time")
    left_config_file = LaunchConfiguration("left_config_file")
    right_config_file = LaunchConfiguration("right_config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="是否使用 /clock 仿真时间。",
            ),
            DeclareLaunchArgument(
                "left_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("ft_sensor_processing"),
                        "config",
                        "left_ft_sensor.yaml",
                    ]
                ),
                description="左侧六维力处理参数文件。",
            ),
            DeclareLaunchArgument(
                "right_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("ft_sensor_processing"),
                        "config",
                        "right_ft_sensor.yaml",
                    ]
                ),
                description="右侧六维力处理参数文件。",
            ),
            _processor("left", left_config_file, use_sim_time),
            _processor("right", right_config_file, use_sim_time),
        ]
    )
