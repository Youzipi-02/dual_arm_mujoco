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

"""启动 S622 全局相机 YOLO-OBB 检测和目标确认节点."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """生成 YOLO-OBB RGB-D 检测和目标确认启动描述."""
    config_file = LaunchConfiguration("config_file")
    model_path = LaunchConfiguration("model_path")
    device = LaunchConfiguration("device")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    target_frame = LaunchConfiguration("target_frame")
    debug_image_topic = LaunchConfiguration("debug_image_topic")
    enabled = LaunchConfiguration("enabled")
    use_sim_time = LaunchConfiguration("use_sim_time")

    def create_perception_nodes(context):
        overrides = {
            "model.path": model_path,
            "model.device": device,
            "tf.target_frame": target_frame,
            "output.debug_image_topic": debug_image_topic,
            "inference.enabled": ParameterValue(enabled, value_type=bool),
            "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
        }
        for parameter, argument in (
            ("input.rgb_topic", rgb_topic),
            ("input.depth_topic", depth_topic),
            ("input.camera_info_topic", camera_info_topic),
        ):
            topic = argument.perform(context)
            if topic:
                overrides[parameter] = ParameterValue(topic, value_type=str)
        return [
            Node(
                package="s622_dual_perception",
                executable="yolo_obb_detector",
                name="yolo_obb_detector",
                output="screen",
                emulate_tty=True,
                parameters=[config_file, overrides],
            ),
            Node(
                package="s622_dual_perception",
                executable="target_acquisition",
                name="target_acquisition",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": ParameterValue(
                            use_sim_time, value_type=bool
                        )
                    },
                ],
            ),
        ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("s622_dual_perception"),
                        "config",
                        "obb_detector.yaml",
                    ]
                ),
                description="YOLO-OBB 检测参数文件。",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("s622_dual_perception"),
                        "models",
                        "yolo_obb.pt",
                    ]
                ),
                description="YOLO-OBB 权重文件路径。",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="",
                description="推理设备，例如 cpu、0 或 cuda:0;空值自动选择。",
            ),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="",
                description="全局相机 RGB 图像话题；空值使用配置文件。",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="",
                description="与 RGB 对齐的深度图话题；空值使用配置文件。",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="",
                description="全局相机内参话题；空值使用配置文件。",
            ),
            DeclareLaunchArgument(
                "target_frame",
                default_value="world",
                description="三维检测转换后的目标坐标系。",
            ),
            DeclareLaunchArgument(
                "debug_image_topic",
                default_value="/yolo_obb/debug_image",
                description="带 OBB 标注的调试图像话题。",
            ),
            DeclareLaunchArgument(
                "enabled",
                default_value="true",
                description="节点启动后是否立即执行 OBB 推理。",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="是否使用 Gazebo 仿真时钟。",
            ),
            OpaqueFunction(function=create_perception_nodes),
        ]
    )
