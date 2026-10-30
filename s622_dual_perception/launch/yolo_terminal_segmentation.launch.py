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

"""启动左相机铜端子孔洞和端面分割节点."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """生成左相机 YOLO segmentation 启动描述."""
    config_file = LaunchConfiguration("config_file")
    model_path = LaunchConfiguration("model_path")
    device = LaunchConfiguration("device")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    target_frame = LaunchConfiguration("target_frame")
    enabled = LaunchConfiguration("enabled")
    use_sim_time = LaunchConfiguration("use_sim_time")

    segmenter = Node(
        package="s622_dual_perception",
        executable="yolo_terminal_segmenter",
        name="yolo_terminal_segmenter",
        output="screen",
        emulate_tty=True,
        parameters=[
            config_file,
            {
                "model.path": model_path,
                "model.device": device,
                "input.rgb_topic": rgb_topic,
                "input.depth_topic": depth_topic,
                "input.camera_info_topic": camera_info_topic,
                "tf.target_frame": target_frame,
                "inference.enabled": ParameterValue(enabled, value_type=bool),
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("s622_dual_perception"),
                        "config",
                        "terminal_segmenter.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("s622_dual_perception"),
                        "models",
                        "yolo_seg.pt",
                    ]
                ),
            ),
            DeclareLaunchArgument("device", default_value=""),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/cameras/left_camera/rgb/image_raw",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/cameras/left_camera/depth/image_raw",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/cameras/left_camera/rgb/camera_info",
            ),
            DeclareLaunchArgument("target_frame", default_value="world"),
            DeclareLaunchArgument("enabled", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            segmenter,
        ]
    )
