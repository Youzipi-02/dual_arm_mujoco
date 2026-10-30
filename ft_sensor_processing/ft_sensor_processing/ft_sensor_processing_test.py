#!/usr/bin/env python3

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

"""依次向左臂轨迹控制器发送用户预设的关节姿态."""

import math
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_NAMES = [
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_j6",
]

# 在这里填写测试姿态，关节角单位均为 rad。
# 每个姿态必须按照 left_j1 到 left_j6 的顺序包含 6 个数值。
PRESET_POSITIONS = [
    (
        "初始姿态",
        [1.0472, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    ),
    (
        "测试姿态一",
        [0.29849236, -1.983992277, 1.67081340, -1.76251875, -1.05756731, -1.5720626],
    ),
    (
        "测试姿态二",
        [-1.6072708, -1.765544708, -1.0875004, -1.91495171, 1.955052222, -1.4265058],
    ),
    (
        "测试姿态二",
        [-1.7541856, -1.678789401, -1.1706773, -2.29815367, -1.31929620, -0.8515357],
    ),
    (
        "测试姿态二",
        [-0.8924098, -2.197264112, -0.8071163, -1.01049590, -2.28728363, -2.0404157],
    ),
]

# 每次运动的轨迹时间，以及到位后的停留时间。
MOVE_DURATION_SEC = 5.0
HOLD_DURATION_SEC = 3.0
CONTROLLER_WAIT_TIMEOUT_SEC = 15.0

# 来自当前机器人 URDF。用于阻止明显越界的测试指令。
JOINT_LIMITS = [
    (-3.0543, 3.0543),
    (-4.6251, 1.4835),
    (-2.8274, 2.8274),
    (-4.6251, 1.4835),
    (-3.0543, 3.0543),
    (-3.0543, 3.0543),
]


class MoveJoints(Node):
    """发布预设左臂关节轨迹."""

    def __init__(self) -> None:
        """创建关节轨迹发布者."""
        super().__init__("ft_sensor_processing_test")
        self.publisher = self.create_publisher(
            JointTrajectory,
            "/left_arm_controller/joint_trajectory",
            1,
        )

    def move_to_position(self, positions, duration: float) -> None:
        """发送一个六关节目标姿态."""
        validate_position(positions)

        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in positions]
        point.time_from_start = Duration(seconds=duration).to_msg()
        trajectory.points.append(point)

        self.publisher.publish(trajectory)
        self.get_logger().info(f"已发送关节目标：{point.positions}")


def validate_position(positions) -> None:
    """检查姿态维数、有限性和 URDF 关节限位."""
    if len(positions) != len(JOINT_NAMES):
        raise ValueError("每个预设姿态必须包含 6 个关节角")

    for name, position, limits in zip(
        JOINT_NAMES,
        positions,
        JOINT_LIMITS,
    ):
        value = float(position)
        lower, upper = limits
        if not math.isfinite(value):
            raise ValueError(f"{name} 的目标值不是有限数值")
        if value < lower or value > upper:
            raise ValueError(
                f"{name}={value:.4f} rad 超出限位 "
                f"[{lower:.4f}, {upper:.4f}] rad"
            )


def spin_for(node: Node, duration: float) -> None:
    """在等待期间继续处理 ROS 事件."""
    end_time = time.monotonic() + duration
    while rclpy.ok() and time.monotonic() < end_time:
        remaining = end_time - time.monotonic()
        rclpy.spin_once(node, timeout_sec=min(0.1, remaining))


def wait_for_controller(node: MoveJoints) -> bool:
    """等待轨迹控制器订阅命令话题."""
    deadline = time.monotonic() + CONTROLLER_WAIT_TIMEOUT_SEC
    while rclpy.ok() and time.monotonic() < deadline:
        if node.publisher.get_subscription_count() > 0:
            return True
        rclpy.spin_once(node, timeout_sec=0.5)

    node.get_logger().error(
        "等待 /left_arm_controller/joint_trajectory 超时，请检查控制器状态"
    )
    return False


def main(args=None) -> int:
    """依次执行所有预设姿态后退出."""
    rclpy.init(args=args)
    node = MoveJoints()
    exit_code = 0

    try:
        if not PRESET_POSITIONS:
            node.get_logger().error("PRESET_POSITIONS 不能为空")
            return 1

        for _, positions in PRESET_POSITIONS:
            validate_position(positions)

        node.get_logger().info("正在等待左臂轨迹控制器……")
        if not wait_for_controller(node):
            return 1

        for index, (pose_name, positions) in enumerate(
            PRESET_POSITIONS,
            start=1,
        ):
            node.get_logger().info(
                f"执行姿态 {index}/{len(PRESET_POSITIONS)}：{pose_name}"
            )
            node.move_to_position(positions, MOVE_DURATION_SEC)
            spin_for(node, MOVE_DURATION_SEC + HOLD_DURATION_SEC)

        node.get_logger().info("全部预设姿态已发送完成")
    except KeyboardInterrupt:
        node.get_logger().warning("用户中断测试")
        exit_code = 130
    except (TypeError, ValueError) as error:
        node.get_logger().error(f"预设姿态无效：{error}")
        exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
