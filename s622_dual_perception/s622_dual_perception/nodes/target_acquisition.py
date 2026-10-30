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

"""Acquire one fresh, continuously visible pair of world-frame targets."""

import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from s622_dual_interfaces.action import AcquireTargets
from vision_msgs.msg import Detection3DArray

from s622_dual_perception.target_confirmation import (
    ContinuousDetectionGate, extract_best_targets, stamp_seconds,
)


class TargetAcquisition(Node):
    """Own target validation, continuity and snapshot selection."""

    def __init__(self):
        super().__init__("target_acquisition")
        group = ReentrantCallbackGroup()
        topic = self.declare_parameter(
            "detections_topic", "/yolo_obb/detections_3d"
        ).value
        self._lock = threading.Lock()
        self._reserved = False
        self._gate = None
        self._expected_frame = ""
        self._maximum_message_age = 0.0
        self._started_ros = 0.0
        self._subscription = self.create_subscription(
            Detection3DArray, topic, self._on_detections,
            qos_profile_sensor_data, callback_group=group,
        )
        self._action = ActionServer(
            self, AcquireTargets, "~/acquire_targets", self._acquire,
            goal_callback=self._goal, cancel_callback=self._cancel,
            callback_group=group,
        )

    def _goal(self, goal):
        if (goal.confirm_duration_sec <= 0 or goal.maximum_gap_sec <= 0
                or goal.maximum_message_age_sec <= 0
                or goal.minimum_confirmations < 1 or goal.timeout_sec <= 0
                or not goal.expected_frame):
            return GoalResponse.REJECT
        with self._lock:
            if self._reserved:
                return GoalResponse.REJECT
            self._reserved = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel(_handle):
        return CancelResponse.ACCEPT

    def _on_detections(self, message):
        with self._lock:
            if self._gate is None:
                return
            stamp = stamp_seconds(message)
            now_ros = self.get_clock().now().nanoseconds * 1e-9
            age = now_ros - stamp
            if (stamp <= 0.0 or stamp < self._started_ros
                    or age > self._maximum_message_age
                    or age < -0.1):
                self._gate.reset("检测消息时间无效或不属于本轮请求")
                return
            targets, missing = extract_best_targets(message, self._expected_frame)
            self._gate.update(stamp, time.monotonic(), targets, missing)

    def _acquire(self, handle):
        result = AcquireTargets.Result()
        goal = handle.request
        deadline = time.monotonic() + goal.timeout_sec
        try:
            with self._lock:
                self._gate = ContinuousDetectionGate(
                    goal.confirm_duration_sec, goal.maximum_gap_sec,
                    goal.minimum_confirmations,
                )
                self._expected_frame = goal.expected_frame
                self._maximum_message_age = goal.maximum_message_age_sec
                self._started_ros = self.get_clock().now().nanoseconds * 1e-9
            while rclpy.ok():
                if handle.is_cancel_requested:
                    result.reason = "target acquisition cancelled"
                    handle.canceled()
                    return result
                with self._lock:
                    self._gate.expire(time.monotonic())
                    locked = self._gate.locked
                    status = self._gate.status()
                if locked is not None:
                    result.success = True
                    result.wire = locked.wire
                    result.terminal = locked.terminal
                    handle.succeed()
                    return result
                if time.monotonic() >= deadline:
                    result.reason = "target confirmation timed out: " + status
                    handle.abort()
                    return result
                feedback = AcquireTargets.Feedback()
                feedback.status = status
                handle.publish_feedback(feedback)
                time.sleep(0.1)
            result.reason = "ROS shutdown"
            handle.abort()
            return result
        finally:
            with self._lock:
                self._gate = None
                self._reserved = False


def main(args=None):
    rclpy.init(args=args)
    node = TargetAcquisition()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
