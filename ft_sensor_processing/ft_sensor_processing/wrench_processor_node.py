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

"""ROS 2 node for force-torque gravity compensation and filtering."""

from collections import deque
import math
from threading import Lock
import time
from typing import Optional, Sequence

from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_updater import Updater
from geometry_msgs.msg import WrenchStamped
import numpy as np
import rclpy
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
)
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from ft_sensor_processing.butterworth_filter import (
    StreamingButterworthFilter,
)
from ft_sensor_processing.gravity_compensator import GravityCompensator


class FTSensorProcessingNode(Node):
    """Compensate payload gravity, optionally tare, filter, and republish."""

    def __init__(self) -> None:
        """Declare parameters and create the complete processing pipeline."""
        super().__init__("wrench_processor")
        self._declare_parameters()
        self._load_parameters()
        self._validate_parameters()

        self._processing_group = MutuallyExclusiveCallbackGroup()
        self._tf_group = ReentrantCallbackGroup()
        self._joint_group = ReentrantCallbackGroup()
        self._tf_lock = Lock()
        self._joint_lock = Lock()
        self._statistics_lock = Lock()

        self._compensator = GravityCompensator(
            payload_weight_n=self._payload_weight_n,
            center_of_mass_m=self._payload_com_m,
            force_bias_n=self._force_bias_n,
            torque_bias_nm=self._torque_bias_nm,
        )
        self._filter = StreamingButterworthFilter(
            sampling_frequency_hz=self._filter_sampling_frequency_hz,
            cutoff_frequency_hz=self._filter_cutoff_frequency_hz,
            order=self._filter_order,
            channels=6,
        )

        self._runtime_offset = np.zeros(6, dtype=float)
        self._tare_buffer = deque(maxlen=self._tare_window_samples)
        self._gravity_wrench: Optional[np.ndarray] = None
        self._tf_stamp_ns: Optional[int] = None
        self._latest_max_joint_velocity: Optional[float] = None
        self._joint_state_received_at: Optional[float] = None
        self._warning_times = {}

        self._received_count = 0
        self._published_count = 0
        self._tf_drop_count = 0
        self._invalid_count = 0
        self._frame_drop_count = 0
        self._last_diagnostic_count = 0
        self._last_diagnostic_time = time.monotonic()
        self._input_rate_hz = 0.0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._compensated_publisher = self.create_publisher(
            WrenchStamped,
            "wrench_compensated",
            sensor_qos,
        )
        self._filtered_publisher = self.create_publisher(
            WrenchStamped,
            "wrench_filtered",
            sensor_qos,
        )
        self._wrench_subscription = self.create_subscription(
            WrenchStamped,
            "wrench_raw",
            self._wrench_callback,
            sensor_qos,
            callback_group=self._processing_group,
        )
        self._joint_subscription = self.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            sensor_qos,
            callback_group=self._joint_group,
        )

        self._tf_buffer = Buffer(
            cache_time=Duration(seconds=2.0),
            node=self,
        )
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=False,
        )
        self._tf_timer = self.create_timer(
            1.0 / self._tf_update_rate_hz,
            self._update_gravity_wrench,
            callback_group=self._tf_group,
        )

        self._tare_service = self.create_service(
            Trigger,
            "tare",
            self._tare_callback,
            callback_group=self._processing_group,
        )
        self._clear_tare_service = self.create_service(
            Trigger,
            "clear_tare",
            self._clear_tare_callback,
            callback_group=self._processing_group,
        )
        self._reset_filter_service = self.create_service(
            Trigger,
            "reset_filter",
            self._reset_filter_callback,
            callback_group=self._processing_group,
        )

        self._diagnostic_updater = Updater(
            self,
            period=self._diagnostic_period_sec,
        )
        self._diagnostic_updater.setHardwareID(self._sensor_frame)
        self._diagnostic_updater.add(
            "force_torque_processing",
            self._diagnose,
        )

        filter_description = (
            f"{self._filter_order}阶 Butterworth, "
            f"{self._filter_cutoff_frequency_hz:.1f} Hz"
            if self._filter_enabled
            else "关闭"
        )
        self.get_logger().info(
            "六维力处理节点已启动："
            f"{self._gravity_frame} -> {self._sensor_frame}，"
            f"重量 {self._payload_weight_n:.6f} N，"
            f"滤波器 {filter_description}"
        )

    def _declare_parameters(self) -> None:
        """Declare every user-facing setting with a description and range."""
        self.declare_parameter(
            "gravity_frame",
            "world",
            ParameterDescriptor(description="定义重力方向的世界坐标系。"),
        )
        self.declare_parameter(
            "sensor_frame",
            "left_ft_sensor_link",
            ParameterDescriptor(description="六维力数据所在的传感器坐标系。"),
        )
        self._declare_nonnegative_double(
            "payload.weight_N",
            5.240600,
            "传感器下游全部结构的已辨识重量，单位 N。",
        )
        self.declare_parameter(
            "payload.com_m",
            [-0.0023576, -0.0106920, 0.0074035],
            ParameterDescriptor(
                description="传感器原点指向下游结构重心的向量，单位 m。"
            ),
        )
        self.declare_parameter(
            "bias.force_N",
            [0.0, 0.0, 0.0],
            ParameterDescriptor(description="三轴力固定零偏，单位 N。"),
        )
        self.declare_parameter(
            "bias.torque_Nm",
            [0.0, 0.0, 0.0],
            ParameterDescriptor(description="三轴力矩固定零偏，单位 N·m。"),
        )
        self.declare_parameter(
            "filter.enabled",
            True,
            ParameterDescriptor(description="是否发布低通滤波结果。"),
        )
        self._declare_positive_double(
            "filter.sampling_frequency_Hz",
            500.0,
            "滤波器设计采样频率，单位 Hz。",
        )
        self._declare_positive_double(
            "filter.cutoff_frequency_Hz",
            20.0,
            "Butterworth低通截止频率，单位 Hz。",
        )
        self._declare_positive_integer(
            "filter.order",
            2,
            "Butterworth滤波器阶数。",
            maximum=8,
        )
        self._declare_positive_double(
            "tf.update_rate_Hz",
            250.0,
            "缓存重力方向的TF查询频率，单位 Hz。",
        )
        self._declare_positive_double(
            "tf.max_age_sec",
            0.02,
            "力消息与缓存TF允许的最大时间差，单位 s。",
        )
        self._declare_positive_integer(
            "tare.window_samples",
            250,
            "清零时使用的最近重力补偿样本数。",
            maximum=100000,
        )
        self._declare_positive_double(
            "tare.max_force_std_N",
            0.10,
            "允许清零的单轴最大力标准差，单位 N。",
        )
        self._declare_positive_double(
            "tare.max_torque_std_Nm",
            0.01,
            "允许清零的单轴最大力矩标准差，单位 N·m。",
        )
        self._declare_nonnegative_double(
            "tare.max_joint_velocity_rad_s",
            0.01,
            "允许清零的目标机械臂最大关节速度，单位 rad/s。",
        )
        self._declare_positive_double(
            "tare.joint_state_timeout_sec",
            0.20,
            "清零时关节状态允许的最大接收延迟，单位 s。",
        )
        self.declare_parameter(
            "tare.joint_names",
            [f"left_j{index}" for index in range(1, 7)],
            ParameterDescriptor(description="清零静止检查使用的关节名称。"),
        )
        self._declare_positive_double(
            "diagnostics.period_sec",
            1.0,
            "处理节点诊断信息发布周期，单位 s。",
        )

    def _declare_positive_double(
        self, name: str, default: float, description: str
    ) -> None:
        """Declare a strictly positive double parameter."""
        descriptor = ParameterDescriptor(description=description)
        descriptor.floating_point_range = [
            FloatingPointRange(
                from_value=1.0e-6,
                to_value=1.0e6,
                step=0.0,
            )
        ]
        self.declare_parameter(name, default, descriptor)

    def _declare_nonnegative_double(
        self, name: str, default: float, description: str
    ) -> None:
        """Declare a non-negative double parameter."""
        descriptor = ParameterDescriptor(description=description)
        descriptor.floating_point_range = [
            FloatingPointRange(
                from_value=0.0,
                to_value=1.0e6,
                step=0.0,
            )
        ]
        self.declare_parameter(name, default, descriptor)

    def _declare_positive_integer(
        self,
        name: str,
        default: int,
        description: str,
        maximum: int,
    ) -> None:
        """Declare a bounded positive integer parameter."""
        descriptor = ParameterDescriptor(description=description)
        descriptor.integer_range = [
            IntegerRange(from_value=1, to_value=maximum, step=1)
        ]
        self.declare_parameter(name, default, descriptor)

    def _load_parameters(self) -> None:
        """Copy parameters into typed runtime attributes."""
        self._gravity_frame = str(
            self.get_parameter("gravity_frame").value
        )
        self._sensor_frame = str(self.get_parameter("sensor_frame").value)
        self._payload_weight_n = float(
            self.get_parameter("payload.weight_N").value
        )
        self._payload_com_m = self._vector_parameter("payload.com_m", 3)
        self._force_bias_n = self._vector_parameter("bias.force_N", 3)
        self._torque_bias_nm = self._vector_parameter("bias.torque_Nm", 3)
        self._filter_enabled = bool(
            self.get_parameter("filter.enabled").value
        )
        self._filter_sampling_frequency_hz = float(
            self.get_parameter("filter.sampling_frequency_Hz").value
        )
        self._filter_cutoff_frequency_hz = float(
            self.get_parameter("filter.cutoff_frequency_Hz").value
        )
        self._filter_order = int(self.get_parameter("filter.order").value)
        self._tf_update_rate_hz = float(
            self.get_parameter("tf.update_rate_Hz").value
        )
        self._tf_max_age_sec = float(
            self.get_parameter("tf.max_age_sec").value
        )
        self._tare_window_samples = int(
            self.get_parameter("tare.window_samples").value
        )
        self._tare_max_force_std_n = float(
            self.get_parameter("tare.max_force_std_N").value
        )
        self._tare_max_torque_std_nm = float(
            self.get_parameter("tare.max_torque_std_Nm").value
        )
        self._tare_max_joint_velocity = float(
            self.get_parameter("tare.max_joint_velocity_rad_s").value
        )
        self._joint_state_timeout_sec = float(
            self.get_parameter("tare.joint_state_timeout_sec").value
        )
        self._tare_joint_names = tuple(
            str(value)
            for value in self.get_parameter("tare.joint_names").value
        )
        self._diagnostic_period_sec = float(
            self.get_parameter("diagnostics.period_sec").value
        )

    def _vector_parameter(self, name: str, size: int) -> np.ndarray:
        """Read a finite fixed-length double-array parameter."""
        values = np.asarray(self.get_parameter(name).value, dtype=float)
        if values.shape != (size,):
            raise ValueError(f"参数 {name} 必须包含 {size} 个数值")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"参数 {name} 包含非有限数值")
        return values.copy()

    def _validate_parameters(self) -> None:
        """Validate relationships that parameter descriptors cannot express."""
        if not self._gravity_frame or not self._sensor_frame:
            raise ValueError("gravity_frame 和 sensor_frame 不能为空")
        if self._gravity_frame == self._sensor_frame:
            raise ValueError("gravity_frame 和 sensor_frame 不能相同")
        if len(self._tare_joint_names) == 0:
            raise ValueError("tare.joint_names 不能为空")
        if len(set(self._tare_joint_names)) != len(self._tare_joint_names):
            raise ValueError("tare.joint_names 不能包含重复名称")
        nyquist = 0.5 * self._filter_sampling_frequency_hz
        if self._filter_cutoff_frequency_hz >= nyquist:
            raise ValueError("滤波截止频率必须小于采样频率的一半")

    def _update_gravity_wrench(self) -> None:
        """Refresh the cached gravity wrench from the latest TF rotation."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._sensor_frame,
                self._gravity_frame,
                Time(),
            )
            rotation = self._rotation_matrix(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )
            gravity_wrench = self._compensator.gravity_wrench(rotation)
            stamp_ns = Time.from_msg(transform.header.stamp).nanoseconds
            if stamp_ns == 0:
                stamp_ns = self.get_clock().now().nanoseconds
        except (TransformException, ValueError) as error:
            self._warn_throttled(
                "tf_lookup",
                f"无法更新重力方向TF：{error}",
            )
            return

        with self._tf_lock:
            self._gravity_wrench = gravity_wrench
            self._tf_stamp_ns = stamp_ns

    @staticmethod
    def _rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convert a normalized xyzw quaternion to a rotation matrix."""
        quaternion = np.array([x, y, z, w], dtype=float)
        if not np.all(np.isfinite(quaternion)):
            raise ValueError("TF四元数包含非有限数值")
        norm = np.linalg.norm(quaternion)
        if norm < 1.0e-12:
            raise ValueError("TF四元数范数为零")
        x, y, z, w = quaternion / norm
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    def _wrench_callback(self, message: WrenchStamped) -> None:
        """Run the bounded per-sample compensation and filter path."""
        with self._statistics_lock:
            self._received_count += 1

        if message.header.frame_id != self._sensor_frame:
            with self._statistics_lock:
                self._frame_drop_count += 1
            self._warn_throttled(
                "frame_mismatch",
                "丢弃坐标系不匹配的数据："
                f"收到 '{message.header.frame_id}'，"
                f"期望 '{self._sensor_frame}'",
            )
            return

        measured = self._wrench_to_array(message)
        if not np.all(np.isfinite(measured)):
            with self._statistics_lock:
                self._invalid_count += 1
            self._warn_throttled("invalid_wrench", "丢弃包含NaN或Inf的六维力数据")
            return

        with self._tf_lock:
            gravity_wrench = (
                None
                if self._gravity_wrench is None
                else self._gravity_wrench.copy()
            )
            tf_stamp_ns = self._tf_stamp_ns

        if gravity_wrench is None or tf_stamp_ns is None:
            with self._statistics_lock:
                self._tf_drop_count += 1
            self._warn_throttled("tf_missing", "等待首个有效重力方向TF")
            return

        message_stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
        if message_stamp_ns == 0:
            message_stamp_ns = self.get_clock().now().nanoseconds
        tf_age_sec = abs(message_stamp_ns - tf_stamp_ns) * 1.0e-9
        if tf_age_sec > self._tf_max_age_sec:
            with self._statistics_lock:
                self._tf_drop_count += 1
            self._warn_throttled(
                "tf_stale",
                f"丢弃TF时间差 {tf_age_sec:.4f} s 的六维力数据",
            )
            return

        model_compensated = self._compensator.compensate(
            measured,
            gravity_wrench,
        )
        self._tare_buffer.append(model_compensated.copy())
        compensated = model_compensated - self._runtime_offset
        filtered = (
            self._filter.update(compensated)
            if self._filter_enabled
            else compensated.copy()
        )

        self._compensated_publisher.publish(
            self._array_to_wrench(message, compensated)
        )
        self._filtered_publisher.publish(
            self._array_to_wrench(message, filtered)
        )
        with self._statistics_lock:
            self._published_count += 1

    @staticmethod
    def _wrench_to_array(message: WrenchStamped) -> np.ndarray:
        """Convert a WrenchStamped payload to Fx,Fy,Fz,Mx,My,Mz."""
        return np.array(
            [
                message.wrench.force.x,
                message.wrench.force.y,
                message.wrench.force.z,
                message.wrench.torque.x,
                message.wrench.torque.y,
                message.wrench.torque.z,
            ],
            dtype=float,
        )

    @staticmethod
    def _array_to_wrench(
        source: WrenchStamped,
        values: Sequence[float],
    ) -> WrenchStamped:
        """Create a WrenchStamped while preserving source stamp and frame."""
        data = np.asarray(values, dtype=float)
        output = WrenchStamped()
        output.header = source.header
        output.wrench.force.x = float(data[0])
        output.wrench.force.y = float(data[1])
        output.wrench.force.z = float(data[2])
        output.wrench.torque.x = float(data[3])
        output.wrench.torque.y = float(data[4])
        output.wrench.torque.z = float(data[5])
        return output

    def _joint_state_callback(self, message: JointState) -> None:
        """Cache the maximum configured arm velocity for safe taring."""
        if len(message.name) != len(message.velocity):
            return
        velocity_by_name = dict(zip(message.name, message.velocity))
        if not all(name in velocity_by_name for name in self._tare_joint_names):
            return
        velocities = [
            float(velocity_by_name[name]) for name in self._tare_joint_names
        ]
        if not all(math.isfinite(value) for value in velocities):
            return
        with self._joint_lock:
            self._latest_max_joint_velocity = max(
                abs(value) for value in velocities
            )
            self._joint_state_received_at = time.monotonic()

    def _tare_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Estimate residual offset from a complete stationary sample window."""
        if len(self._tare_buffer) < self._tare_window_samples:
            response.success = False
            response.message = (
                "样本不足："
                f"{len(self._tare_buffer)}/{self._tare_window_samples}"
            )
            return response

        with self._joint_lock:
            joint_velocity = self._latest_max_joint_velocity
            received_at = self._joint_state_received_at
        if received_at is None or joint_velocity is None:
            response.success = False
            response.message = "尚未收到完整的左臂关节速度"
            return response
        if time.monotonic() - received_at > self._joint_state_timeout_sec:
            response.success = False
            response.message = "关节状态已过期，拒绝清零"
            return response
        if joint_velocity > self._tare_max_joint_velocity:
            response.success = False
            response.message = (
                f"机械臂仍在运动：最大速度 {joint_velocity:.5f} rad/s"
            )
            return response

        samples = np.vstack(self._tare_buffer)
        standard_deviation = np.std(samples, axis=0)
        if np.any(standard_deviation[:3] > self._tare_max_force_std_n):
            response.success = False
            response.message = (
                "力信号波动过大：最大标准差 "
                f"{np.max(standard_deviation[:3]):.6f} N"
            )
            return response
        if np.any(standard_deviation[3:] > self._tare_max_torque_std_nm):
            response.success = False
            response.message = (
                "力矩信号波动过大：最大标准差 "
                f"{np.max(standard_deviation[3:]):.6f} N·m"
            )
            return response

        self._runtime_offset = np.mean(samples, axis=0)
        self._filter.reset(np.zeros(6, dtype=float))
        self._tare_buffer.clear()
        response.success = True
        response.message = (
            "已保存重力补偿后的残余偏置："
            + np.array2string(self._runtime_offset, precision=6)
        )
        self.get_logger().info(response.message)
        return response

    def _clear_tare_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Clear only the runtime tare; keep identified calibration values."""
        self._runtime_offset.fill(0.0)
        self._filter.reset()
        self._tare_buffer.clear()
        response.success = True
        response.message = "已清除运行时残余偏置"
        return response

    def _reset_filter_callback(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Reset filter history without modifying any compensation parameter."""
        self._filter.reset()
        response.success = True
        response.message = "已重置低通滤波器状态"
        return response

    def _diagnose(self, status):
        """Populate periodic health and throughput diagnostics."""
        now = time.monotonic()
        with self._statistics_lock:
            elapsed = max(now - self._last_diagnostic_time, 1.0e-9)
            delta = self._received_count - self._last_diagnostic_count
            self._input_rate_hz = delta / elapsed
            self._last_diagnostic_count = self._received_count
            self._last_diagnostic_time = now
            received = self._received_count
            published = self._published_count
            tf_drops = self._tf_drop_count
            invalid = self._invalid_count
            frame_drops = self._frame_drop_count

        with self._tf_lock:
            tf_available = self._gravity_wrench is not None
            tf_stamp_ns = self._tf_stamp_ns

        current_ns = self.get_clock().now().nanoseconds
        tf_age_sec = None
        if tf_stamp_ns is not None and current_ns > 0:
            tf_age_sec = abs(current_ns - tf_stamp_ns) * 1.0e-9

        if not tf_available:
            status.summary(DiagnosticStatus.ERROR, "没有可用的重力方向TF")
        elif tf_age_sec is not None and tf_age_sec > self._tf_max_age_sec:
            status.summary(DiagnosticStatus.WARN, "重力方向TF时间差过大")
        elif self._input_rate_hz < 0.8 * self._filter_sampling_frequency_hz:
            status.summary(DiagnosticStatus.WARN, "六维力输入频率偏低")
        else:
            status.summary(DiagnosticStatus.OK, "六维力处理正常")

        status.add("input_rate_hz", f"{self._input_rate_hz:.2f}")
        status.add(
            "expected_rate_hz",
            f"{self._filter_sampling_frequency_hz:.2f}",
        )
        status.add("received_messages", str(received))
        status.add("published_messages", str(published))
        status.add("tf_dropped_messages", str(tf_drops))
        status.add("invalid_messages", str(invalid))
        status.add("frame_mismatch_messages", str(frame_drops))
        status.add("tare_buffer_samples", str(len(self._tare_buffer)))
        status.add("filter_enabled", str(self._filter_enabled))
        status.add("sensor_frame", self._sensor_frame)
        if tf_age_sec is not None:
            status.add("latest_tf_age_sec", f"{tf_age_sec:.6f}")
        return status

    def _warn_throttled(
        self,
        key: str,
        message: str,
        period_sec: float = 5.0,
    ) -> None:
        """Log a warning at most once per key and wall-clock period."""
        now = time.monotonic()
        previous = self._warning_times.get(key, -math.inf)
        if now - previous >= period_sec:
            self._warning_times[key] = now
            self.get_logger().warning(message)


def main(args=None) -> None:
    """Run the processing node with separate TF and data callback threads."""
    rclpy.init(args=args)
    node = FTSensorProcessingNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
