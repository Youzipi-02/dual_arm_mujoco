"""ROS 2 Action server for right-arm planar spiral search."""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

from geometry_msgs.msg import TwistStamped, WrenchStamped
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from s622_dual_interfaces.action import PlanarSearch
from s622_dual_interfaces.msg import AdmittanceStatus
from s622_dual_interfaces.srv import ConfigureAdmittance
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from .planar_search_core import (
    PlanarSearchConfig,
    PlanarSearchCore,
    SearchCommand,
    SearchPhase,
    WrenchSample,
)


class PlanarSearchNode(Node):
    """Coordinate force contact, spiral velocity and safe controller handoff."""

    FAULT_STATES = {
        "STALE_WRENCH",
        "STALE_COMMAND",
        "TF_UNAVAILABLE",
        "CALC_ERROR",
    }

    def __init__(self) -> None:
        """Load parameters and create the planar-search ROS interfaces."""
        super().__init__("planar_search_node")
        self._group = ReentrantCallbackGroup()
        self._load_parameters()
        self._core = PlanarSearchCore(self._config)
        self._lock = threading.Lock()
        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._latest_wrench: Optional[WrenchSample] = None
        self._wrench_received_at = 0.0
        self._last_wrench_stamp_ns = 0
        self._admittance_status = ""
        self._admittance_detail = ""
        self._status_received_at = 0.0
        self._grasp_from_sensor = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=False
        )
        self._velocity_publisher = self.create_publisher(
            TwistStamped,
            self._velocity_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._state_publisher = self.create_publisher(String, "~/state", state_qos)
        self._wrench_subscription = self.create_subscription(
            WrenchStamped,
            self._wrench_topic,
            self._on_wrench,
            rclpy.qos.qos_profile_sensor_data,
            callback_group=self._group,
        )
        self._status_subscription = self.create_subscription(
            AdmittanceStatus,
            self._status_topic,
            self._on_admittance_status,
            state_qos,
            callback_group=self._group,
        )
        self._configure_client = self.create_client(
            ConfigureAdmittance,
            self._configure_service,
            callback_group=self._group,
        )
        self._enable_client = self.create_client(
            SetBool,
            self._enable_service,
            callback_group=self._group,
        )
        self._action_server = ActionServer(
            self,
            PlanarSearch,
            self._action_name,
            self._execute,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self._group,
        )
        self._publish_state("IDLE", "等待平面搜孔请求")

    def _parameter(self, name: str, default):
        """Declare a parameter once and return its configured value."""
        return self.declare_parameter(name, default).value

    def _load_parameters(self) -> None:
        """Read ROS parameters and construct the validated core configuration."""
        self._update_rate = float(self._parameter("update_rate_hz", 50.0))
        self._wrench_timeout = float(self._parameter("wrench_timeout", 0.05))
        self._service_timeout = float(self._parameter("service_timeout", 3.0))
        self._status_timeout = float(self._parameter("status_timeout", 0.5))
        self._wrench_sign = float(self._parameter("wrench_sign", 1.0))
        self._grasp_frame = str(
            self._parameter("command_frame", "right_grasp_frame")
        )
        self._sensor_frame = str(
            self._parameter("sensor_frame", "right_ft_sensor_link")
        )
        self._action_name = str(
            self._parameter("action_name", "/planar_search")
        )
        self._wrench_topic = str(self._parameter(
            "wrench_topic", "/right_ft_sensor_processing/wrench_filtered"
        ))
        self._velocity_topic = str(self._parameter(
            "velocity_topic", "/right_admittance/nominal_velocity_command"
        ))
        self._status_topic = str(self._parameter(
            "status_topic", "/right_admittance/status"
        ))
        self._configure_service = str(self._parameter(
            "configure_service", "/right_admittance/configure"
        ))
        self._enable_service = str(self._parameter(
            "enable_service", "/right_admittance/enable"
        ))

        if (
            not math.isfinite(self._update_rate)
            or self._update_rate <= 0.0
            or not math.isfinite(self._wrench_timeout)
            or self._wrench_timeout <= 0.0
            or not math.isfinite(self._service_timeout)
            or self._service_timeout <= 0.0
            or not math.isfinite(self._status_timeout)
            or self._status_timeout <= 0.0
        ):
            raise ValueError("rates and timeouts must be finite and positive")
        if self._wrench_sign not in (-1.0, 1.0):
            raise ValueError("wrench_sign must be either -1.0 or 1.0")
        if not self._grasp_frame or not self._sensor_frame:
            raise ValueError("sensor and command frames must not be empty")

        self._config = PlanarSearchConfig(
            spacing=float(self._parameter("spacing", 0.00015)),
            tangential_speed=float(
                self._parameter("tangential_speed", 0.002)
            ),
            max_radius=float(self._parameter("max_radius", 0.020)),
            approach_speed=float(self._parameter("approach_speed", 0.001)),
            max_approach_distance=float(
                self._parameter("max_approach_distance", 0.010)
            ),
            desired_contact_force=float(
                self._parameter("desired_contact_force", 10.0)
            ),
            contact_threshold=float(
                self._parameter("contact_threshold", 9.0)
            ),
            hole_force_threshold=float(
                self._parameter("hole_force_threshold", 1.0)
            ),
            contact_stable_duration=float(
                self._parameter("contact_stable_duration", 0.3)
            ),
            hole_confirm_duration=float(
                self._parameter("hole_confirm_duration", 0.1)
            ),
            timeout=float(self._parameter("timeout", 30.0)),
            force_limit=float(self._parameter("force_limit", 40.0)),
            torque_limit=float(self._parameter("torque_limit", 2.0)),
            contact_offset_z=float(self._parameter("contact_offset_z", 0.0)),
        )

    def _accept_goal(self, goal_request) -> GoalResponse:
        """Accept one executable search at a time."""
        if not goal_request.execute:
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _accept_cancel(_goal_handle) -> CancelResponse:
        """Allow the task layer to cancel an active search."""
        return CancelResponse.ACCEPT

    def _on_admittance_status(self, message: AdmittanceStatus) -> None:
        """Cache the latest right-admittance status heartbeat."""
        with self._lock:
            self._admittance_status = message.state
            self._admittance_detail = message.detail
            self._status_received_at = time.monotonic()

    @staticmethod
    def _rotate(vector, quaternion):
        """Rotate a three-vector by a normalized quaternion."""
        qx, qy, qz, qw = quaternion
        vx, vy, vz = vector
        dot_uv = qx * vx + qy * vy + qz * vz
        dot_uu = qx * qx + qy * qy + qz * qz
        cross = (
            qy * vz - qz * vy,
            qz * vx - qx * vz,
            qx * vy - qy * vx,
        )
        scale = qw * qw - dot_uu
        return (
            2.0 * dot_uv * qx + scale * vx + 2.0 * qw * cross[0],
            2.0 * dot_uv * qy + scale * vy + 2.0 * qw * cross[1],
            2.0 * dot_uv * qz + scale * vz + 2.0 * qw * cross[2],
        )

    @staticmethod
    def _cross(left, right):
        """Return the cross product of two three-vectors."""
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def _on_wrench(self, message: WrenchStamped) -> None:
        """Transform a processed sensor wrench into ``right_grasp_frame``."""
        if message.header.frame_id != self._sensor_frame:
            return
        source_time = Time.from_msg(message.header.stamp)
        source_stamp_ns = source_time.nanoseconds
        source_age = (self.get_clock().now() - source_time).nanoseconds * 1.0e-9
        if (
            source_stamp_ns == 0
            or not math.isfinite(source_age)
            or source_age < -0.01
            or source_age > self._wrench_timeout
        ):
            return
        with self._lock:
            if source_stamp_ns <= self._last_wrench_stamp_ns:
                return
        try:
            if self._grasp_from_sensor is None:
                self._grasp_from_sensor = self._tf_buffer.lookup_transform(
                    self._grasp_frame, self._sensor_frame, Time()
                )
            transform = self._grasp_from_sensor.transform
            quaternion = (
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            )
            norm = math.sqrt(sum(value * value for value in quaternion))
            if norm <= 1.0e-12:
                return
            quaternion = tuple(value / norm for value in quaternion)
            force = self._rotate(
                (
                    message.wrench.force.x,
                    message.wrench.force.y,
                    message.wrench.force.z,
                ),
                quaternion,
            )
            rotated_torque = self._rotate(
                (
                    message.wrench.torque.x,
                    message.wrench.torque.y,
                    message.wrench.torque.z,
                ),
                quaternion,
            )
            lever = (
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            )
            lever_torque = self._cross(lever, force)
            values = tuple(
                self._wrench_sign * value
                for value in (
                    *force,
                    rotated_torque[0] + lever_torque[0],
                    rotated_torque[1] + lever_torque[1],
                    rotated_torque[2] + lever_torque[2],
                )
            )
            sample = WrenchSample(*values)
        except (TransformException, ValueError):
            return
        with self._lock:
            if source_stamp_ns <= self._last_wrench_stamp_ns:
                return
            self._latest_wrench = sample
            self._wrench_received_at = time.monotonic()
            self._last_wrench_stamp_ns = source_stamp_ns

    def _interfaces_ready(self) -> bool:
        """Return whether all controller handoff services are available."""
        return all(
            client.service_is_ready()
            for client in (
                self._configure_client,
                self._enable_client,
            )
        )

    def _wait_for_interfaces(self) -> bool:
        """Wait for all required services within the configured deadline."""
        deadline = time.monotonic() + self._service_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self._interfaces_ready():
                return True
            time.sleep(0.02)
        return False

    def _call(self, client, request, heartbeat_command=None):
        """Call a service without blocking subscription callbacks."""
        future = client.call_async(request)
        deadline = time.monotonic() + self._service_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                response = future.result()
                if response is None or not response.success:
                    detail = "empty service response"
                    if response is not None:
                        detail = response.message
                    raise RuntimeError(detail)
                return response
            if heartbeat_command is not None:
                self._publish_velocity(heartbeat_command)
            time.sleep(0.01)
        raise RuntimeError("service response timed out")

    def _configure_admittance(
        self, desired_force_y: float, heartbeat_command=None
    ) -> None:
        """Set the right-arm force reference and require live XYZ velocity."""
        request = ConfigureAdmittance.Request()
        request.nominal_velocity = [0.0] * 6
        request.desired_wrench = [
            0.0,
            desired_force_y,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        request.require_velocity_command = True
        self._call(
            self._configure_client, request,
            heartbeat_command=heartbeat_command,
        )

    def _publish_velocity(
        self, command: SearchCommand, *, fresh_feedback: bool = True
    ) -> None:
        """Publish approach or spiral velocity in the right grasp frame."""
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._grasp_frame
        if fresh_feedback:
            message.twist.linear.x = command.velocity_x
            message.twist.linear.y = command.velocity_y
            message.twist.linear.z = command.velocity_z
        self._velocity_publisher.publish(message)

    def _publish_state(self, phase: str, detail: str) -> None:
        """Publish a transient-local human-readable search state."""
        message = String()
        message.data = f"{phase}: {detail}"
        self._state_publisher.publish(message)

    def _snapshot(self):
        """Read wrench and controller status caches atomically."""
        with self._lock:
            return (
                self._latest_wrench,
                self._wrench_received_at,
                self._admittance_status,
                self._admittance_detail,
                self._status_received_at,
            )

    def _cleanup(self, controller_enabled: bool) -> str:
        """Stop the contact chain through its coordinated enable endpoint."""
        errors = []
        self._publish_velocity(SearchCommand(
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            desired_force_y=self._config.desired_contact_force,
            phase=SearchPhase.CANCELED,
            radius=self._core.radius,
        ))
        if controller_enabled:
            try:
                self._call(self._enable_client, SetBool.Request(data=False))
            except Exception as error:
                errors.append(f"disable admittance: {error}")
        return "; ".join(errors)

    def _execute(self, goal_handle):
        """Execute one bounded planar search and always return controller ownership."""
        result = PlanarSearch.Result()
        controller_enabled = False
        command = self._core.start()
        self._publish_state(command.phase.value, "准备导纳控制接口")
        try:
            if not self._wait_for_interfaces():
                raise RuntimeError("planar search controller interfaces unavailable")
            self._configure_admittance(command.desired_force_y)
            configured_force_y = command.desired_force_y
            # A failed or timed-out enable can have reached the controller; always clean it up.
            controller_enabled = True
            self._call(self._enable_client, SetBool.Request(data=True))
            self._publish_state(command.phase.value, "沿 -Y 建立 +Fy 接触")

            period = 1.0 / self._update_rate
            previous = time.monotonic()
            started = previous
            while rclpy.ok():
                cycle_started = time.monotonic()
                dt = cycle_started - previous
                previous = cycle_started
                if goal_handle.is_cancel_requested:
                    command = self._core.cancel()
                    self._publish_velocity(command)
                    goal_handle.canceled()
                    result.reason = command.reason
                    result.final_radius = command.radius
                    return result
                if dt > max(0.1, 5.0 * period):
                    raise RuntimeError("planar search update loop stalled")

                wrench, wrench_time, status, detail, status_time = self._snapshot()
                if status in self.FAULT_STATES:
                    raise RuntimeError(f"admittance {status}: {detail}")

                status_fresh = (
                    status == "RUNNING"
                    and cycle_started - status_time <= self._status_timeout
                )
                wrench_fresh = (
                    wrench is not None
                    and cycle_started - wrench_time <= self._wrench_timeout
                    and wrench_time >= started
                )
                if status_fresh and wrench_fresh:
                    command = self._core.update(max(dt, 1.0e-6), wrench)
                    self._publish_velocity(command)
                    if command.desired_force_y != configured_force_y:
                        self._configure_admittance(
                            command.desired_force_y,
                            heartbeat_command=command,
                        )
                        configured_force_y = command.desired_force_y
                        previous = time.monotonic()
                elif cycle_started - started > self._service_timeout:
                    raise RuntimeError("fresh wrench or RUNNING status unavailable")
                else:
                    self._publish_velocity(command, fresh_feedback=False)

                feedback = PlanarSearch.Feedback()
                feedback.phase = command.phase.value
                feedback.radius = command.radius
                feedback.force_y = 0.0 if wrench is None else wrench.force_y
                goal_handle.publish_feedback(feedback)
                self._publish_state(
                    command.phase.value,
                    f"approach={self._core.approach_distance:.4f} m, "
                    f"radius={command.radius:.6f} m, Fy={feedback.force_y:.3f} N",
                )

                if command.phase == SearchPhase.FOUND:
                    cleanup_error = self._cleanup(controller_enabled)
                    controller_enabled = False
                    if cleanup_error:
                        raise RuntimeError(
                            "controller cleanup failed: " + cleanup_error
                        )
                    goal_handle.succeed()
                    result.success = True
                    result.reason = command.reason
                    result.final_radius = command.radius
                    return result
                if command.phase == SearchPhase.FAILED:
                    goal_handle.abort()
                    result.reason = command.reason
                    result.final_radius = command.radius
                    return result
                remaining = period - (time.monotonic() - cycle_started)
                if remaining > 0.0:
                    time.sleep(remaining)
            raise RuntimeError("ROS context stopped")
        except Exception as error:
            self.get_logger().error(f"Planar search failed: {error}")
            if goal_handle.is_active:
                goal_handle.abort()
            result.reason = str(error)
            result.final_radius = self._core.radius
            return result
        finally:
            cleanup_error = self._cleanup(controller_enabled)
            if cleanup_error:
                self.get_logger().error(cleanup_error)
                result.success = False
                prefix = f"{result.reason}; " if result.reason else ""
                result.reason = prefix + "controller cleanup failed: " + cleanup_error
            self._publish_state("IDLE", "平面搜孔已结束，等待下一次请求")
            with self._goal_lock:
                self._goal_active = False

    def destroy_node(self):
        """Destroy the Action server before the node resources."""
        self._action_server.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the planar search node with concurrent action and sensor callbacks."""
    rclpy.init(args=args)
    node = PlanarSearchNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
