"""Track attached workpiece frames from locked grasp targets and live EEF TF."""

from dataclasses import dataclass
import math

from geometry_msgs.msg import Pose, TransformStamped
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from .grasp_geometry import GraspPoseSet, compose_pose, pose_from_transform, relative_pose


@dataclass(frozen=True)
class GraspRelations:
    wire_model: Pose
    terminal_model: Pose
    wire_object_frame: Pose
    terminal_object_frame: Pose
    error_detail: str


class GraspTracker:
    """Publish object frames and capture EEF-to-object relations after grasp."""

    def __init__(self, node, planning_frame, left_eef, right_eef,
                 wire_frame, terminal_frame):
        self._node = node
        self._planning_frame = planning_frame
        self._left_eef = left_eef
        self._right_eef = right_eef
        self._wire_frame = wire_frame
        self._terminal_frame = terminal_frame
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node, spin_thread=False)
        self._broadcaster = TransformBroadcaster(node)
        self.reset()

    def reset(self):
        self.poses = None
        self.relations = None
        self._tf_unavailable = False

    @property
    def attached(self):
        return self.relations is not None

    def set_targets(self, poses: GraspPoseSet):
        self.poses = poses
        self.relations = None

    def lookup_eef_pose(self, left: bool):
        link = self._left_eef if left else self._right_eef
        try:
            transform = self._buffer.lookup_transform(self._planning_frame, link, Time())
        except TransformException:
            return None
        stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
        age = self._node.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp <= 0.0 or age < -0.1 or age > 1.0:
            return None
        return pose_from_transform(transform)

    @staticmethod
    def _pose_error(actual: Pose, expected: Pose):
        position = math.sqrt(
            (actual.position.x - expected.position.x) ** 2 +
            (actual.position.y - expected.position.y) ** 2 +
            (actual.position.z - expected.position.z) ** 2
        )
        dot = abs(
            actual.orientation.x * expected.orientation.x +
            actual.orientation.y * expected.orientation.y +
            actual.orientation.z * expected.orientation.z +
            actual.orientation.w * expected.orientation.w
        )
        angle = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
        return position, angle

    def capture(self, position_tolerance: float, angular_tolerance: float):
        """Return None while TF is missing; otherwise store checked relations."""
        if self.poses is None:
            raise ValueError("locked visual poses are unavailable")
        left_eef = self.lookup_eef_pose(True)
        right_eef = self.lookup_eef_pose(False)
        if left_eef is None or right_eef is None:
            return None
        wire_position, wire_angle = self._pose_error(left_eef, self.poses.wire_grasp)
        terminal_position, terminal_angle = self._pose_error(
            right_eef, self.poses.terminal_grasp)
        detail = (
            f"wire={wire_position * 1000.0:.1f}mm/"
            f"{math.degrees(wire_angle):.1f}deg, "
            f"terminal={terminal_position * 1000.0:.1f}mm/"
            f"{math.degrees(terminal_angle):.1f}deg"
        )
        if (max(wire_position, terminal_position) > position_tolerance or
                max(wire_angle, terminal_angle) > angular_tolerance):
            raise ValueError(f"end effectors did not reach visual grasp targets: {detail}")
        self.relations = GraspRelations(
            wire_model=relative_pose(left_eef, self.poses.wire_model),
            terminal_model=relative_pose(right_eef, self.poses.terminal_model),
            wire_object_frame=relative_pose(left_eef, self.poses.wire_object_frame),
            terminal_object_frame=relative_pose(
                right_eef, self.poses.terminal_object_frame),
            error_detail=detail,
        )
        return self.relations

    @staticmethod
    def _transform(pose, stamp, parent, child):
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation = pose.orientation
        return transform

    def broadcast(self):
        if self.poses is None:
            return
        wire_pose = self.poses.wire_object_frame
        terminal_pose = self.poses.terminal_object_frame
        if self.relations is not None:
            left_eef = self.lookup_eef_pose(True)
            right_eef = self.lookup_eef_pose(False)
            if left_eef is None or right_eef is None:
                if not self._tf_unavailable:
                    self._node.get_logger().warning(
                        "工件中心坐标系跟随暂停：末端 TF 暂时不可用")
                    self._tf_unavailable = True
                return
            if self._tf_unavailable:
                self._node.get_logger().info("末端 TF 已恢复，继续发布跟随工件坐标系")
                self._tf_unavailable = False
            wire_pose = compose_pose(left_eef, self.relations.wire_object_frame)
            terminal_pose = compose_pose(right_eef, self.relations.terminal_object_frame)
        stamp = self._node.get_clock().now().to_msg()
        self._broadcaster.sendTransform([
            self._transform(wire_pose, stamp, self._planning_frame, self._wire_frame),
            self._transform(terminal_pose, stamp, self._planning_frame, self._terminal_frame),
        ])
