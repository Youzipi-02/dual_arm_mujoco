"""视觉抓取与双臂装配任务使用的纯几何计算工具。"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from geometry_msgs.msg import Pose, Quaternion, TransformStamped
from vision_msgs.msg import Detection3D

VectorTuple = Tuple[float, float, float]
QuaternionTuple = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TargetPair:
    """The two confirmed observations returned by perception."""

    wire: Detection3D
    terminal: Detection3D


@dataclass(frozen=True)
class GraspPoseSet:
    """由一次视觉快照生成的工件中心、碰撞体和双臂抓取目标。"""

    wire_model: Pose
    terminal_model: Pose
    wire_object_frame: Pose
    terminal_object_frame: Pose
    wire_grasp_point: Pose
    terminal_grasp_point: Pose
    wire_pregrasp: Pose
    terminal_pregrasp: Pose
    wire_grasp: Pose
    terminal_grasp: Pose


def normalize_quaternion(values: Sequence[float]) -> QuaternionTuple:
    """归一化 xyzw 四元数，并拒绝零四元数。"""

    if len(values) != 4:
        raise ValueError("a quaternion must contain four values")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm < 1.0e-12:
        raise ValueError("a quaternion cannot be zero")
    return tuple(float(value) / norm for value in values)  # type: ignore[return-value]


def quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> QuaternionTuple:
    """返回两个 xyzw 四元数的 Hamilton 积。"""

    lx, ly, lz, lw = normalize_quaternion(left)
    rx, ry, rz, rw = normalize_quaternion(right)
    return normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def quaternion_conjugate(values: Sequence[float]) -> QuaternionTuple:
    """返回归一化 xyzw 四元数的共轭。"""

    x, y, z, w = normalize_quaternion(values)
    return -x, -y, -z, w


def rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> VectorTuple:
    """使用 xyzw 四元数旋转三维向量。"""

    qx, qy, qz, qw = normalize_quaternion(quaternion)
    vx, vy, vz = (float(value) for value in vector)

    # Expanded q * v * q^-1 avoids normalizing the zero-w vector quaternion.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def pose_from_vector(values: Sequence[float], name: str = "pose") -> Pose:
    """将 ``[x, y, z, qx, qy, qz, qw]`` 转换为四元数已归一化的 Pose。"""

    if len(values) != 7:
        raise ValueError(f"{name} must contain seven values")
    pose = Pose()
    pose.position.x = float(values[0])
    pose.position.y = float(values[1])
    pose.position.z = float(values[2])
    qx, qy, qz, qw = normalize_quaternion(values[3:])
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def pose_to_vector(pose: Pose) -> Tuple[float, ...]:
    """将 Pose 转换为 xyz-xyzw 七元组。"""

    qx, qy, qz, qw = normalize_quaternion(
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
    )
    return (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        qx,
        qy,
        qz,
        qw,
    )


def make_pose(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
) -> Pose:
    """由标量分量构造四元数已归一化的 Pose。"""

    return pose_from_vector((x, y, z, qx, qy, qz, qw))


def compose_pose(parent: Pose, child: Pose) -> Pose:
    """计算 ``world_T_parent`` 与 ``parent_T_child`` 的位姿复合。"""

    parent_q = pose_to_vector(parent)[3:]
    child_q = pose_to_vector(child)[3:]
    child_translation = rotate_vector(
        parent_q,
        (child.position.x, child.position.y, child.position.z),
    )
    result_q = quaternion_multiply(parent_q, child_q)
    return make_pose(
        parent.position.x + child_translation[0],
        parent.position.y + child_translation[1],
        parent.position.z + child_translation[2],
        *result_q,
    )


def inverse_pose(pose: Pose) -> Pose:
    """返回 Pose 所表示刚体变换的逆变换。"""

    inverse_q = quaternion_conjugate(pose_to_vector(pose)[3:])
    translation = rotate_vector(
        inverse_q,
        (-pose.position.x, -pose.position.y, -pose.position.z),
    )
    return make_pose(*translation, *inverse_q)


def relative_pose(parent_world: Pose, child_world: Pose) -> Pose:
    """由两个世界坐标系位姿计算 ``parent_T_child``。"""

    return compose_pose(inverse_pose(parent_world), child_world)


def axis_from_pose(
    pose: Pose, local_axis: Sequence[float] = (0.0, 0.0, 1.0)
) -> VectorTuple:
    """将位姿的局部单位轴表示到父坐标系中。"""

    axis = rotate_vector(pose_to_vector(pose)[3:], local_axis)
    norm = math.sqrt(sum(value * value for value in axis))
    if norm < 1.0e-12:
        raise ValueError("axis cannot be zero")
    return tuple(value / norm for value in axis)  # type: ignore[return-value]


def pose_from_transform(transform: TransformStamped) -> Pose:
    """将 TransformStamped 消息中的变换转换为 Pose。"""

    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return make_pose(
        translation.x,
        translation.y,
        translation.z,
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    )


def canonical_horizontal_axis(pose: Pose) -> VectorTuple:
    """读取局部 +X 轴，投影到桌面并固定为同一个无向表示。"""

    return canonical_horizontal_vector(axis_from_pose(pose, (1.0, 0.0, 0.0)))


def canonical_horizontal_vector(vector: Sequence[float]) -> VectorTuple:
    """将任意三维轴投影到桌面并固定为同一个无向表示。"""

    x_value, y_value = float(vector[0]), float(vector[1])
    norm = math.hypot(x_value, y_value)
    if norm < 1.0e-9:
        raise ValueError("detected object axis has no horizontal component")
    axis = (x_value / norm, y_value / norm, 0.0)
    if axis[0] < -1.0e-12 or (abs(axis[0]) <= 1.0e-12 and axis[1] < 0.0):
        axis = (-axis[0], -axis[1], 0.0)
    return axis


def _yaw_quaternion(axis: Sequence[float]) -> QuaternionTuple:
    yaw = math.atan2(float(axis[1]), float(axis[0]))
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def make_model_pose_from_detection(detection: Detection3D, length: float) -> Pose:
    """把几何中心/+X 轴检测转换为端部原点/+Z 轴模型位姿。"""

    axis = canonical_horizontal_axis(detection.bbox.center)
    center = detection.bbox.center.position
    model_rotation = quaternion_multiply(
        _yaw_quaternion(axis),
        (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
    )
    return make_pose(
        center.x - 0.5 * length * axis[0],
        center.y - 0.5 * length * axis[1],
        center.z,
        *model_rotation,
    )


def make_grasp_frame(
    detection: Detection3D,
    axial_offset: float,
    height_offset: float,
    reverse_axis: bool = False,
) -> Pose:
    """由检测结果生成 X 沿工件轴、Z 竖直向上的抓取坐标系。"""

    return make_grasp_frame_from_center_axis(
        (
            detection.bbox.center.position.x,
            detection.bbox.center.position.y,
            detection.bbox.center.position.z,
        ),
        canonical_horizontal_axis(detection.bbox.center),
        axial_offset,
        height_offset,
        reverse_axis,
    )


def make_object_frame(detection: Detection3D) -> Pose:
    """以 YOLO 三维检测中心为原点构造工件坐标系。"""

    return make_object_frame_from_center_axis(
        (
            detection.bbox.center.position.x,
            detection.bbox.center.position.y,
            detection.bbox.center.position.z,
        ),
        canonical_horizontal_axis(detection.bbox.center),
    )


def make_object_frame_from_center_axis(
    center: Sequence[float],
    detected_axis: Sequence[float],
) -> Pose:
    """构造右手工件系：原点为检测中心，+X 沿水平轴，+Z 沿世界竖直。"""

    center_values = tuple(float(value) for value in center)
    if len(center_values) != 3 or not all(
        math.isfinite(value) for value in center_values
    ):
        raise ValueError("object center must contain three finite values")
    axis = canonical_horizontal_vector(detected_axis)
    return make_pose(*center_values, *_yaw_quaternion(axis))


def make_grasp_frame_from_center_axis(
    center: Sequence[float],
    detected_axis: Sequence[float],
    axial_offset: float,
    height_offset: float,
    reverse_axis: bool = False,
) -> Pose:
    """构造右手抓取系：原点为抓取点，+X 沿水平轴，+Z 沿世界竖直。"""

    if not math.isfinite(axial_offset) or not math.isfinite(height_offset):
        raise ValueError("grasp offsets must be finite")

    object_frame = make_object_frame_from_center_axis(center, detected_axis)
    center_values = pose_to_vector(object_frame)[:3]
    axis = axis_from_pose(object_frame, (1.0, 0.0, 0.0))
    if reverse_axis:
        axis = (-axis[0], -axis[1], 0.0)
    orientation = _yaw_quaternion(axis)
    return make_pose(
        center_values[0] + axial_offset * axis[0],
        center_values[1] + axial_offset * axis[1],
        center_values[2] + height_offset,
        *orientation,
    )


def make_grasp_pose_from_frame(
    grasp_frame: Pose,
    reference_orientation: Pose,
) -> Pose:
    """通过抓取系到 EEF 的标定旋转生成机械臂末端目标位姿。"""

    orientation = quaternion_multiply(
        pose_to_vector(grasp_frame)[3:],
        pose_to_vector(reference_orientation)[3:],
    )
    return make_pose(
        grasp_frame.position.x,
        grasp_frame.position.y,
        grasp_frame.position.z,
        *orientation,
    )


def make_grasp_pose_set(
    targets: TargetPair,
    *,
    wire_length: float,
    terminal_length: float,
    wire_axial_offset: float,
    terminal_axial_offset: float,
    wire_height_offset: float,
    terminal_height_offset: float,
    pregrasp_height: float,
    minimum_grasp_height: float,
    reference_orientation: Pose,
    reverse_axis: bool = False,
) -> GraspPoseSet:
    """一次生成两个模型位姿以及双臂预抓取/抓取位姿。"""

    wire_object_frame = make_object_frame(targets.wire)
    terminal_object_frame = make_object_frame(targets.terminal)
    wire_grasp_point = make_grasp_frame(
        targets.wire,
        wire_axial_offset,
        wire_height_offset,
        reverse_axis,
    )
    terminal_grasp_point = make_grasp_frame(
        targets.terminal,
        terminal_axial_offset,
        terminal_height_offset,
        reverse_axis,
    )
    wire_grasp_point.position.z = max(
        wire_grasp_point.position.z, minimum_grasp_height
    )
    terminal_grasp_point.position.z = max(
        terminal_grasp_point.position.z, minimum_grasp_height
    )
    wire_grasp = make_grasp_pose_from_frame(
        wire_grasp_point, reference_orientation
    )
    terminal_grasp = make_grasp_pose_from_frame(
        terminal_grasp_point, reference_orientation
    )
    wire_pregrasp = deepcopy(wire_grasp)
    terminal_pregrasp = deepcopy(terminal_grasp)
    wire_pregrasp.position.z += pregrasp_height
    terminal_pregrasp.position.z += pregrasp_height
    return GraspPoseSet(
        wire_model=make_model_pose_from_detection(targets.wire, wire_length),
        terminal_model=make_model_pose_from_detection(
            targets.terminal, terminal_length
        ),
        wire_object_frame=wire_object_frame,
        terminal_object_frame=terminal_object_frame,
        wire_grasp_point=wire_grasp_point,
        terminal_grasp_point=terminal_grasp_point,
        wire_pregrasp=wire_pregrasp,
        terminal_pregrasp=terminal_pregrasp,
        wire_grasp=wire_grasp,
        terminal_grasp=terminal_grasp,
    )
