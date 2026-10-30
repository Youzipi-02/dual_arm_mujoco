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

"""OBB 像素到相机/世界坐标的纯二维和三维几何计算。."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from s622_dual_perception.visual_perception_utils.depth_estimation import (
    DepthEstimate,
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """保存针孔相机反投影所需的内参和图像尺寸。."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def validate(self) -> None:
        """检查相机内参和图像尺寸是否可以用于反投影。."""
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("相机内参必须是有限数值")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("相机焦距必须大于零")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("相机图像尺寸必须大于零")


@dataclass(frozen=True)
class ObbPose3D:
    """保存 OBB 中心和局部 +X 长轴方向。."""

    center: np.ndarray
    axis: np.ndarray


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    """返回单位向量，拒绝零向量或非有限输入。."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("三维向量必须包含三个有限数值")
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        raise ValueError("无法归一化零向量")
    return value / norm


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """按照 x、y、z、w 顺序返回单位四元数。."""
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("四元数必须包含四个有限数值")
    norm = float(np.linalg.norm(value))
    if norm < 1.0e-12:
        raise ValueError("无法归一化零四元数")
    return value / norm


def obb_axis_endpoints(
    center: Sequence[float],
    size: Sequence[float],
    angle_rad: float,
    half_span_ratio: float = 0.32,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 OBB 长轴上的两个内部采样像素。."""
    center_value = np.asarray(center, dtype=np.float64)
    size_value = np.asarray(size, dtype=np.float64)
    if center_value.shape != (2,) or size_value.shape != (2,):
        raise ValueError("OBB 中心和尺寸都必须包含两个数值")
    if not np.all(np.isfinite(center_value)) or not np.all(
        np.isfinite(size_value)
    ):
        raise ValueError("OBB 中心和尺寸必须是有限数值")
    if not math.isfinite(angle_rad):
        raise ValueError("OBB 角度必须是有限数值")
    if np.any(size_value <= 0.0):
        raise ValueError("OBB 尺寸必须大于零")
    if not 0.0 < half_span_ratio < 0.5:
        raise ValueError("长轴采样比例必须位于零到二分之一之间")

    width, height = size_value
    long_angle = float(angle_rad)
    long_size = float(width)
    if height > width:
        long_angle += 0.5 * math.pi
        long_size = float(height)
    direction = np.array(
        (math.cos(long_angle), math.sin(long_angle)),
        dtype=np.float64,
    )
    offset = direction * long_size * half_span_ratio
    return center_value - offset, center_value + offset


def back_project_pixel(
    pixel_x: float,
    pixel_y: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """使用针孔模型把像素和轴向深度反投影到光学坐标系。."""
    intrinsics.validate()
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("反投影深度必须是大于零的有限数值")
    return np.array(
        (
            (float(pixel_x) - intrinsics.cx) * depth_m / intrinsics.fx,
            (float(pixel_y) - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ),
        dtype=np.float64,
    )


def calculate_obb_pose_3d(
    center_pixel: Sequence[float],
    endpoint_pixels: tuple[Sequence[float], Sequence[float]],
    depth_estimate: DepthEstimate,
    intrinsics: CameraIntrinsics,
) -> ObbPose3D:
    """将 OBB 中心和长轴端点转换为相机光学坐标系中的中心和轴。."""
    center_value = np.asarray(center_pixel, dtype=np.float64)
    first_pixel = np.asarray(endpoint_pixels[0], dtype=np.float64)
    second_pixel = np.asarray(endpoint_pixels[1], dtype=np.float64)
    for pixel in (center_value, first_pixel, second_pixel):
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError("像素坐标必须是两个有限数值")

    center = back_project_pixel(
        center_value[0],
        center_value[1],
        depth_estimate.center_depth_m,
        intrinsics,
    )
    first = back_project_pixel(
        first_pixel[0],
        first_pixel[1],
        depth_estimate.first_endpoint_depth_m,
        intrinsics,
    )
    second = back_project_pixel(
        second_pixel[0],
        second_pixel[1],
        depth_estimate.second_endpoint_depth_m,
        intrinsics,
    )
    return ObbPose3D(center=center, axis=normalize_vector(second - first))


def rotate_vector(
    vector: Sequence[float],
    quaternion: Sequence[float],
) -> np.ndarray:
    """使用 x、y、z、w 四元数旋转一个三维向量。."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("待旋转向量必须包含三个有限数值")
    qx, qy, qz, qw = normalize_quaternion(quaternion)
    q_vector = np.array((qx, qy, qz), dtype=np.float64)
    intermediate = 2.0 * np.cross(q_vector, value)
    return value + qw * intermediate + np.cross(q_vector, intermediate)


def transform_point(
    point: Sequence[float],
    translation: Sequence[float],
    quaternion: Sequence[float],
) -> np.ndarray:
    """先旋转再平移，把一个点转换到目标坐标系。."""
    offset = np.asarray(translation, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("平移量必须包含三个有限数值")
    return rotate_vector(point, quaternion) + offset


def transform_obb_pose(
    pose: ObbPose3D,
    translation: Sequence[float],
    quaternion: Sequence[float],
) -> ObbPose3D:
    """使用普通平移和四元数把相机下 OBB 转换到目标坐标系。."""
    return ObbPose3D(
        center=transform_point(pose.center, translation, quaternion),
        axis=normalize_vector(rotate_vector(pose.axis, quaternion)),
    )


def quaternion_from_x_axis(axis: Sequence[float]) -> np.ndarray:
    """生成把局部 +X 轴旋转到目标方向的最小旋转四元数。."""
    direction = normalize_vector(axis)
    dot = float(direction[0])
    if dot < -1.0 + 1.0e-10:
        return np.array((0.0, 0.0, 1.0, 0.0), dtype=np.float64)

    cross = np.array((0.0, -direction[2], direction[1]))
    return normalize_quaternion((cross[0], cross[1], cross[2], 1.0 + dot))


def symmetric_axis_error(
    first_axis: Sequence[float],
    second_axis: Sequence[float],
) -> float:
    """返回无方向轴线的最小夹角，范围为零到 π/2。."""
    first = normalize_vector(first_axis)
    second = normalize_vector(second_axis)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.acos(cosine)
