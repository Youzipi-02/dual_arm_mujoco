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

"""由端面主深度和孔洞像素中心估计孔中心三维坐标."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np

from s622_dual_perception.visual_perception_utils.obb_geometry import (
    CameraIntrinsics,
    transform_point,
)


@dataclass(frozen=True)
class DepthEstimate:
    """保存端面掩膜主深度簇的稳健统计量."""

    depth_m: float
    selected_count: int
    valid_count: int
    cluster_count: int
    valid_ratio: float
    cluster_ratio: float
    mad_m: float


@dataclass(frozen=True)
class HoleCenter3D:
    """保存单帧孔中心三维坐标及其深度质量."""

    center: np.ndarray
    depth: DepthEstimate


@dataclass(frozen=True)
class StableHoleCenter:
    """保存多帧稳定确认后的孔中心和离散程度."""

    center: np.ndarray
    sample_count: int
    duration_s: float
    position_range_m: float
    mean_depth_m: float
    mean_depth_mad_m: float


def estimate_depth_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    minimum_valid_ratio: float,
    depth_cluster_tolerance_m: float,
    minimum_depth_cluster_ratio: float,
) -> DepthEstimate:
    """从端面掩膜中选择最密集深度簇并返回其中值."""
    image = np.asarray(depth_m, dtype=np.float64)
    binary = np.asarray(mask, dtype=np.uint8)
    if image.ndim != 2 or binary.shape != image.shape:
        raise ValueError("深度图与端面掩膜尺寸不一致")
    if maximum_depth_m <= minimum_depth_m:
        raise ValueError("最大有效深度必须大于最小有效深度")
    if not 0.0 <= minimum_valid_ratio <= 1.0:
        raise ValueError("端面最小深度有效比例必须位于零到一之间")
    if depth_cluster_tolerance_m <= 0.0:
        raise ValueError("端面主深度簇容差必须大于零")
    if not 0.0 <= minimum_depth_cluster_ratio <= 1.0:
        raise ValueError("端面主深度簇最小比例必须位于零到一之间")

    selected = binary > 0
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        raise ValueError("端面掩膜为空")
    valid = (
        selected
        & np.isfinite(image)
        & (image >= minimum_depth_m)
        & (image <= maximum_depth_m)
    )
    valid_depths = image[valid]
    valid_count = int(valid_depths.size)
    valid_ratio = float(valid_count) / float(selected_count)
    if valid_count == 0:
        raise ValueError("端面掩膜中没有有效深度")
    if valid_ratio < minimum_valid_ratio:
        raise ValueError(
            f"端面有效深度比例不足: {valid_ratio:.3f} < "
            f"{minimum_valid_ratio:.3f}"
        )

    sorted_depths = np.sort(valid_depths)
    left = 0
    best_left = 0
    best_right = 1
    cluster_width = 2.0 * depth_cluster_tolerance_m
    for right in range(sorted_depths.size):
        while sorted_depths[right] - sorted_depths[left] > cluster_width:
            left += 1
        if right - left + 1 > best_right - best_left:
            best_left = left
            best_right = right + 1
    cluster_seed = sorted_depths[best_left:best_right]
    seed_center = float(np.median(cluster_seed))
    clustered_depths = valid_depths[
        np.abs(valid_depths - seed_center) <= depth_cluster_tolerance_m
    ]
    cluster_count = int(clustered_depths.size)
    cluster_ratio = float(cluster_count) / float(valid_count)
    if cluster_ratio < minimum_depth_cluster_ratio:
        raise ValueError(
            f"端面主深度簇比例不足: {cluster_ratio:.3f} < "
            f"{minimum_depth_cluster_ratio:.3f}"
        )
    depth = float(np.median(clustered_depths))
    mad = float(np.median(np.abs(clustered_depths - depth)))
    return DepthEstimate(
        depth_m=depth,
        selected_count=selected_count,
        valid_count=valid_count,
        cluster_count=cluster_count,
        valid_ratio=valid_ratio,
        cluster_ratio=cluster_ratio,
        mad_m=mad,
    )


def estimate_hole_center(
    center_pixel: Sequence[float],
    depth: DepthEstimate,
    intrinsics: CameraIntrinsics,
) -> HoleCenter3D:
    """使用端面主深度把孔洞椭圆中心反投影到相机坐标系."""
    pixel = np.asarray(center_pixel, dtype=np.float64)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        raise ValueError("孔洞中心像素必须包含两个有限数值")
    intrinsics.validate()
    z_value = float(depth.depth_m)
    if not math.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("孔中心深度必须是正有限数值")
    center = np.array(
        (
            (pixel[0] - intrinsics.cx) * z_value / intrinsics.fx,
            (pixel[1] - intrinsics.cy) * z_value / intrinsics.fy,
            z_value,
        ),
        dtype=np.float64,
    )
    return HoleCenter3D(center=center, depth=depth)


def transform_hole_center(
    result: HoleCenter3D,
    translation: Sequence[float],
    quaternion: Sequence[float],
) -> HoleCenter3D:
    """把相机光学坐标下的孔中心转换到目标坐标系."""
    return HoleCenter3D(
        center=transform_point(result.center, translation, quaternion),
        depth=result.depth,
    )


class HoleCenterStabilityGate:
    """仅按世界坐标位置范围锁定稳定孔中心."""

    def __init__(
        self,
        *,
        minimum_samples: int,
        confirm_duration_s: float,
        maximum_gap_s: float,
        maximum_position_range_m: float,
    ) -> None:
        if minimum_samples < 1:
            raise ValueError("稳定确认帧数必须大于零")
        if min(
            confirm_duration_s,
            maximum_gap_s,
            maximum_position_range_m,
        ) <= 0.0:
            raise ValueError("稳定确认阈值必须大于零")
        self.minimum_samples = int(minimum_samples)
        self.confirm_duration_s = float(confirm_duration_s)
        self.maximum_gap_s = float(maximum_gap_s)
        self.maximum_position_range_m = float(maximum_position_range_m)
        self.reset()

    def reset(self) -> None:
        """清空当前稳定窗口和已锁定结果."""
        self._samples: list[tuple[float, HoleCenter3D]] = []
        self.locked: Optional[StableHoleCenter] = None
        self.last_reason = "等待首帧"

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def duration_s(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return max(0.0, self._samples[-1][0] - self._samples[0][0])

    def update(self, stamp_s: float, result: HoleCenter3D) -> bool:
        """加入一帧世界系孔中心并在满足阈值时锁定结果."""
        if self.locked is not None:
            return True
        if not math.isfinite(stamp_s):
            self.reset()
            self.last_reason = "图像时间戳无效"
            return False
        if self._samples:
            gap = stamp_s - self._samples[-1][0]
            if gap <= 0.0 or gap > self.maximum_gap_s:
                self.reset()
                self.last_reason = "检测帧时间不连续"
        self._samples.append((stamp_s, result))
        if not self._window_is_stable():
            return False
        if (
            self.sample_count < self.minimum_samples
            or self.duration_s < self.confirm_duration_s
        ):
            self.last_reason = "稳定帧数或持续时间不足"
            return False
        self.locked = self._summarize()
        self.last_reason = "孔中心已稳定锁定"
        return True

    def _window_is_stable(self) -> bool:
        centers = np.asarray(
            [sample.center for _, sample in self._samples],
            dtype=np.float64,
        )
        reference = np.median(centers, axis=0)
        position_range = 2.0 * float(
            np.max(np.linalg.norm(centers - reference, axis=1))
        )
        if position_range > self.maximum_position_range_m:
            latest = self._samples[-1]
            self._samples = [latest]
            self.last_reason = "孔中心离散程度超限"
            return False
        return True

    def _summarize(self) -> StableHoleCenter:
        centers = np.asarray([item.center for _, item in self._samples])
        center = np.median(centers, axis=0)
        position_range = 2.0 * float(
            np.max(np.linalg.norm(centers - center, axis=1))
        )
        return StableHoleCenter(
            center=center,
            sample_count=self.sample_count,
            duration_s=self.duration_s,
            position_range_m=position_range,
            mean_depth_m=float(
                np.mean([item.depth.depth_m for _, item in self._samples])
            ),
            mean_depth_mad_m=float(
                np.mean([item.depth.mad_m for _, item in self._samples])
            ),
        )

    def status(self) -> str:
        state = "STABLE" if self.locked is not None else "STABILIZING"
        return (
            f"{state} {self.sample_count}/{self.minimum_samples} "
            f"{self.duration_s:.2f}/{self.confirm_duration_s:.2f}s"
        )
