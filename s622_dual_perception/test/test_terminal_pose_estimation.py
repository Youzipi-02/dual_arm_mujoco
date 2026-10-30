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

"""测试端面深度、孔中心反投影、坐标变换和位置稳定门."""

import numpy as np
import pytest

from s622_dual_perception.visual_perception_utils.obb_geometry import (
    CameraIntrinsics,
)
from s622_dual_perception.visual_perception_utils.terminal_pose_estimation import (
    HoleCenterStabilityGate,
    estimate_depth_from_mask,
    estimate_hole_center,
    transform_hole_center,
)


def _intrinsics():
    return CameraIntrinsics(
        fx=120.0,
        fy=120.0,
        cx=50.0,
        cy=40.0,
        width=100,
        height=80,
    )


def _center():
    depth_image = np.full((80, 100), 0.5, dtype=np.float32)
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask[15:65, 20:80] = 255
    depth = estimate_depth_from_mask(
        depth_image,
        mask,
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        minimum_valid_ratio=0.8,
        depth_cluster_tolerance_m=0.01,
        minimum_depth_cluster_ratio=0.8,
    )
    return estimate_hole_center((50.0, 40.0), depth, _intrinsics())


def test_planar_depth_returns_backprojected_hole_center():
    """正对相机的半米端面应得到中心像素反投影点."""
    result = _center()
    assert result.center == pytest.approx((0.0, 0.0, 0.5), abs=1.0e-6)
    assert result.depth.valid_ratio == pytest.approx(1.0)
    assert result.depth.cluster_ratio == pytest.approx(1.0)


def test_depth_estimate_uses_dominant_cluster():
    """端面深度中混入远处背景时应保留占优的近端深度簇."""
    depth_image = np.full((80, 100), 0.5, dtype=np.float32)
    mask = np.zeros_like(depth_image, dtype=np.uint8)
    mask[10:70, 15:85] = 255
    depth_image[10:28, 15:85] = 0.9

    result = estimate_depth_from_mask(
        depth_image,
        mask,
        minimum_depth_m=0.1,
        maximum_depth_m=1.0,
        minimum_valid_ratio=0.8,
        depth_cluster_tolerance_m=0.01,
        minimum_depth_cluster_ratio=0.6,
    )

    assert result.depth_m == pytest.approx(0.5)
    assert result.cluster_ratio == pytest.approx(0.7)


def test_hole_center_backprojection_uses_pixel_and_depth():
    """非主点像素应按相机模型反投影."""
    result = _center()
    shifted = estimate_hole_center((62.0, 34.0), result.depth, _intrinsics())
    assert shifted.center == pytest.approx((0.05, -0.025, 0.5))


def test_transform_hole_center_preserves_depth_quality():
    """世界坐标转换只改变中心坐标，不改变深度统计."""
    result = _center()
    transformed = transform_hole_center(
        result,
        (0.1, 0.2, 0.3),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert transformed.center == pytest.approx((0.1, 0.2, 0.8))
    assert transformed.depth is result.depth


def test_stability_gate_locks_consistent_world_centers():
    """持续时间、帧数和位置均合格时应锁定稳定孔中心."""
    result = _center()
    gate = HoleCenterStabilityGate(
        minimum_samples=5,
        confirm_duration_s=0.5,
        maximum_gap_s=0.2,
        maximum_position_range_m=0.003,
    )
    for index, stamp in enumerate((1.0, 1.15, 1.30, 1.45, 1.60)):
        jittered = type(result)(
            center=result.center + np.array((index * 0.0001, 0.0, 0.0)),
            depth=result.depth,
        )
        gate.update(stamp, jittered)
    assert gate.locked is not None
    assert gate.locked.sample_count == 5
    assert gate.locked.center[2] == pytest.approx(0.5)


def test_stability_gate_restarts_on_large_position_jump():
    """孔中心跳变超过阈值时应只保留最新候选帧."""
    result = _center()
    gate = HoleCenterStabilityGate(
        minimum_samples=3,
        confirm_duration_s=0.1,
        maximum_gap_s=0.2,
        maximum_position_range_m=0.001,
    )
    gate.update(1.0, result)
    jumped = type(result)(
        center=result.center + np.array((0.01, 0.0, 0.0)),
        depth=result.depth,
    )
    gate.update(1.1, jumped)
    assert gate.sample_count == 1
    assert gate.locked is None
