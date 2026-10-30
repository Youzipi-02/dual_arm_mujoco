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

"""测试 OBB 深度估计。."""

import numpy as np
import pytest

from s622_dual_perception.visual_perception_utils.depth_estimation import (
    depth_image_to_meters,
    estimate_obb_depth,
    robust_surface_depth,
)


def test_depth_image_to_meters_supports_common_encodings():
    """毫米和米两种标准深度编码都应转换为米。."""
    millimeters = np.array([[1000, 2500]], dtype=np.uint16)
    meters = np.array([[1.0, 2.5]], dtype=np.float32)
    assert np.allclose(depth_image_to_meters(millimeters, "16UC1"), meters)
    assert np.allclose(depth_image_to_meters(meters, "32FC1"), meters)


def test_robust_surface_depth_rejects_background_cluster():
    """靠近相机的分位数和窄带应排除更远的桌面深度。."""
    values = np.array([1.0] * 60 + [1.2] * 40, dtype=np.float32)
    surface, mad = robust_surface_depth(values, 35.0, 0.02)
    assert surface == pytest.approx(1.0)
    assert mad == pytest.approx(0.0)


def test_estimate_obb_depth_returns_center_and_endpoint_depths():
    """OBB 内部及长轴端点应得到稳定的米制深度。."""
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    depth[44:57, 29:72] = 1.0
    corners = np.array(
        [[30.0, 45.0], [70.0, 45.0], [70.0, 55.0], [30.0, 55.0]]
    )
    result = estimate_obb_depth(
        depth,
        corners,
        (np.array([37.2, 50.0]), np.array([62.8, 50.0])),
        roi_scale=0.8,
        minimum_depth_m=0.15,
        maximum_depth_m=4.0,
        minimum_valid_ratio=0.2,
        surface_percentile=35.0,
        surface_band_m=0.02,
        endpoint_radius_px=2,
        center_depth_offset_m=0.005,
    )
    assert result.center_depth_m == pytest.approx(1.005)
    assert result.first_endpoint_depth_m == pytest.approx(1.005)
    assert result.second_endpoint_depth_m == pytest.approx(1.005)
    assert result.valid_ratio == pytest.approx(1.0)
    assert result.mad_m == pytest.approx(0.0)


def test_estimate_obb_depth_rejects_missing_depth():
    """OBB 内没有有效深度时必须拒绝发布三维目标。."""
    depth = np.zeros((100, 100), dtype=np.float32)
    corners = np.array(
        [[30.0, 45.0], [70.0, 45.0], [70.0, 55.0], [30.0, 55.0]]
    )
    with pytest.raises(ValueError, match="有效深度比例不足"):
        estimate_obb_depth(
            depth,
            corners,
            (np.array([37.2, 50.0]), np.array([62.8, 50.0])),
            roi_scale=0.8,
            minimum_depth_m=0.15,
            maximum_depth_m=4.0,
            minimum_valid_ratio=0.2,
            surface_percentile=35.0,
            surface_band_m=0.02,
            endpoint_radius_px=2,
            center_depth_offset_m=0.005,
        )
