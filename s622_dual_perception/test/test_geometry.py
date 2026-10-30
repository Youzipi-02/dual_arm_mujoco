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

"""测试 OBB 像素到三维点、方向轴和四元数工具."""

import math

import pytest

from s622_dual_perception.visual_perception_utils.depth_estimation import (
    DepthEstimate,
)
from s622_dual_perception.visual_perception_utils.obb_geometry import (
    CameraIntrinsics,
    back_project_pixel,
    calculate_obb_pose_3d,
    quaternion_from_x_axis,
    rotate_vector,
    symmetric_axis_error,
    transform_point,
)


def test_quaternion_aligns_x_axis_to_y_axis():
    """构造的四元数应把局部 X 轴旋转到目标 Y 轴."""
    quaternion = quaternion_from_x_axis((0.0, 1.0, 0.0))
    rotated = rotate_vector((1.0, 0.0, 0.0), quaternion)
    assert rotated == pytest.approx((0.0, 1.0, 0.0), abs=1.0e-9)


def test_transform_point_applies_rotation_before_translation():
    """点变换应先旋转源点再添加目标坐标系平移."""
    half = math.sqrt(0.5)
    transformed = transform_point(
        (1.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (0.0, 0.0, half, half),
    )
    assert transformed == pytest.approx((1.0, 3.0, 3.0), abs=1.0e-9)


def test_axis_error_ignores_axis_sign():
    """圆柱轴正反方向应视为同一方向."""
    assert symmetric_axis_error(
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    ) == pytest.approx(0.0)


def test_calculate_obb_pose_uses_center_and_endpoint_depths():
    """中心和端点深度应共同确定相机坐标下的中心与长轴。."""
    intrinsics = CameraIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    depth = DepthEstimate(2.0, 2.0, 2.0, 1.0, 0.0)
    pose = calculate_obb_pose_3d(
        (50.0, 50.0),
        ((40.0, 50.0), (60.0, 50.0)),
        depth,
        intrinsics,
    )
    assert pose.center == pytest.approx((0.0, 0.0, 2.0))
    assert abs(pose.axis[0]) == pytest.approx(1.0)
    assert pose.axis[1] == pytest.approx(0.0, abs=1.0e-9)
    assert pose.axis[2] == pytest.approx(0.0, abs=1.0e-9)


def test_back_project_optical_center():
    """主点像素应反投影到相机光轴上。."""
    intrinsics = CameraIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    point = back_project_pixel(50.0, 50.0, 2.0, intrinsics)
    assert point == pytest.approx((0.0, 0.0, 2.0))
