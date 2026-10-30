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

"""测试分割调试图和稳定孔特征 Marker."""

import cv2
import numpy as np
from std_msgs.msg import Header
import pytest

from s622_dual_perception.visual_perception_utils.segmentation_postprocess import (
    make_detection,
)
from s622_dual_perception.visual_perception_utils.visualization import (
    build_hole_center_marker_array,
    render_segmentation_status_image,
)


def test_hole_center_marker_is_single_blue_reference_point():
    """孔中心 Marker 只渲染一个蓝色球点，不包含文字或轴向箭头."""
    header = Header()
    header.frame_id = "world"
    markers = build_hole_center_marker_array(
        header,
        (0.1, 0.2, 0.3),
        diameter_m=0.006,
    )
    # 第一项是清除旧 Marker 的控制消息，RViz 中只渲染第二项蓝点。
    assert len(markers.markers) == 2
    sphere = markers.markers[1]
    assert sphere.header.frame_id == "world"
    assert sphere.ns == "computed_hole_center"
    assert sphere.type == sphere.SPHERE
    position = sphere.pose.position
    assert (position.x, position.y, position.z) == (
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
    )
    assert sphere.scale.x == pytest.approx(0.006)
    assert sphere.scale.y == pytest.approx(0.006)
    assert sphere.scale.z == pytest.approx(0.006)
    assert sphere.color.r == pytest.approx(0.0)
    assert sphere.color.g == pytest.approx(0.0)
    assert sphere.color.b == pytest.approx(1.0)
    assert sphere.color.a == pytest.approx(1.0)


def test_pair_failure_image_draws_candidate_mask_and_diagnostics():
    """配对失败图仍应显示候选掩膜、框和筛选信息."""
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    cv2.circle(mask, (70, 55), 8, 1.0, thickness=-1)
    detection = make_detection(
        index=0,
        class_name="hole",
        confidence=0.82,
        bbox_xyxy=(62, 47, 78, 63),
        probability_mask=mask,
        image_shape=frame.shape[:2],
        threshold=0.5,
        opening_kernel_px=0,
        closing_kernel_px=0,
    )

    annotated = render_segmentation_status_image(
        frame,
        "当前帧缺少端面",
        [detection],
        minimum_hole_area_px=20.0,
        minimum_face_area_px=30.0,
    )

    assert annotated.shape == frame.shape
    assert np.count_nonzero(annotated) > 0
    assert np.any(annotated[55, 70] != frame[55, 70])
