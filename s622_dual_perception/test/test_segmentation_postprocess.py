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

"""测试孔洞端面分割掩膜后处理."""

import cv2
import numpy as np
import pytest

from s622_dual_perception.visual_perception_utils.segmentation_postprocess import (
    make_detection,
    select_hole_face_pair,
)


def _detection(index, class_name, mask):
    rows, columns = np.nonzero(mask)
    return make_detection(
        index=index,
        class_name=class_name,
        confidence=0.9,
        bbox_xyxy=(columns.min(), rows.min(), columns.max(), rows.max()),
        probability_mask=mask.astype(np.float32),
        image_shape=mask.shape,
        threshold=0.5,
        opening_kernel_px=0,
        closing_kernel_px=0,
    )


def test_select_pair_and_fit_hole_ellipse():
    """孔洞位于端面内部时应正确配对并拟合中心."""
    face = np.zeros((120, 160), dtype=np.uint8)
    hole = np.zeros_like(face)
    cv2.rectangle(face, (35, 25), (125, 95), 1, thickness=-1)
    cv2.circle(hole, (80, 60), 15, 1, thickness=-1)
    pair = select_hole_face_pair(
        [_detection(0, "hole", hole), _detection(1, "terminal_face", face)],
        hole_class="hole",
        face_class="terminal_face",
        minimum_hole_area_px=20.0,
        minimum_face_area_px=100.0,
    )
    assert pair.ellipse[0] == pytest.approx((80.0, 60.0), abs=0.5)
    assert pair.hole.class_name == "hole"
    assert pair.face.class_name == "terminal_face"


def test_pair_accepts_partial_face_smaller_than_hole():
    """端面分割不完整时允许其面积小于孔洞，但仍受最大面积比约束."""
    face = np.zeros((120, 160), dtype=np.uint8)
    hole = np.zeros_like(face)
    cv2.rectangle(face, (74, 44), (91, 76), 1, thickness=-1)
    cv2.circle(hole, (82, 60), 15, 1, thickness=-1)

    pair = select_hole_face_pair(
        [_detection(0, "hole", hole), _detection(1, "terminal_face", face)],
        hole_class="hole",
        face_class="terminal_face",
        minimum_hole_area_px=20.0,
        minimum_face_area_px=100.0,
        maximum_hole_to_face_area_ratio=3.0,
    )

    assert pair.hole_to_face_area_ratio > 1.0
    assert pair.match_mode == "inside_hull"


def test_pair_accepts_hole_just_outside_incomplete_face_hull():
    """孔中心轻微越出端面轮廓时使用 near_hull 回退配对."""
    face = np.zeros((120, 160), dtype=np.uint8)
    hole = np.zeros_like(face)
    cv2.rectangle(face, (20, 20), (60, 80), 1, thickness=-1)
    cv2.circle(hole, (68, 50), 5, 1, thickness=-1)

    pair = select_hole_face_pair(
        [_detection(0, "hole", hole), _detection(1, "terminal_face", face)],
        hole_class="hole",
        face_class="terminal_face",
        minimum_hole_area_px=20.0,
        minimum_face_area_px=100.0,
        face_margin_ratio=0.25,
    )

    assert pair.match_mode == "near_hull"
    assert pair.center_distance_ratio < 1.0


def test_pair_rejects_distant_instances_with_diagnostics():
    """相距过远的孔洞和端面不得误配，错误应给出拒绝统计."""
    face = np.zeros((120, 180), dtype=np.uint8)
    hole = np.zeros_like(face)
    cv2.rectangle(face, (10, 30), (50, 90), 1, thickness=-1)
    cv2.circle(hole, (150, 60), 8, 1, thickness=-1)

    with pytest.raises(ValueError, match="spatial_rejected=1"):
        select_hole_face_pair(
            [
                _detection(0, "hole", hole),
                _detection(1, "terminal_face", face),
            ],
            hole_class="hole",
            face_class="terminal_face",
            minimum_hole_area_px=20.0,
            minimum_face_area_px=100.0,
            face_margin_ratio=0.10,
            maximum_center_distance_ratio=0.50,
        )


def test_small_face_rejection_reports_candidate_area():
    """面积门槛过滤端面时应报告原始候选数量、面积和阈值."""
    face = np.zeros((80, 100), dtype=np.uint8)
    hole = np.zeros_like(face)
    cv2.rectangle(face, (46, 34), (53, 45), 1, thickness=-1)
    cv2.circle(hole, (50, 40), 8, 1, thickness=-1)

    with pytest.raises(
        ValueError,
        match=r"face=0/1 areas=\[77\] min=100px",
    ):
        select_hole_face_pair(
            [
                _detection(0, "hole", hole),
                _detection(1, "terminal_face", face),
            ],
            hole_class="hole",
            face_class="terminal_face",
            minimum_hole_area_px=20.0,
            minimum_face_area_px=100.0,
        )
