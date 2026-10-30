"""测试连续可见门和 YOLO 三维检测到抓取位姿的转换。"""

import math

import pytest
from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from vision_pick.grasp_geometry import axis_from_pose, make_grasp_pose_set, make_pose
from s622_dual_perception.target_confirmation import (
    ContinuousDetectionGate,
    extract_best_targets,
)


def detection(class_id, score, x, y, *, frame="world", yaw=0.0):
    result = Detection3D()
    result.header.frame_id = frame
    result.bbox.center.position.x = x
    result.bbox.center.position.y = y
    result.bbox.center.position.z = 0.755
    result.bbox.center.orientation.z = math.sin(0.5 * yaw)
    result.bbox.center.orientation.w = math.cos(0.5 * yaw)
    result.bbox.size.x = 0.06 if class_id == "wire" else 0.03
    result.bbox.size.y = 0.007 if class_id == "wire" else 0.015
    result.bbox.size.z = result.bbox.size.y
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_id
    hypothesis.hypothesis.score = score
    result.results = [hypothesis]
    return result


def message(*detections):
    result = Detection3DArray()
    result.header.frame_id = "world"
    result.detections = list(detections)
    return result


def target_pair():
    targets, missing = extract_best_targets(
        message(
            detection("wire", 0.8, -0.15, 0.35),
            detection("copper_terminal", 0.9, 0.195, 0.35),
        ),
        "world",
    )
    assert missing == ()
    return targets


def test_each_class_uses_only_highest_confidence_valid_detection():
    targets, missing = extract_best_targets(
        message(
            detection("wire", 0.4, -0.25, 0.3),
            detection("wire", 0.9, -0.12, 0.4),
            detection("copper_terminal", 0.8, 0.2, 0.35),
        ),
        "world",
    )

    assert missing == ()
    assert targets.wire.bbox.center.position.x == pytest.approx(-0.12)


def test_wrong_frame_and_nonfinite_detection_are_not_confirmed():
    invalid = detection("wire", 0.9, -0.1, 0.3, frame="camera")
    invalid.bbox.center.position.x = float("nan")
    targets, missing = extract_best_targets(
        message(invalid, detection("copper_terminal", 0.8, 0.2, 0.3)),
        "world",
    )

    assert targets is None
    assert missing == ("wire",)


def test_continuous_visibility_locks_latest_frame_after_two_seconds():
    gate = ContinuousDetectionGate(2.0, 0.75, 3)
    first = target_pair()
    second = target_pair()
    third = target_pair()
    latest = target_pair()

    assert not gate.update(10.0, 1.0, first)
    assert not gate.update(10.7, 1.7, second)
    assert not gate.update(11.4, 2.4, third)
    assert gate.update(12.0, 3.0, latest)
    latest.wire.bbox.center.position.x = 99.0

    assert gate.locked is not None
    assert gate.locked.wire.bbox.center.position.x != 99.0
    assert gate.elapsed == pytest.approx(2.0)


def test_missing_target_or_large_gap_immediately_restarts_confirmation():
    gate = ContinuousDetectionGate(2.0, 0.75, 3)
    targets = target_pair()

    gate.update(1.0, 1.0, targets)
    gate.update(1.5, 1.5, targets)
    gate.update(1.6, 1.6, None, ("wire",))
    assert gate.confirmations == 0
    assert gate.elapsed == 0.0

    gate.update(2.0, 2.0, targets)
    gate.update(3.0, 3.0, targets)
    assert gate.confirmations == 1
    assert "间隔超限" in gate.last_reason


def test_reception_stream_timeout_restarts_confirmation():
    gate = ContinuousDetectionGate(2.0, 0.75, 3)
    gate.update(1.0, 5.0, target_pair())

    assert gate.expire(5.8)
    assert gate.confirmations == 0
    assert "中断" in gate.last_reason


def test_visual_pose_set_rotates_grasp_and_builds_model_endpoint():
    targets = target_pair()
    targets.wire.bbox.center.orientation.z = math.sqrt(0.5)
    targets.wire.bbox.center.orientation.w = math.sqrt(0.5)
    poses = make_grasp_pose_set(
        targets,
        wire_length=0.06,
        terminal_length=0.03,
        wire_axial_offset=-0.022,
        terminal_axial_offset=0.010,
        wire_height_offset=0.0075,
        terminal_height_offset=0.0035,
        pregrasp_height=0.12,
        minimum_grasp_height=0.0,
        reference_orientation=make_pose(
            0.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0
        ),
    )

    assert poses.wire_model.position.x == pytest.approx(-0.15)
    assert poses.wire_model.position.y == pytest.approx(0.32)
    assert axis_from_pose(poses.wire_model) == pytest.approx((0.0, 1.0, 0.0))
    assert poses.wire_object_frame.position.x == pytest.approx(-0.15)
    assert poses.wire_object_frame.position.y == pytest.approx(0.35)
    assert poses.wire_object_frame.position.z == pytest.approx(0.755)
    assert axis_from_pose(
        poses.wire_object_frame, (1.0, 0.0, 0.0)
    ) == pytest.approx(
        (0.0, 1.0, 0.0)
    )
    assert axis_from_pose(
        poses.wire_object_frame, (0.0, 0.0, 1.0)
    ) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert poses.wire_grasp_point.position.y == pytest.approx(0.328)
    assert poses.wire_grasp.position.y == pytest.approx(0.328)
    assert poses.wire_pregrasp.position.z - poses.wire_grasp.position.z == pytest.approx(
        0.12
    )


def test_reverse_candidate_flips_only_the_selected_axis_solution():
    targets = target_pair()
    common = dict(
        wire_length=0.06,
        terminal_length=0.03,
        wire_axial_offset=-0.022,
        terminal_axial_offset=0.010,
        wire_height_offset=0.0075,
        terminal_height_offset=0.0035,
        pregrasp_height=0.12,
        minimum_grasp_height=0.0,
        reference_orientation=make_pose(
            0.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0
        ),
    )
    normal = make_grasp_pose_set(targets, **common)
    reverse = make_grasp_pose_set(targets, reverse_axis=True, **common)

    assert normal.wire_model.position.x == pytest.approx(reverse.wire_model.position.x)
    assert normal.wire_grasp.position.x < targets.wire.bbox.center.position.x
    assert reverse.wire_grasp.position.x > targets.wire.bbox.center.position.x
    assert axis_from_pose(
        normal.wire_object_frame, (1.0, 0.0, 0.0)
    ) == pytest.approx(
        axis_from_pose(reverse.wire_object_frame, (1.0, 0.0, 0.0))
    )
    assert axis_from_pose(
        normal.wire_grasp_point, (1.0, 0.0, 0.0)
    ) == pytest.approx(
        tuple(
            -value
            for value in axis_from_pose(
                reverse.wire_grasp_point, (1.0, 0.0, 0.0)
            )
        )
    )


def test_tabletop_grasp_height_is_clamped_above_depth_noise():
    targets = target_pair()
    targets.wire.bbox.center.position.z = 0.7505
    poses = make_grasp_pose_set(
        targets,
        wire_length=0.06,
        terminal_length=0.03,
        wire_axial_offset=-0.022,
        terminal_axial_offset=0.010,
        wire_height_offset=0.0075,
        terminal_height_offset=0.0035,
        pregrasp_height=0.12,
        minimum_grasp_height=0.762,
        reference_orientation=make_pose(
            0.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0
        ),
    )

    assert poses.wire_grasp.position.z == pytest.approx(0.762)
    assert poses.wire_grasp_point.position.z == pytest.approx(0.762)
    assert poses.wire_object_frame.position.z == pytest.approx(0.7505)
    assert poses.wire_pregrasp.position.z == pytest.approx(0.882)
