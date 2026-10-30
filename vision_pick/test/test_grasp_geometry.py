"""测试两工件坐标系、抓取坐标系及其刚体关系计算。"""

import math

import pytest

from vision_pick.grasp_geometry import (
    axis_from_pose,
    compose_pose,
    inverse_pose,
    make_grasp_frame_from_center_axis,
    make_grasp_pose_from_frame,
    make_pose,
    make_object_frame_from_center_axis,
    pose_from_vector,
    relative_pose,
)


def test_pose_quaternion_is_normalized():
    pose = pose_from_vector([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0])
    assert pose.orientation.w == 1.0


def test_grasp_frame_uses_grasp_point_axis_and_world_up():
    half_sqrt = math.sqrt(0.5)
    frame = make_grasp_frame_from_center_axis(
        center=(0.10, 0.20, 0.75),
        detected_axis=(1.0, 1.0, 0.4),
        axial_offset=0.02,
        height_offset=0.01,
    )

    assert frame.position.x == pytest.approx(0.10 + 0.02 * half_sqrt)
    assert frame.position.y == pytest.approx(0.20 + 0.02 * half_sqrt)
    assert frame.position.z == pytest.approx(0.76)
    assert axis_from_pose(frame, (1.0, 0.0, 0.0)) == pytest.approx(
        (half_sqrt, half_sqrt, 0.0)
    )
    assert axis_from_pose(frame, (0.0, 1.0, 0.0)) == pytest.approx(
        (-half_sqrt, half_sqrt, 0.0)
    )
    assert axis_from_pose(frame, (0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )


def test_object_frame_origin_is_exact_yolo_center_without_grasp_offsets():
    frame = make_object_frame_from_center_axis(
        center=(0.10, 0.20, 0.75),
        detected_axis=(1.0, 1.0, 0.4),
    )
    half_sqrt = math.sqrt(0.5)

    assert frame.position.x == pytest.approx(0.10)
    assert frame.position.y == pytest.approx(0.20)
    assert frame.position.z == pytest.approx(0.75)
    assert axis_from_pose(frame, (1.0, 0.0, 0.0)) == pytest.approx(
        (half_sqrt, half_sqrt, 0.0)
    )
    assert axis_from_pose(frame, (0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, 1.0)
    )


def test_eef_pose_is_derived_from_grasp_frame_without_moving_its_origin():
    frame = make_grasp_frame_from_center_axis(
        center=(0.10, 0.20, 0.75),
        detected_axis=(0.0, 1.0, 0.0),
        axial_offset=0.02,
        height_offset=0.01,
    )
    reference = make_pose(
        0.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0
    )
    eef_pose = make_grasp_pose_from_frame(frame, reference)

    assert eef_pose.position.x == pytest.approx(frame.position.x)
    assert eef_pose.position.y == pytest.approx(frame.position.y)
    assert eef_pose.position.z == pytest.approx(frame.position.z)
    assert axis_from_pose(eef_pose, (0.0, 0.0, 1.0)) != pytest.approx(
        (0.0, 0.0, 1.0)
    )


def test_tabletop_tcp_calibration_points_outward_axis_down_without_translation():
    """传感器长度/固定旋转只能来自 URDF，不能再次进入视觉抓取偏移。"""

    half_sqrt = math.sqrt(0.5)
    grasp_point = make_grasp_frame_from_center_axis(
        center=(0.10, 0.20, 0.80),
        detected_axis=(1.0, 0.0, 0.0),
        axial_offset=0.0,
        height_offset=0.0,
    )
    reference = make_pose(
        0.0, 0.0, 0.0, half_sqrt, half_sqrt, 0.0, 0.0
    )

    tcp_target = make_grasp_pose_from_frame(grasp_point, reference)

    assert (
        tcp_target.position.x,
        tcp_target.position.y,
        tcp_target.position.z,
    ) == pytest.approx((0.10, 0.20, 0.80))
    assert axis_from_pose(tcp_target, (0.0, 0.0, 1.0)) == pytest.approx(
        (0.0, 0.0, -1.0)
    )
    assert axis_from_pose(tcp_target, (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_relative_pose_reconstructs_child_pose():
    parent = make_pose(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    child = make_pose(1.5, 1.5, 4.0, 0.0, 0.0, 0.0, 1.0)
    reconstructed = compose_pose(parent, relative_pose(parent, child))
    assert reconstructed.position.x == pytest.approx(child.position.x)
    assert reconstructed.position.y == pytest.approx(child.position.y)
    assert reconstructed.position.z == pytest.approx(child.position.z)


def test_captured_object_frame_relation_follows_moving_eef_rigidly():
    captured_eef = make_pose(0.40, 0.20, 0.80)
    captured_frame = make_pose(0.45, 0.20, 0.80)
    eef_to_frame = relative_pose(captured_eef, captured_frame)

    # 切换到附着跟随的瞬间必须重建出完全相同的世界位姿。
    switch_frame = compose_pose(captured_eef, eef_to_frame)
    assert switch_frame.position.x == pytest.approx(captured_frame.position.x)
    assert switch_frame.position.y == pytest.approx(captured_frame.position.y)
    assert switch_frame.position.z == pytest.approx(captured_frame.position.z)

    half_sqrt = math.sqrt(0.5)
    moved_eef = make_pose(
        0.50, -0.10, 1.00, 0.0, 0.0, half_sqrt, half_sqrt
    )
    moved_frame = compose_pose(moved_eef, eef_to_frame)

    assert moved_frame.position.x == pytest.approx(0.50)
    assert moved_frame.position.y == pytest.approx(-0.05)
    assert moved_frame.position.z == pytest.approx(1.00)
    assert axis_from_pose(moved_frame, (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_pose_inverse_produces_identity():
    pose = make_pose(0.4, -0.2, 0.8, 0.2, 0.3, -0.1, 0.9)
    identity = compose_pose(pose, inverse_pose(pose))
    assert identity.position.x == pytest.approx(0.0, abs=1.0e-9)
    assert identity.position.y == pytest.approx(0.0, abs=1.0e-9)
    assert identity.position.z == pytest.approx(0.0, abs=1.0e-9)
    assert identity.orientation.w == pytest.approx(1.0, abs=1.0e-9)


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        pose_from_vector([0.0] * 7)
