"""Verify captured object frames follow the measured end effector poses."""

from types import SimpleNamespace

from builtin_interfaces.msg import Time

from vision_pick.grasp_geometry import GraspPoseSet, make_pose
from vision_pick.grasp_tracker import GraspTracker


def test_object_frames_follow_eef_after_checked_capture():
    wire_eef = make_pose(0.0, 0.0, 0.0)
    terminal_eef = make_pose(1.0, 0.0, 0.0)
    tracker = GraspTracker.__new__(GraspTracker)
    tracker.poses = GraspPoseSet(
        wire_model=make_pose(0.01, 0.0, 0.0),
        terminal_model=make_pose(1.01, 0.0, 0.0),
        wire_object_frame=make_pose(0.02, 0.0, 0.0),
        terminal_object_frame=make_pose(1.03, 0.0, 0.0),
        wire_grasp_point=wire_eef,
        terminal_grasp_point=terminal_eef,
        wire_pregrasp=wire_eef,
        terminal_pregrasp=terminal_eef,
        wire_grasp=wire_eef,
        terminal_grasp=terminal_eef,
    )
    eef = {True: wire_eef, False: terminal_eef}
    tracker.lookup_eef_pose = lambda left: eef[left]
    tracker.relations = None
    tracker._tf_unavailable = False
    tracker._planning_frame = "world"
    tracker._wire_frame = "wire_object_frame"
    tracker._terminal_frame = "terminal_object_frame"
    sent = []
    tracker._broadcaster = SimpleNamespace(sendTransform=lambda value: sent.extend(value))
    tracker._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=1))
        )
    )

    relations = tracker.capture(0.001, 0.01)
    assert relations is not None
    assert relations.wire_object_frame.position.x == 0.02
    assert abs(relations.terminal_object_frame.position.x - 0.03) < 1.0e-9

    eef[True] = make_pose(0.1, 0.0, 0.0)
    eef[False] = make_pose(1.1, 0.0, 0.0)
    tracker.broadcast()

    assert len(sent) == 2
    assert sent[0].header.frame_id == "world"
    assert sent[0].child_frame_id == "wire_object_frame"
    assert abs(sent[0].transform.translation.x - 0.12) < 1.0e-9
    assert abs(sent[1].transform.translation.x - 1.13) < 1.0e-9


def test_stale_eef_tf_is_not_republished_as_fresh_object_pose():
    from geometry_msgs.msg import TransformStamped
    transform = TransformStamped()
    transform.header.stamp.sec = 1
    transform.transform.rotation.w = 1.0
    tracker = GraspTracker.__new__(GraspTracker)
    tracker._left_eef, tracker._right_eef, tracker._planning_frame = 'left', 'right', 'world'
    tracker._buffer = SimpleNamespace(lookup_transform=lambda *args: transform)
    tracker._node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=3_000_000_000)))
    assert tracker.lookup_eef_pose(True) is None
    transform.header.stamp.sec = 3
    assert tracker.lookup_eef_pose(True) is not None
