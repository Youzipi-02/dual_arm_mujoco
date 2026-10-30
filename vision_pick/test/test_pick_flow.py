"""Planning is delegated to pymoveit2; test grasp policy and failure boundaries."""
from types import SimpleNamespace as NS
import threading
import time

from moveit_msgs.msg import MoveItErrorCodes
from pymoveit2 import MoveIt2State
import pytest
from rclpy.action import GoalResponse
from rclpy.task import Future
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from vision_pick.node import VisionPickNode, PlanningFailed, Cancelled, slow_cartesian_trajectory


def completed(value):
    result = Future()
    result.set_result(value)
    return result


def test_humble_cartesian_scaling_preserves_positions_and_limits_derivatives():
    trajectory = JointTrajectory(points=[JointTrajectoryPoint(positions=[1.0], velocities=[2.0], accelerations=[4.0],
                                                              time_from_start=Duration(sec=1))])
    slow_cartesian_trajectory(trajectory, 0.1, 0.25)
    point = trajectory.points[0]
    assert list(point.positions) == [1.0]
    assert point.time_from_start.sec == 10
    assert point.velocities == pytest.approx([0.2])
    assert point.accelerations == pytest.approx([0.04])


@pytest.mark.parametrize('busy,fault,holding,execute,releasing,accepted', [
    (False, False, False, True, False, True),
    (True, False, False, True, False, False),
    (False, True, False, True, False, False),
    (False, False, True, True, False, False),
    (False, False, False, False, False, False),
    (False, True, True, True, True, True),
])
def test_goal_gate_and_explicit_release_of_faulted_hold(busy, fault, holding, execute, releasing, accepted):
    node = NS(_lock=threading.RLock(), _busy=busy, _fault=fault, _holding=[holding, holding],
              _pending=set(), _lease_held=False, _idle=lambda: True,
              vision=NS(cancellation_pending=False), p=lambda key: True)
    result = VisionPickNode._reserve(node, NS(execute=execute), releasing)
    assert (result == GoalResponse.ACCEPT) == accepted


def flow():
    events = []
    cfg = {'pick.scene_timeout': 1.0, 'perception.service_timeout': 1.0, 'vision.wait_timeout': 1.0,
           'frames.planning': 'world', 'frames.left_eef': 'left_tip', 'frames.right_eef': 'right_tip'}
    def parameter(key):
        return cfg.get(key, [0.0, 0.0] if key.startswith('gripper.') else [0.0] * 6 if key.endswith(('joints', '.home')) else 1.0)
    poses = NS(wire_model='wire', terminal_model='terminal', wire_pregrasp='left_pre', terminal_pregrasp='right_pre',
               wire_grasp='left_grasp', terminal_grasp='right_grasp')
    def vision_start(**kwargs):
        vision.state = 'SUCCEEDED'
        return True
    vision = NS(reset=lambda: None, start=vision_start, state='IDLE', snapshot=object(), status='done')
    scene = NS(query_task_attachments_async=lambda: completed(NS(scene=NS(robot_state=NS(attached_collision_objects=[]), world=NS(collision_objects=[])))),
               sync_world_async=lambda **kw: completed(NS(success=True)), update_workpiece_poses=lambda *a: None,
               apply_workpiece_poses_async=lambda: completed(NS(success=True)),
               attach_workpieces_async=lambda *a: events.append('attach') or completed(NS(success=True)))
    tracker = NS(set_targets=lambda value: events.append('targets'), capture=lambda *a: True,
                 relations=NS(wire_model='wire_relative', terminal_model='terminal_relative'))
    node = NS(p=parameter, scene=scene, tracker=tracker, vision=vision, _lock=threading.RLock(),
              _obb=NS(call_async=lambda req: completed(NS(success=True))),
              _holding=[False, False], _check=lambda handle: None,
              _publish=lambda phase, detail, handle: events.append(phase),
              _wait=lambda future, timeout, handle=None: future.result(),
              _scene_result=lambda future, handle: future.result(),
              _remove_workpieces=lambda handle: events.append('remove'),
              _poses=lambda snapshot, reverse: events.append(('candidate', snapshot, reverse)) or poses,
              _motion=lambda group, handle, **kw: events.append(('motion', group, kw)))
    return node, events


def test_pick_locks_one_snapshot_and_uses_sequential_cartesian_approach_and_lift():
    node, events = flow()
    VisionPickNode._pick_steps(node, object())
    candidates = [event for event in events if isinstance(event, tuple) and event[0] == 'candidate']
    assert len(candidates) == 1
    assert candidates[0][1] is node.vision.snapshot
    cartesian = [event[1] for event in events if isinstance(event, tuple) and event[0] == 'motion' and event[2].get('cartesian')]
    assert cartesian == ['left_arm', 'right_arm', 'left_arm', 'right_arm']
    assert events.index('CLOSE_GRIPPERS') < events.index('attach') < events.index('LIFT_LEFT')
    assert node._holding == [True, True]
    assert events[-1][1] == 'both_arms'


@pytest.mark.parametrize('exception,expected_candidates', [(PlanningFailed('no plan'), 2), (RuntimeError('execution failed'), 1)])
def test_only_planning_failure_retries_the_same_snapshot_with_reverse_axis(exception, expected_candidates):
    node, events = flow()
    original = node._motion
    attempted = []
    def motion(group, handle, **kw):
        if group == 'both_arms' and 'poses' in kw and not attempted:
            attempted.append(True)
            raise exception
        original(group, handle, **kw)
    node._motion = motion
    if isinstance(exception, PlanningFailed):
        VisionPickNode._pick_steps(node, object())
    else:
        with pytest.raises(RuntimeError):
            VisionPickNode._pick_steps(node, object())
    candidates = [event for event in events if isinstance(event, tuple) and event[0] == 'candidate']
    assert len(candidates) == expected_candidates
    assert all(event[1] is node.vision.snapshot for event in candidates)


def test_close_failure_keeps_conservative_hold_and_does_not_lift():
    node, events = flow()
    original = node._motion
    def motion(group, handle, **kw):
        if 'CLOSE_GRIPPERS' in events:
            raise RuntimeError('one gripper failed')
        original(group, handle, **kw)
    node._motion = motion
    with pytest.raises(RuntimeError):
        VisionPickNode._pick_steps(node, object())
    assert node._holding == [True, True]
    assert 'attach' not in events and 'LIFT_LEFT' not in events


def test_release_scene_failure_does_not_clear_tracking_or_hold():
    events = []
    node = NS(_holding=[True, True], _publish=lambda *a: None, p=lambda k: [0.0, 0.0],
              _motion=lambda *a, **kw: events.append('open'),
              tracker=NS(reset=lambda: events.append('clear')))
    def fail(handle):
        raise RuntimeError('scene apply failed')
    node._remove_workpieces = fail
    with pytest.raises(RuntimeError):
        VisionPickNode._release_steps(node, object())
    assert events == ['open'] and node._holding == [True, True]


def test_pregrasp_uses_native_subgroup_ik_and_one_combined_joint_plan():
    events = []
    trajectory = JointTrajectory(points=[JointTrajectoryPoint(positions=[0.1, 0.2])])
    group = NS(joint_state=object(), joint_names=['left_j1', 'right_j1'],
               plan_async=lambda **kw: events.append(('plan', kw)) or completed(object()),
               get_trajectory=lambda *a, **kw: trajectory,
               query_state=lambda: MoveIt2State.IDLE,
               get_last_execution_error_code=lambda: NS(val=MoveItErrorCodes.SUCCESS))
    def arm(side):
        def ik(**kwargs):
            events.append(('ik', side, kwargs['start_joint_state']))
            # Full-robot responses contain the other arm's seed. Do not overwrite
            # the first arm's solution when consuming the second IK response.
            return completed(NS(name=['left_j1', 'right_j1'],
                                position=[0.1, -1.0] if side == 'left' else [-1.0, 0.2]))
        return NS(joint_names=[side + '_j1'], compute_ik_async=ik,
                  get_compute_ik_result=lambda future: future.result())
    def execute(value):
        events.append('execute')
        group.motion_suceeded = True
    group.execute = execute
    node = NS(groups={'both_arms': group, 'left_arm': arm('left'), 'right_arm': arm('right')},
              _joint_received=time.monotonic(), p=lambda k: 1.0,
              _check=lambda handle: None, _wait=lambda future, *a: future.result())
    pose = NS(position=object(), orientation=object())
    VisionPickNode._motion(node, 'both_arms', object(), poses=[('left_tip', pose), ('right_tip', pose)])
    ik_calls = [event for event in events if isinstance(event, tuple) and event[0] == 'ik']
    assert len(ik_calls) == 2 and ik_calls[0][2] is ik_calls[1][2]
    plans = [event[1] for event in events if isinstance(event, tuple) and event[0] == 'plan']
    assert len(plans) == 1 and plans[0]['joint_positions'] == [0.1, 0.2]
    assert events[-1] == 'execute'


def test_gripper_planning_region_cannot_consume_contact_preload():
    calls = []
    group = NS(joint_state=object(), joint_names=['finger1', 'finger2'],
               plan_async=lambda **kw: calls.append(kw) or completed(object()),
               get_trajectory=lambda *a, **kw: JointTrajectory(points=[JointTrajectoryPoint()]),
               query_state=lambda: MoveIt2State.IDLE,
               get_last_execution_error_code=lambda: NS(val=MoveItErrorCodes.SUCCESS))
    group.execute = lambda trajectory: setattr(group, 'motion_suceeded', True)
    node = NS(groups={'both_grippers': group}, _joint_received=time.monotonic(),
              p=lambda key: 1.0, _check=lambda handle: None,
              _wait=lambda future, *a: future.result())
    VisionPickNode._motion(node, 'both_grippers', object(), joints=[0.0025, -0.0025])
    request = calls[0]
    # Even the widest accepted endpoint leaves at least 0.95 mm compression
    # relative to the wire's 3.5 mm radius.
    assert request['joint_positions'][0] + request['tolerance_joint_position'] <= 0.00255
    assert request['joint_positions'][1] - request['tolerance_joint_position'] >= -0.00255


def test_cancelled_or_unstopped_pick_retains_lease_and_never_invokes_release():
    from s622_dual_interfaces.action import VisionPick
    events = []
    node = NS(p=lambda key: 1.0, _interfaces_ready=lambda **kw: True,
        _acquire_lease=lambda handle: None, _holding=[True, False],
        _release_steps=lambda handle: events.append('release'), _check=lambda handle: None,
        stop_active=lambda: events.append('stop'), _idle=lambda: False,
        _release_lease=lambda: events.append('give_up_lease'), _lock=threading.RLock(), _busy=True,
        vision=NS(cancellation_pending=False), _pending=set(), _publish=lambda *a: None)
    def cancel(handle):
        raise Cancelled('cancelled')
    node._pick_steps = cancel
    handle = NS(is_cancel_requested=True, canceled=lambda: events.append('cancelled'),
                abort=lambda: events.append('aborted'))
    result = VisionPickNode._run(node, handle, False)
    assert not result.success and result.left_holding and not result.right_holding
    assert result.error_code == 'OPERATOR_CANCELLED'
    assert events == ['stop', 'cancelled']


def test_release_readiness_does_not_depend_on_perception_being_alive():
    node = NS(_joint_received=time.monotonic(), scene=NS(interfaces_ready=lambda: True),
              _lease=NS(service_is_ready=lambda: True), vision=NS(interfaces_ready=lambda: False),
              _obb=NS(service_is_ready=lambda: False), groups={'arm': NS(joint_state=object())})
    assert not VisionPickNode._interfaces_ready(node)
    assert VisionPickNode._interfaces_ready(node, require_vision=False)


def test_cancel_during_final_handoff_returns_cancelled_instead_of_success():
    events = []
    handle = NS(is_cancel_requested=False, canceled=lambda: events.append('cancelled'),
                succeed=lambda: events.append('succeeded'), abort=lambda: events.append('aborted'))
    node = NS(p=lambda key: 1.0, _interfaces_ready=lambda **kw: True,
        _acquire_lease=lambda handle: None, _holding=[True, True],
        _pick_steps=lambda handle: None, _check=lambda handle: None,
        stop_active=lambda: None, _idle=lambda: True,
        _release_lease=lambda: setattr(handle, 'is_cancel_requested', True),
        _lock=threading.RLock(), _busy=True, vision=NS(cancellation_pending=False),
        _pending=set(), _publish=lambda *a: None)
    result = VisionPickNode._run(node, handle, False)
    assert not result.success and result.error_code == 'OPERATOR_CANCELLED'
    assert events == ['cancelled']
