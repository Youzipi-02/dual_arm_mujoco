"""The directly used pymoveit2 path must remain pollable on rejected/late goals."""
from types import SimpleNamespace as NS
import threading

from pymoveit2 import MoveIt2, MoveIt2State
from rclpy.task import Future
import pytest


def interface(ready=True):
    group = MoveIt2.__new__(MoveIt2)
    group._MoveIt2__execution_mutex = threading.Lock()
    group._MoveIt2__is_motion_requested = False
    group._MoveIt2__is_executing = False
    group._node = NS(get_logger=lambda: NS(warn=lambda msg: None, error=lambda msg: None))
    group._execute_trajectory_action_client = NS(server_is_ready=lambda: ready)
    return group


def test_missing_execute_server_and_send_failure_do_not_lock_state_queries():
    group = interface(False)
    group._send_goal_async_execute_trajectory(object())
    assert group.query_state() == MoveIt2State.IDLE
    group = interface()
    def fail(**kwargs):
        raise RuntimeError('send failed')
    group._execute_trajectory_action_client.send_goal_async = fail
    with pytest.raises(RuntimeError):
        group._send_goal_async_execute_trajectory(object())
    assert group.query_state() == MoveIt2State.IDLE


@pytest.mark.parametrize('accepted,error', [(False, False), (False, True)])
def test_rejected_and_failed_acceptance_clear_request_state(accepted, error):
    group = interface()
    group._MoveIt2__is_motion_requested = True
    future = Future()
    if error:
        future.set_exception(RuntimeError('accept failed'))
    else:
        future.set_result(NS(accepted=accepted))
    group._MoveIt2__response_callback_execute_trajectory(future)
    assert group.query_state() == MoveIt2State.IDLE
    assert not group.motion_suceeded
