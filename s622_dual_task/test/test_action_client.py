from types import SimpleNamespace as NS

from action_msgs.msg import GoalStatus
import pytest
from rclpy.task import Future

from s622_dual_task.action_client import TaskActionClient


def client():
    adapter = TaskActionClient.__new__(TaskActionClient)
    adapter._pending = set()
    adapter._request_id = 0
    adapter.reset()
    acceptance = Future()
    adapter._action = NS(server_is_ready=lambda: True, send_goal_async=lambda *a, **kw: acceptance)
    return adapter, acceptance


def test_cancel_before_acceptance_waits_for_terminal_and_blocks_reset():
    adapter, acceptance = client()
    terminal = Future()
    assert adapter.start(object())
    adapter.cancel()
    with pytest.raises(RuntimeError):
        adapter.reset()
    cancelled = []
    acceptance.set_result(NS(accepted=True, cancel_goal_async=lambda: cancelled.append(True), get_result_async=lambda: terminal))
    assert cancelled == [True]
    assert adapter.cancellation_pending
    terminal.set_result(NS(status=GoalStatus.STATUS_CANCELED, result=NS(success=False, reason='cancelled')))
    assert not adapter.cancellation_pending
    assert adapter.state == 'CANCELLED'
    adapter.reset()
    assert adapter.state == 'IDLE'


@pytest.mark.parametrize('status,success,expected', [
    (GoalStatus.STATUS_SUCCEEDED, True, 'SUCCEEDED'),
    (GoalStatus.STATUS_ABORTED, True, 'FAILED'),
    (GoalStatus.STATUS_SUCCEEDED, False, 'FAILED'),
])
def test_ros_terminal_status_and_business_result_must_both_succeed(status, success, expected):
    adapter, acceptance = client()
    terminal = Future()
    adapter.start(object())
    acceptance.set_result(NS(accepted=True, get_result_async=lambda: terminal))
    terminal.set_result(NS(status=status, result=NS(success=success, reason='result')))
    assert adapter.state == expected


def test_rejection_and_send_exception_clear_pending_requests():
    adapter, acceptance = client()
    adapter.start(object())
    acceptance.set_result(NS(accepted=False))
    assert adapter.state == 'FAILED' and not adapter.cancellation_pending
    def fail(*a, **kw):
        raise RuntimeError('send failed')
    adapter._action.send_goal_async = fail
    assert not adapter.start(object())
    assert not adapter.cancellation_pending


def test_stale_feedback_cannot_change_current_goal():
    adapter, _ = client()
    old = adapter._request_id
    adapter.reset()
    adapter._feedback(NS(feedback=NS(phase='late', detail='old')), old)
    assert adapter.status == ''
