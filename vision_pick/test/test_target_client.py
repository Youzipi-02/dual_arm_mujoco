"""Check that a cancelled target action cannot return an obsolete snapshot."""

from rclpy.task import Future

from vision_pick.target_client import TargetClient


class AcceptedGoal:
    """Record cancellation of an accepted acquisition goal."""

    accepted = True

    def __init__(self):
        self.cancel_calls = 0

    def cancel_goal_async(self):
        self.cancel_calls += 1


def test_cancel_before_target_goal_response():
    """Cancel an accepted goal even when its response arrives after cancel."""

    client = TargetClient.__new__(TargetClient)
    client.reset()
    client.state = "WAITING"
    goal = AcceptedGoal()
    response = Future()
    response.set_result(goal)

    client.cancel()
    client._goal_response(response, client._request_id)

    assert goal.cancel_calls == 1
    assert client.state == "CANCELLED"
    assert client.snapshot is None


def test_cancelled_target_stays_pending_until_terminal_result():
    from types import SimpleNamespace
    client = TargetClient.__new__(TargetClient)
    client.reset()
    acceptance, terminal = Future(), Future()
    client._action = SimpleNamespace(server_is_ready=lambda: True,
                                    send_goal_async=lambda *_args, **_kwargs: acceptance)
    assert client.start(expected_frame='world', confirm_duration=2.0, maximum_gap=0.75,
                        maximum_message_age=1.0, minimum_confirmations=3, timeout=15.0)
    client.cancel()
    acceptance.set_result(SimpleNamespace(accepted=True, cancel_goal_async=lambda: None,
                                         get_result_async=lambda: terminal))
    assert client.cancellation_pending
    terminal.set_result(SimpleNamespace(result=SimpleNamespace(success=False)))
    assert not client.cancellation_pending
    assert client.state == 'CANCELLED' and client.snapshot is None
