"""Nonblocking clients for task capabilities, including late-goal cancellation."""
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient


class TaskActionClient:
    def __init__(self, node, group, action_type, name):
        self._action = ActionClient(node, action_type, name, callback_group=group)
        self._pending = set()
        self._request_id = 0
        self.reset()

    def interfaces_ready(self):
        return self._action.server_is_ready()

    @property
    def cancellation_pending(self):
        return bool(self._pending)

    def reset(self):
        if self._pending:
            raise RuntimeError('previous Action has not terminated')
        self._request_id += 1
        self._handle = None
        self._cancel_requested = False
        self.state, self.status, self.result = 'IDLE', '', None

    def start(self, goal):
        if self._pending or not self.interfaces_ready():
            return False
        self.reset()
        request_id = self._request_id
        self._pending.add(request_id)
        self.state = 'WAITING'
        try:
            self._action.send_goal_async(goal, feedback_callback=lambda msg: self._feedback(msg, request_id)).add_done_callback(
                lambda future: self._accepted(future, request_id))
        except Exception as error:
            self._pending.discard(request_id)
            self.state, self.status = 'FAILED', str(error)
            return False
        return True

    def _feedback(self, message, request_id):
        if request_id == self._request_id and not self._cancel_requested:
            feedback = message.feedback
            self.status = f'{feedback.phase}: {getattr(feedback, "detail", "")}'

    def _accepted(self, future, request_id):
        try:
            handle = future.result()
            if not handle.accepted:
                raise RuntimeError('goal rejected')
            self._handle = handle
            if self._cancel_requested or request_id != self._request_id:
                handle.cancel_goal_async()
            handle.get_result_async().add_done_callback(lambda done: self._finished(done, request_id))
        except Exception as error:
            self._pending.discard(request_id)
            self.state, self.status = 'FAILED', str(error)

    def _finished(self, future, request_id):
        if request_id != self._request_id:
            self._pending.discard(request_id)
            return
        try:
            wrapped = future.result()
            self.result = wrapped.result
            self.status = self.result.reason
            self.state = ('CANCELLED' if self._cancel_requested else
                          'SUCCEEDED' if wrapped.status == GoalStatus.STATUS_SUCCEEDED and self.result.success else 'FAILED')
        except Exception as error:
            self.state, self.status = 'FAILED', str(error)
        finally:
            self._pending.discard(request_id)

    def cancel(self):
        if self._pending:
            self._cancel_requested = True
            if self._handle is not None:
                self._handle.cancel_goal_async()
