"""Task-facing client for perception's confirmed-target action."""

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from s622_dual_interfaces.action import AcquireTargets
from .grasp_geometry import TargetPair


class TargetClient:
    """Keep acquisition feedback and the immutable result outside the FSM."""

    def __init__(self, node, callback_group, action_name):
        """创建双目标确认 Action 客户端并初始化请求状态。

        参数:
            node (Node): 持有 Action 客户端的 ROS 节点。
            callback_group (CallbackGroup): Action 回调使用的回调组。
            action_name (str): ``AcquireTargets`` Action 的名称。

        返回:
            None: 客户端和初始状态保存在当前实例中。
        """
        self._action = ActionClient(
            node, AcquireTargets, action_name, callback_group=callback_group
        )
        self.reset()

    def reset(self):
        """清空目标快照并使之前请求的异步回调失效。

        返回:
            None: 状态重置为 ``IDLE``，请求编号递增。
        """
        self._request_id = getattr(self, "_request_id", 0) + 1
        self.state = "IDLE"
        self.status = "等待视觉目标"
        self.snapshot = None
        self._handle = None
        self._cancel_requested = False


    def interfaces_ready(self):
        return self._action.server_is_ready()

    @property
    def cancellation_pending(self):
        return bool(getattr(self, "_pending_requests", set()))

    def start(self, *, expected_frame, confirm_duration, maximum_gap,
              maximum_message_age, minimum_confirmations, timeout):
        """向感知节点异步请求一对连续可见的目标。

        参数:
            expected_frame (str): 返回目标所需的坐标系。
            confirm_duration (float): 连续确认所需时长，单位为秒。
            maximum_gap (float): 连续检测允许的最大间隔，单位为秒。
            maximum_message_age (float): 检测消息允许的最大年龄，单位为秒。
            minimum_confirmations (int): 连续确认所需的最少帧数。
            timeout (float): 感知端完成请求的时限，单位为秒。

        返回:
            bool: Action 服务端就绪并已提交请求时为 ``True``；
                服务端未就绪时为 ``False``。结果需通过 ``state`` 和
                ``snapshot`` 异步读取。
        """
        if not self._action.server_is_ready():
            return False
        goal = AcquireTargets.Goal()
        goal.expected_frame = expected_frame
        goal.confirm_duration_sec = confirm_duration
        goal.maximum_gap_sec = maximum_gap
        goal.maximum_message_age_sec = maximum_message_age
        goal.minimum_confirmations = minimum_confirmations
        goal.timeout_sec = timeout
        self._request_id += 1
        request_id = self._request_id
        if not hasattr(self, "_pending_requests"):
            self._pending_requests = set()
        self._pending_requests.add(request_id)
        self._cancel_requested = False
        self._handle = None
        self.state = "WAITING"
        try:
            self._action.send_goal_async(
                goal, feedback_callback=lambda message: self._feedback(message, request_id)
            ).add_done_callback(lambda future: self._goal_response(future, request_id))
        except Exception as error:
            self._pending_requests.discard(request_id)
            self.state = "FAILED"
            self.status = str(error)
            return False
        return True

    def _feedback(self, message, request_id):
        """更新当前请求的感知反馈，忽略已取消或过期的请求。

        参数:
            message (AcquireTargets.FeedbackMessage): Action 反馈消息。
            request_id (int): 发送请求时记录的编号。

        返回:
            None: 有效反馈的文字状态写入 ``status``。
        """
        if request_id != self._request_id or self._cancel_requested:
            return
        self.status = message.feedback.status

    def _goal_response(self, future, request_id):
        """处理目标接受结果，并为已接受的请求注册结果回调。

        参数:
            future (Future): ``send_goal_async`` 返回的 Future。
            request_id (int): 发送请求时记录的编号。

        返回:
            None: 拒绝或回调异常会将当前请求设为 ``FAILED``；
                迟到的已接受目标会被取消。
        """
        try:
            handle = future.result()
            if not handle.accepted:
                raise RuntimeError("target acquisition rejected")
            if request_id != self._request_id or self._cancel_requested:
                handle.cancel_goal_async()
                handle.get_result_async().add_done_callback(
                    lambda result_future: self._result(result_future, request_id))
                return
            self._handle = handle
            handle.get_result_async().add_done_callback(
                lambda result_future: self._result(result_future, request_id)
            )
        except Exception as error:
            getattr(self, "_pending_requests", set()).discard(request_id)
            if request_id != self._request_id or self._cancel_requested:
                return
            self.state = "FAILED"
            self.status = str(error)

    def _result(self, future, request_id):
        """保存当前请求返回的双目标快照或失败原因。

        参数:
            future (Future): ``get_result_async`` 返回的 Future。
            request_id (int): 发送请求时记录的编号。

        返回:
            None: 成功时写入 ``snapshot`` 并设为 ``SUCCEEDED``；
                结果失败或回调异常时设为 ``FAILED``。
        """
        getattr(self, "_pending_requests", set()).discard(request_id)
        if request_id != self._request_id or self._cancel_requested:
            return
        try:
            wrapped = future.result()
            result = wrapped.result
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED or not result.success:
                raise RuntimeError(result.reason)
            self.snapshot = TargetPair(result.wire, result.terminal)
            self.state = "SUCCEEDED"
            self.status = "双目标确认完成"
        except Exception as error:
            self.state = "FAILED"
            self.status = str(error)

    def cancel(self):
        """取消正在等待的目标确认请求并忽略后续迟到回调。

        返回:
            None: 仅在 ``WAITING`` 状态下标记取消；如果已取得目标句柄，
                同时向 Action 服务端发送取消请求。
        """
        if self.state != "WAITING":
            return
        self._cancel_requested = True
        self.state = "CANCELLED"
        if self._handle is not None:
            self._handle.cancel_goal_async()
