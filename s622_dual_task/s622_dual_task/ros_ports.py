"""Existing nonblocking admittance shutdown sequence."""
import time
from geometry_msgs.msg import TwistStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from s622_dual_interfaces.msg import AdmittanceStatus
from s622_dual_interfaces.srv import ConfigureAdmittance
from std_srvs.srv import SetBool


class DualAdmittancePort:
    """Nonblocking reference, mode, enable and status endpoints."""

    FAULTS = {"STALE_WRENCH", "TF_UNAVAILABLE", "CALC_ERROR"}

    def __init__(self, node, callback_group):
        """创建左右导纳及运动控制模式的服务和状态订阅。

        参数:
            node (Node): 持有服务客户端和订阅的 ROS 节点。
            callback_group (CallbackGroup): 异步服务与状态订阅使用的回调组。

        返回:
            None: 通信端点和内部状态保存在当前实例中。
        """
        self._node = node
        self._zero_velocity = {
            arm: node.create_publisher(TwistStamped, f"/{arm}_admittance/nominal_velocity_command", 1)
            for arm in ("left", "right")
        }
        self._mode = node.create_client(
            SetBool, "/assembly_control/set_compliant_mode", callback_group=callback_group
        )
        self._configure = {
            arm: node.create_client(
                ConfigureAdmittance, f"/{arm}_admittance/configure",
                callback_group=callback_group,
            ) for arm in ("left", "right")
        }
        self._enable = {
            arm: node.create_client(
                SetBool, f"/{arm}_admittance/enable", callback_group=callback_group
            ) for arm in ("left", "right")
        }
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status = {}
        self._running_once = set()
        self._subscriptions = [
            node.create_subscription(
                AdmittanceStatus, f"/{arm}_admittance/status",
                lambda message, arm=arm: self._on_status(arm, message),
                qos, callback_group=callback_group,
            ) for arm in ("left", "right")
        ]
        self._futures = []
        self._stop_stage = None
        self._stop_started = 0.0
        self.last_error = ""

    def _on_status(self, arm, message):
        """记录指定机械臂最新的导纳状态及本地接收时间。

        参数:
            arm (str): 机械臂名称，取 ``left`` 或 ``right``。
            message (AdmittanceStatus): 导纳节点发布的状态消息。

        返回:
            None: 更新状态缓存；首次运行时记录该机械臂。
        """
        self._status[arm] = (message, time.monotonic())
        if message.state == "RUNNING":
            self._running_once.add(arm)

    def ready(self):
        """检查控制模式切换、配置和启停服务是否全部就绪。

        返回:
            bool: 全部服务就绪时为 ``True``，否则为 ``False``。
        """
        return self._mode.service_is_ready() and all(
            client.service_is_ready()
            for client in (*self._configure.values(), *self._enable.values())
        )

    def configure(self, nominal_velocity, desired_wrench):
        """异步向左右导纳节点发送相同的参考运动和期望力。

        参数:
            nominal_velocity (Sequence[float]): 六维名义速度。
            desired_wrench (Sequence[float]): 六维期望力与力矩。

        返回:
            None: 请求结果需随后通过 ``poll()`` 获取。
        """
        self._futures = []
        for client in self._configure.values():
            request = ConfigureAdmittance.Request()
            request.nominal_velocity = list(nominal_velocity)
            request.desired_wrench = list(desired_wrench)
            self._futures.append(client.call_async(request))

    def switch_mode(self, enabled):
        """异步请求运动服务切换柔顺控制模式。

        参数:
            enabled (bool): ``True`` 进入柔顺模式，``False`` 退出。

        返回:
            None: 请求结果需随后通过 ``poll()`` 获取。
        """
        self._futures = [self._mode.call_async(SetBool.Request(data=enabled))]

    def enable(self, enabled):
        """异步启停左右导纳节点，并在启用时清除旧状态。

        参数:
            enabled (bool): ``True`` 启用双侧导纳，``False`` 停止。

        返回:
            None: 请求结果需随后通过 ``poll()`` 获取。
        """
        if enabled:
            self._status.clear()
            self._running_once.clear()
        self._futures = [
            client.call_async(SetBool.Request(data=enabled))
            for client in self._enable.values()
        ]

    def poll(self):
        """查询最近一批异步服务请求是否全部完成且成功。

        返回:
            Optional[bool]: 尚无请求或仍有请求未完成时为 ``None``；
                全部成功时为 ``True``，服务失败或回调异常时为 ``False``。
                失败原因同时写入 ``last_error``。
        """
        if not self._futures or not all(future.done() for future in self._futures):
            return None
        try:
            responses = [future.result() for future in self._futures]
            failed = [response.message for response in responses if not response.success]
        except Exception as error:
            failed = [str(error)]
        self._futures = []
        self.last_error = "; ".join(failed)
        return not failed

    def running(self, maximum_status_age):
        """判断左右导纳是否都在运行且状态消息足够新鲜。

        参数:
            maximum_status_age (float): 允许的最大本地接收间隔，单位为秒。

        返回:
            bool: 两侧最近状态均为 ``RUNNING`` 且未超时时为 ``True``。
        """
        now = time.monotonic()
        for arm in ("left", "right"):
            item = self._status.get(arm)
            if item is None or item[0].state != "RUNNING":
                return False
            if now - item[1] > maximum_status_age:
                return False
        return True

    def fault(self, maximum_status_age):
        """检查导纳故障、退出运行状态及运行心跳超时。

        参数:
            maximum_status_age (float): 运行状态允许的最大接收间隔，单位为秒。

        返回:
            str: 检测到问题时返回包含机械臂和原因的描述；否则返回空串。
        """
        now = time.monotonic()
        for arm in ("left", "right"):
            item = self._status.get(arm)
            if item is None:
                continue
            message, received = item
            if message.state in self.FAULTS:
                return f"{arm}: {message.state}: {message.detail}"
            if arm in self._running_once and message.state != "RUNNING":
                return f"{arm}: admittance left RUNNING: {message.state}"
            if message.state == "RUNNING" and now - received > maximum_status_age:
                return f"{arm}: admittance status heartbeat timed out"
        return ""

    def stop_output(self):
        """Zero nominal velocity and request both controllers to stop."""
        self._stop_stage = None
        for arm, publisher in self._zero_velocity.items():
            message = TwistStamped()
            message.header.frame_id = f"{arm}_grasp_frame"
            message.header.stamp = self._node.get_clock().now().to_msg()
            publisher.publish(message)
        self.enable(False)

    def start_stop(self):
        """启动非阻塞停止流程，先停导纳再退出柔顺模式。

        返回:
            None: 只发起停止或等待在途请求；后续调用 ``tick_stop()`` 推进。
        """
        if self._stop_stage is not None:
            return
        if self._futures and not all(future.done() for future in self._futures):
            self._stop_stage = "WAIT_PENDING"
        else:
            self.stop_output()
            self._stop_stage = "DISABLE_ADMITTANCE"
        self._stop_started = time.monotonic()

    def tick_stop(self):
        """推进停止流程并查询双侧导纳和控制模式是否已关闭。

        返回:
            Optional[bool]: 尚在停止过程中为 ``None``；全部完成为 ``True``；
                服务失败或停止阶段超时为 ``False``，原因保存在 ``last_error``。
        """
        if self._stop_stage is None:
            return True
        if self._stop_stage == "WAIT_PENDING":
            if (all(future.done() for future in self._futures) or
                    time.monotonic() - self._stop_started > 3.0):
                self.stop_output()
                self._stop_stage = "DISABLE_ADMITTANCE"
                self._stop_started = time.monotonic()
            return None
        if time.monotonic() - self._stop_started > 3.0:
            self.last_error = f"{self._stop_stage} timed out"
            self._stop_stage = None
            self._futures = []
            return False
        if self._stop_stage == "DISABLE_ADMITTANCE":
            result = self.poll()
            if result is None:
                return None
            if not result:
                self._stop_stage = None
                return False
            self.switch_mode(False)
            self._stop_stage = "DISABLE_MODE"
            self._stop_started = time.monotonic()
            return None
        result = self.poll()
        if result is not None:
            self._stop_stage = None
        return result
