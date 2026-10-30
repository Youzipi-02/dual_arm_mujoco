"""Check task transitions and compliant handoff without starting controllers."""

import time
from types import SimpleNamespace

from s622_dual_interfaces.msg import AdmittanceStatus
from s622_dual_task.ros_ports import DualAdmittancePort
from s622_dual_task.states import FIRST_PHASE, TaskState, may_transition


class CompletedFuture:
    def done(self):
        return True

    def result(self):
        return SimpleNamespace(success=True, message="ok")


class PendingFuture(CompletedFuture):
    ready = False

    def done(self):
        return self.ready


class RecordingClient:
    def __init__(self, label, history):
        self.label = label
        self.history = history

    def call_async(self, request):
        self.history.append((self.label, request.data))
        return CompletedFuture()


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class RecordingNode:
    def __init__(self):
        self.publishers = {}
        self.callbacks = {}

    def create_publisher(self, _message_type, topic, _qos):
        publisher = RecordingPublisher()
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(
        self, _message_type, topic, callback, _qos, *, callback_group
    ):
        del callback_group
        self.callbacks[topic] = callback
        return callback






def test_stop_disables_both_admittance_nodes_before_followers():
    history = []
    port = DualAdmittancePort.__new__(DualAdmittancePort)
    port._zero_velocity = {}
    port._enable = {
        arm: RecordingClient(arm, history) for arm in ("left", "right")
    }
    port._mode = RecordingClient("mode", history)
    port._futures = []
    port._stop_stage = None
    port._stop_started = 0.0
    port.last_error = ""

    port.start_stop()
    assert history == [("left", False), ("right", False)]
    assert port.tick_stop() is None
    assert history[-1] == ("mode", False)
    assert port.tick_stop() is True


def test_running_requires_fresh_status_from_both_arms():
    port = DualAdmittancePort.__new__(DualAdmittancePort)
    port._status = {}
    port._running_once = set()
    message = AdmittanceStatus()
    message.state = "RUNNING"
    now = time.monotonic()
    port._status["left"] = (message, now)
    assert not port.running(0.5)
    port._status["right"] = (message, now)
    assert port.running(0.5)
    port._status["right"] = (message, now - 1.0)
    assert not port.running(0.5)
    assert "heartbeat timed out" in port.fault(0.5)


def test_stop_waits_for_inflight_enable_before_disabling():
    history = []
    pending = PendingFuture()
    port = DualAdmittancePort.__new__(DualAdmittancePort)
    port._zero_velocity = {}
    port._enable = {
        arm: RecordingClient(arm, history) for arm in ("left", "right")
    }
    port._mode = RecordingClient("mode", history)
    port._futures = [pending]
    port._stop_stage = None
    port._stop_started = 0.0
    port.last_error = ""
    port.start_stop()
    assert history == []
    assert port.tick_stop() is None
    pending.ready = True
    assert port.tick_stop() is None
    assert history == [("left", False), ("right", False)]
    assert port.tick_stop() is None
    assert port.tick_stop() is True


def test_fault_state_is_reported_to_task():
    port = DualAdmittancePort.__new__(DualAdmittancePort)
    message = AdmittanceStatus()
    message.state = "STALE_WRENCH"
    message.detail = "source-stamped wrench timed out"
    port._status = {"left": (message, time.monotonic())}
    port._running_once = set()
    assert "left: STALE_WRENCH" in port.fault(0.5)


def test_unexpected_disable_after_running_is_fault():
    port = DualAdmittancePort.__new__(DualAdmittancePort)
    message = AdmittanceStatus()
    message.state = "DISABLED"
    port._status = {"left": (message, time.monotonic())}
    port._running_once = {"left"}
    assert "left RUNNING" in port.fault(0.5)
