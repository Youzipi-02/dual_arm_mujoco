"""Control ownership must survive partial failures and reject competing writers."""
import threading
from types import SimpleNamespace as NS

from rclpy.task import Future

from s622_dual_assembly.control_mode import ControlMode


class Client:
    def __init__(self, label, history):
        self.label = label
        self.history = history
        self.ready = True
        self.success = True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.history.append((self.label, request.data))
        future = Future()
        future.set_result(NS(success=self.success, message='ok' if self.success else 'TF unavailable'))
        return future


def manager():
    node = ControlMode.__new__(ControlMode)
    node._lock = threading.Lock()
    node._owner = ''
    node._compliant = False
    node._initialized = True
    node._inflight = []
    node._followers_enabled = True
    node._settle = 0.0
    node._active_arms = set()
    node.history = []
    node._followers = [Client(f'{arm} follower', node.history) for arm in ('left', 'right')]
    node._admittance = {arm: Client(arm, node.history) for arm in ('left', 'right')}
    return node


def lease(node, owner, acquire):
    return node._lease(NS(owner=owner, acquire=acquire), NS())


def enable(node, arm, enabled=True):
    return node._enable_arm(arm, NS(data=enabled), NS())


def test_only_one_planner_can_own_and_release_the_robot():
    node = manager()
    assert lease(node, 'pick', True).success
    assert not lease(node, 'task', True).success
    assert not lease(node, 'task', False).success
    assert not node._mode(NS(data=True), NS()).success
    assert lease(node, 'pick', False).success
    assert lease(node, 'task', True).success


def test_compliant_mode_excludes_planning_and_disable_restores_access():
    node = manager()
    node._switch_followers = lambda enabled: None
    assert node._mode(NS(data=True), NS()).success
    assert not lease(node, 'pick', True).success
    assert node._mode(NS(data=False), NS()).success
    assert lease(node, 'pick', True).success


def test_unknown_follower_state_blocks_planners_until_explicit_disable_succeeds():
    node = manager()
    def fail(enabled):
        node._initialized = False
        raise RuntimeError('partial enable')
    node._switch_followers = fail
    assert not node._mode(NS(data=True), NS()).success
    assert not lease(node, 'pick', True).success
    node._switch_followers = lambda enabled: setattr(node, '_initialized', True)
    assert node._mode(NS(data=False), NS()).success
    assert lease(node, 'pick', True).success


def test_late_enable_must_finish_before_a_disable_can_be_sent():
    node = manager()
    node._inflight = [Future()]
    node._initialized = False
    node._compliant = True
    assert not node._mode(NS(data=False), NS()).success
    assert not lease(node, 'pick', True).success
    node._inflight[0].set_result(NS(success=True))
    assert node._mode(NS(data=False), NS()).success
    assert lease(node, 'pick', True).success


def test_unconfigured_followers_cannot_enter_compliant_mode():
    node = manager()
    node._followers_enabled = False
    assert not node._mode(NS(data=True), NS()).success
    assert lease(node, 'pick', True).success


def test_one_enable_request_starts_execution_before_admittance():
    node = manager()
    assert enable(node, 'right').success
    assert node.history == [('left follower', True), ('right follower', True), ('right', True)]
    assert node._active_arms == {'right'}
    assert not lease(node, 'pick', True).success
    assert enable(node, 'right', False).success
    assert node.history[-3:] == [('right', False), ('left follower', False), ('right follower', False)]
    assert lease(node, 'pick', True).success


def test_enabling_second_arm_does_not_reset_first_arm_or_followers():
    node = manager()
    assert enable(node, 'left').success
    node.history.clear()
    assert enable(node, 'right').success
    assert node.history == [('right', True)]
    assert enable(node, 'left', False).success
    assert node.history[-1] == ('left', False)
    assert node._active_arms == {'right'}
    assert node._compliant
    assert not lease(node, 'pick', True).success
    assert enable(node, 'right', False).success
    assert lease(node, 'pick', True).success


def test_planning_lease_rejects_arm_enable_without_sending_commands():
    node = manager()
    assert lease(node, 'pick', True).success
    assert not enable(node, 'right').success
    assert node.history == []


def test_first_request_clears_stale_controllers_after_manager_restart():
    node = manager()
    node._initialized = False
    assert enable(node, 'right').success
    assert node.history == [
        ('left follower', False), ('right follower', False),
        ('left', False), ('right', False),
        ('left follower', True), ('right follower', True), ('right', True),
    ]


def test_first_planning_lease_also_confirms_a_full_stop():
    node = manager()
    node._initialized = False
    assert lease(node, 'pick', True).success
    assert node.history == [
        ('left follower', False), ('right follower', False), ('left', False), ('right', False),
    ]


def test_rejected_enable_blocks_planning_until_disable_is_confirmed():
    node = manager()
    node._admittance['right'].success = False
    response = enable(node, 'right')
    assert not response.success
    assert 'TF unavailable' in response.message
    assert not lease(node, 'pick', True).success
    assert not enable(node, 'left').success
    node._admittance['right'].success = True
    assert enable(node, 'right', False).success
    assert not node._active_arms
    assert lease(node, 'pick', True).success


def test_missing_follower_sends_no_partial_enable_and_can_recover():
    node = manager()
    node._followers[1].ready = False
    assert not enable(node, 'right').success
    assert node.history == []
    assert not lease(node, 'pick', True).success
    node._followers[1].ready = True
    assert enable(node, 'right', False).success
    assert lease(node, 'pick', True).success


def test_late_arm_enable_cannot_override_a_new_disable():
    node = manager()
    pending = Future()
    node._inflight = [pending]
    node._initialized = False
    node._compliant = True
    assert not enable(node, 'right', False).success
    assert node.history == []
    assert not lease(node, 'pick', True).success
    pending.set_result(NS(success=True))
    assert enable(node, 'right', False).success
    assert node.history == [
        ('left follower', False), ('right follower', False), ('left', False), ('right', False),
    ]
    assert lease(node, 'pick', True).success


def test_legacy_mode_sequence_is_compatible_and_full_disable_clears_all_arms():
    node = manager()
    assert node._mode(NS(data=True), NS()).success
    assert enable(node, 'left').success
    assert enable(node, 'right').success
    node.history.clear()
    assert node._mode(NS(data=True), NS()).success
    assert node.history == []
    assert node._mode(NS(data=False), NS()).success
    assert node.history == [
        ('left follower', False), ('right follower', False), ('left', False), ('right', False),
    ]
    assert not node._active_arms
    assert lease(node, 'pick', True).success
