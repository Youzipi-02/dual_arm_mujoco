"""Exercise full task rounds through capability boundaries without robot motion."""
from pathlib import Path
from types import SimpleNamespace as NS
import time

import pytest
from rclpy.task import Future
import yaml

from s622_dual_task.states import FIRST_PHASE, VALID_PHASES, TaskPhase as P, TaskState as S
from s622_dual_task.task_runner import TaskRunnerMixin


class Action:
    def __init__(self, history, name):
        self.history, self.name = history, name
        self.cancellation_pending = False
        self.reset()

    def reset(self):
        self.state, self.status = 'IDLE', ''

    def interfaces_ready(self):
        return True

    def start(self, goal):
        self.history.append(self.name)
        self.state = 'WAITING'
        return True

    def cancel(self):
        self.history.append('cancel_' + self.name)
        self.state = 'CANCELLED'


class Runner(TaskRunnerMixin):
    def __init__(self, execute=True, search=True):
        self.cfg = yaml.safe_load((Path(__file__).parents[1] / 'config/task.yaml').read_text())['/**']['ros__parameters']
        self.cfg.update(yaml.safe_load((Path(__file__).parents[2] / 's622_dual_descriptions/config/shared_model.yaml').read_text())['/**']['ros__parameters'])
        self.execute, self.enable_planar_search, self.autostart = execute, search, False
        self.history = []
        self._pick, self._release, self._search = [Action(self.history, name) for name in ('pick', 'release', 'search')]
        self._actions = (self._pick, self._release, self._search)
        self._pick_ready = self._pick_reset = self._lease = self._scene = NS(
            service_is_ready=lambda: True, call_async=self.service)
        self._admittance = NS(start_stop=lambda: self.history.append('stop'), tick_stop=lambda: True)
        self.groups, self._operations = {}, set()
        self._lease_held = False
        self._current_state, self._current_phase = S.INIT, P.RESET_CONTEXT
        self._phase_started = time.monotonic()
        self._future, self._step = None, ''
        self._state_detail, self.error_code = '', ''
        self._autostart_pending = False
        self._state_publisher = NS(publish=lambda msg: None)
        self._tick_joints = lambda name, target: self.history.append('motion_' + name) or True
        self._guard = lambda enabled: self.history.append('guard_' + str(enabled)) or True

    def service(self, request):
        future = Future()
        future.set_result(NS(success=True, message='ready'))
        return future

    def get_fully_qualified_name(self):
        return "/s622_dual_task_fsm"

    def get_logger(self):
        return NS(error=lambda msg: None)

    def until(self, state, phase, limit=60):
        for _ in range(limit):
            if (self.state, self.current_phase) == (state, phase):
                return
            self._tick()
        raise AssertionError((self.state, self.current_phase, self._state_detail))

    def complete_action(self, action, next_state, next_phase):
        action.state, action.status = 'SUCCEEDED', 'done'
        self.until(next_state, next_phase)


@pytest.mark.parametrize('search', [False, True])
def test_two_rounds_call_pick_and_release_once_and_preserve_cleanup_order(search):
    runner = Runner(search=search)
    runner.until(S.IDLE, P.WAIT_TASK)
    for _ in range(2):
        assert runner._on_start(None, NS()).success
        runner.until(S.GRASP_TARGET, P.WAIT_PICK)
        runner.complete_action(runner._pick, S.RIGHT_DOCK, P.START_PLANAR_SEARCH)
        if search:
            runner.until(S.RIGHT_DOCK, P.WAIT_PLANAR_SEARCH)
            runner.complete_action(runner._search, S.COMPLETE, P.RELEASE_ASSEMBLY)
        runner.until(S.COMPLETE, P.WAIT_RELEASE)
        runner.complete_action(runner._release, S.IDLE, P.WAIT_TASK)
    assert runner.history.count('pick') == runner.history.count('release') == 2
    assert runner.history.count('search') == (2 if search else 0)
    assert runner.history.index('release') < runner.history.index('motion_both_arms')
    assert runner.history.index('guard_True') < runner.history.index('motion_right_arm')
    assert runner.history.index('motion_both_arms') < runner.history.index('guard_False')


def test_execute_false_never_dispatches_motion_or_actions():
    runner = Runner(execute=False)
    runner.until(S.IDLE, P.WAIT_TASK)
    assert not runner._on_start(None, NS()).success
    assert runner.history == []


@pytest.mark.parametrize('state', list(S))
def test_failure_in_each_state_never_releases_and_requires_explicit_reset(state):
    runner = Runner(search=False)
    runner._current_state, runner._current_phase = state, FIRST_PHASE[state]
    if state == S.ERROR_HOLD:
        runner._current_phase = P.CANCEL_ACTIVE_GOALS
    else:
        runner._fail('injected')
    runner.until(S.ERROR_HOLD, P.HOLD)
    assert 'release' not in runner.history
    assert runner._on_reset(None, NS()).success
    assert runner.state == S.INIT


def test_cancel_waits_for_terminal_action_and_keeps_objects():
    runner = Runner()
    runner._current_state, runner._current_phase = S.GRASP_TARGET, P.WAIT_PICK
    runner._pick.cancellation_pending = True
    assert runner._on_cancel(None, NS()).success
    runner._tick()
    assert runner.current_phase == P.CANCEL_ACTIVE_GOALS
    assert 'stop' not in runner.history
    assert not runner._on_reset(None, NS()).success
    runner._pick.cancellation_pending = False
    runner.until(S.ERROR_HOLD, P.HOLD)
    assert runner.error_code == 'OPERATOR_CANCELLED'
    assert 'stop' in runner.history and 'release' not in runner.history


def test_release_failure_blocks_retreat_and_home():
    runner = Runner(search=False)
    runner._current_state, runner._current_phase = S.COMPLETE, P.WAIT_RELEASE
    runner._release.state = 'FAILED'
    runner.until(S.ERROR_HOLD, P.HOLD)
    assert not any(value.startswith('motion_') for value in runner.history)


def test_pick_timeout_cancels_instead_of_advancing():
    runner = Runner(search=False)
    runner._current_state, runner._current_phase = S.GRASP_TARGET, P.WAIT_PICK
    runner._phase_started -= runner.cfg['task.pick_timeout'] + 1
    runner.until(S.ERROR_HOLD, P.HOLD)
    assert 'motion_right_arm' not in runner.history


def test_corrupted_position_enters_error_and_invalid_transition_is_atomic():
    runner = Runner(search=False)
    with pytest.raises(ValueError):
        runner._goto_state(S.COMPLETE, 'invalid')
    assert runner.state == S.INIT
    runner._current_phase = P.WAIT_PICK
    runner.until(S.ERROR_HOLD, P.HOLD)


def test_cancel_timeout_still_stops_admittance_but_retains_lease_and_blocks_reset():
    runner = Runner(search=True)
    runner._current_state, runner._current_phase = S.GRASP_TARGET, P.WAIT_PICK
    runner._pick.cancellation_pending = True
    runner._lease_held = True
    runner._fail('cancel did not terminate')
    runner._phase_started -= runner.cfg['task.cancel_timeout'] + 1
    runner.until(S.ERROR_HOLD, P.HOLD)
    assert 'stop' in runner.history
    assert runner._lease_held
    assert not runner._on_reset(None, NS()).success
    runner._pick.cancellation_pending = False
    runner._tick()
    assert not runner._lease_held
    assert runner._on_reset(None, NS()).success


def test_secondary_stop_exception_keeps_original_error_and_reaches_hold():
    runner = Runner(search=True)
    runner._admittance.start_stop = lambda: (_ for _ in ()).throw(RuntimeError('stop failed'))
    runner._fail('original failure')
    runner.until(S.ERROR_HOLD, P.HOLD)
    assert runner._state_detail == 'original failure'
