"""Task milestones; grasp phases belong exclusively to vision_pick."""
from enum import Enum, auto


class TaskState(Enum):
    INIT = auto()
    IDLE = auto()
    GRASP_TARGET = auto()
    RIGHT_PREDOCK = auto()
    RIGHT_DOCK = auto()
    COMPLETE = auto()
    ERROR_HOLD = auto()


class TaskPhase(Enum):
    RESET_CONTEXT = auto()
    WAIT_INTERFACES = auto()
    WAIT_TASK = auto()
    START_PICK = auto()
    WAIT_PICK = auto()
    ADD_CLEARANCE_GUARD = auto()
    MOVE_TO_PREDOCK = auto()
    START_PLANAR_SEARCH = auto()
    WAIT_PLANAR_SEARCH = auto()
    RELEASE_ASSEMBLY = auto()
    WAIT_RELEASE = auto()
    RETREAT = auto()
    RESET_SCENE = auto()
    MOVE_HOME = auto()
    CANCEL_ACTIVE_GOALS = auto()
    STOP_ADMITTANCE = auto()
    HOLD = auto()


_PHASES = {
    TaskState.INIT: (TaskPhase.RESET_CONTEXT, TaskPhase.WAIT_INTERFACES),
    TaskState.IDLE: (TaskPhase.WAIT_TASK,),
    TaskState.GRASP_TARGET: (TaskPhase.START_PICK, TaskPhase.WAIT_PICK),
    TaskState.RIGHT_PREDOCK: (TaskPhase.ADD_CLEARANCE_GUARD, TaskPhase.MOVE_TO_PREDOCK),
    TaskState.RIGHT_DOCK: (TaskPhase.START_PLANAR_SEARCH, TaskPhase.WAIT_PLANAR_SEARCH),
    TaskState.COMPLETE: (TaskPhase.RELEASE_ASSEMBLY, TaskPhase.WAIT_RELEASE, TaskPhase.RETREAT, TaskPhase.RESET_SCENE, TaskPhase.MOVE_HOME),
    TaskState.ERROR_HOLD: (TaskPhase.CANCEL_ACTIVE_GOALS, TaskPhase.STOP_ADMITTANCE, TaskPhase.HOLD),
}
FIRST_PHASE = {state: phases[0] for state, phases in _PHASES.items()}
VALID_PHASES = {state: frozenset(phases) for state, phases in _PHASES.items()}
SUCCESSORS = {
    TaskState.INIT: {TaskState.IDLE}, TaskState.IDLE: {TaskState.GRASP_TARGET},
    TaskState.GRASP_TARGET: {TaskState.RIGHT_PREDOCK}, TaskState.RIGHT_PREDOCK: {TaskState.RIGHT_DOCK},
    TaskState.RIGHT_DOCK: {TaskState.COMPLETE}, TaskState.COMPLETE: {TaskState.IDLE},
    TaskState.ERROR_HOLD: {TaskState.INIT},
}


def valid_position(state, phase):
    return phase in VALID_PHASES.get(state, ())


def may_transition(previous, next_state):
    return (next_state == TaskState.ERROR_HOLD and previous != next_state
            or next_state in SUCCESSORS.get(previous, ()))
