"""Task orchestration; perception and grasp execution live in vision_pick."""
import time

from pymoveit2 import MoveIt2
from moveit_msgs.srv import ApplyPlanningScene
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from s622_dual_interfaces.action import PlanarSearch, ReleaseGrasp, VisionPick
from s622_dual_interfaces.srv import PlanningLease
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .action_client import TaskActionClient
from .ros_ports import DualAdmittancePort
from .states import FIRST_PHASE, TaskState
from .task_config import TaskConfigMixin
from .task_runner import TaskRunnerMixin


class FSM(TaskRunnerMixin, TaskConfigMixin, Node):
    def __init__(self):
        super().__init__('s622_dual_task_fsm')
        self._load_parameters()
        self._group = ReentrantCallbackGroup()
        self._fsm_group = MutuallyExclusiveCallbackGroup()
        self._pick = TaskActionClient(self, self._group, VisionPick, self.cfg['pick.action'])
        self._release = TaskActionClient(self, self._group, ReleaseGrasp, self.cfg['pick.release_action'])
        self._search = TaskActionClient(self, self._group, PlanarSearch, self.cfg['assembly.planar_search_action'])
        self._actions = (self._pick, self._release, self._search)
        self._pick_ready = self.create_client(Trigger, '/vision_pick/ready', callback_group=self._group)
        self._pick_reset = self.create_client(Trigger, '/vision_pick/reset', callback_group=self._group)
        self._lease = self.create_client(PlanningLease, '/assembly_control/planning_lease', callback_group=self._group)
        self._scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene', callback_group=self._group)
        self._admittance = DualAdmittancePort(self, self._group)
        self.groups = {
            name: MoveIt2(self, joints, self.cfg['frames.planning'], tip,
                          group_name=name, callback_group=self._group)
            for name, joints, tip in (
                ('right_arm', [f'right_j{i}' for i in range(1, 7)], self.cfg['frames.right_eef']),
                ('both_arms', [f'{side}_j{i}' for side in ('left', 'right') for i in range(1, 7)], self.cfg['frames.left_eef']))
        }
        self._operations = set()
        self._joint_received = 0.0
        self._last_joint_stamp = 0.0
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10, callback_group=self._group)
        self._lease_held = False
        self._current_state = TaskState.INIT
        self._current_phase = FIRST_PHASE[self._current_state]
        self._phase_started = time.monotonic()
        self._future = None
        self._step = ''
        self._autostart_pending = self.autostart
        self.error_code = ''
        self._state_detail = '初始化'
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state_publisher = self.create_publisher(String, '~/state', qos)
        for name, callback in (('start', self._on_start), ('cancel', self._on_cancel), ('reset', self._on_reset)):
            self.create_service(Trigger, '~/' + name, callback, callback_group=self._fsm_group)
        self.create_timer(1.0 / self.cfg['task.fsm_rate'], self._tick, callback_group=self._fsm_group)
        self._publish_state()
        if self.cfg['status.enabled']:
            self.create_timer(self.cfg['status.period'], self._publish_state, callback_group=self._fsm_group)

    def _on_joints(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp > self._last_joint_stamp and -0.1 <= age <= 1.0:
            self._last_joint_stamp = stamp
            self._joint_received = time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    node = FSM()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for client in node._actions:
            client.cancel()
        for group in node.groups.values():
            group.cancel_execution()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
