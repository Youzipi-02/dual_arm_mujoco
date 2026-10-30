"""A persistent pick/release Action server; MoveIt owns all path planning."""
from __future__ import annotations

from copy import deepcopy
import math
import threading
import time

from builtin_interfaces.msg import Duration
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from pymoveit2 import MoveIt2, MoveIt2State
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from s622_dual_interfaces.action import ReleaseGrasp, VisionPick
from s622_dual_interfaces.srv import PlanningLease
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from .config import load_defaults
from .grasp_geometry import make_grasp_pose_set, pose_from_vector
from .grasp_tracker import GraspTracker
from .scene_manager import SceneManager
from .target_client import TargetClient


class PlanningFailed(RuntimeError):
    """A failed plan can try the other axial grasp candidate before execution."""


class Cancelled(RuntimeError):
    pass


def slow_cartesian_trajectory(trajectory, velocity, acceleration):
    """Humble GetCartesianPath lacks scaling fields; only stretch its timestamps."""
    scale = min(velocity, math.sqrt(acceleration))
    for point in trajectory.points:
        ns = point.time_from_start.sec * 1_000_000_000 + point.time_from_start.nanosec
        ns = round(ns / scale)
        point.time_from_start = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        point.velocities = [v * scale for v in point.velocities]
        point.accelerations = [a * scale * scale for a in point.accelerations]
    return trajectory


class VisionPickNode(Node):
    def __init__(self):
        super().__init__('vision_pick')
        defaults = load_defaults('s622_dual_descriptions', 'shared_model.yaml')
        defaults.update(load_defaults('vision_pick', 'vision_pick.yaml'))
        self.cfg = {key: self.declare_parameter(key, value).value
                    for key, value in defaults.items() if key != 'use_sim_time'}
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        self._validate_config()
        self._group = ReentrantCallbackGroup()
        self._lock = threading.RLock()
        self._busy = False
        self._fault = False
        self._holding = [False, False]
        self._lease_held = False
        self._pending = set()
        self._phase = 'IDLE'
        self._abort_reason = ''
        self._last_joint_stamp = 0.0
        self._joint_received = 0.0
        self.groups = {}
        for name, joints, tip in (
            ('left_arm', [f'left_j{i}' for i in range(1, 7)], self.p('frames.left_eef')),
            ('right_arm', [f'right_j{i}' for i in range(1, 7)], self.p('frames.right_eef')),
            ('both_arms', [f'{side}_j{i}' for side in ('left', 'right') for i in range(1, 7)], self.p('frames.left_eef')),
            ('both_grippers', [f'{side}_finger{i}_joint' for side in ('left', 'right') for i in (1, 2)], self.p('frames.left_eef')),
        ):
            group = MoveIt2(self, joints, self.p('frames.planning'), tip,
                            group_name=name, callback_group=self._group)
            group.allowed_planning_time = self.p('motion.planning_time')
            group.num_planning_attempts = self.p('motion.planning_attempts')
            group.cartesian_avoid_collisions = True
            self.groups[name] = group
        # Arrival time and source time both matter when the simulation pauses.
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10,
                                 callback_group=self._group)
        self.tracker = GraspTracker(self, self.p('frames.planning'), self.p('frames.left_eef'),
                                    self.p('frames.right_eef'), self.p('frames.wire_object'),
                                    self.p('frames.terminal_object'))
        self.vision = TargetClient(self, self._group, self.p('perception.acquire_targets_action'))
        self._obb = self.create_client(SetBool, self.p('perception.obb_enable_service'), callback_group=self._group)
        self._lease = self.create_client(PlanningLease, '/assembly_control/planning_lease', callback_group=self._group)
        self.scene = SceneManager(
            self, self._group, self.p('frames.planning'), self.p('frames.left_eef'), self.p('frames.right_eef'),
            pose_from_vector(self.p('wire.model_pose')), pose_from_vector(self.p('terminal.model_pose')),
            self.p('wire.radius'), self.p('wire.length'), self.p('terminal.outer_radius'),
            self.p('terminal.inner_radius'), self.p('terminal.length'),
            pose_from_vector(self.p('fixture.model_pose')), self.p('fixture.mesh_resource'),
            self.p('safety.clearance_guard.center'), self.p('safety.clearance_guard.size'),
            self.p('scene.workpiece_collisions_enabled'))
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state = self.create_publisher(String, '~/state', qos)
        self._pick_server = ActionServer(self, VisionPick, '~/pick', self._pick,
            goal_callback=lambda goal: self._reserve(goal, False),
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self._group)
        self._release_server = ActionServer(self, ReleaseGrasp, '~/release', self._release,
            goal_callback=lambda goal: self._reserve(goal, True),
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self._group)
        self.create_service(Trigger, '~/ready', self._ready, callback_group=self._group)
        self.create_service(Trigger, '~/reset', self._reset, callback_group=self._group)
        self.create_timer(0.02, self._maintenance)
        self._publish('IDLE', '等待抓取请求')

    def p(self, key):
        return self.cfg[key]

    def _validate_config(self):
        for key, value in self.cfg.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
            if isinstance(value, list) and any(isinstance(v, float) and not math.isfinite(v) for v in value):
                raise ValueError(f'{key} contains nonfinite values')
        for key in ('pick.motion_timeout', 'pick.scene_timeout', 'pick.cancel_timeout', 'pick.interface_timeout',
                    'vision.confirm_duration', 'vision.wait_timeout', 'vision.pregrasp_height',
                    'motion.cartesian_max_step', 'motion.planning_time'):
            if self.p(key) <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('motion.velocity_scale', 'motion.acceleration_scale', 'motion.approach_velocity_scale',
                    'motion.cartesian_fraction_threshold'):
            if not 0 < self.p(key) <= 1:
                raise ValueError(f'{key} must lie in (0, 1]')
        for key, length in (('left_arm.home', 6), ('right_arm.home', 6), ('wire.stage_joints', 6),
                            ('terminal.stage_joints', 6), ('gripper.open', 2), ('gripper.wire_closed', 2),
                            ('gripper.terminal_closed', 2)):
            if len(self.p(key)) != length:
                raise ValueError(f'{key} needs {length} values')

    def _on_joints(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp > self._last_joint_stamp and -0.1 <= age <= 1.0:
            self._last_joint_stamp = stamp
            self._joint_received = time.monotonic()

    def _idle(self):
        return all(group.query_state() == MoveIt2State.IDLE for group in self.groups.values())

    def _interfaces_ready(self, require_vision=True):
        return (self._joint_received > 0 and time.monotonic() - self._joint_received < 1.0
                and self.scene.interfaces_ready() and self._lease.service_is_ready()
                and (not require_vision or (self.vision.interfaces_ready() and self._obb.service_is_ready()))
                and all(group.joint_state is not None for group in self.groups.values()))

    def _reserve(self, goal, releasing):
        with self._lock:
            if (not goal.execute or not self.p('pick.execute') or self._busy or self._pending
                    or self.vision.cancellation_pending or self._lease_held or not self._idle()
                    or (not releasing and (self._fault or any(self._holding)))):
                return GoalResponse.REJECT
            self._busy = True
            self._abort_reason = ''
        return GoalResponse.ACCEPT

    def _ready(self, _request, response):
        with self._lock:
            response.success = (not self._busy and not self._fault and not any(self._holding)
                                and not self._pending and not self._lease_held and self._idle()
                                and self._interfaces_ready())
        response.message = self._phase
        return response

    def _reset(self, _request, response):
        with self._lock:
            response.success = (not self._busy and not any(self._holding) and not self._pending
                                and not self.vision.cancellation_pending and not self._lease_held and self._idle())
            if response.success:
                self._fault = False
                self.tracker.reset()
                self.vision.reset()
                self._publish('IDLE', '已复位；下一次抓取重新获取快照')
        response.message = 'reset' if response.success else 'release held objects and wait for active operations first'
        return response

    def _publish(self, phase, detail, handle=None):
        self._phase = phase
        self._state.publish(String(data=f'{phase}: {detail}; holding={self._holding}'))
        self.get_logger().info(f'{phase}: {detail}')
        if handle is not None:
            feedback = (VisionPick.Feedback() if isinstance(handle.request, VisionPick.Goal)
                        else ReleaseGrasp.Feedback())
            feedback.phase, feedback.detail = phase, detail
            handle.publish_feedback(feedback)

    def _check(self, handle):
        if not rclpy.ok() or (handle is not None and handle.is_cancel_requested):
            raise Cancelled('operator cancelled; grippers retain their current state')
        if handle is not None and self._abort_reason:
            raise RuntimeError(self._abort_reason)

    def _wait(self, future, timeout, handle=None):
        if future is None:
            raise RuntimeError('ROS interface unavailable')
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(self._forget)
        deadline = time.monotonic() + timeout
        while not future.done():
            self._check(handle)
            if time.monotonic() > deadline:
                raise TimeoutError(f'{self._phase} timed out')
            time.sleep(0.01)
        self._check(handle)
        self._forget(future)
        return future.result()

    def _forget(self, future):
        with self._lock:
            self._pending.discard(future)

    def _scene_result(self, future, handle):
        if not self._wait(future, self.p('pick.scene_timeout'), handle).success:
            raise RuntimeError('MoveIt rejected planning scene update')

    def _remove_workpieces(self, handle):
        response = self._wait(self.scene.query_task_attachments_async(), self.p('pick.scene_timeout'), handle)
        attached = {item.object.id for item in response.scene.robot_state.attached_collision_objects}
        world = {item.id for item in response.scene.world.collision_objects}
        self._scene_result(self.scene.remove_workpieces_async(attached, world), handle)

    def _acquire_lease(self, handle):
        # Do not abandon a lease request on cancel: learn its outcome before cleanup.
        future = self._lease.call_async(PlanningLease.Request(owner=self.get_fully_qualified_name(), acquire=True))
        def granted(done):
            try:
                if done.result().success:
                    self._lease_held = True
            except Exception:
                pass
        future.add_done_callback(granted)
        result = self._wait(future, self.p('pick.interface_timeout'))
        if not result.success:
            raise RuntimeError(result.message)
        self._lease_held = True
        self._check(handle)

    def _release_lease(self):
        if not self._lease_held:
            return
        response = self._wait(self._lease.call_async(PlanningLease.Request(
            owner=self.get_fully_qualified_name(), acquire=False)), self.p('pick.cancel_timeout'))
        if not response.success:
            raise RuntimeError(response.message)
        self._lease_held = False

    def _motion(self, name, handle, *, joints=None, poses=None, cartesian=False, slow=False):
        """Submit this grasp step to pymoveit2, then check its actual result."""
        self._check(handle)
        if time.monotonic() - self._joint_received > 1.0:
            raise RuntimeError('joint feedback is stale')
        group = self.groups[name]
        velocity = self.p('motion.approach_velocity_scale' if slow else 'motion.velocity_scale')
        acceleration = self.p('motion.acceleration_scale')
        group.max_velocity, group.max_acceleration = velocity, acceleration
        kwargs = {'start_joint_state': deepcopy(group.joint_state), 'cartesian': cartesian}
        if joints is not None:
            if len(joints) != len(group.joint_names) or not all(math.isfinite(v) for v in joints):
                raise ValueError('invalid joint goal')
            kwargs.update(joint_positions=list(joints), joint_names=group.joint_names)
            if name == 'both_grippers':
                # Prismatic units are metres: the default 1 mm goal region
                # can consume the entire contact preload of a closed jaw.
                kwargs['tolerance_joint_position'] = 0.00005
        elif cartesian:
            kwargs.update(pose=poses[0][1], max_step=self.p('motion.cartesian_max_step'))
        else:
            # Humble's combined KDL constraint sampler can reject a dirty state.
            # Use pymoveit2's native IK per subgroup, then ONE joint-space plan.
            seed = kwargs['start_joint_state']
            joint_targets = {}
            for side, (link, pose) in zip(('left', 'right'), poses):
                arm = self.groups[side + '_arm']
                ik_future = arm.compute_ik_async(position=pose.position, quat_xyzw=pose.orientation,
                                                ik_link_name=link, start_joint_state=seed)
                self._wait(ik_future, self.p('pick.motion_timeout'), handle)
                solution = arm.get_compute_ik_result(ik_future)
                if solution is None:
                    raise PlanningFailed(f'{side} pregrasp IK failed')
                values = dict(zip(solution.name, solution.position))
                try:
                    joint_targets.update({joint: values[joint] for joint in arm.joint_names})
                except KeyError as error:
                    raise PlanningFailed(f'{side} IK result is missing {error}') from error
            target = [joint_targets[joint] for joint in group.joint_names]
            if not all(math.isfinite(value) for value in target):
                raise PlanningFailed('nonfinite pregrasp IK result')
            kwargs.update(joint_positions=target, joint_names=group.joint_names)
        future = group.plan_async(**kwargs)
        self._wait(future, self.p('pick.motion_timeout'), handle)
        trajectory = group.get_trajectory(future, cartesian=cartesian,
            cartesian_fraction_threshold=self.p('motion.cartesian_fraction_threshold'))
        if trajectory is None or not trajectory.points:
            raise PlanningFailed(f'{name} planning failed or Cartesian path incomplete')
        if cartesian and not hasattr(GetCartesianPath.Request(), 'max_velocity_scaling_factor'):
            slow_cartesian_trajectory(trajectory, velocity, acceleration)
        self._check(handle)
        group.motion_suceeded = False
        group.execute(trajectory)
        deadline = time.monotonic() + self.p('pick.motion_timeout')
        while group.query_state() != MoveIt2State.IDLE:
            self._check(handle)
            if time.monotonic() > deadline:
                raise TimeoutError(f'{name} execution timed out')
            if time.monotonic() - self._joint_received > 1.0:
                raise RuntimeError('joint feedback stopped during execution')
            time.sleep(0.01)
        code = group.get_last_execution_error_code()
        if not group.motion_suceeded or code is None or code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'{name} trajectory execution failed')

    def _poses(self, snapshot, reverse=False):
        return make_grasp_pose_set(snapshot, wire_length=self.p('wire.length'), terminal_length=self.p('terminal.length'),
            wire_axial_offset=self.p('vision.wire_grasp_axial_offset'), terminal_axial_offset=self.p('vision.terminal_grasp_axial_offset'),
            wire_height_offset=self.p('vision.wire_grasp_height_offset'), terminal_height_offset=self.p('vision.terminal_grasp_height_offset'),
            pregrasp_height=self.p('vision.pregrasp_height'), minimum_grasp_height=self.p('vision.minimum_grasp_height'),
            reference_orientation=pose_from_vector(self.p('vision.tabletop_grasp_orientation')), reverse_axis=reverse)

    def _pick_steps(self, handle):
        self._publish('PREPARE', '同步场景、回 Home 并张开夹爪', handle)
        query = self._wait(self.scene.query_task_attachments_async(), self.p('pick.scene_timeout'), handle)
        if any(item.object.id in ('wire', 'copper_terminal') for item in query.scene.robot_state.attached_collision_objects):
            raise RuntimeError('previous workpieces remain attached; explicitly release first')
        world = {item.id for item in query.scene.world.collision_objects}
        self._scene_result(self.scene.sync_world_async(world_ids=world), handle)
        self._motion('both_arms', handle, joints=self.p('left_arm.home') + self.p('right_arm.home'))
        self._motion('both_grippers', handle, joints=self.p('gripper.open') * 2)
        self._publish('ACQUIRE_TARGETS', '获取本轮新鲜视觉快照', handle)
        if not self._wait(self._obb.call_async(SetBool.Request(data=True)), self.p('perception.service_timeout'), handle).success:
            raise RuntimeError('OBB enable failed')
        self.vision.reset()
        if not self.vision.start(expected_frame=self.p('frames.planning'), confirm_duration=self.p('vision.confirm_duration'),
                maximum_gap=self.p('vision.maximum_detection_gap'), maximum_message_age=self.p('vision.maximum_message_age'),
                minimum_confirmations=self.p('vision.minimum_confirmations'), timeout=self.p('vision.wait_timeout')):
            raise RuntimeError('target acquisition unavailable')
        deadline = time.monotonic() + self.p('vision.wait_timeout') + self.p('perception.service_timeout')
        while self.vision.state == 'WAITING':
            self._check(handle)
            if time.monotonic() > deadline:
                raise TimeoutError('target acquisition timed out')
            time.sleep(0.02)
        if self.vision.state != 'SUCCEEDED':
            raise RuntimeError(self.vision.status)
        snapshot = self.vision.snapshot
        for reverse in (False, True):
            poses = self._poses(snapshot, reverse)
            with self._lock:
                self.tracker.set_targets(poses)
            self.scene.update_workpiece_poses(poses.wire_model, poses.terminal_model)
            self._scene_result(self.scene.apply_workpiece_poses_async(), handle)
            self._publish('PREGRASP', '联合规划预抓取位姿' + ('，使用反向候选' if reverse else ''), handle)
            try:
                self._motion('both_arms', handle, poses=[(self.p('frames.left_eef'), poses.wire_pregrasp),
                                                        (self.p('frames.right_eef'), poses.terminal_pregrasp)])
                break
            except PlanningFailed:
                if reverse:
                    raise
        self._remove_workpieces(handle)
        for side, pose in (('left', poses.wire_grasp), ('right', poses.terminal_grasp)):
            self._publish('APPROACH_' + side.upper(), '单臂笛卡尔下探，另一臂保持', handle)
            self._motion(side + '_arm', handle, poses=[(self.p('frames.' + side + '_eef'), pose)], cartesian=True, slow=True)
        self._publish('CLOSE_GRIPPERS', '闭合双夹爪；夹持标志为保守估计', handle)
        self._holding = [True, True]  # Set before dispatch: failure may leave one gripper closed.
        self._motion('both_grippers', handle, joints=self.p('gripper.wire_closed') + self.p('gripper.terminal_closed'), slow=True)
        deadline = time.monotonic() + self.p('pick.scene_timeout')
        while True:
            with self._lock:
                captured = self.tracker.capture(self.p('safety.grasp_eef_position_tolerance'), self.p('safety.grasp_eef_angular_tolerance'))
                relations = self.tracker.relations
            if captured is not None:
                break
            self._check(handle)
            if time.monotonic() > deadline:
                raise TimeoutError('EEF TF unavailable while capturing grasp')
            time.sleep(0.02)
        self._scene_result(self.scene.attach_workpieces_async(relations.wire_model, relations.terminal_model), handle)
        for side, pose in (('left', poses.wire_pregrasp), ('right', poses.terminal_pregrasp)):
            self._publish('LIFT_' + side.upper(), '单臂笛卡尔抬升', handle)
            self._motion(side + '_arm', handle, poses=[(self.p('frames.' + side + '_eef'), pose)], cartesian=True, slow=True)
        self._publish('MOVE_TO_STAGE', '联合规划到暂歇位', handle)
        # Keep the loaded transfer at the same conservative scale as lift.
        # Fast lateral acceleration can walk a cylindrical part off the pad.
        self._motion('both_arms', handle,
                     joints=self.p('wire.stage_joints') + self.p('terminal.stage_joints'),
                     slow=True)

    def _release_steps(self, handle):
        self._publish('OPEN_GRIPPERS', '张开夹爪，解除 MoveIt 工件附着', handle)
        self._motion('both_grippers', handle, joints=self.p('gripper.open') * 2)
        self._remove_workpieces(handle)
        with self._lock:
            self._holding = [False, False]
            self.tracker.reset()
        self.scene.reset_workpiece_poses()
        self._fault = False

    def _run(self, handle, releasing):
        result = ReleaseGrasp.Result() if releasing else VisionPick.Result()
        error = None
        try:
            deadline = time.monotonic() + self.p('pick.interface_timeout')
            while not self._interfaces_ready(require_vision=not releasing):
                self._check(handle)
                if time.monotonic() > deadline:
                    raise TimeoutError('MoveIt, perception or control interfaces unavailable')
                time.sleep(0.05)
            self._acquire_lease(handle)
            (self._release_steps if releasing else self._pick_steps)(handle)
            self._check(handle)
        except Exception as exc:
            error = exc
            self._fault = True
        finally:
            try:
                self.stop_active()
                if not self._idle() or self.vision.cancellation_pending or self._pending:
                    raise RuntimeError('operation termination unconfirmed; planning lease retained')
                self._release_lease()
            except Exception as exc:
                error = error or exc
                self._fault = True
            with self._lock:
                self._busy = False
        if error is None and handle.is_cancel_requested:
            error = Cancelled('cancelled during control handoff')
            self._fault = True
        result.left_holding, result.right_holding = self._holding
        result.success = error is None
        result.error_code = '' if error is None else ('OPERATOR_CANCELLED' if isinstance(error, Cancelled)
                                                     else 'RELEASE_FAILED' if releasing else 'PICK_FAILED')
        result.reason = 'released' if releasing and error is None else ('picked and staged' if error is None else str(error))
        self._publish('HOLDING' if error is None and not releasing else ('IDLE' if error is None else 'ERROR_HOLD'), result.reason)
        if isinstance(error, Cancelled) and handle.is_cancel_requested:
            handle.canceled()
        elif error is None:
            handle.succeed()
        else:
            handle.abort()
        return result

    def _pick(self, handle):
        return self._run(handle, False)

    def _release(self, handle):
        return self._run(handle, True)

    def stop_active(self):
        self.vision.cancel()
        deadline = time.monotonic() + self.p('pick.cancel_timeout')
        while (not self._idle() or self.vision.cancellation_pending or self._pending) and time.monotonic() < deadline:
            for group in self.groups.values():
                if group.query_state() == MoveIt2State.EXECUTING:
                    group.cancel_execution()
            time.sleep(0.02)

    def _maintenance(self):
        try:
            with self._lock:
                self.tracker.broadcast()
                if not self._busy and self._fault:
                    for group in self.groups.values():
                        if group.query_state() == MoveIt2State.EXECUTING:
                            group.cancel_execution()
                if (not self._busy and self._lease_held and self._idle()
                        and not self._pending and not self.vision.cancellation_pending):
                    self._release_lease()
        except Exception as exc:
            self._fault = True
            self._abort_reason = str(exc)
            self.get_logger().error(f'grasp maintenance failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = VisionPickNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_active()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
