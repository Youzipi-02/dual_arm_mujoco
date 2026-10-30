"""Nonblocking orchestration of pick, predock, search and release."""
import time

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from pymoveit2 import MoveIt2State
from s622_dual_interfaces.action import PlanarSearch, ReleaseGrasp, VisionPick
from s622_dual_interfaces.srv import PlanningLease
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .states import FIRST_PHASE, TaskPhase as P, TaskState as S, may_transition, valid_position


class TaskRunnerMixin:
    @property
    def current_state(self):
        return self._current_state

    @property
    def current_phase(self):
        return self._current_phase

    @property
    def state(self):
        return self.current_state

    @property
    def phase(self):
        return self.current_phase.name

    def _on_start(self, _request, response):
        response.success = self.execute and (self.state, self.current_phase) == (S.IDLE, P.WAIT_TASK)
        if response.success:
            self._autostart_pending = False
            self._goto_state(S.GRASP_TARGET, '启动视觉抓取')
        response.message = 'started' if response.success else 'task is not ready or task.execute is false'
        return response

    def _on_cancel(self, _request, response):
        response.success = self.state not in (S.IDLE, S.ERROR_HOLD)
        response.message = 'cancellation requested' if response.success else 'no active task'
        if response.success:
            self._fail('操作人员取消；保持夹爪', 'OPERATOR_CANCELLED')
        return response

    def _on_reset(self, _request, response):
        response.success = ((self.state, self.current_phase) == (S.ERROR_HOLD, P.HOLD)
                            and not self._cancellation_pending() and not self._lease_held)
        response.message = 'resetting; vision_pick must also be free of held objects' if response.success else 'wait for ERROR_HOLD/HOLD and all operations to stop'
        if response.success:
            self._autostart_pending = False
            self._goto_state(S.INIT, '显式复位，重新检查抓取节点和控制模式')
        return response

    def _set_position(self, state, phase, detail):
        if not valid_position(state, phase) or (state != self.state and not may_transition(self.state, state)):
            raise ValueError(f'invalid task transition to {state}/{phase}')
        self._current_state, self._current_phase = state, phase
        self._phase_started = time.monotonic()
        self._state_detail = detail
        self._future = None
        self._step = ''
        self._publish_state()

    def _goto_state(self, state, detail):
        self._set_position(state, FIRST_PHASE[state], detail)

    def _goto_phase(self, phase, detail):
        self._set_position(self.state, phase, detail)

    def _publish_state(self):
        self._state_publisher.publish(String(data=f'{self.state.name}/{self.phase}: [{self.error_code}] {self._state_detail}'))

    def _detail(self, detail):
        if detail != self._state_detail:
            self._state_detail = detail
            self._publish_state()

    def _fail(self, detail, error_code='TASK_FAILED'):
        self.get_logger().error(detail)
        if self.state != S.ERROR_HOLD:
            self.error_code = error_code
            self._goto_state(S.ERROR_HOLD, detail)

    def _cancellation_pending(self):
        return (any(client.cancellation_pending for client in self._actions)
                or any(group.query_state() != MoveIt2State.IDLE for group in self.groups.values())
                or any(not future.done() for future in tuple(self._operations)))

    def _track(self, future):
        self._operations.add(future)
        future.add_done_callback(lambda done: self._operations.discard(done))
        return future

    def _service(self, client, request):
        if self._future is None:
            if not client.service_is_ready():
                return None
            self._future = client.call_async(request)
            if isinstance(request, PlanningLease.Request) and request.acquire:
                self._future.add_done_callback(self._lease_granted)
            self._track(self._future)
        if not self._future.done():
            return None
        response = self._future.result()
        self._future = None
        return response

    def _lease_granted(self, future):
        if future.result().success:
            self._lease_held = True

    def _guard(self, enabled):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        collision = CollisionObject(id='central_clearance_guard')
        collision.header.frame_id = self.cfg['frames.planning']
        collision.operation = CollisionObject.ADD if enabled else CollisionObject.REMOVE
        if enabled:
            shape = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=self.cfg['safety.clearance_guard.size'])
            pose = Pose()
            pose.orientation.w = 1.0
            pose.position.x, pose.position.y, pose.position.z = self.cfg['safety.clearance_guard.center']
            collision.primitives, collision.primitive_poses = [shape], [pose]
        request.scene.world.collision_objects = [collision]
        response = self._service(self._scene, request)
        if response is not None and not response.success:
            raise RuntimeError('clearance guard update failed')
        return response is not None

    def _tick(self):
        try:
            if not valid_position(self.state, self.current_phase):
                raise ValueError('corrupted task position')
            if self.state == S.ERROR_HOLD:
                self._tick_error()
                return
            timeout = self.cfg['task.interface_timeout']
            if self.state == S.GRASP_TARGET:
                timeout = self.cfg['task.pick_timeout']
            elif self.state == S.RIGHT_DOCK:
                timeout = self.cfg['task.search_timeout']
            elif self.current_phase in (P.RELEASE_ASSEMBLY, P.WAIT_RELEASE):
                timeout = self.cfg['task.release_timeout']
            elif self.state in (S.RIGHT_PREDOCK, S.COMPLETE):
                timeout = self.cfg['task.motion_timeout']
            if self.state != S.IDLE and time.monotonic() - self._phase_started > timeout:
                raise TimeoutError(f'{self.state.name}/{self.phase} timed out')
            if self.state == S.INIT:
                self._tick_init()
            elif self.state == S.IDLE:
                if self.execute and self._autostart_pending:
                    self._autostart_pending = False
                    self._goto_state(S.GRASP_TARGET, '自动启动一次')
            elif self.state == S.GRASP_TARGET:
                if self.current_phase == P.START_PICK:
                    if self._pick.start(VisionPick.Goal(execute=True)):
                        self._goto_phase(P.WAIT_PICK, 'vision_pick 抓取中')
                elif self._action_done(self._pick):
                    self._goto_state(S.RIGHT_PREDOCK, '抓取成功，工件 TF 由 vision_pick 持续维护')
            elif self.state == S.RIGHT_PREDOCK:
                if self.current_phase == P.ADD_CLEARANCE_GUARD:
                    if self._guard(True):
                        self._goto_phase(P.MOVE_TO_PREDOCK, '中央净空约束已加入')
                elif self._tick_joints('right_arm', self.cfg['terminal.predock_joints']):
                    self._goto_state(S.RIGHT_DOCK, '右臂已到预设预对接位')
            elif self.state == S.RIGHT_DOCK:
                self._tick_search()
            elif self.state == S.COMPLETE:
                self._tick_complete()
        except Exception as error:
            if self.state == S.ERROR_HOLD:
                self.get_logger().error(f'error recovery failed: {error}')
                if self.current_phase != P.HOLD:
                    self._goto_phase(P.HOLD, self._state_detail)
            else:
                self._fail(str(error))

    def _tick_init(self):
        if self.current_phase == P.RESET_CONTEXT:
            response = self._service(self._pick_reset, Trigger.Request())
            if response is None:
                return
            if not response.success:
                raise RuntimeError(response.message)
            self.error_code = ''
            for client in self._actions:
                client.reset()
            self._goto_phase(P.WAIT_INTERFACES, '等待抓取和控制接口')
        elif all(client.interfaces_ready() for client in (self._pick, self._release)) and self._lease.service_is_ready():
            response = self._service(self._pick_ready, Trigger.Request())
            if response is not None and response.success:
                self._goto_state(S.IDLE, '接口就绪；等待任务')

    def _action_done(self, client):
        self._detail(client.status)
        if client.state in ('FAILED', 'CANCELLED'):
            raise RuntimeError(client.status)
        return client.state == 'SUCCEEDED'

    def _tick_search(self):
        if self.current_phase == P.START_PLANAR_SEARCH:
            if not self.enable_planar_search:
                self._goto_state(S.COMPLETE, '跳过平面搜孔')
            elif self._search.start(PlanarSearch.Goal(execute=True)):
                self._goto_phase(P.WAIT_PLANAR_SEARCH, '装配 Action 搜孔中')
        elif self._action_done(self._search):
            self._goto_state(S.COMPLETE, '搜孔完成')

    def _tick_complete(self):
        if self.current_phase == P.RELEASE_ASSEMBLY:
            if self.enable_planar_search:
                if not self._step:
                    self._admittance.start_stop()
                    self._step = 'STOP'
                result = self._admittance.tick_stop()
                if result is None:
                    return
                if not result:
                    raise RuntimeError(self._admittance.last_error)
            if self._release.start(ReleaseGrasp.Goal(execute=True)):
                self._goto_phase(P.WAIT_RELEASE, '等待释放和场景清理')
        elif self.current_phase == P.WAIT_RELEASE:
            if self._action_done(self._release):
                self._goto_phase(P.RETREAT, '释放完成，撤退到暂歇位')
        elif self.current_phase == P.RETREAT:
            if self._tick_joints('both_arms', self.cfg['wire.stage_joints'] + self.cfg['terminal.stage_joints']):
                self._goto_phase(P.RESET_SCENE, '撤退完成，移除中央临时约束')
        elif self.current_phase == P.RESET_SCENE:
            if self._guard(False):
                self._goto_phase(P.MOVE_HOME, '回 Home')
        elif self.current_phase == P.MOVE_HOME:
            if self._tick_joints('both_arms', self.cfg['left_arm.home'] + self.cfg['right_arm.home']):
                self._goto_state(S.IDLE, '本轮完成；等待下一次启动')

    def _tick_joints(self, name, target):
        # These are the task's predock/cleanup motions; no generic motion Action layer.
        group = self.groups[name]
        if not self._step:
            if group.joint_state is None or time.monotonic() - self._joint_received > 1.0:
                raise RuntimeError('joint feedback unavailable or stale')
            response = self._service(self._lease, PlanningLease.Request(owner=self.get_fully_qualified_name(), acquire=True))
            if response is None:
                return False
            if not response.success:
                raise RuntimeError(response.message)
            self._lease_held = True
            self._step = 'PLAN'
            group.max_velocity = self.cfg['motion.velocity_scale']
            group.max_acceleration = self.cfg['motion.acceleration_scale']
            self._future = group.plan_async(joint_positions=target, start_joint_state=group.joint_state)
            if self._future is None:
                raise RuntimeError('MoveIt planning unavailable')
            self._track(self._future)
        if self._step == 'PLAN':
            if not self._future.done():
                return False
            trajectory = group.get_trajectory(self._future)
            self._future = None
            if trajectory is None:
                raise RuntimeError('joint planning failed')
            group.motion_suceeded = False
            group.execute(trajectory)
            self._step = 'EXECUTE'
        if self._step == 'EXECUTE':
            if time.monotonic() - self._joint_received > 1.0:
                raise RuntimeError('joint feedback stopped during execution')
            if group.query_state() != MoveIt2State.IDLE:
                return False
            code = group.get_last_execution_error_code()
            if not group.motion_suceeded or code is None or code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError('joint execution failed')
            self._step = 'RELEASE_LEASE'
        if self._step == 'RELEASE_LEASE':
            response = self._service(self._lease, PlanningLease.Request(owner=self.get_fully_qualified_name(), acquire=False))
            if response is None:
                return False
            if not response.success:
                raise RuntimeError(response.message)
            self._lease_held = False
            return True
        return False

    def _tick_error(self):
        if self.current_phase in (P.CANCEL_ACTIVE_GOALS, P.HOLD):
            for client in self._actions:
                client.cancel()
            for group in self.groups.values():
                if group.query_state() == MoveIt2State.EXECUTING:
                    group.cancel_execution()
            pending = self._cancellation_pending()
            # Try to stop admittance even if cancellation has not been acknowledged.
            # The lease stays reserved, and HOLD keeps cancelling late executions.
            if pending and (self.current_phase == P.HOLD or
                            time.monotonic() - self._phase_started < self.cfg['task.cancel_timeout']):
                return
            if not pending and self._lease_held:
                response = self._service(self._lease, PlanningLease.Request(owner=self.get_fully_qualified_name(), acquire=False))
                if response is None or not response.success:
                    return
                self._lease_held = False
            if self.current_phase == P.CANCEL_ACTIVE_GOALS:
                self._goto_phase(P.STOP_ADMITTANCE, self._state_detail)
        elif self.current_phase == P.STOP_ADMITTANCE:
            if not self.enable_planar_search:
                self._goto_phase(P.HOLD, self._state_detail)
                return
            if not self._step:
                self._admittance.start_stop()
                self._step = 'STOP'
            result = self._admittance.tick_stop()
            if result is not None:
                if not result:
                    self.get_logger().error(self._admittance.last_error)
                self._goto_phase(P.HOLD, self._state_detail)
