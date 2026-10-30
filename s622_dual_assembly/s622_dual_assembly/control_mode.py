"""Coordinate admittance start/stop and exclude competing planned motion."""
from functools import partial
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from s622_dual_interfaces.srv import PlanningLease
from std_srvs.srv import SetBool


class ControlMode(Node):
    def __init__(self):
        super().__init__('assembly_control')
        self._followers_enabled = self.declare_parameter('followers_enabled', False).value
        self._settle = float(self.declare_parameter('continuous_settle_sec', 0.1).value)
        self._lock = threading.Lock()
        self._owner = ''
        self._compliant = False
        self._initialized = not self._followers_enabled
        self._inflight = []
        self._active_arms = set()
        group = ReentrantCallbackGroup()
        service_group = MutuallyExclusiveCallbackGroup()
        self._followers = [self.create_client(SetBool, f'/{side}_cartesian_follower/cartesian_pose_executor/enable',
                                              callback_group=group) for side in ('left', 'right')]
        self._admittance = {}
        if self._followers_enabled:
            for side in ('left', 'right'):
                self._admittance[side] = self.create_client(
                    SetBool, f'/{side}_admittance/enable_controller', callback_group=group)
                self.create_service(SetBool, f'/{side}_admittance/enable',
                                    partial(self._enable_arm, side), callback_group=service_group)
        self.create_service(PlanningLease, '~/planning_lease', self._lease, callback_group=service_group)
        self.create_service(SetBool, '~/set_compliant_mode', self._mode, callback_group=service_group)

    def _switch(self, clients, enabled, label):
        # Unknown or late transitions must be resolved before sending another request.
        self._initialized = False
        if any(not future.done() for future in self._inflight):
            raise RuntimeError('previous control transition is still in flight; retry disable after it finishes')
        self._inflight = []
        if any(not client.service_is_ready() for client in clients):
            raise RuntimeError(f'{label} service unavailable')
        for client in clients:
            self._inflight.append(client.call_async(SetBool.Request(data=enabled)))
        deadline = time.monotonic() + 3.0
        while not all(future.done() for future in self._inflight) and time.monotonic() < deadline:
            time.sleep(0.01)
        if not all(future.done() for future in self._inflight):
            raise RuntimeError(f'{label} transition timed out; disable before planning')
        responses = [future.result() for future in self._inflight]
        errors = [response.message for response in responses if not response.success]
        if errors:
            raise RuntimeError(f'{label}: ' + '; '.join(errors))
        self._inflight = []
        self._initialized = True

    def _switch_followers(self, enabled):
        if not self._followers_enabled:
            if enabled:
                raise RuntimeError('continuous followers are not launched')
            return
        self._switch(self._followers, enabled, 'Cartesian follower')
        if not enabled:
            time.sleep(self._settle)

    def _stop_all(self):
        # Stop joint output first, then clear all targets, including after a manager restart.
        self._switch_followers(False)
        if self._followers_enabled:
            self._switch(list(self._admittance.values()), False, 'admittance')
        self._active_arms.clear()
        self._compliant = False
        self._initialized = True

    def _enter_compliant(self):
        if not self._initialized:
            if self._compliant:
                raise RuntimeError('control state unconfirmed; disable admittance before retrying')
            self._stop_all()
        if not self._compliant:
            self._switch_followers(True)
            self._compliant = True

    def _enable_arm(self, arm, request, response):
        """One public request handles both computation and joint execution."""
        with self._lock:
            try:
                if self._owner:
                    raise RuntimeError(f'planned motion is owned by {self._owner}')
                if request.data:
                    self._enter_compliant()
                    self._active_arms.add(arm)
                    self._switch([self._admittance[arm]], True, f'{arm} admittance')
                elif not self._initialized:
                    # A failed transition may have affected either side; confirm a full stop.
                    self._stop_all()
                else:
                    self._switch([self._admittance[arm]], False, f'{arm} admittance')
                    self._active_arms.discard(arm)
                    if not self._active_arms:
                        self._switch_followers(False)
                        self._compliant = False
                response.success = True
                response.message = f'{arm} admittance enabled; waiting for fresh wrench' if request.data else (
                    f'{arm} admittance disabled; ' + ('COMPLIANT' if self._compliant else 'IDLE'))
            except Exception as error:
                if not self._initialized:
                    self._compliant = True
                response.success = False
                response.message = str(error)
        return response

    def _lease(self, request, response):
        with self._lock:
            try:
                if not request.owner:
                    raise RuntimeError('planning owner is required')
                if request.acquire:
                    if self._owner or self._compliant:
                        raise RuntimeError('planned or compliant motion already owns the robot')
                    if not self._initialized:
                        self._stop_all()
                    self._owner = request.owner
                else:
                    if self._owner and self._owner != request.owner:
                        raise RuntimeError('only the planning owner may release its lease')
                    self._owner = ''
                response.success = True
                response.message = self._owner or 'IDLE'
            except Exception as error:
                response.success = False
                response.message = str(error)
        return response

    def _mode(self, request, response):
        with self._lock:
            try:
                if self._owner:
                    raise RuntimeError(f'planned motion is owned by {self._owner}')
                if request.data:
                    self._enter_compliant()
                else:
                    self._stop_all()
                response.success = True
                response.message = 'COMPLIANT' if request.data else 'IDLE'
            except Exception as error:
                # Unknown follower state prevents a planner from entering.
                if not self._initialized:
                    self._compliant = True
                response.success = False
                response.message = str(error)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ControlMode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
