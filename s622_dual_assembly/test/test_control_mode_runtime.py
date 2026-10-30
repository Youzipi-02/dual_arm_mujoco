"""Exercise the public service chain with fake backends, without robot commands."""
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from s622_dual_interfaces.srv import PlanningLease
from std_srvs.srv import SetBool

from s622_dual_assembly.control_mode import ControlMode


def test_public_enable_coordinates_backends_and_excludes_planning():
    # Keep fake backend names separate from any running robot graph.
    rclpy.init(args=['test', '--ros-args', '-p', 'followers_enabled:=true',
                     '-p', 'continuous_settle_sec:=0.0'], domain_id=217)
    manager = ControlMode()
    backend = Node('admittance_backend_probe', use_global_arguments=False)
    executor = MultiThreadedExecutor(num_threads=4)
    thread = None
    history = []
    state = {}

    def callback(label):
        def switch(request, response):
            history.append((label, request.data))
            state[label] = request.data
            response.success = True
            response.message = 'ok'
            return response
        return switch

    try:
        for arm in ('left', 'right'):
            backend.create_service(SetBool, f'/{arm}_admittance/enable_controller', callback(arm))
            backend.create_service(SetBool, f'/{arm}_cartesian_follower/cartesian_pose_executor/enable',
                                   callback(f'{arm} follower'))
        clients = {
            arm: backend.create_client(SetBool, f'/{arm}_admittance/enable')
            for arm in ('left', 'right')
        }
        planning = backend.create_client(PlanningLease, '/assembly_control/planning_lease')
        executor.add_node(manager)
        executor.add_node(backend)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        for client in (*clients.values(), planning, *manager._admittance.values(), *manager._followers):
            assert client.wait_for_service(timeout_sec=5.0)

        def call(client, request):
            future = client.call_async(request)
            deadline = time.monotonic() + 5.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert future.done(), 'coordinated service did not respond'
            return future.result()

        assert call(clients['right'], SetBool.Request(data=True)).success
        assert state == {'left': False, 'right': True,
                         'left follower': True, 'right follower': True}
        assert not call(planning, PlanningLease.Request(owner='pick', acquire=True)).success
        history.clear()
        assert call(clients['left'], SetBool.Request(data=True)).success
        assert history == [('left', True)]
        assert call(clients['right'], SetBool.Request(data=False)).success
        assert state['left'] and state['left follower']
        assert not call(planning, PlanningLease.Request(owner='pick', acquire=True)).success
        assert call(clients['left'], SetBool.Request(data=False)).success
        assert not any(state.values())
        assert call(planning, PlanningLease.Request(owner='pick', acquire=True)).success
        history.clear()
        assert not call(clients['right'], SetBool.Request(data=True)).success
        assert history == []
        assert call(planning, PlanningLease.Request(owner='pick', acquire=False)).success
    finally:
        executor.shutdown(timeout_sec=5.0)
        if thread is not None:
            thread.join(timeout=5.0)
        manager.destroy_node()
        backend.destroy_node()
        rclpy.shutdown()
