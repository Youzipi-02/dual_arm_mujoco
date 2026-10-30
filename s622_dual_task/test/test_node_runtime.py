"""Real nonmoving ROS endpoints for the new pick capability and task boundary."""
import time

import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from sensor_msgs.msg import JointState
from s622_dual_interfaces.action import AcquireTargets, VisionPick
from std_srvs.srv import SetBool, Trigger

from vision_pick.node import VisionPickNode
from s622_dual_assembly.control_mode import ControlMode
from s622_dual_task.node import FSM
from s622_dual_task.states import TaskPhase, TaskState


def test_real_nodes_initialize_reset_and_reject_execution_when_disabled():
    rclpy.init(args=['test', '--ros-args', '-p', 'use_sim_time:=false', '-p', 'status.enabled:=false'], domain_id=219)
    support = Node('nonmoving_support', use_global_arguments=False)
    executor = MultiThreadedExecutor(num_threads=8)
    nodes, actions, requests = [support], [], []
    def scene(request, response):
        requests.append('scene')
        response.success = True
        return response
    def obb(request, response):
        requests.append('obb')
        response.success = True
        return response
    def targets(handle):
        requests.append('targets')
        handle.abort()
        return AcquireTargets.Result(success=False)
    support.create_service(ApplyPlanningScene, '/apply_planning_scene', scene)
    support.create_service(GetPlanningScene, '/get_planning_scene', lambda req, res: res)
    support.create_service(SetBool, '/s622_dual_perception/obb/set_enabled', obb)
    actions.append(ActionServer(support, AcquireTargets, '/target_acquisition/acquire_targets', targets))
    publisher = support.create_publisher(JointState, '/joint_states', 10)
    def joints():
        message = JointState()
        message.header.stamp = support.get_clock().now().to_msg()
        message.name = [f'{side}_j{i}' for side in ('left', 'right') for i in range(1, 7)]
        message.name += [f'{side}_finger{i}_joint' for side in ('left', 'right') for i in (1, 2)]
        message.position = [0.0] * len(message.name)
        publisher.publish(message)
    support.create_timer(0.02, joints)
    def until(predicate):
        deadline = time.monotonic() + 8.0
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert predicate()
    try:
        mode, pick, task = ControlMode(), VisionPickNode(), FSM()
        nodes += [mode, pick, task]
        for node in nodes:
            executor.add_node(node)
        until(lambda: task.current_state == TaskState.IDLE)
        start = support.create_client(Trigger, '/s622_dual_task_fsm/start')
        reset = support.create_client(Trigger, '/s622_dual_task_fsm/reset')
        until(lambda: start.service_is_ready() and reset.service_is_ready())
        future = start.call_async(Trigger.Request())
        until(future.done)
        assert not future.result().success
        client = ActionClient(support, VisionPick, '/vision_pick/pick')
        until(client.server_is_ready)
        goal = client.send_goal_async(VisionPick.Goal(execute=True))
        until(goal.done)
        assert not goal.result().accepted
        task._fail('injected smoke failure')
        until(lambda: task.current_phase == TaskPhase.HOLD)
        future = reset.call_async(Trigger.Request())
        until(future.done)
        assert future.result().success
        until(lambda: task.current_state == TaskState.IDLE)
        assert requests == []  # Initialization never plans, closes grippers or mutates the scene.
    finally:
        executor.shutdown()
        for action in actions:
            action.destroy()
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()
