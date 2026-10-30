"""Real ROS Action/service wiring against nonmoving fake MoveIt endpoints."""
from copy import deepcopy
import time

from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetMotionPlan, GetPlanningScene, GetPositionIK
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from s622_dual_interfaces.action import AcquireTargets, ReleaseGrasp, VisionPick
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from s622_dual_assembly.control_mode import ControlMode
from vision_pick.node import VisionPickNode


def test_two_pick_release_cycles_use_native_planning_and_keep_tf_after_action():
    rclpy.init(args=['test', '--ros-args', '-p', 'use_sim_time:=false', '-p', 'pick.execute:=true'], domain_id=220)
    support = Node('fake_moveit', use_global_arguments=False)
    group = ReentrantCallbackGroup()
    events, acquisitions, feedback = [], [], []
    arms = {side: [f'{side}_j{i}' for i in range(1, 7)] for side in ('left', 'right')}
    fingers = [f'{side}_finger{i}_joint' for side in ('left', 'right') for i in (1, 2)]
    positions = {name: 0.0 for name in arms['left'] + arms['right'] + fingers}
    eef = {side: Pose() for side in ('left', 'right')}
    for pose in eef.values():
        pose.orientation.w = 1.0
    publisher = support.create_publisher(JointState, '/joint_states', 10)
    broadcaster = TransformBroadcaster(support)
    def sensors():
        stamp = support.get_clock().now().to_msg()
        message = JointState()
        message.header.stamp = stamp
        message.name, message.position = list(positions), list(positions.values())
        publisher.publish(message)
        transforms = []
        for side, pose in eef.items():
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = 'world'
            transform.child_frame_id = f'{side}_grasp_frame'
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation = pose.orientation
            transforms.append(transform)
        broadcaster.sendTransform(transforms)
    support.create_timer(0.01, sensors, callback_group=group)

    def trajectory(names, values):
        return JointTrajectory(joint_names=names, points=[
            JointTrajectoryPoint(positions=values, time_from_start=Duration(nanosec=10_000_000))])

    def plan(request, response):
        req = request.motion_plan_request
        constraints = req.goal_constraints[0]
        events.append(('plan', req.group_name, len(constraints.position_constraints)))
        if constraints.joint_constraints:
            names = [item.joint_name for item in constraints.joint_constraints]
            values = [item.position for item in constraints.joint_constraints]
        else:
            raise AssertionError('pregrasp must submit a combined joint goal')
        response.motion_plan_response.error_code.val = MoveItErrorCodes.SUCCESS
        response.motion_plan_response.trajectory.joint_trajectory = trajectory(names, values)
        return response

    def ik(request, response):
        req = request.ik_request
        events.append(('ik', req.group_name, req.avoid_collisions))
        side = 'left' if req.group_name == 'left_arm' else 'right'
        eef[side] = deepcopy(req.pose_stamped.pose)
        response.error_code.val = MoveItErrorCodes.SUCCESS
        response.solution.joint_state.name = list(positions)
        response.solution.joint_state.position = [
            (0.1 if side == 'left' else 0.2) if name in arms[side] else positions[name]
            for name in positions]
        return response

    def cartesian(request, response):
        events.append(('cartesian', request.group_name, request.avoid_collisions))
        side = 'left' if request.group_name == 'left_arm' else 'right'
        eef[side] = deepcopy(request.waypoints[-1])
        response.fraction = 1.0
        response.error_code.val = MoveItErrorCodes.SUCCESS
        response.solution.joint_trajectory = trajectory(arms[side], [0.0] * 6)
        return response

    def execute(handle):
        traj = handle.request.trajectory.joint_trajectory
        positions.update(zip(traj.joint_names, traj.points[-1].positions))
        events.append(('execute', len(traj.joint_names)))
        handle.succeed()
        result = ExecuteTrajectory.Result()
        result.error_code.val = MoveItErrorCodes.SUCCESS
        return result

    def targets(handle):
        acquisitions.append(len(acquisitions) + 1)
        result = AcquireTargets.Result(success=True)
        for detection, x in ((result.wire, -0.20), (result.terminal, 0.20)):
            detection.header.frame_id = 'world'
            detection.header.stamp = support.get_clock().now().to_msg()
            detection.bbox.center.position.x = x
            detection.bbox.center.position.y = 0.35
            detection.bbox.center.position.z = 0.77
            detection.bbox.center.orientation.w = 1.0
        handle.succeed()
        return result

    def success(request, response):
        response.success = True
        return response
    support.create_service(GetPositionIK, '/compute_ik', ik, callback_group=group)
    support.create_service(GetMotionPlan, '/plan_kinematic_path', plan, callback_group=group)
    support.create_service(GetCartesianPath, '/compute_cartesian_path', cartesian, callback_group=group)
    support.create_service(ApplyPlanningScene, '/apply_planning_scene', success, callback_group=group)
    support.create_service(GetPlanningScene, '/get_planning_scene', lambda req, res: res, callback_group=group)
    support.create_service(SetBool, '/s622_dual_perception/obb/set_enabled', success, callback_group=group)
    servers = [ActionServer(support, ExecuteTrajectory, '/execute_trajectory', execute, callback_group=group),
               ActionServer(support, AcquireTargets, '/target_acquisition/acquire_targets', targets, callback_group=group)]
    nodes = [support]
    executor = MultiThreadedExecutor(num_threads=10)
    def until(predicate):
        deadline = time.monotonic() + 8.0
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert predicate()
    try:
        mode, pick = ControlMode(), VisionPickNode()
        nodes += [mode, pick]
        for node in nodes:
            executor.add_node(node)
        pick_client = ActionClient(support, VisionPick, '/vision_pick/pick', callback_group=group)
        release_client = ActionClient(support, ReleaseGrasp, '/vision_pick/release', callback_group=group)
        until(lambda: pick_client.server_is_ready() and release_client.server_is_ready() and pick._interfaces_ready())
        for _ in range(2):
            acceptance = pick_client.send_goal_async(VisionPick.Goal(execute=True),
                feedback_callback=lambda msg: feedback.append(msg.feedback.phase))
            until(acceptance.done)
            assert acceptance.result().accepted
            result = acceptance.result().get_result_async()
            until(result.done)
            assert result.result().result.success, result.result().result.reason
            assert pick.tracker.attached and pick._holding == [True, True]
            assert mode._owner == ''
            # An external duplicate pick cannot move a robot that is still holding parts.
            duplicate = pick_client.send_goal_async(VisionPick.Goal(execute=True))
            until(duplicate.done)
            assert not duplicate.result().accepted
            acceptance = release_client.send_goal_async(ReleaseGrasp.Goal(execute=True))
            until(acceptance.done)
            assert acceptance.result().accepted
            result = acceptance.result().get_result_async()
            until(result.done)
            assert result.result().result.success, result.result().result.reason
            assert not pick.tracker.attached and pick._holding == [False, False]
        assert acquisitions == [1, 2]
        assert feedback.count('ACQUIRE_TARGETS') == 2
        assert [item[1] for item in events if item[0] == 'cartesian'] == ['left_arm', 'right_arm', 'left_arm', 'right_arm'] * 2
        assert all(item[2] for item in events if item[0] == 'cartesian')
        assert [item[1] for item in events if item[0] == 'ik'] == ['left_arm', 'right_arm'] * 2
        assert all(item[2] for item in events if item[0] == 'ik')
        assert not any(item[2] for item in events if item[0] == 'plan')
    finally:
        executor.shutdown()
        for server in servers:
            server.destroy()
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()
