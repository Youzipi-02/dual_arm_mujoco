"""Verify the ROS parameter layers used by the combined task launch."""

from pathlib import Path

import rclpy
from rclpy.context import Context
from rclpy.node import Node

from s622_dual_task.task_config import TaskConfigMixin


SRC = Path(__file__).parents[2]
SHARED = SRC / "s622_dual_descriptions" / "config" / "shared_model.yaml"


def _probe_node(name, *parameter_files):
    args = ["probe", "--ros-args", "-r", f"__node:={name}"]
    for path in parameter_files:
        args.extend(("--params-file", str(path)))
    context = Context()
    rclpy.init(args=args, context=context)
    return Node(name, context=context), context


def test_task_launch_overrides_defaults_and_loads_shared_float_arrays(tmp_path):
    task_config = SRC / "s622_dual_task" / "config" / "task.yaml"
    launch_override = tmp_path / "task_override.yaml"
    launch_override.write_text(
        "/**:\n  ros__parameters:\n    task.execute: true\n"
        "    task.autostart: true\n",
        encoding="utf-8",
    )
    node, context = _probe_node(
        "s622_dual_task_fsm", SHARED, task_config, launch_override
    )
    try:
        assert node.declare_parameter("task.execute", False).value is True
        assert node.declare_parameter("task.autostart", False).value is True
        assert node.declare_parameter("gripper.open", [0.0, 0.0]).value == [
            0.025, -0.025
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown(context=context)


def test_obb_launch_model_path_overrides_empty_config_default(tmp_path):
    obb_config = SRC / "s622_dual_perception" / "config" / "obb_detector.yaml"
    launch_override = tmp_path / "obb_override.yaml"
    model_path = "/tmp/task_test_yolo_obb.pt"
    launch_override.write_text(
        f"/**:\n  ros__parameters:\n    model.path: {model_path}\n",
        encoding="utf-8",
    )
    node, context = _probe_node(
        "yolo_obb_detector", SHARED, obb_config, launch_override
    )
    try:
        assert node.declare_parameter("model.path", "").value == model_path
    finally:
        node.destroy_node()
        rclpy.shutdown(context=context)
