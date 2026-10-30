# Copyright 2026 dsl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""测试 OBB launch 的配置文件和相机话题覆盖规则。"""

import importlib.util
from pathlib import Path

import pytest
import yaml

LaunchContext = pytest.importorskip("launch").LaunchContext
launch_actions = pytest.importorskip("launch.actions")
parameter_utilities = pytest.importorskip("launch_ros.utilities")


def _effective_parameters(monkeypatch, config_file, **arguments):
    launch_file = Path(__file__).parents[1] / "launch" / "yolo_obb_perception.launch.py"
    spec = importlib.util.spec_from_file_location("obb_launch", launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)
    context = LaunchContext()
    context.launch_configurations.update(
        config_file=str(config_file), model_path="/tmp/test_obb.pt", **arguments
    )
    for action in module.generate_launch_description().entities:
        if isinstance(action, launch_actions.DeclareLaunchArgument):
            action.execute(context)
        elif isinstance(action, launch_actions.OpaqueFunction):
            node = action.execute(context)[0]
    parameters = {}
    entries = parameter_utilities.evaluate_parameters(
        context, parameter_utilities.normalize_parameters(node["parameters"])
    )
    for entry in entries:
        if isinstance(entry, Path):
            parameters.update(yaml.safe_load(entry.read_text())["/**"]["ros__parameters"])
        else:
            parameters.update(entry)
    return parameters


def test_camera_topics_follow_yaml(monkeypatch, tmp_path):
    """修改配置文件应直接改变三个订阅话题，不受 launch 默认值覆盖。"""
    topics = {
        "input.rgb_topic": "/custom/rgb",
        "input.depth_topic": "/custom/depth",
        "input.camera_info_topic": "/custom/info",
    }
    config_file = tmp_path / "obb.yaml"
    config_file.write_text(yaml.safe_dump({"/**": {"ros__parameters": topics}}))
    parameters = _effective_parameters(monkeypatch, config_file)
    assert {key: parameters[key] for key in topics} == topics


@pytest.mark.parametrize("argument", ["rgb_topic", "depth_topic", "camera_info_topic"])
def test_explicit_camera_topic_override(monkeypatch, argument):
    """命令行可以覆盖单个话题，其他话题继续采用配置文件。"""
    config_file = Path(__file__).parents[1] / "config" / "obb_detector.yaml"
    expected = yaml.safe_load(config_file.read_text())["/**"]["ros__parameters"]
    expected[f"input.{argument}"] = "/override"
    parameters = _effective_parameters(monkeypatch, config_file, **{argument: "/override"})
    for name in ("rgb_topic", "depth_topic", "camera_info_topic"):
        assert parameters[f"input.{name}"] == expected[f"input.{name}"]
