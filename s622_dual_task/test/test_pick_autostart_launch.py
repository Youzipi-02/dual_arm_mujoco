"""The standalone launch must actually send a goal, with execution gated."""
import importlib.util
from pathlib import Path

from launch import LaunchService
from launch.actions import OpaqueFunction
import pytest


@pytest.mark.parametrize('execute,autostart,count', [
    ('false', 'true', 0), ('true', 'false', 0), ('true', 'true', 1)])
def test_standalone_launch_sends_exactly_one_goal_only_when_enabled(monkeypatch, execute, autostart, count):
    path = Path(__file__).parents[2] / 's622_dual_bringup/launch/vision_pick.launch.py'
    spec = importlib.util.spec_from_file_location('standalone_pick_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sent = []
    monkeypatch.setattr(module, 'Node', lambda **kwargs: OpaqueFunction(function=lambda context: []))
    monkeypatch.setattr(module, 'IncludeLaunchDescription', lambda *args, **kwargs: OpaqueFunction(function=lambda context: []))
    def process(**kwargs):
        def record(context):
            sent.append(kwargs['cmd'])
            return []
        return OpaqueFunction(function=record, condition=kwargs['condition'])
    monkeypatch.setattr(module, 'ExecuteProcess', process)
    service = LaunchService()
    service.context.launch_configurations.update(execute=execute, autostart=autostart)
    service.include_launch_description(module.generate_launch_description())
    assert service.run() == 0
    assert len(sent) == count
    if count:
        assert sent[0][:4] == ['ros2', 'action', 'send_goal', '/vision_pick/pick']
        assert '{execute: true}' in sent[0]
