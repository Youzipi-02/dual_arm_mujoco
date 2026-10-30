"""Package boundaries and shared configuration after removing the motion server."""
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

SRC = Path(__file__).parents[2]


def params(package, name):
    return yaml.safe_load((SRC / package / 'config' / name).read_text())['/**']['ros__parameters']


def test_shared_specification_has_one_source():
    shared = params('s622_dual_descriptions', 'shared_model.yaml')
    pick = params('vision_pick', 'vision_pick.yaml')
    task = params('s622_dual_task', 'task.yaml')
    for key in ('wire.length', 'terminal.length', 'gripper.open', 'gripper.wire_closed', 'gripper.terminal_closed',
                'frames.planning', 'frames.left_eef', 'frames.right_eef', 'wire.stage_joints', 'terminal.stage_joints'):
        assert key in shared and key not in pick and key not in task
    assert not any(key.startswith(('vision.', 'perception.')) for key in task)


def test_no_package_has_a_runtime_dependency_on_the_removed_executor():
    for manifest in SRC.glob('*/package.xml'):
        dependencies = {element.text for element in ET.parse(manifest).getroot()}
        assert 'motion_executor' not in dependencies
    assert not (SRC / 'motion_executor').exists()
    cmake = (SRC / 's622_dual_interfaces/CMakeLists.txt').read_text()
    assert 'VisionPick.action' in cmake and 'ReleaseGrasp.action' in cmake
    assert 'ExecuteMotion.action' not in cmake and 'ManageScene.action' not in cmake


def test_bringup_wires_task_search_and_one_control_mode_owner():
    text = (SRC / 's622_dual_bringup/launch/peg_in_hole.launch.py').read_text()
    target_acquisition = (
        SRC / 's622_dual_perception/s622_dual_perception/nodes/target_acquisition.py'
    ).read_text()
    assert "'start_control_mode': 'true'" in text
    assert "executable='control_mode'" not in text
    assert "'ft_sensor_processing.launch.py'" in text
    assert '"detections_topic", "/yolo_obb/detections_3d"' in target_acquisition
    assert "'planar_search.launch.py'" in text
    assert "executable='fsm'" in text
