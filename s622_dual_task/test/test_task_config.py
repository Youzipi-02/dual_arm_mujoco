from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

SRC = Path(__file__).parents[2]


def test_execute_defaults_are_disabled_at_both_boundaries():
    task = yaml.safe_load((SRC / 's622_dual_task/config/task.yaml').read_text())['/**']['ros__parameters']
    pick = yaml.safe_load((SRC / 'vision_pick/config/vision_pick.yaml').read_text())['/**']['ros__parameters']
    assert task['task.execute'] is False and task['task.autostart'] is False
    assert pick['pick.execute'] is False
    assert 0.99 <= pick['motion.cartesian_fraction_threshold'] <= 1.0


def test_fixture_and_workpieces_match_mujoco_model_poses():
    pick = yaml.safe_load((SRC / 'vision_pick/config/vision_pick.yaml').read_text())['/**']['ros__parameters']
    shared = yaml.safe_load((SRC / 's622_dual_descriptions/config/shared_model.yaml').read_text())['/**']['ros__parameters']
    root = ET.parse(SRC / 's622_dual_descriptions/robot_model/objects.xml').getroot()
    for body_name, key in [('wire', 'wire.model_pose'), ('copper_terminal', 'terminal.model_pose'), ('middleware', 'fixture.model_pose')]:
        expected = [float(v) for v in root.find(f"worldbody/body[@name='{body_name}']").attrib['pos'].split()]
        assert (shared if key in shared else pick)[key][:3] == pytest.approx(expected)
    assert pick['fixture.mesh_resource'] == 'package://s622_dual_descriptions/meshes/objects/middleware.stl'


def test_robot_description_keeps_joint_planning_groups_and_cross_arm_collisions():
    root = ET.parse(SRC / 's622_dual_moveit_config/config/s622_dual_descriptions.srdf').getroot()
    assert {x.attrib['name'] for x in root.find("group[@name='both_arms']")} == {'left_arm', 'right_arm'}
    for item in root.findall('disable_collisions'):
        a, b = item.attrib['link1'], item.attrib['link2']
        assert not (a.startswith('left_') and b.startswith('right_') or a.startswith('right_') and b.startswith('left_'))


def test_contact_grasp_commands_are_inside_workpiece_radius():
    shared = yaml.safe_load((SRC / 's622_dual_descriptions/config/shared_model.yaml').read_text())['/**']['ros__parameters']
    for key, radius in [('gripper.wire_closed', shared['wire.radius']),
                        ('gripper.terminal_closed', shared['terminal.outer_radius'])]:
        positive, negative = shared[key]
        assert 0 < positive < radius
        assert negative == pytest.approx(-positive)
        assert radius - positive == pytest.approx(0.001)
