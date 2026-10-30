"""测试规划场景所使用的碰撞几何。"""

import struct

import pytest

from builtin_interfaces.msg import Time
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from moveit_msgs.srv import GetPlanningScene
from shape_msgs.msg import Mesh

from vision_pick.grasp_geometry import make_pose
from vision_pick.scene_manager import (
    SceneManager,
    load_binary_stl_mesh,
    make_tube_mesh,
)


class DoneFuture:
    """用于解析场景结果的最简已完成 Future。"""

    def __init__(self, result):
        self.result_value = result

    def done(self):
        return True

    def result(self):
        return self.result_value


def test_terminal_mesh_preserves_a_real_bore():
    mesh = make_tube_mesh(0.0075, 0.0055, 0.03, segments=48)
    assert len(mesh.vertices) == 192
    assert len(mesh.triangles) == 384
    radii = sorted({round((point.x**2 + point.y**2) ** 0.5, 6) for point in mesh.vertices})
    assert radii == [0.0055, 0.0075]
    assert sorted({point.z for point in mesh.vertices}) == [0.0, 0.03]


def test_binary_stl_loader_preserves_all_fixture_triangles(tmp_path):
    stl_path = tmp_path / "fixture.stl"
    triangle = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
    stl_path.write_bytes(bytes(80) + struct.pack("<I", 1) + triangle)

    mesh = load_binary_stl_mesh(stl_path)

    assert len(mesh.vertices) == 3
    assert len(mesh.triangles) == 1
    assert list(mesh.triangles[0].vertex_indices) == [0, 1, 2]


def test_attachment_query_filters_out_unrelated_objects():
    response = GetPlanningScene.Response()
    for object_id in ("wire", "copper_terminal", "unrelated_tool"):
        attached = AttachedCollisionObject()
        attached.object.id = object_id
        response.scene.robot_state.attached_collision_objects.append(attached)
    for object_id in ("work_table", "central_clearance_guard"):
        collision = CollisionObject()
        collision.id = object_id
        response.scene.world.collision_objects.append(collision)

    result = SceneManager.attachment_query_result(DoneFuture(response))

    assert result == (
        True,
        {"wire", "copper_terminal"},
        {"work_table", "central_clearance_guard"},
    )


def test_attachment_query_reports_pending_future():
    future = type("PendingFuture", (), {"done": lambda self: False})()
    assert SceneManager.attachment_query_result(future) is None


def test_visual_workpiece_poses_replace_scene_defaults():
    manager = SceneManager.__new__(SceneManager)
    wire = make_pose(-0.1, 0.2, 0.75)
    terminal = make_pose(0.1, 0.3, 0.75)

    manager.update_workpiece_poses(wire, terminal)

    assert manager.wire_model_pose is wire
    assert manager.terminal_model_pose is terminal


def test_grasped_workpieces_are_attached_with_local_touch_links_only():
    manager = SceneManager.__new__(SceneManager)
    manager.workpiece_collisions_enabled = True
    manager.left_eef_link = "left_grasp_frame"
    manager.right_eef_link = "right_grasp_frame"
    manager.wire_radius = 0.0035
    manager.wire_length = 0.060
    manager.terminal_outer_radius = 0.0075
    manager.terminal_inner_radius = 0.0055
    manager.terminal_length = 0.030
    manager._node = type(
        "FakeNode",
        (),
        {
            "get_clock": lambda self: type(
                "FakeClock",
                (),
                {
                    "now": lambda self: type(
                        "FakeNow", (), {"to_msg": lambda self: Time()}
                    )()
                },
            )()
        },
    )()
    manager._apply = lambda scene: scene

    scene = manager.attach_workpieces_async(
        make_pose(0.01, 0.0, 0.0), make_pose(0.0, 0.0, 0.0)
    )
    attached = scene.robot_state.attached_collision_objects

    assert scene.robot_state.is_diff is True
    assert [item.object.id for item in attached] == ["wire", "copper_terminal"]
    assert attached[0].link_name == "left_grasp_frame"
    assert attached[1].link_name == "right_grasp_frame"
    assert set(attached[0].touch_links) == {
        "left_finger1",
        "left_finger2",
        "left_grasp_frame",
    }
    assert set(attached[1].touch_links) == {
        "right_finger1",
        "right_finger2",
        "right_grasp_frame",
    }
    assert all("right" not in name for name in attached[0].touch_links)
    assert all("left" not in name for name in attached[1].touch_links)


def test_disabled_workpiece_collisions_do_not_add_or_remove_missing_objects():
    manager = SceneManager.__new__(SceneManager)
    manager.workpiece_collisions_enabled = False
    manager._apply = lambda scene: scene

    scene = manager.attach_workpieces_async(
        make_pose(0.0, 0.0, 0.0),
        make_pose(0.0, 0.0, 0.0),
    )

    assert scene.is_diff is True
    assert scene.robot_state.is_diff is True
    assert not scene.robot_state.attached_collision_objects
    assert not scene.world.collision_objects


def test_disabled_scene_sync_only_removes_existing_workpiece_collisions():
    manager = SceneManager.__new__(SceneManager)
    manager.workpiece_collisions_enabled = False
    manager.planning_frame = "world"
    manager.left_eef_link = "left_grasp_frame"
    manager.right_eef_link = "right_grasp_frame"
    manager.table_object = lambda: CollisionObject(id="work_table")
    manager.fixture_object = lambda: CollisionObject(id="middleware")
    manager.clearance_guard_object = lambda add: CollisionObject(
        id="central_clearance_guard",
        operation=CollisionObject.ADD if add else CollisionObject.REMOVE,
    )
    manager._apply = lambda scene: scene

    scene = manager.sync_world_async(
        attached_ids={"wire"},
        world_ids={"wire", "central_clearance_guard"},
    )

    assert [
        item.object.id
        for item in scene.robot_state.attached_collision_objects
    ] == ["wire"]
    assert all(
        item.object.operation == CollisionObject.REMOVE
        for item in scene.robot_state.attached_collision_objects
    )
    assert [item.id for item in scene.world.collision_objects[:3]] == [
        "work_table",
        "middleware",
        "wire",
    ]
    assert scene.world.collision_objects[2].operation == CollisionObject.REMOVE
    assert scene.world.collision_objects[3].id == "central_clearance_guard"
    assert scene.world.collision_objects[3].operation == CollisionObject.REMOVE


def test_fixture_collision_uses_configured_mesh_and_pose():
    manager = SceneManager.__new__(SceneManager)
    manager.planning_frame = "world"
    manager.fixture_mesh = Mesh()
    manager.fixture_model_pose = make_pose(
        0.0, 0.25, 0.851, 0.0, 0.70710678, 0.70710678, 0.0
    )
    manager._node = type(
        "FakeNode",
        (),
        {
            "get_clock": lambda self: type(
                "FakeClock",
                (),
                {
                    "now": lambda self: type(
                        "FakeNow", (), {"to_msg": lambda self: Time()}
                    )()
                },
            )()
        },
    )()

    collision = manager.fixture_object()

    assert collision.id == "middleware"
    assert collision.header.frame_id == "world"
    assert collision.operation == CollisionObject.ADD
    assert collision.meshes == [manager.fixture_mesh]
    assert collision.mesh_poses == [manager.fixture_model_pose]


@pytest.mark.parametrize(
    "outer,inner,length,segments",
    [
        (0.005, 0.006, 0.03, 48),
        (0.005, 0.0, 0.03, 48),
        (0.007, 0.005, 0.0, 48),
        (0.007, 0.005, 0.03, 2),
    ],
)
def test_invalid_terminal_mesh_dimensions_are_rejected(
    outer, inner, length, segments
):
    with pytest.raises(ValueError):
        make_tube_mesh(outer, inner, length, segments)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('attached,world', [
    (set(), set()),
    (set(), {'wire', 'copper_terminal', 'work_table'}),
    ({'wire'}, {'work_table'}),
    ({'copper_terminal'}, {'wire', 'work_table'}),
    ({'wire', 'copper_terminal', 'unrelated_tool'}, {'work_table'}),
])
def test_remove_workpieces_only_deletes_existing_task_objects(enabled, attached, world):
    manager = SceneManager.__new__(SceneManager)
    manager.planning_frame = 'world'
    manager.left_eef_link = 'left_grasp_frame'
    manager.right_eef_link = 'right_grasp_frame'
    manager.workpiece_collisions_enabled = enabled
    manager._apply = lambda scene: scene
    scene = manager.remove_workpieces_async(attached, world)
    task_ids = {'wire', 'copper_terminal'}
    assert scene.is_diff and scene.robot_state.is_diff
    assert {item.object.id for item in scene.robot_state.attached_collision_objects} == attached & task_ids
    assert all(item.object.operation == CollisionObject.REMOVE for item in scene.robot_state.attached_collision_objects)
    # MoveIt places detached bodies in the world, so remove those in the same diff.
    assert {item.id for item in scene.world.collision_objects} == (attached | world) & task_ids
    assert all(item.operation == CollisionObject.REMOVE for item in scene.world.collision_objects)


def test_repeated_removal_of_an_empty_scene_sends_no_invalid_remove_operations():
    manager = SceneManager.__new__(SceneManager)
    manager.planning_frame = 'world'
    manager.left_eef_link = 'left_grasp_frame'
    manager.right_eef_link = 'right_grasp_frame'
    manager._apply = lambda scene: scene
    for _ in range(2):
        scene = manager.remove_workpieces_async(set(), set())
        assert not scene.robot_state.attached_collision_objects
        assert not scene.world.collision_objects


def test_base_scene_reset_also_removes_objects_created_by_detachment():
    manager = SceneManager.__new__(SceneManager)
    manager.planning_frame = 'world'
    manager.left_eef_link = 'left_grasp_frame'
    manager.right_eef_link = 'right_grasp_frame'
    manager.workpiece_collisions_enabled = False
    manager.table_object = lambda: CollisionObject(id='work_table')
    manager.fixture_object = lambda: CollisionObject(id='middleware')
    manager._apply = lambda scene: scene
    scene = manager.sync_world_async(attached_ids={'wire'}, world_ids={'work_table'})
    assert [(item.id, item.operation) for item in scene.world.collision_objects[2:]] == [
        ('wire', CollisionObject.REMOVE)]


def test_reset_discards_the_previous_visual_workpiece_poses():
    manager = SceneManager.__new__(SceneManager)
    initial = (make_pose(-0.1, 0.2, 0.75), make_pose(0.1, 0.2, 0.75))
    manager._initial_workpiece_poses = initial
    manager.update_workpiece_poses(make_pose(2.0, 3.0, 4.0), make_pose(5.0, 6.0, 7.0))
    manager.reset_workpiece_poses()
    assert (manager.wire_model_pose, manager.terminal_model_pose) == initial




def test_moveit_and_mujoco_tables_keep_three_mm_below_robot_mounts():
    from pathlib import Path
    from types import SimpleNamespace
    import xml.etree.ElementTree as ET

    src = Path(__file__).parents[2]
    models = src / 's622_dual_descriptions/robot_model'
    manager = SceneManager.__new__(SceneManager)
    manager.planning_frame = 'world'
    manager._node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time())))
    table = manager.table_object()
    moveit_top = table.primitive_poses[0].position.z + table.primitives[0].dimensions[2] / 2
    scene = ET.parse(models / 'scene.xml').getroot()
    tabletop = scene.find("worldbody/body[@name='work_table']/geom[@name='table_top']")
    position = [float(v) for v in tabletop.attrib['pos'].split()]
    half_size = [float(v) for v in tabletop.attrib['size'].split()]
    assert moveit_top == pytest.approx(0.750)
    assert position[2] + half_size[2] == pytest.approx(moveit_top)
    urdf = ET.parse(models / 's622_dual.urdf').getroot()
    mount = float(urdf.find("joint[@name='world_to_dual_base']/origin").attrib['xyz'].split()[2])
    mjcf = ET.parse(models / 's622_dual.xml').getroot()
    assert float(mjcf.find("worldbody/body[@name='dual_base']").attrib['pos'].split()[2]) == pytest.approx(mount)
    mesh = load_binary_stl_mesh(src / 's622_dual_descriptions/meshes/robot/base_link.STL')
    assert mount + min(v.z for v in mesh.vertices) - moveit_top == pytest.approx(0.003, abs=1e-10)
    for leg in scene.findall("worldbody/body[@name='work_table']/geom"):
        if leg.attrib['name'].startswith('table_leg_'):
            z = float(leg.attrib['pos'].split()[2])
            half = float(leg.attrib['size'].split()[2])
            assert z + half == pytest.approx(position[2] - half_size[2])
