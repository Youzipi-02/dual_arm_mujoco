"""管理视觉抓取任务使用的 MoveIt 规划场景。"""

import math
from pathlib import Path
import struct
from typing import Optional, Set, Tuple

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive

from .grasp_geometry import compose_pose, make_pose


WIRE_ID = "wire"
TERMINAL_ID = "copper_terminal"
TABLE_ID = "work_table"
FIXTURE_ID = "middleware"
CLEARANCE_GUARD_ID = "central_clearance_guard"


def resolve_mesh_resource(resource: str) -> Path:
    """将 package://、file:// 或普通路径解析为本地网格文件。"""

    if resource.startswith("package://"):
        package_path = resource.removeprefix("package://")
        package_name, separator, relative_path = package_path.partition("/")
        if not separator or not package_name or not relative_path:
            raise ValueError(f"invalid package mesh resource: {resource}")
        path = Path(get_package_share_directory(package_name)) / relative_path
    elif resource.startswith("file://"):
        path = Path(resource.removeprefix("file://"))
    else:
        path = Path(resource)
    if not path.is_file():
        raise FileNotFoundError(
            f"fixture mesh resource does not exist: {path}"
        )
    return path


def load_binary_stl_mesh(path: Path) -> Mesh:
    """读取二进制 STL，保留单侧开口等非凸几何特征。"""

    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"binary STL is truncated: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if triangle_count == 0 or len(data) != expected_size:
        raise ValueError(
            f"invalid binary STL size for {path}: "
            f"triangles={triangle_count}, bytes={len(data)}, "
            f"expected={expected_size}"
        )

    mesh = Mesh()
    for triangle_index in range(triangle_count):
        record = struct.unpack_from("<12fH", data, 84 + triangle_index * 50)
        base_index = len(mesh.vertices)
        vertices = (record[3:6], record[6:9], record[9:12])
        finite = all(
            math.isfinite(value) for vertex in vertices for value in vertex
        )
        if not finite:
            raise ValueError(
                f"binary STL contains a non-finite vertex: {path}"
            )
        for x_value, y_value, z_value in vertices:
            mesh.vertices.append(
                Point(x=float(x_value), y=float(y_value), z=float(z_value))
            )
        triangle = MeshTriangle()
        triangle.vertex_indices = [base_index, base_index + 1, base_index + 2]
        mesh.triangles.append(triangle)
    return mesh


def make_tube_mesh(
    outer_radius: float,
    inner_radius: float,
    length: float,
    segments: int = 48,
) -> Mesh:
    """创建局部轴线沿 +Z 方向的封闭三角面空心管网格。"""

    if not 0.0 < inner_radius < outer_radius:
        raise ValueError("tube radii must satisfy 0 < inner < outer")
    if length <= 0.0 or segments < 3:
        raise ValueError("tube length and segment count must be positive")
    mesh = Mesh()
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for radius, z_value in (
            (outer_radius, 0.0),
            (outer_radius, length),
            (inner_radius, 0.0),
            (inner_radius, length),
        ):
            mesh.vertices.append(
                Point(x=radius * cosine, y=radius * sine, z=z_value)
            )

    def add_triangle(a: int, b: int, c: int) -> None:
        """向网格中追加一个三角面片。"""

        triangle = MeshTriangle()
        triangle.vertex_indices = [a, b, c]
        mesh.triangles.append(triangle)

    for index in range(segments):
        next_index = (index + 1) % segments
        outer_bottom = index * 4
        outer_top = outer_bottom + 1
        inner_bottom = outer_bottom + 2
        inner_top = outer_bottom + 3
        next_outer_bottom = next_index * 4
        next_outer_top = next_outer_bottom + 1
        next_inner_bottom = next_outer_bottom + 2
        next_inner_top = next_outer_bottom + 3
        for vertices in (
            (outer_bottom, next_outer_bottom, next_outer_top),
            (outer_bottom, next_outer_top, outer_top),
            (inner_bottom, inner_top, next_inner_top),
            (inner_bottom, next_inner_top, next_inner_bottom),
            (outer_bottom, inner_bottom, next_inner_bottom),
            (outer_bottom, next_inner_bottom, next_outer_bottom),
            (outer_top, next_outer_top, next_inner_top),
            (outer_top, next_inner_top, inner_top),
        ):
            add_triangle(*vertices)
    return mesh


def _box(x_size: float, y_size: float, z_size: float) -> SolidPrimitive:
    """创建指定尺寸的盒体碰撞几何。"""

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [x_size, y_size, z_size]
    return primitive


class SceneManager:
    """管理 MoveIt 世界物体，并清理旧任务遗留的附着物。"""

    def __init__(
        self,
        node: Node,
        service_callback_group: CallbackGroup,
        planning_frame: str,
        left_eef_link: str,
        right_eef_link: str,
        wire_model_pose: Pose,
        terminal_model_pose: Pose,
        wire_radius: float,
        wire_length: float,
        terminal_outer_radius: float,
        terminal_inner_radius: float,
        terminal_length: float,
        fixture_model_pose: Pose,
        fixture_mesh_resource: str,
        clearance_guard_center: Tuple[float, float, float],
        clearance_guard_size: Tuple[float, float, float],
        workpiece_collisions_enabled: bool = False,
    ) -> None:
        """创建 MoveIt 规划场景服务接口。"""

        self._node = node
        self.planning_frame = planning_frame
        self.left_eef_link = left_eef_link
        self.right_eef_link = right_eef_link
        self._initial_workpiece_poses = (wire_model_pose, terminal_model_pose)
        self.wire_model_pose = wire_model_pose
        self.terminal_model_pose = terminal_model_pose
        self.wire_radius = wire_radius
        self.wire_length = wire_length
        self.terminal_outer_radius = terminal_outer_radius
        self.terminal_inner_radius = terminal_inner_radius
        self.terminal_length = terminal_length
        self.fixture_model_pose = fixture_model_pose
        self.fixture_mesh = load_binary_stl_mesh(
            resolve_mesh_resource(fixture_mesh_resource)
        )
        self.clearance_guard_center = clearance_guard_center
        self.clearance_guard_size = clearance_guard_size
        self.workpiece_collisions_enabled = bool(workpiece_collisions_enabled)
        self._apply_client = node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=service_callback_group,
        )
        self._get_client = node.create_client(
            GetPlanningScene,
            "/get_planning_scene",
            callback_group=service_callback_group,
        )

    def interfaces_ready(self) -> bool:
        """返回 MoveIt 是否已能接收规划场景差量。"""

        return (
            self._apply_client.service_is_ready()
            and self._get_client.service_is_ready()
        )

    def update_workpiece_poses(self, wire_pose: Pose, terminal_pose: Pose) -> None:
        """更新后续规划场景差量使用的工件模型位姿。"""

        self.wire_model_pose = wire_pose
        self.terminal_model_pose = terminal_pose

    def reset_workpiece_poses(self):
        """Discard the previous task's visual poses before rebuilding the base scene."""
        self.wire_model_pose, self.terminal_model_pose = self._initial_workpiece_poses

    def apply_workpiece_poses_async(self):
        """以一个规划场景差量应用当前视觉工件位姿。"""

        scene = PlanningScene()
        scene.is_diff = True
        if self.workpiece_collisions_enabled:
            scene.world.collision_objects = [
                self.wire_object(self.planning_frame, self.wire_model_pose),
                self.terminal_object(
                    self.planning_frame, self.terminal_model_pose
                ),
            ]
        return self._apply(scene)

    def query_task_attachments_async(self):
        """在幂等重置前查询任务附着物和世界物体名称。"""

        if not self._get_client.service_is_ready():
            return None
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.WORLD_OBJECT_NAMES
        )
        return self._get_client.call_async(request)

    @staticmethod
    def attachment_query_result(
        future,
    ) -> Optional[Tuple[bool, Set[str], Set[str]]]:
        """返回查询结果、已附着任务物体 ID 和世界物体 ID。"""

        if future is None:
            return False, set(), set()
        if not future.done():
            return None
        try:
            attached_ids = {
                attached.object.id
                for attached in future.result().scene.robot_state.attached_collision_objects
                if attached.object.id in (WIRE_ID, TERMINAL_ID)
            }
            world_ids = {
                collision.id
                for collision in future.result().scene.world.collision_objects
            }
            return True, attached_ids, world_ids
        except Exception:
            return False, set(), set()

    def sync_world_async(self, attached_ids=(), world_ids=()):
        """替换任务碰撞物体，并移除失效的附着关系。"""

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for object_id, link_name in (
            (WIRE_ID, self.left_eef_link),
            (TERMINAL_ID, self.right_eef_link),
        ):
            if object_id not in attached_ids:
                continue
            attached = AttachedCollisionObject()
            attached.link_name = link_name
            attached.object.id = object_id
            attached.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        scene.world.collision_objects = [
            self.table_object(),
            self.fixture_object(),
        ]
        if self.workpiece_collisions_enabled:
            scene.world.collision_objects.extend(
                [
                    self.wire_object(
                        self.planning_frame, self.wire_model_pose
                    ),
                    self.terminal_object(
                        self.planning_frame, self.terminal_model_pose
                    ),
                ]
            )
        else:
            scene.world.collision_objects.extend(
                self._workpiece_world_removals(
                    object_id
                    for object_id in (WIRE_ID, TERMINAL_ID)
                    if object_id in set(world_ids) | set(attached_ids)
                )
            )
        if CLEARANCE_GUARD_ID in world_ids:
            scene.world.collision_objects.append(
                self.clearance_guard_object(add=False)
            )
        return self._apply(scene)

    def remove_workpieces_async(self, attached_ids, world_ids):
        """只移除查询确认存在的任务工件，兼容未建模及重复清理。

        MoveIt 拒绝删除不存在的附着体。解除真实附着会把物体放回世界，
        因此世界删除集合也必须包含本次解除附着的工件。
        """

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for object_id, link_name in ((WIRE_ID, self.left_eef_link),
                                     (TERMINAL_ID, self.right_eef_link)):
            if object_id not in attached_ids:
                continue
            attached = AttachedCollisionObject()
            attached.link_name = link_name
            attached.object.id = object_id
            attached.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(attached)
        removed_ids = set(attached_ids) | set(world_ids)
        scene.world.collision_objects = self._workpiece_world_removals(
            object_id for object_id in (WIRE_ID, TERMINAL_ID) if object_id in removed_ids
        )
        return self._apply(scene)

    def attach_workpieces_async(
        self,
        eef_to_wire_model: Pose,
        eef_to_terminal_model: Pose,
    ):
        """将抓取后的两个工件作为 MoveIt 附着碰撞体加入机器人状态。"""

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        if self.workpiece_collisions_enabled:
            scene.robot_state.attached_collision_objects = [
                self._wire_attachment(eef_to_wire_model),
                self._terminal_attachment(eef_to_terminal_model),
            ]
        return self._apply(scene)

    def _workpiece_world_removals(self, object_ids=(WIRE_ID, TERMINAL_ID)):
        """构造两个工件世界碰撞体的幂等删除差量。"""

        removals = []
        for object_id in object_ids:
            collision = CollisionObject()
            collision.header.frame_id = self.planning_frame
            collision.id = object_id
            collision.operation = CollisionObject.REMOVE
            removals.append(collision)
        return removals

    def _wire_attachment(self, eef_to_wire_model: Pose) -> AttachedCollisionObject:
        """构造只允许电线接触左侧抓持部件的附着碰撞体。"""

        attached = AttachedCollisionObject()
        attached.link_name = self.left_eef_link
        attached.touch_links = [
            "left_finger1",
            "left_finger2",
            self.left_eef_link,
        ]
        attached.object = self.wire_object(self.left_eef_link, eef_to_wire_model)
        return attached

    def _terminal_attachment(
        self, eef_to_terminal_model: Pose
    ) -> AttachedCollisionObject:
        """构造只允许端子接触右侧抓持部件的附着碰撞体。"""

        attached = AttachedCollisionObject()
        attached.link_name = self.right_eef_link
        attached.touch_links = [
            "right_finger1",
            "right_finger2",
            self.right_eef_link,
        ]
        attached.object = self.terminal_object(
            self.right_eef_link, eef_to_terminal_model
        )
        return attached

    def set_clearance_guard_async(self, add: bool):
        """添加或移除双臂撤退后使用的中央虚拟地板。"""

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [self.clearance_guard_object(add)]
        return self._apply(scene)

    def _apply(self, scene: PlanningScene):
        """以非阻塞方式发送 ApplyPlanningScene 请求。"""

        if not self._apply_client.service_is_ready():
            return None
        request = ApplyPlanningScene.Request()
        request.scene = scene
        return self._apply_client.call_async(request)

    @staticmethod
    def future_succeeded(future) -> Optional[bool]:
        """等待期间返回 None，否则返回 ApplyPlanningScene 的执行结果。"""

        if future is None:
            return False
        if not future.done():
            return None
        try:
            return bool(future.result().success)
        except Exception:  # The FSM reports the contextual operation and retries.
            return False

    def table_object(self) -> CollisionObject:
        """桌面顶面 0.750 m，与 0.753 m 的底座安装面保留 3 mm 间隙。"""

        collision = CollisionObject()
        collision.header.frame_id = self.planning_frame
        collision.header.stamp = self._node.get_clock().now().to_msg()
        collision.id = TABLE_ID
        collision.operation = CollisionObject.ADD
        collision.primitives = [_box(1.6, 1.0, 0.06)]
        collision.primitive_poses = [make_pose(0.0, 0.0, 0.720)]
        return collision

    def fixture_object(self) -> CollisionObject:
        """构造与 MuJoCo 位姿和 STL 一致的固定中间件。"""

        collision = CollisionObject()
        collision.header.frame_id = self.planning_frame
        collision.header.stamp = self._node.get_clock().now().to_msg()
        collision.id = FIXTURE_ID
        collision.operation = CollisionObject.ADD
        collision.meshes = [self.fixture_mesh]
        collision.mesh_poses = [self.fixture_model_pose]
        return collision

    def clearance_guard_object(self, add: bool = True) -> CollisionObject:
        """构造避开两个抬高底座的中央禁入板。"""

        collision = CollisionObject()
        collision.header.frame_id = self.planning_frame
        collision.header.stamp = self._node.get_clock().now().to_msg()
        collision.id = CLEARANCE_GUARD_ID
        collision.operation = CollisionObject.ADD if add else CollisionObject.REMOVE
        if add:
            collision.primitives = [_box(*self.clearance_guard_size)]
            collision.primitive_poses = [make_pose(*self.clearance_guard_center)]
        return collision

    def wire_object(self, frame_id: str, wire_model_pose: Pose) -> CollisionObject:
        """构造直径 7 mm、长度 60 mm 的电线碰撞圆柱。"""

        collision = CollisionObject()
        collision.header.frame_id = frame_id
        collision.header.stamp = self._node.get_clock().now().to_msg()
        collision.id = WIRE_ID
        collision.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [self.wire_length, self.wire_radius]
        collision.primitives = [primitive]
        center = compose_pose(
            wire_model_pose, make_pose(0.0, 0.0, self.wire_length / 2.0)
        )
        collision.primitive_poses = [center]
        return collision

    def terminal_object(
        self, frame_id: str, terminal_model_pose: Pose
    ) -> CollisionObject:
        """构造保留真实内孔的空心端子碰撞网格。"""

        collision = CollisionObject()
        collision.header.frame_id = frame_id
        collision.header.stamp = self._node.get_clock().now().to_msg()
        collision.id = TERMINAL_ID
        collision.operation = CollisionObject.ADD
        collision.meshes = [
            make_tube_mesh(
                self.terminal_outer_radius,
                self.terminal_inner_radius,
                self.terminal_length,
            )
        ]
        collision.mesh_poses = [terminal_model_pose]
        return collision
