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

"""生成 OpenCV 调试图像和 RViz Marker,不负责创建发布器。."""

from collections.abc import Sequence
import cv2
from builtin_interfaces.msg import Duration as DurationMessage
from geometry_msgs.msg import Point
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from s622_dual_perception.visual_perception_utils.obb_geometry import (
    normalize_vector,
)


def _class_color(class_name: str) -> tuple[int, int, int]:
    """返回 OpenCV BGR 颜色。."""
    if class_name == "wire":
        return 80, 130, 230
    if class_name == "copper_terminal":
        return 40, 170, 255
    return 80, 220, 80


def draw_obb(
    image,
    corners,
    color: tuple[int, int, int],
) -> None:
    """在图像上绘制四边形 OBB。."""
    points = corners.astype("int32")
    cv2.polylines(
        image,
        [points.reshape((-1, 1, 2))],
        isClosed=True,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_center(
    image,
    center_pixel: Sequence[float],
    color: tuple[int, int, int],
) -> None:
    """在图像上绘制检测中心。."""
    center = tuple(int(round(value)) for value in center_pixel)
    cv2.circle(image, center, 4, color, thickness=-1, lineType=cv2.LINE_AA)


def draw_major_axis(
    image,
    endpoint_pixels: tuple[Sequence[float], Sequence[float]],
    color: tuple[int, int, int],
) -> None:
    """在图像上绘制 OBB 长轴的内部采样线段。."""
    first = tuple(int(round(value)) for value in endpoint_pixels[0])
    second = tuple(int(round(value)) for value in endpoint_pixels[1])
    cv2.line(image, first, second, color, 2, cv2.LINE_AA)


def _format_vector(prefix: str, value: Sequence[float]) -> str:
    """把三维坐标格式化为短文本。."""
    return f"{prefix}[{value[0]:.3f},{value[1]:.3f},{value[2]:.3f}]"


def draw_detection_text(
    image,
    center_pixel: Sequence[float],
    class_name: str,
    confidence: float,
    center_camera=None,
    center_world=None,
    depth_quality=None,
) -> None:
    """绘制类别、置信度、坐标和深度质量文本。."""
    text_parts = [f"{class_name} {confidence:.2f}"]
    # if center_camera is not None:
    #     text_parts.append(_format_vector("Camera", center_camera))
    if center_world is not None:
        text_parts.append(_format_vector("World", center_world))
    if depth_quality is not None:
        valid_ratio = getattr(depth_quality, "valid_ratio", None)
        mad_m = getattr(depth_quality, "mad_m", None)
        if valid_ratio is not None and mad_m is not None:
            text_parts.append(f"Depth:{valid_ratio:.2f}/{mad_m * 1000.0:.1f}mm")

    # origin_x = max(0, int(round(center_pixel[0])) - 100)
    # origin_y = max(18, int(round(center_pixel[1])) - 8)
    origin_x = int(round(center_pixel[0])) - 40
    origin_y = int(round(center_pixel[1])) + 20
    for line_index, text in enumerate(text_parts):
        origin = (origin_x, origin_y + 15 * line_index)
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def render_debug_image(frame, detections, localized_targets):
    """返回绘制 OBB、长轴、三维坐标和深度质量的 BGR 图像。."""
    annotated = frame.copy()
    localized_by_index = {
        target.detection.index: target for target in localized_targets
    }
    for detection in detections:
        color = _class_color(detection.class_name)
        draw_obb(annotated, detection.corners, color)
        draw_center(annotated, detection.center, color)
        target = localized_by_index.get(detection.index)
        center_camera = None
        center_world = None
        depth_quality = None
        if target is not None:
            draw_major_axis(annotated, target.endpoint_pixels, color)
            center_camera = target.pose_camera.center
            depth_quality = target.depth
            if target.pose_world is not None:
                center_world = target.pose_world.center
        draw_detection_text(
            annotated,
            detection.center,
            detection.class_name,
            detection.confidence,
            center_camera=center_camera,
            center_world=center_world,
            depth_quality=depth_quality,
        )
    return annotated


def _append_target_markers(
    marker_array: MarkerArray,
    header,
    marker_id: int,
    class_name: str,
    center,
    axis,
    length: float,
    color: tuple[float, float, float],
) -> int:
    """追加一个中心球、长轴箭头和类别文字 Marker。."""
    lifetime = DurationMessage(sec=0, nanosec=500_000_000)
    center_marker = Marker()
    center_marker.header = header
    center_marker.ns = "estimated"
    center_marker.id = marker_id
    center_marker.type = Marker.SPHERE
    center_marker.action = Marker.ADD
    center_marker.pose.position = Point(
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2]),
    )
    center_marker.pose.orientation.w = 1.0
    center_marker.scale.x = 0.012
    center_marker.scale.y = 0.012
    center_marker.scale.z = 0.012
    center_marker.color.r = color[0]
    center_marker.color.g = color[1]
    center_marker.color.b = color[2]
    center_marker.color.a = 0.95
    center_marker.lifetime = lifetime
    marker_array.markers.append(center_marker)
    marker_id += 1

    axis_value = normalize_vector(axis)
    first = center - 0.5 * length * axis_value
    second = center + 0.5 * length * axis_value
    axis_marker = Marker()
    axis_marker.header = header
    axis_marker.ns = "estimated"
    axis_marker.id = marker_id
    axis_marker.type = Marker.ARROW
    axis_marker.action = Marker.ADD
    axis_marker.points = [
        Point(x=float(first[0]), y=float(first[1]), z=float(first[2])),
        Point(x=float(second[0]), y=float(second[1]), z=float(second[2])),
    ]
    axis_marker.scale.x = 0.003
    axis_marker.scale.y = 0.007
    axis_marker.scale.z = 0.009
    axis_marker.color.r = color[0]
    axis_marker.color.g = color[1]
    axis_marker.color.b = color[2]
    axis_marker.color.a = 0.95
    axis_marker.lifetime = lifetime
    marker_array.markers.append(axis_marker)
    marker_id += 1

    text_marker = Marker()
    text_marker.header = header
    text_marker.ns = "estimated"
    text_marker.id = marker_id
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position = Point(
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2] + 0.025),
    )
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.018
    text_marker.color.r = color[0]
    text_marker.color.g = color[1]
    text_marker.color.b = color[2]
    text_marker.color.a = 0.95
    text_marker.text = class_name
    text_marker.lifetime = lifetime
    marker_array.markers.append(text_marker)
    return marker_id + 1


def build_marker_array(header, world_targets) -> MarkerArray:
    """构造世界坐标下的目标中心、长轴和文字 Marker。."""
    marker_array = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)

    marker_id = 0
    for target in world_targets:
        if target.pose_world is None:
            continue
        color = (
            (0.45, 0.20, 0.08)
            if target.detection.class_name == "wire"
            else (1.0, 0.45, 0.05)
        )
        marker_id = _append_target_markers(
            marker_array,
            header,
            marker_id,
            target.detection.class_name,
            target.pose_world.center,
            target.pose_world.axis,
            target.geometry.length,
            color,
        )
    return marker_array


def blend_segmentation_masks(
    image,
    hole_mask,
    face_mask,
    *,
    alpha: float = 0.35,
) -> None:
    """把孔洞和端面掩膜以不同颜色叠加到 BGR 图像."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("掩膜透明度必须位于零到一之间")
    hole = np.asarray(hole_mask) > 0
    face = np.asarray(face_mask) > 0
    if hole.shape != image.shape[:2] or face.shape != image.shape[:2]:
        raise ValueError("分割掩膜与调试图像尺寸不一致")
    overlay = image.copy()
    overlay[face] = (40, 170, 255)
    overlay[hole] = (255, 170, 40)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def draw_hole_ellipse(image, ellipse) -> None:
    """绘制孔洞拟合椭圆和中心十字."""
    center, axes, angle = ellipse
    cv2.ellipse(
        image,
        (
            (float(center[0]), float(center[1])),
            (float(axes[0]), float(axes[1])),
            float(angle),
        ),
        (80, 255, 80),
        1,
        cv2.LINE_AA,
    )
    pixel = tuple(int(round(value)) for value in center)
    # cv2.drawMarker(
    #     image,
    #     pixel,
    #     (80, 255, 80),
    #     markerType=cv2.MARKER_CROSS,
    #     markerSize=14,
    #     thickness=2,
    #     line_type=cv2.LINE_AA,
    # )
    cv2.circle(
    image,
    pixel,
    radius=2,
    color=(80, 255, 80),
    thickness=-1,  # -1 表示填充实心圆
    lineType=cv2.LINE_AA,
)

def draw_hole_center_text(
    image,
    pair,
    center_camera=None,
    center_world=None,
    stability_status: str = "DISABLED",
) -> None:
    """绘制类别置信度、孔中心、深度质量和稳定状态."""
    lines = [
        f"hole conf: {pair.hole.confidence:.2f}",
        f"face conf: {pair.face.confidence:.2f}",
        # (
        #     f"pair: {pair.match_mode} "
        #     f"distance: {pair.center_distance_ratio:.2f} "
        #     f"area: {pair.hole_to_face_area_ratio:.2f}"
        # ),
    ]
    # if center_camera is not None:
        # lines.append(_format_vector("center camera: ", center_camera.center))
        # lines.append(
        #     f"depth median: {center_camera.depth.depth_m:.3f}m "
        #     f"mad: {center_camera.depth.mad_m * 1000.0:.2f}mm"
        # )
        # lines.append(
        #     f"depth valid: {center_camera.depth.valid_ratio:.2f} "
        #     f"cluster: {center_camera.depth.cluster_ratio:.2f}"
        # )
    if center_world is not None:
        lines.append(_format_vector("center world: ", center_world.center))
    lines.append(f"stability: {stability_status}")

    stable = stability_status.startswith("STABLE")
    status_color = (80, 255, 80) if stable else (40, 220, 255)
    for index, line in enumerate(lines):
        color = status_color if index == len(lines) - 1 else (255, 255, 255)
        cv2.putText(
            image,
            line,
            (16, 24 + 18 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

def render_segmentation_status_image(
    frame,
    status: str,
    detections=(),
    *,
    hole_class: str = "hole",
    face_class: str = "terminal_face",
    minimum_hole_area_px: float = 0.0,
    minimum_face_area_px: float = 0.0,
):
    """在配对失败时直接发布原始图像."""
    annotated = frame.copy()
    return annotated


def render_segmentation_debug_image(
    frame,
    pair,
    *,
    center_camera=None,
    center_world=None,
    stability_status: str = "DISABLED",
):
    """生成孔洞端面掩膜、中心和深度质量信息调试图."""
    annotated = frame.copy()
    # blend_segmentation_masks(
    #     annotated,
    #     pair.hole.mask,
    #     pair.face.mask,
    # )
    # cv2.drawContours(
    #     annotated,
    #     [pair.face.contour],
    #     -1,
    #     (40, 170, 255),
    #     2,
    #     cv2.LINE_AA,
    # )
    # cv2.drawContours(
    #     annotated,
    #     [pair.hole.contour],
    #     -1,
    #     (255, 170, 40),
    #     2,
    #     cv2.LINE_AA,
    # )
    draw_hole_ellipse(annotated, pair.ellipse)
    draw_hole_center_text(
        annotated,
        pair,
        center_camera,
        center_world,
        stability_status,
    )
    return annotated


def clear_hole_marker_array(header) -> MarkerArray:
    """构造清除全部孔特征 Marker 的消息."""
    marker_array = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)
    return marker_array


def build_hole_center_marker_array(
    header,
    center,
    *,
    diameter_m: float,
) -> MarkerArray:
    """构造仅用于 RViz 参考、不带碰撞属性的稳定孔中心蓝点."""
    center_value = np.asarray(center, dtype=np.float64)
    marker_array = clear_hole_marker_array(header)
    lifetime = DurationMessage(sec=0, nanosec=0)

    center_marker = Marker()
    center_marker.header = header
    center_marker.ns = "computed_hole_center"
    center_marker.id = 0
    center_marker.type = Marker.SPHERE
    center_marker.action = Marker.ADD
    center_marker.pose.position = Point(
        x=float(center_value[0]),
        y=float(center_value[1]),
        z=float(center_value[2]),
    )
    center_marker.pose.orientation.w = 1.0
    center_marker.scale.x = float(diameter_m)
    center_marker.scale.y = float(diameter_m)
    center_marker.scale.z = float(diameter_m)
    center_marker.color.b = 1.0
    center_marker.color.a = 1.0
    center_marker.lifetime = lifetime
    marker_array.markers.append(center_marker)
    return marker_array
