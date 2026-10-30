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

"""YOLO 分割掩膜的清理、轮廓提取和孔洞端面配对."""

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationDetection:
    """保存一项与 Ultralytics 解耦的二维分割检测."""

    index: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray
    contour: np.ndarray
    area_px: float
    center: tuple[float, float]


@dataclass(frozen=True)
class HoleFacePair:
    """保存一个孔洞及其对应端面和孔口拟合椭圆."""

    hole: SegmentationDetection
    face: SegmentationDetection
    ellipse: tuple
    match_mode: str
    center_distance_ratio: float
    hole_to_face_area_ratio: float


def clean_binary_mask(
    mask: np.ndarray,
    image_shape: Sequence[int],
    *,
    threshold: float,
    opening_kernel_px: int,
    closing_kernel_px: int,
) -> np.ndarray:
    """将概率掩膜缩放到原图并执行可配置的形态学清理."""
    value = np.asarray(mask, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("分割掩膜必须是二维数组")
    if len(image_shape) < 2:
        raise ValueError("图像尺寸至少需要包含高度和宽度")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须大于零")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("掩膜阈值必须位于零到一之间")

    resized = cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (resized >= threshold).astype(np.uint8) * 255
    for operation, kernel_size in (
        (cv2.MORPH_OPEN, opening_kernel_px),
        (cv2.MORPH_CLOSE, closing_kernel_px),
    ):
        if kernel_size <= 1:
            continue
        size = int(kernel_size)
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.morphologyEx(binary, operation, kernel)
    return binary


def largest_contour(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """返回二值掩膜的最大外轮廓及其面积."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("二值掩膜必须是二维数组")
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("掩膜中没有有效轮廓")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        raise ValueError("掩膜轮廓面积必须大于零")
    return contour, area


def contour_center(contour: np.ndarray) -> tuple[float, float]:
    """使用轮廓矩计算二维中心."""
    moments = cv2.moments(contour)
    if abs(float(moments["m00"])) < 1.0e-9:
        raise ValueError("轮廓矩面积为零")
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def make_detection(
    *,
    index: int,
    class_name: str,
    confidence: float,
    bbox_xyxy: Sequence[float],
    probability_mask: np.ndarray,
    image_shape: Sequence[int],
    threshold: float,
    opening_kernel_px: int,
    closing_kernel_px: int,
) -> SegmentationDetection:
    """从一项模型输出构造经过清理的分割检测."""
    bbox = tuple(float(value) for value in bbox_xyxy)
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("检测框必须包含四个有限数值")
    mask = clean_binary_mask(
        probability_mask,
        image_shape,
        threshold=threshold,
        opening_kernel_px=opening_kernel_px,
        closing_kernel_px=closing_kernel_px,
    )
    contour, area = largest_contour(mask)
    return SegmentationDetection(
        index=int(index),
        class_name=str(class_name),
        confidence=float(confidence),
        bbox_xyxy=bbox,
        mask=mask,
        contour=contour,
        area_px=area,
        center=contour_center(contour),
    )


def fit_hole_ellipse(contour: np.ndarray) -> tuple:
    """拟合孔洞轮廓椭圆并统一为有限浮点结果."""
    value = np.asarray(contour)
    if len(value) < 5:
        raise ValueError("拟合孔洞椭圆至少需要五个轮廓点")
    ellipse = cv2.fitEllipse(value.astype(np.float32))
    center, axes, angle = ellipse
    numbers = (*center, *axes, angle)
    if not all(math.isfinite(float(number)) for number in numbers):
        raise ValueError("孔洞椭圆包含非有限数值")
    if min(float(axes[0]), float(axes[1])) <= 0.0:
        raise ValueError("孔洞椭圆轴长必须大于零")
    return (
        (float(center[0]), float(center[1])),
        (float(axes[0]), float(axes[1])),
        float(angle),
    )


def select_hole_face_pair(
    detections: Sequence[SegmentationDetection],
    *,
    hole_class: str,
    face_class: str,
    minimum_hole_area_px: float,
    minimum_face_area_px: float,
    maximum_hole_to_face_area_ratio: float = 3.0,
    face_margin_ratio: float = 0.25,
    maximum_center_distance_ratio: float = 1.25,
) -> HoleFacePair:
    """使用分层空间关系、面积和置信度选择最佳孔洞端面组合."""
    if maximum_hole_to_face_area_ratio <= 0.0:
        raise ValueError("孔洞端面最大面积比必须大于零")
    if face_margin_ratio < 0.0:
        raise ValueError("端面匹配边界扩展比例不能小于零")
    if maximum_center_distance_ratio <= 0.0:
        raise ValueError("孔洞端面最大中心距离比必须大于零")

    raw_holes = [
        item for item in detections if item.class_name == hole_class
    ]
    raw_faces = [
        item for item in detections if item.class_name == face_class
    ]
    holes = [
        item
        for item in raw_holes
        if item.area_px >= minimum_hole_area_px
    ]
    faces = [
        item
        for item in raw_faces
        if item.area_px >= minimum_face_area_px
    ]
    if not holes or not faces:
        hole_areas = ",".join(f"{item.area_px:.0f}" for item in raw_holes) or "none"
        face_areas = ",".join(f"{item.area_px:.0f}" for item in raw_faces) or "none"
        raise ValueError(
            "当前帧缺少有效孔洞或端面掩膜："
            f"hole={len(holes)}/{len(raw_holes)} "
            f"areas=[{hole_areas}] min={minimum_hole_area_px:.0f}px, "
            f"face={len(faces)}/{len(raw_faces)} "
            f"areas=[{face_areas}] min={minimum_face_area_px:.0f}px"
        )

    candidates = []
    area_rejections = 0
    spatial_rejections = 0
    for hole in holes:
        ellipse = fit_hole_ellipse(hole.contour)
        hole_center = ellipse[0]
        for face in faces:
            area_ratio = hole.area_px / max(face.area_px, 1.0)
            if area_ratio > maximum_hole_to_face_area_ratio:
                area_rejections += 1
                continue

            x_min, y_min, x_max, y_max = face.bbox_xyxy
            width = max(float(x_max - x_min), 1.0)
            height = max(float(y_max - y_min), 1.0)
            diagonal = max(math.hypot(width, height), 1.0)
            face_center = (
                0.5 * float(x_min + x_max),
                0.5 * float(y_min + y_max),
            )
            center_distance = math.hypot(
                hole_center[0] - face_center[0],
                hole_center[1] - face_center[1],
            )
            normalized_distance = center_distance / diagonal

            hull = cv2.convexHull(face.contour)
            signed_hull_distance = float(
                cv2.pointPolygonTest(hull, hole_center, True)
            )
            margin_x = face_margin_ratio * width
            margin_y = face_margin_ratio * height
            inside_expanded_bbox = (
                x_min - margin_x <= hole_center[0] <= x_max + margin_x
                and y_min - margin_y <= hole_center[1] <= y_max + margin_y
            )
            near_hull = signed_hull_distance >= -face_margin_ratio * diagonal

            if signed_hull_distance >= 0.0:
                match_mode = "inside_hull"
                relation_bonus = 2.0
            elif near_hull and inside_expanded_bbox:
                match_mode = "near_hull"
                relation_bonus = 1.25
            elif inside_expanded_bbox:
                match_mode = "expanded_bbox"
                relation_bonus = 0.75
            elif normalized_distance <= maximum_center_distance_ratio:
                match_mode = "center_distance"
                relation_bonus = 0.0
            else:
                spatial_rejections += 1
                continue

            area_penalty = 0.15 * max(0.0, area_ratio - 1.0)
            score = (
                hole.confidence
                + face.confidence
                + relation_bonus
                - normalized_distance
                - area_penalty
            )
            candidates.append(
                (
                    score,
                    hole,
                    face,
                    ellipse,
                    match_mode,
                    normalized_distance,
                    area_ratio,
                )
            )
    if not candidates:
        raise ValueError(
            "未找到空间关系合理的孔洞端面组合："
            f"holes={len(holes)}, faces={len(faces)}, "
            f"area_rejected={area_rejections}, "
            f"spatial_rejected={spatial_rejections}, "
            f"limits(area_ratio<={maximum_hole_to_face_area_ratio:.2f}, "
            f"center_ratio<={maximum_center_distance_ratio:.2f})"
        )
    (
        _,
        hole,
        face,
        ellipse,
        match_mode,
        normalized_distance,
        area_ratio,
    ) = max(candidates, key=lambda item: item[0])
    return HoleFacePair(
        hole=hole,
        face=face,
        ellipse=ellipse,
        match_mode=match_mode,
        center_distance_ratio=normalized_distance,
        hole_to_face_area_ratio=area_ratio,
    )
