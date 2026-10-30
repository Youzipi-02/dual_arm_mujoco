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

"""从对齐深度图中估计 OBB 表面和采样端点的可靠深度。."""

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthEstimate:
    """保存 OBB 中心、长轴端点和深度质量指标，单位均为米。."""

    center_depth_m: float
    first_endpoint_depth_m: float
    second_endpoint_depth_m: float
    valid_ratio: float
    mad_m: float


def depth_image_to_meters(
    depth_image: np.ndarray,
    encoding: str,
) -> np.ndarray:
    """将 16UC1/mono16 或 32FC1 深度图统一转换为米。."""
    image = np.asarray(depth_image)
    normalized_encoding = str(encoding).upper()
    if normalized_encoding in ("16UC1", "MONO16"):
        return image.astype(np.float32) * 0.001
    if normalized_encoding == "32FC1":
        return image.astype(np.float32, copy=False)
    raise ValueError(f"不支持的深度图编码: {encoding}")


def robust_surface_depth(
    depth_values: np.ndarray,
    percentile: float,
    surface_band_m: float,
) -> tuple[float, float]:
    """利用分位数、带宽和中位数估计表面深度及 MAD。."""
    values = np.asarray(depth_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("OBB 中没有有效深度")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("深度分位数必须位于零到一百之间")
    if not math.isfinite(surface_band_m) or surface_band_m <= 0.0:
        raise ValueError("表面深度带宽必须大于零")

    reference = float(np.percentile(values, percentile))
    inliers = values[np.abs(values - reference) <= surface_band_m]
    if inliers.size == 0:
        raise ValueError("OBB 中没有满足表面带宽的深度")
    surface_depth = float(np.median(inliers))
    mad = float(np.median(np.abs(inliers - surface_depth)))
    return surface_depth, mad


def _scaled_polygon(corners: np.ndarray, scale: float) -> np.ndarray:
    """围绕 OBB 中心向内缩放四边形。."""
    polygon = np.asarray(corners, dtype=np.float64)
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        raise ValueError("OBB 顶点必须是形状为 4×2 的有限数值数组")
    if not 0.0 < scale <= 1.0:
        raise ValueError("OBB 掩膜缩放比例必须位于零到一之间")
    center = np.mean(polygon, axis=0)
    return center + scale * (polygon - center)


def _sample_surface_depth(
    depth_m: np.ndarray,
    pixel: np.ndarray,
    radius_px: int,
    minimum_depth_m: float,
    maximum_depth_m: float,
    percentile: float,
    surface_band_m: float,
    fallback_depth_m: float,
) -> float:
    """在端点邻域采样表面深度，无有效值时返回 OBB 表面深度。."""
    height, width = depth_m.shape
    center_x = int(round(float(pixel[0])))
    center_y = int(round(float(pixel[1])))
    x_min = max(0, center_x - radius_px)
    x_max = min(width, center_x + radius_px + 1)
    y_min = max(0, center_y - radius_px)
    y_max = min(height, center_y + radius_px + 1)
    if x_min >= x_max or y_min >= y_max:
        return fallback_depth_m

    values = depth_m[y_min:y_max, x_min:x_max].reshape(-1)
    valid = values[
        np.isfinite(values)
        & (values >= minimum_depth_m)
        & (values <= maximum_depth_m)
    ]
    if valid.size == 0:
        return fallback_depth_m
    try:
        surface, _ = robust_surface_depth(
            valid,
            percentile,
            surface_band_m,
        )
    except ValueError:
        return fallback_depth_m
    return surface


def estimate_obb_depth(
    depth_m: np.ndarray,
    corners: np.ndarray,
    endpoint_pixels: tuple[Sequence[float], Sequence[float]],
    *,
    roi_scale: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
    minimum_valid_ratio: float,
    surface_percentile: float,
    surface_band_m: float,
    endpoint_radius_px: int,
    center_depth_offset_m: float,
) -> DepthEstimate:
    """
    估计 OBB 中心和长轴两端的深度及质量。.

    该函数只处理深度数值和像素区域，不执行相机反投影或坐标变换。
    """
    image = np.asarray(depth_m, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("深度图必须是单通道二维数组")
    if not 0.0 <= minimum_valid_ratio <= 1.0:
        raise ValueError("最小有效深度比例必须位于零到一之间")
    if (
        not math.isfinite(minimum_depth_m)
        or not math.isfinite(maximum_depth_m)
        or minimum_depth_m <= 0.0
        or maximum_depth_m <= minimum_depth_m
    ):
        raise ValueError("深度有效范围配置错误")
    if center_depth_offset_m < 0.0 or not math.isfinite(center_depth_offset_m):
        raise ValueError("中心深度偏移不能为负数")
    if endpoint_radius_px < 0:
        raise ValueError("端点采样半径不能为负数")

    polygon = _scaled_polygon(corners, roi_scale)
    height, width = image.shape
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    mask_count = int(np.count_nonzero(mask))
    if mask_count == 0:
        raise ValueError("缩放后的 OBB 掩膜为空")

    values = image[mask.astype(bool)]
    valid = values[
        np.isfinite(values)
        & (values >= minimum_depth_m)
        & (values <= maximum_depth_m)
    ]
    valid_ratio = float(valid.size) / float(mask_count)
    if valid.size == 0 or valid_ratio < minimum_valid_ratio:
        raise ValueError(
            f"OBB 有效深度比例不足: {valid_ratio:.3f} < "
            f"{minimum_valid_ratio:.3f}"
        )

    surface_depth, depth_mad = robust_surface_depth(
        valid,
        surface_percentile,
        surface_band_m,
    )
    first_pixel = np.asarray(endpoint_pixels[0], dtype=np.float64)
    second_pixel = np.asarray(endpoint_pixels[1], dtype=np.float64)
    for pixel in (first_pixel, second_pixel):
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError("OBB 长轴端点必须是两个有限像素坐标")

    first_surface_depth = _sample_surface_depth(
        image,
        first_pixel,
        endpoint_radius_px,
        minimum_depth_m,
        maximum_depth_m,
        surface_percentile,
        surface_band_m,
        surface_depth,
    )
    second_surface_depth = _sample_surface_depth(
        image,
        second_pixel,
        endpoint_radius_px,
        minimum_depth_m,
        maximum_depth_m,
        surface_percentile,
        surface_band_m,
        surface_depth,
    )
    return DepthEstimate(
        center_depth_m=surface_depth + center_depth_offset_m,
        first_endpoint_depth_m=first_surface_depth + center_depth_offset_m,
        second_endpoint_depth_m=second_surface_depth + center_depth_offset_m,
        valid_ratio=valid_ratio,
        mad_m=depth_mad,
    )
