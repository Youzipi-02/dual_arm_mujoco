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

"""测试节点内的 YOLO-OBB 类别名称处理函数。."""

from dataclasses import replace
import math
from threading import Condition
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

node_module = pytest.importorskip(
    "s622_dual_perception.nodes.yolo_detector_obb"
)
_normalize_model_names = node_module._normalize_model_names
_resolve_class_ids = node_module._resolve_class_ids


def test_normalize_mapping_names():
    """字典形式的模型类别能够转换为整数键。."""
    assert _normalize_model_names({"0": "terminal", 1: "wire"}) == {
        0: "terminal",
        1: "wire",
    }


def test_resolve_requested_class_ids():
    """目标类别按照请求顺序转换并去除重复项。."""
    assert _resolve_class_ids(
        {0: "terminal", 1: "wire"},
        ["wire", "terminal", "wire"],
    ) == [1, 0]


def test_empty_requested_classes_selects_all():
    """空类别列表使用 None 表示不限制模型类别。."""
    assert _resolve_class_ids({0: "terminal", 1: "wire"}, []) is None


def test_unknown_class_is_rejected():
    """请求不存在类别时应给出明确异常。."""
    with pytest.raises(ValueError, match="不存在目标类别"):
        _resolve_class_ids({0: "terminal", 1: "wire"}, ["cable"])


@pytest.mark.parametrize("angle, expected", [
    (0.0, (0.8, 0.1)),
    (math.pi / 4.0, (0.632455532, 0.158113883)),
    (math.pi / 2.0, (0.4, 0.2)),
])
def test_visible_geometry_uses_camera_intrinsics(angle, expected):
    """像素边长按各自焦距和旋转方向转换为米制长度。"""
    intrinsics = node_module.CameraIntrinsics(100.0, 200.0, 50.0, 50.0, 100, 100)
    detection = SimpleNamespace(size=(40.0, 10.0), angle=angle)
    geometry = node_module._estimate_object_geometry(detection, 2.0, intrinsics)
    assert (geometry.length, geometry.diameter) == pytest.approx(expected)


def test_rgbd_localization_without_shared_dimensions():
    """不提供工件尺寸时仍生成表面三维位置、有效检测框和 Marker。"""
    from std_msgs.msg import Header
    from s622_dual_perception.target_confirmation import detection_is_valid

    detection = node_module.DetectedObb(
        index=0, class_name="wire", confidence=0.9,
        center=(50.0, 50.0), size=(40.0, 10.0), angle=0.0,
        corners=np.array([[30., 45.], [70., 45.], [70., 55.], [30., 55.]]),
    )
    config = SimpleNamespace(
        roi_scale=0.8, minimum_depth_m=0.15, maximum_depth_m=4.0,
        minimum_valid_ratio=0.2, surface_percentile=35.0, surface_band_m=0.02,
        endpoint_radius_px=2, summary_interval=30,
    )
    intrinsics = node_module.CameraIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100)
    node = SimpleNamespace(
        _config=config, _enabled=True, _frame_condition=Condition(), _processed_frames=0,
        _convert_inputs=Mock(return_value=(None, np.ones((100, 100)), intrinsics)),
        _predict=Mock(), _parse_detections=Mock(return_value=[detection]),
        _transform_targets=lambda header, targets: [
            replace(target, pose_world=target.pose_camera) for target in targets
        ],
        _publish_results=Mock(), _log_localization_warning=Mock(),
    )
    header = Header(frame_id="world")
    node_module.YoloDetectorObb._process_rgbd(node, SimpleNamespace(header=header), None, None)
    target = node._publish_results.call_args.args[3][0]
    assert target.pose_camera.center == pytest.approx((0.0, 0.0, 1.0))
    message = node_module.YoloDetectorObb._build_detection_3d(
        node, header, target, target.pose_camera
    )
    assert (message.bbox.size.x, message.bbox.size.y, message.bbox.size.z) == pytest.approx(
        (0.4, 0.1, 0.1)
    )
    assert detection_is_valid(message, "world")
    world_target = node._publish_results.call_args.args[4][0]
    markers = node_module.build_marker_array(header, [world_target])
    assert len(markers.markers) == 4
    node._log_localization_warning.assert_not_called()
