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

"""左腕 RGB-D YOLO 分割、孔中心三维定位和稳定结果发布节点."""

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
import numpy as np
from rcl_interfaces.msg import FloatingPointRange
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import SetBool
from std_srvs.srv import Trigger
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray

from s622_dual_perception.visual_perception_utils.depth_estimation import (
    depth_image_to_meters,
)
from s622_dual_perception.visual_perception_utils.obb_geometry import (
    CameraIntrinsics,
)
from s622_dual_perception.visual_perception_utils.segmentation_postprocess import (
    HoleFacePair,
    SegmentationDetection,
    make_detection,
    select_hole_face_pair,
)
from s622_dual_perception.visual_perception_utils.terminal_pose_estimation import (
    HoleCenterStabilityGate,
    StableHoleCenter,
    estimate_depth_from_mask,
    estimate_hole_center,
    transform_hole_center,
)
from s622_dual_perception.visual_perception_utils.visualization import (
    build_hole_center_marker_array,
    clear_hole_marker_array,
    render_segmentation_debug_image,
    render_segmentation_status_image,
)

try:
    from ultralytics import YOLO
except ImportError as import_error:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = import_error
else:
    ULTRALYTICS_IMPORT_ERROR = None


@dataclass(frozen=True)
class SegmenterConfig:
    """保存左相机分割节点启动后使用的不变配置."""

    model_path: Path
    device: str
    hole_class: str
    face_class: str
    confidence: float
    iou: float
    image_size: int
    max_detections: int
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    sync_queue_size: int
    sync_slop: float
    mask_threshold: float
    opening_kernel_px: int
    closing_kernel_px: int
    minimum_hole_area_px: float
    minimum_face_area_px: float
    maximum_hole_to_face_area_ratio: float
    face_margin_ratio: float
    maximum_center_distance_ratio: float
    minimum_depth_m: float
    maximum_depth_m: float
    minimum_valid_ratio: float
    depth_cluster_tolerance_m: float
    minimum_depth_cluster_ratio: float
    target_frame: str
    tf_lookup_timeout: float
    minimum_stable_samples: int
    stable_confirm_duration_s: float
    maximum_stable_gap_s: float
    maximum_position_range_m: float
    detections_2d_topic: str
    hole_center_topic: str
    hole_mask_topic: str
    face_mask_topic: str
    debug_image_topic: str
    markers_topic: str
    publish_debug_image: bool
    initially_enabled: bool
    enable_service: str
    reset_stability_service: str
    maximum_rate_hz: float
    inference_verbose: bool
    summary_interval: int
    hole_center_marker_diameter_m: float


def _normalize_model_names(names) -> dict[int, str]:
    """将 Ultralytics 类别名称转换为整数键字典."""
    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(name) for index, name in enumerate(names)}
    raise TypeError("模型类别名称必须是字典或序列")


class YoloTerminalSegmenter(Node):
    """编排左腕分割、端面深度拟合、TF、稳定门和可视化."""

    def __init__(self) -> None:
        """声明参数、加载分割模型并启动最新帧后台推理."""
        super().__init__("yolo_terminal_segmenter")
        self._declare_parameters()
        self._config = self._load_parameters()
        self._model = self._load_model()
        self._model_names = _normalize_model_names(self._model.names)
        self._class_ids = self._resolve_class_ids()
        self._initialize_worker_state()
        self._create_interfaces()
        self._worker.start()
        self.get_logger().info(
            "左相机铜端子分割节点已启动："
            f"model={self._config.model_path}, "
            f"classes={self._config.hole_class}/{self._config.face_class}, "
            f"enabled={self._enabled}, "
            f"target_frame={self._config.target_frame}"
        )

    @staticmethod
    def _descriptor(description: str) -> ParameterDescriptor:
        return ParameterDescriptor(description=description, read_only=True)

    @staticmethod
    def _float_descriptor(
        description: str,
        minimum: float,
        maximum: float,
    ) -> ParameterDescriptor:
        return ParameterDescriptor(
            description=description,
            read_only=True,
            floating_point_range=[
                FloatingPointRange(
                    from_value=minimum,
                    to_value=maximum,
                    step=0.0,
                )
            ],
        )

    @staticmethod
    def _integer_descriptor(
        description: str,
        minimum: int,
        maximum: int,
    ) -> ParameterDescriptor:
        return ParameterDescriptor(
            description=description,
            read_only=True,
            integer_range=[IntegerRange(from_value=minimum, to_value=maximum)],
        )

    def _declare_parameters(self) -> None:
        """声明模型、掩膜、深度、稳定门、接口和可视化参数."""
        declare = self.declare_parameter
        describe = self._descriptor
        float_range = self._float_descriptor
        integer_range = self._integer_descriptor

        declare("model.path", "", describe("YOLO segmentation 权重路径。"))
        declare("model.device", "", describe("Ultralytics 推理设备。"))
        declare("model.hole_class", "hole", describe("孔洞类别名称。"))
        declare(
            "model.face_class",
            "terminal_face",
            describe("铜端子端面类别名称。"),
        )
        declare(
            "model.confidence",
            0.30,
            float_range("YOLO 置信度阈值。", 0.0, 1.0),
        )
        declare("model.iou", 0.50, float_range("YOLO IoU 阈值。", 0.0, 1.0))
        declare(
            "model.image_size",
            640,
            integer_range("YOLO 输入尺寸。", 32, 4096),
        )
        declare(
            "model.max_detections",
            10,
            integer_range("单帧最大分割数量。", 1, 1000),
        )
        declare(
            "input.rgb_topic",
            "/cameras/left_camera/rgb/image_raw",
            describe("左相机彩色图话题。"),
        )
        declare(
            "input.depth_topic",
            "/cameras/left_camera/depth/image_raw",
            describe("对齐到左相机彩色图的深度话题。"),
        )
        declare(
            "input.camera_info_topic",
            "/cameras/left_camera/rgb/camera_info",
            describe("左相机彩色内参话题。"),
        )
        declare(
            "synchronization.queue_size",
            10,
            integer_range("RGB-D 同步队列长度。", 2, 100),
        )
        declare(
            "synchronization.slop",
            0.03,
            float_range("RGB-D 同步容差，单位秒。", 0.0, 1.0),
        )
        declare(
            "mask.threshold",
            0.50,
            float_range("分割掩膜二值阈值。", 0.0, 1.0),
        )
        declare(
            "mask.opening_kernel_px",
            3,
            integer_range("掩膜开运算核尺寸。", 0, 51),
        )
        declare(
            "mask.closing_kernel_px",
            5,
            integer_range("掩膜闭运算核尺寸。", 0, 51),
        )
        declare(
            "mask.minimum_hole_area_px",
            20.0,
            float_range("孔洞最小像素面积。", 1.0, 1.0e8),
        )
        declare(
            "mask.minimum_face_area_px",
            30.0,
            float_range("端面最小像素面积。", 1.0, 1.0e8),
        )
        declare(
            "pairing.maximum_hole_to_face_area_ratio",
            5.0,
            float_range("孔洞面积相对端面面积的最大比例。", 0.01, 100.0),
        )
        declare(
            "pairing.face_margin_ratio",
            0.25,
            float_range("端面凸包和检测框的匹配扩展比例。", 0.0, 2.0),
        )
        declare(
            "pairing.maximum_center_distance_ratio",
            1.25,
            float_range("孔洞到端面中心相对端面对角线的最大距离。", 0.01, 10.0),
        )
        declare(
            "depth.minimum_m",
            0.08,
            float_range("最小有效深度，单位米。", 0.001, 100.0),
        )
        declare(
            "depth.maximum_m",
            1.50,
            float_range("最大有效深度，单位米。", 0.01, 100.0),
        )
        declare(
            "depth.minimum_valid_ratio",
            0.20,
            float_range("端面掩膜最小深度有效比例。", 0.0, 1.0),
        )
        declare(
            "depth.cluster_tolerance_m",
            0.01,
            float_range("端面主深度簇半宽，单位米。", 1.0e-5, 1.0),
        )
        declare(
            "depth.minimum_cluster_ratio",
            0.35,
            float_range("端面主深度簇占有效深度的最小比例。", 0.0, 1.0),
        )
        declare("tf.target_frame", "world", describe("稳定结果目标坐标系。"))
        declare(
            "tf.lookup_timeout",
            0.10,
            float_range("图像时刻 TF 查询超时。", 0.0, 5.0),
        )
        declare(
            "stability.minimum_samples",
            5,
            integer_range("稳定确认最少帧数。", 1, 1000),
        )
        declare(
            "stability.confirm_duration",
            0.50,
            float_range("稳定确认最短持续时间。", 0.01, 60.0),
        )
        declare(
            "stability.maximum_gap",
            0.20,
            float_range("稳定帧最大间隔。", 0.01, 10.0),
        )
        declare(
            "stability.maximum_position_range_m",
            0.003,
            float_range("孔中心最大位置全范围。", 1.0e-5, 1.0),
        )
        defaults = {
            "output.detections_2d_topic": (
                "/s622_dual_perception/left_camera/features_2d"
            ),
            "output.hole_center_topic": (
                "/s622_dual_perception/left_camera/hole_center"
            ),
            "output.hole_mask_topic": (
                "/s622_dual_perception/left_camera/hole_mask"
            ),
            "output.face_mask_topic": (
                "/s622_dual_perception/left_camera/end_face_mask"
            ),
            "output.debug_image_topic": (
                "/s622_dual_perception/left_camera/debug_image"
            ),
            "output.markers_topic": (
                "/s622_dual_perception/left_camera/markers"
            ),
        }
        for name, default in defaults.items():
            declare(name, default, describe(f"分割输出 {name}。"))
        declare(
            "output.publish_debug_image",
            True,
            describe("是否发布分割调试图像。"),
        )
        declare(
            "visualization.hole_center_marker_diameter_m",
            0.006,
            float_range("RViz 孔中心蓝点直径，单位米。", 0.0001, 1.0),
        )
        declare(
            "inference.enabled",
            False,
            describe("节点启动后是否立即执行分割推理。"),
        )
        declare(
            "inference.max_rate_hz",
            10.0,
            float_range("最大推理频率。", 0.1, 120.0),
        )
        declare(
            "inference.verbose",
            False,
            describe("是否输出 Ultralytics 单帧日志。"),
        )
        declare(
            "control.enable_service",
            "/s622_dual_perception/left_camera/set_enabled",
            describe("分割推理启停服务。"),
        )
        declare(
            "control.reset_stability_service",
            "/s622_dual_perception/left_camera/reset_stability",
            describe("稳定窗口重置服务。"),
        )
        declare(
            "logging.summary_interval",
            30,
            integer_range("每多少帧打印一次摘要。", 1, 100000),
        )

    def _load_parameters(self) -> SegmenterConfig:
        """加载参数并转换为不可变配置."""
        def value(name: str):
            return self.get_parameter(name).value

        config = SegmenterConfig(
            model_path=Path(str(value("model.path"))).expanduser(),
            device=str(value("model.device")).strip(),
            hole_class=str(value("model.hole_class")).strip(),
            face_class=str(value("model.face_class")).strip(),
            confidence=float(value("model.confidence")),
            iou=float(value("model.iou")),
            image_size=int(value("model.image_size")),
            max_detections=int(value("model.max_detections")),
            rgb_topic=str(value("input.rgb_topic")),
            depth_topic=str(value("input.depth_topic")),
            camera_info_topic=str(value("input.camera_info_topic")),
            sync_queue_size=int(value("synchronization.queue_size")),
            sync_slop=float(value("synchronization.slop")),
            mask_threshold=float(value("mask.threshold")),
            opening_kernel_px=int(value("mask.opening_kernel_px")),
            closing_kernel_px=int(value("mask.closing_kernel_px")),
            minimum_hole_area_px=float(value("mask.minimum_hole_area_px")),
            minimum_face_area_px=float(value("mask.minimum_face_area_px")),
            maximum_hole_to_face_area_ratio=float(
                value("pairing.maximum_hole_to_face_area_ratio")
            ),
            face_margin_ratio=float(value("pairing.face_margin_ratio")),
            maximum_center_distance_ratio=float(
                value("pairing.maximum_center_distance_ratio")
            ),
            minimum_depth_m=float(value("depth.minimum_m")),
            maximum_depth_m=float(value("depth.maximum_m")),
            minimum_valid_ratio=float(value("depth.minimum_valid_ratio")),
            depth_cluster_tolerance_m=float(
                value("depth.cluster_tolerance_m")
            ),
            minimum_depth_cluster_ratio=float(
                value("depth.minimum_cluster_ratio")
            ),
            target_frame=str(value("tf.target_frame")).strip(),
            tf_lookup_timeout=float(value("tf.lookup_timeout")),
            minimum_stable_samples=int(value("stability.minimum_samples")),
            stable_confirm_duration_s=float(
                value("stability.confirm_duration")
            ),
            maximum_stable_gap_s=float(value("stability.maximum_gap")),
            maximum_position_range_m=float(
                value("stability.maximum_position_range_m")
            ),
            detections_2d_topic=str(value("output.detections_2d_topic")),
            hole_center_topic=str(value("output.hole_center_topic")),
            hole_mask_topic=str(value("output.hole_mask_topic")),
            face_mask_topic=str(value("output.face_mask_topic")),
            debug_image_topic=str(value("output.debug_image_topic")),
            markers_topic=str(value("output.markers_topic")),
            publish_debug_image=bool(value("output.publish_debug_image")),
            initially_enabled=bool(value("inference.enabled")),
            enable_service=str(value("control.enable_service")),
            reset_stability_service=str(
                value("control.reset_stability_service")
            ),
            maximum_rate_hz=float(value("inference.max_rate_hz")),
            inference_verbose=bool(value("inference.verbose")),
            summary_interval=int(value("logging.summary_interval")),
            hole_center_marker_diameter_m=float(
                value("visualization.hole_center_marker_diameter_m")
            ),
        )
        if config.maximum_depth_m <= config.minimum_depth_m:
            raise ValueError("depth.maximum_m 必须大于 depth.minimum_m")
        if not config.hole_class or not config.face_class:
            raise ValueError("孔洞和端面类别名称不能为空")
        if config.hole_class == config.face_class:
            raise ValueError("孔洞和端面类别名称不能相同")
        if not config.target_frame:
            raise ValueError("tf.target_frame 不能为空")
        return config

    def _load_model(self):
        """加载并验证 YOLO segmentation 模型."""
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics，无法启动分割节点") \
                from ULTRALYTICS_IMPORT_ERROR
        if not self._config.model_path.is_file():
            raise RuntimeError(
                f"YOLO segmentation 权重不存在: {self._config.model_path}"
            )
        try:
            model = YOLO(str(self._config.model_path))
        except Exception as error:
            raise RuntimeError("加载 YOLO segmentation 权重失败") from error
        if getattr(model, "task", None) != "segment":
            raise RuntimeError(
                "模型任务类型不是 segment: "
                f"task={getattr(model, 'task', None)}"
            )
        return model

    def _resolve_class_ids(self) -> list[int]:
        """验证孔洞和端面类别并返回模型类别编号."""
        name_to_id = {
            name: class_id for class_id, name in self._model_names.items()
        }
        requested = (self._config.hole_class, self._config.face_class)
        missing = [name for name in requested if name not in name_to_id]
        if missing:
            raise ValueError(
                "分割模型缺少类别 "
                + ", ".join(missing)
                + "；可用类别: "
                + ", ".join(self._model_names.values())
            )
        return [name_to_id[name] for name in requested]

    def _initialize_worker_state(self) -> None:
        """创建推理线程共享状态和世界系稳定门."""
        self._frame_condition = threading.Condition()
        self._stability_lock = threading.Lock()
        self._latest_rgbd = None
        self._stop_requested = False
        self._enabled = self._config.initially_enabled
        self._processed_frames = 0
        self._last_inference_time = 0.0
        self._last_error_text = ""
        self._last_error_time = 0.0
        self._warning_times = {}
        self._stability_minimum_stamp_s = -math.inf
        self._stability_gate = HoleCenterStabilityGate(
            minimum_samples=self._config.minimum_stable_samples,
            confirm_duration_s=self._config.stable_confirm_duration_s,
            maximum_gap_s=self._config.maximum_stable_gap_s,
            maximum_position_range_m=self._config.maximum_position_range_m,
        )
        self._worker = threading.Thread(
            target=self._inference_loop,
            name="yolo_left_camera_segmentation",
            daemon=True,
        )

    def _create_interfaces(self) -> None:
        """创建发布器、服务、同步订阅和 TF 接口."""
        self._bridge = CvBridge()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=False,
        )
        stable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        debug_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._detections_2d_publisher = self.create_publisher(
            Detection2DArray,
            self._config.detections_2d_topic,
            10,
        )
        self._hole_center_publisher = self.create_publisher(
            PointStamped,
            self._config.hole_center_topic,
            stable_qos,
        )
        self._hole_mask_publisher = self.create_publisher(
            Image,
            self._config.hole_mask_topic,
            qos_profile_sensor_data,
        )
        self._face_mask_publisher = self.create_publisher(
            Image,
            self._config.face_mask_topic,
            qos_profile_sensor_data,
        )
        self._debug_image_publisher = self.create_publisher(
            Image,
            self._config.debug_image_topic,
            debug_image_qos,
        )
        self._markers_publisher = self.create_publisher(
            MarkerArray,
            self._config.markers_topic,
            stable_qos,
        )
        self._enable_service = self.create_service(
            SetBool,
            self._config.enable_service,
            self._on_set_enabled,
        )
        self._reset_service = self.create_service(
            Trigger,
            self._config.reset_stability_service,
            self._on_reset_stability,
        )

        self._rgb_subscription = Subscriber(
            self,
            Image,
            self._config.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._depth_subscription = Subscriber(
            self,
            Image,
            self._config.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._camera_info_subscription = Subscriber(
            self,
            CameraInfo,
            self._config.camera_info_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._synchronizer = ApproximateTimeSynchronizer(
            [
                self._rgb_subscription,
                self._depth_subscription,
                self._camera_info_subscription,
            ],
            queue_size=self._config.sync_queue_size,
            slop=self._config.sync_slop,
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._rgbd_callback)

    def _on_set_enabled(self, request, response):
        """原子启停推理；禁用时清空旧帧和稳定结果."""
        with self._frame_condition:
            changed = self._enabled != bool(request.data)
            self._enabled = bool(request.data)
            self._latest_rgbd = None
            self._frame_condition.notify_all()
        if changed:
            self._reset_stability(self._clock_seconds())
            self._publish_marker_clear()
        response.success = True
        response.message = (
            "left camera segmentation enabled"
            if self._enabled
            else "left camera segmentation disabled"
        )
        self.get_logger().info(response.message)
        return response

    def _on_reset_stability(self, request, response):
        """清除机械臂运动期间的历史并重新开始稳定确认."""
        del request
        with self._frame_condition:
            self._latest_rgbd = None
        self._reset_stability(self._clock_seconds())
        self._publish_marker_clear()
        response.success = True
        response.message = "left camera hole stability window reset"
        self.get_logger().info(response.message)
        return response

    def _reset_stability(self, minimum_stamp_s: float) -> None:
        with self._stability_lock:
            self._stability_gate.reset()
            self._stability_minimum_stamp_s = minimum_stamp_s

    def _publish_marker_clear(self) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._config.target_frame
        self._markers_publisher.publish(clear_hole_marker_array(header))

    def _rgbd_callback(
        self,
        rgb_message: Image,
        depth_message: Image,
        camera_info: CameraInfo,
    ) -> None:
        """仅在启用时缓存最新同步帧，覆盖未处理的旧数据."""
        with self._frame_condition:
            if not self._enabled:
                return
            self._latest_rgbd = (rgb_message, depth_message, camera_info)
            self._frame_condition.notify()

    def _inference_loop(self) -> None:
        """限频处理最新 RGB-D 帧，并在禁用时休眠."""
        period = 1.0 / self._config.maximum_rate_hz
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: self._stop_requested
                    or (self._enabled and self._latest_rgbd is not None)
                )
                if self._stop_requested:
                    return
                wait_time = (
                    self._last_inference_time + period - time.monotonic()
                )
                if wait_time > 0.0:
                    self._frame_condition.wait(timeout=wait_time)
                    continue
                messages = self._latest_rgbd
                self._latest_rgbd = None
            try:
                self._process_rgbd(*messages)
            except Exception as error:  # noqa: B902
                self._log_processing_error(error)
            self._last_inference_time = time.monotonic()

    def _process_rgbd(
        self,
        rgb_message: Image,
        depth_message: Image,
        camera_info: CameraInfo,
    ) -> None:
        """完成一帧分割、孔中心反投影、TF、稳定门和结果发布."""
        frame, depth_m, intrinsics = self._convert_inputs(
            rgb_message,
            depth_message,
            camera_info,
        )
        result = self._predict(frame)
        detections = self._parse_detections(result, frame.shape[:2])
        if not self._is_enabled():
            return
        self._detections_2d_publisher.publish(
            self._build_detection_2d(rgb_message.header, detections)
        )
        try:
            pair = select_hole_face_pair(
                detections,
                hole_class=self._config.hole_class,
                face_class=self._config.face_class,
                minimum_hole_area_px=self._config.minimum_hole_area_px,
                minimum_face_area_px=self._config.minimum_face_area_px,
                maximum_hole_to_face_area_ratio=(
                    self._config.maximum_hole_to_face_area_ratio
                ),
                face_margin_ratio=self._config.face_margin_ratio,
                maximum_center_distance_ratio=(
                    self._config.maximum_center_distance_ratio
                ),
            )
        except ValueError as error:
            self._publish_status_debug(
                rgb_message.header,
                frame,
                str(error),
                detections,
            )
            self._log_warning("pair", error)
            return

        self._publish_masks(rgb_message.header, pair)
        try:
            depth = estimate_depth_from_mask(
                depth_m,
                pair.face.mask,
                minimum_depth_m=self._config.minimum_depth_m,
                maximum_depth_m=self._config.maximum_depth_m,
                minimum_valid_ratio=self._config.minimum_valid_ratio,
                depth_cluster_tolerance_m=(
                    self._config.depth_cluster_tolerance_m
                ),
                minimum_depth_cluster_ratio=(
                    self._config.minimum_depth_cluster_ratio
                ),
            )
            center_camera = estimate_hole_center(
                pair.ellipse[0],
                depth,
                intrinsics,
            )
            center_world, world_header = self._transform_center(
                rgb_message.header,
                center_camera,
            )
        except (ValueError, TransformException) as error:
            debug_reason = (
                "TF unavailable"
                if isinstance(error, TransformException)
                else "depth invalid"
            )
            # self._publish_pair_debug(
            #     rgb_message.header,
            #     frame,
            #     pair,
            #     status=f"INVALID: {debug_reason}; see node warning",
            # )
            self._log_warning("localization", error)
            return

        stamp_s = self._stamp_seconds(rgb_message.header)
        with self._stability_lock:
            if stamp_s <= self._stability_minimum_stamp_s:
                stability_status = "WAITING FOR FRESH FRAME"
            else:
                self._stability_gate.update(stamp_s, center_world)
                stability_status = self._stability_gate.status()
            stable = self._stability_gate.locked
        if stable is not None:
            self._publish_stable(world_header, stable)
        self._publish_pair_debug(
            rgb_message.header,
            frame,
            pair,
            center_camera=center_camera,
            center_world=center_world,
            status=stability_status,
        )
        self._processed_frames += 1
        if self._processed_frames % self._config.summary_interval == 0:
            self.get_logger().info(
                f"左相机分割已处理 {self._processed_frames} 帧，"
                f"stability={stability_status}"
            )

    def _is_enabled(self) -> bool:
        with self._frame_condition:
            return self._enabled

    def _convert_inputs(
        self,
        rgb_message: Image,
        depth_message: Image,
        camera_info: CameraInfo,
    ) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """转换并校验已对齐的 RGB、深度和相机内参."""
        try:
            frame = self._bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding="bgr8",
            )
            depth_raw = self._bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding="passthrough",
            )
        except CvBridgeError as error:
            raise RuntimeError("无法转换左相机同步 RGB-D 图像") from error
        depth_m = depth_image_to_meters(depth_raw, depth_message.encoding)
        if frame.shape[:2] != depth_m.shape:
            raise RuntimeError("左相机 RGB 与深度图尺寸不一致")
        if rgb_message.header.frame_id != depth_message.header.frame_id:
            raise RuntimeError("左相机深度图未对齐到彩色光学坐标系")
        intrinsics = CameraIntrinsics(
            fx=float(camera_info.k[0]),
            fy=float(camera_info.k[4]),
            cx=float(camera_info.k[2]),
            cy=float(camera_info.k[5]),
            width=int(camera_info.width),
            height=int(camera_info.height),
        )
        intrinsics.validate()
        if (intrinsics.height, intrinsics.width) != depth_m.shape:
            raise RuntimeError("左相机 CameraInfo 与图像尺寸不一致")
        return frame, depth_m, intrinsics

    def _predict(self, frame: np.ndarray):
        """执行单帧 YOLO segmentation 推理."""
        arguments = {
            "source": frame,
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "imgsz": self._config.image_size,
            "max_det": self._config.max_detections,
            "classes": self._class_ids,
            "verbose": self._config.inference_verbose,
        }
        if self._config.device:
            arguments["device"] = self._config.device
        results = self._model.predict(**arguments)
        if not results:
            raise RuntimeError("YOLO segmentation 未返回结果")
        return results[0]

    def _parse_detections(
        self,
        result,
        image_shape: Sequence[int],
    ) -> list[SegmentationDetection]:
        """解析模型框和概率掩膜并执行纯二维后处理."""
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or len(boxes) == 0:
            return []
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        probability_masks = masks.data.cpu().numpy()
        count = min(
            len(boxes_xyxy),
            len(class_ids),
            len(confidences),
            len(probability_masks),
        )
        detections = []
        for index in range(count):
            try:
                detection = make_detection(
                    index=index,
                    class_name=self._model_names.get(
                        int(class_ids[index]),
                        str(class_ids[index]),
                    ),
                    confidence=float(confidences[index]),
                    bbox_xyxy=boxes_xyxy[index],
                    probability_mask=probability_masks[index],
                    image_shape=image_shape,
                    threshold=self._config.mask_threshold,
                    opening_kernel_px=self._config.opening_kernel_px,
                    closing_kernel_px=self._config.closing_kernel_px,
                )
            except ValueError as error:
                self._log_warning("mask", error)
                continue
            detections.append(detection)
        return detections

    def _transform_center(self, header, center_camera):
        """使用图像时间戳把孔中心转换到目标坐标系."""
        if not header.frame_id:
            raise ValueError("左相机图像 frame_id 为空")
        transform = self._tf_buffer.lookup_transform(
            self._config.target_frame,
            header.frame_id,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=self._config.tf_lookup_timeout),
        )
        translation = np.array(
            (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            )
        )
        rotation = np.array(
            (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )
        )
        world_header = copy.deepcopy(header)
        world_header.frame_id = self._config.target_frame
        return (
            transform_hole_center(center_camera, translation, rotation),
            world_header,
        )

    @staticmethod
    def _stamp_seconds(header) -> float:
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1.0e-9

    def _clock_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    def _build_detection_2d(self, header, detections) -> Detection2DArray:
        """构造孔洞和端面二维检测数组."""
        result = Detection2DArray()
        result.header = header
        for item in detections:
            detection = Detection2D()
            detection.header = header
            x_min, y_min, x_max, y_max = item.bbox_xyxy
            detection.bbox.center.position.x = 0.5 * (x_min + x_max)
            detection.bbox.center.position.y = 0.5 * (y_min + y_max)
            detection.bbox.size_x = max(0.0, x_max - x_min)
            detection.bbox.size_y = max(0.0, y_max - y_min)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = item.class_name
            hypothesis.hypothesis.score = item.confidence
            detection.results.append(hypothesis)
            result.detections.append(detection)
        return result

    def _publish_masks(self, header, pair: HoleFacePair) -> None:
        for mask, publisher in (
            (pair.hole.mask, self._hole_mask_publisher),
            (pair.face.mask, self._face_mask_publisher),
        ):
            message = self._bridge.cv2_to_imgmsg(mask, encoding="mono8")
            message.header = header
            publisher.publish(message)

    def _publish_stable(
        self,
        header,
        stable: StableHoleCenter,
    ) -> None:
        center_message = PointStamped()
        center_message.header = header
        center_message.point.x = float(stable.center[0])
        center_message.point.y = float(stable.center[1])
        center_message.point.z = float(stable.center[2])
        self._hole_center_publisher.publish(center_message)
        self._markers_publisher.publish(
            build_hole_center_marker_array(
                header,
                stable.center,
                diameter_m=self._config.hole_center_marker_diameter_m,
            )
        )

    def _publish_pair_debug(
        self,
        header,
        frame,
        pair,
        *,
        center_camera=None,
        center_world=None,
        status: str,
    ) -> None:
        if not self._config.publish_debug_image:
            return
        annotated = render_segmentation_debug_image(
            frame,
            pair,
            center_camera=center_camera,
            center_world=center_world,
            stability_status=status,
        )
        message = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        message.header = header
        self._debug_image_publisher.publish(message)

    def _publish_status_debug(
        self,
        header,
        frame,
        status: str,
        detections=(),
    ) -> None:
        if not self._config.publish_debug_image:
            return
        annotated = render_segmentation_status_image(
            frame,
            status,
            detections,
            hole_class=self._config.hole_class,
            face_class=self._config.face_class,
            minimum_hole_area_px=self._config.minimum_hole_area_px,
            minimum_face_area_px=self._config.minimum_face_area_px,
        )
        message = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        message.header = header
        self._debug_image_publisher.publish(message)

    def _log_warning(self, key: str, error: Exception) -> None:
        now = time.monotonic()
        previous = self._warning_times.get(key, 0.0)
        if now - previous >= 5.0:
            self.get_logger().warning(f"左相机 {key} 暂不可用：{error}")
            self._warning_times[key] = now

    def _log_processing_error(self, error: Exception) -> None:
        text = f"{type(error).__name__}: {error}"
        now = time.monotonic()
        if text != self._last_error_text or now - self._last_error_time >= 5.0:
            self.get_logger().error(f"左相机分割处理失败：{text}")
            self._last_error_text = text
            self._last_error_time = now

    def destroy_node(self):
        """停止后台推理线程并销毁 ROS 节点."""
        with self._frame_condition:
            self._stop_requested = True
            self._frame_condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            self.get_logger().warning("左相机分割线程未在五秒内退出")
        return super().destroy_node()


def main(args=None) -> None:
    """启动左相机铜端子分割节点."""
    rclpy.init(args=args)
    node = None
    try:
        node = YoloTerminalSegmenter()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(f"左相机分割节点启动失败：{error}")
        else:
            print(f"[yolo_terminal_segmenter] 启动失败：{error}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
