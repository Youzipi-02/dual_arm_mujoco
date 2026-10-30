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

"""YOLO-OBB RGB-D 检测节点及其 ROS 2 数据流编排。."""

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time

from cv_bridge import CvBridge, CvBridgeError
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
from std_srvs.srv import SetBool
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection3D
from vision_msgs.msg import Detection3DArray
from vision_msgs.msg import ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray

from s622_dual_perception.visual_perception_utils.depth_estimation import (
    DepthEstimate,
    depth_image_to_meters,
    estimate_obb_depth,
)
from s622_dual_perception.visual_perception_utils.obb_geometry import (
    CameraIntrinsics,
    ObbPose3D,
    calculate_obb_pose_3d,
    normalize_vector,
    quaternion_from_x_axis,
    transform_obb_pose,
    obb_axis_endpoints,
)
from s622_dual_perception.visual_perception_utils.visualization import (
    build_marker_array,
    render_debug_image,
)
try:
    from ultralytics import YOLO
except ImportError as import_error:
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = import_error
else:
    ULTRALYTICS_IMPORT_ERROR = None


@dataclass(frozen=True)
class ObjectGeometry:
    """保存由二维 OBB 和深度估算的可见长度和宽度。."""

    length: float
    diameter: float


@dataclass(frozen=True)
class DetectorConfig:
    """保存节点启动后使用的不变配置。."""

    model_path: Path
    device: str
    target_classes: tuple[str, ...]
    confidence: float
    iou: float
    image_size: int
    max_detections: int
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    sync_queue_size: int
    sync_slop: float
    minimum_depth_m: float
    maximum_depth_m: float
    roi_scale: float
    minimum_valid_ratio: float
    surface_percentile: float
    surface_band_m: float
    endpoint_radius_px: int
    target_frame: str
    tf_lookup_timeout: float
    enforce_horizontal_axis: bool
    detections_2d_topic: str
    detections_3d_camera_topic: str
    detections_3d_world_topic: str
    debug_image_topic: str
    markers_topic: str
    publish_debug_image: bool
    initially_enabled: bool
    enable_service: str
    inference_verbose: bool
    summary_interval: int


@dataclass(frozen=True)
class DetectedObb:
    """保存一项经过 Ultralytics 解析的二维 OBB 检测。."""

    index: int
    class_name: str
    confidence: float
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float
    corners: np.ndarray


@dataclass(frozen=True)
class LocalizedTarget:
    """保存检测的深度、相机坐标和可选目标坐标结果。."""

    detection: DetectedObb
    geometry: ObjectGeometry
    endpoint_pixels: tuple[np.ndarray, np.ndarray]
    depth: DepthEstimate
    pose_camera: ObbPose3D
    pose_world: ObbPose3D | None = None


def _normalize_model_names(names) -> dict[int, str]:
    """将 Ultralytics 类别名称统一转换为整数键字典。."""
    if isinstance(names, Mapping):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {class_id: str(name) for class_id, name in enumerate(names)}
    raise TypeError("模型类别名称必须是字典或序列")


def _resolve_class_ids(
    names,
    requested_names: Sequence[str],
) -> list[int] | None:
    """将目标类别名称转换为模型编号；空列表表示检测全部类别。."""
    model_names = _normalize_model_names(names)
    requested = [
        str(name).strip()
        for name in requested_names
        if str(name).strip()
    ]
    if not requested:
        return None

    name_to_id = {name: class_id for class_id, name in model_names.items()}
    unknown = sorted(set(requested) - set(name_to_id))
    if unknown:
        available = ", ".join(model_names.values())
        missing = ", ".join(unknown)
        raise ValueError(
            f"模型中不存在目标类别: {missing}；可用类别: {available}"
        )
    return [name_to_id[name] for name in dict.fromkeys(requested)]


def _estimate_object_geometry(
    detection: DetectedObb,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> ObjectGeometry:
    """在目标表面深度处反投影 OBB 两条边，估算可见尺寸。."""
    cosine = math.cos(detection.angle)
    sine = math.sin(detection.angle)
    width = detection.size[0] * depth_m * math.hypot(
        cosine / intrinsics.fx, sine / intrinsics.fy
    )
    height = detection.size[1] * depth_m * math.hypot(
        sine / intrinsics.fx, cosine / intrinsics.fy
    )
    return ObjectGeometry(length=max(width, height), diameter=min(width, height))


class YoloDetectorObb(Node):
    """编排 YOLO-OBB、RGB-D 深度定位、TF 和结果发布。."""

    def __init__(self) -> None:
        """声明参数、加载模型、建立接口并启动后台推理线程。."""
        super().__init__("yolo_obb_detector")
        self._declare_parameters()
        self._config = self._load_parameters()
        self._model = self._load_model()
        self._model_names = _normalize_model_names(self._model.names)
        self._class_ids = _resolve_class_ids(
            self._model_names,
            self._config.target_classes,
        )
        self._enabled = self._config.initially_enabled
        self._create_interfaces()
        self._start_worker()

        selected_classes = (
            "全部"
            if self._class_ids is None
            else ", ".join(
                self._model_names[class_id] for class_id in self._class_ids
            )
        )
        self.get_logger().info(
            "YOLO-OBB RGB-D 检测节点已启动："
            f"model={self._config.model_path}, "
            f"rgb={self._config.rgb_topic}, "
            f"depth={self._config.depth_topic}, "
            f"classes={selected_classes}, "
            f"enabled={self._enabled}, "
            f"target_frame={self._config.target_frame}"
        )

    @staticmethod
    def _descriptor(description: str) -> ParameterDescriptor:
        """创建带说明的只读 ROS 参数描述。."""
        return ParameterDescriptor(description=description, read_only=True)

    @staticmethod
    def _float_descriptor(
        description: str,
        minimum: float,
        maximum: float,
    ) -> ParameterDescriptor:
        """创建带闭区间约束的只读浮点参数描述。."""
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
        """创建带闭区间约束的只读整数参数描述。."""
        return ParameterDescriptor(
            description=description,
            read_only=True,
            integer_range=[IntegerRange(from_value=minimum, to_value=maximum)]
        )

    def _declare_parameters(self) -> None:
        """声明模型、RGB-D、深度、TF 和输出参数。."""
        declare = self.declare_parameter
        describe = self._descriptor
        float_range = self._float_descriptor
        integer_range = self._integer_descriptor

        declare("model.path", "", describe("YOLO-OBB 权重文件路径。"))
        declare(
            "model.device",
            "",
            describe("Ultralytics 推理设备；空字符串表示自动选择。"),
        )
        declare(
            "model.target_classes",
            ["copper_terminal", "wire"],
            describe("需要发布的模型类别名称；空列表表示全部类别。"),
        )
        declare(
            "model.confidence",
            0.60,
            float_range("YOLO 置信度阈值。", 0.0, 1.0),
        )
        declare(
            "model.iou",
            0.50,
            float_range("非极大值抑制的 IoU 阈值。", 0.0, 1.0),
        )
        declare(
            "model.image_size",
            640,
            integer_range("YOLO 推理输入尺寸。", 32, 4096),
        )
        declare(
            "model.max_detections",
            20,
            integer_range("单帧最多保留的检测框数量。", 1, 1000),
        )

        declare(
            "input.rgb_topic",
            "/global_camera/rgb/image_raw",
            describe("全局相机彩色图像话题。"),
        )
        declare(
            "input.depth_topic",
            "/global_camera/depth/image_raw",
            describe("与彩色图像对齐的深度图话题。"),
        )
        declare(
            "input.camera_info_topic",
            "/global_camera/camera_info",
            describe("彩色相机内参话题。"),
        )
        declare(
            "synchronization.queue_size",
            10,
            integer_range("RGB-D 近似同步队列长度。", 2, 100),
        )
        declare(
            "synchronization.slop",
            0.05,
            float_range("RGB-D 时间戳允许偏差，单位秒。", 0.0, 1.0),
        )

        declare(
            "depth.minimum_m",
            0.15,
            float_range("允许使用的最小深度，单位米。", 0.001, 100.0),
        )
        declare(
            "depth.maximum_m",
            4.0,
            float_range("允许使用的最大深度，单位米。", 0.01, 1000.0),
        )
        declare(
            "depth.roi_scale",
            0.72,
            float_range("OBB 深度掩膜的向内缩放比例。", 0.05, 1.0),
        )
        declare(
            "depth.minimum_valid_ratio",
            0.20,
            float_range("OBB 内最小有效深度比例。", 0.0, 1.0),
        )
        declare(
            "depth.surface_percentile",
            35.0,
            float_range("选择可见表面深度使用的分位数。", 0.0, 100.0),
        )
        declare(
            "depth.surface_band_m",
            0.015,
            float_range("表面深度离群过滤带宽，单位米。", 0.0001, 1.0),
        )
        declare(
            "depth.endpoint_radius_px",
            3,
            integer_range("长轴端点深度采样半径，单位像素。", 0, 50),
        )

        declare("tf.target_frame", "world", describe("三维检测目标坐标系。"))
        declare(
            "tf.lookup_timeout",
            0.10,
            float_range("等待图像时刻 TF 的最长时间，单位秒。", 0.0, 5.0),
        )
        declare(
            "localization.enforce_horizontal_axis",
            True,
            describe("是否把目标坐标下的工件轴投影到水平面。"),
        )

        declare(
            "output.detections_2d_topic",
            "/s622_dual_perception/detections_2d",
            describe("二维 OBB 检测结果话题。"),
        )
        declare(
            "output.detections_3d_camera_topic",
            "/s622_dual_perception/detections_3d_camera",
            describe("相机光学坐标系三维检测结果话题。"),
        )
        declare(
            "output.detections_3d_world_topic",
            "/s622_dual_perception/detections_3d",
            describe("目标坐标系三维检测结果话题。"),
        )
        declare(
            "output.debug_image_topic",
            "/s622_dual_perception/debug_image",
            describe("带 OBB 和三维坐标标注的调试图像话题。"),
        )
        declare(
            "output.markers_topic",
            "/s622_dual_perception/markers",
            describe("RViz 三维目标 Marker 话题。"),
        )
        declare(
            "output.publish_debug_image",
            True,
            describe("是否发布带检测标注的调试图像。"),
        )
        declare(
            "inference.verbose",
            False,
            describe("是否启用 Ultralytics 单帧详细日志。"),
        )
        declare(
            "inference.enabled",
            True,
            describe("节点启动后是否立即执行 OBB 推理。"),
        )
        declare(
            "control.enable_service",
            "/s622_dual_perception/obb/set_enabled",
            describe("OBB 推理启停服务。"),
        )
        declare(
            "logging.summary_interval",
            30,
            integer_range("每处理多少帧打印一次摘要。", 1, 100000),
        )

    def _load_parameters(self) -> DetectorConfig:
        """把 ROS 参数复制并校验为不变的内部配置。."""
        def value(name: str):
            return self.get_parameter(name).value

        target_classes = tuple(
            str(name).strip()
            for name in value("model.target_classes")
            if str(name).strip()
        )
        config = DetectorConfig(
            model_path=Path(str(value("model.path"))).expanduser(),
            device=str(value("model.device")).strip(),
            target_classes=target_classes,
            confidence=float(value("model.confidence")),
            iou=float(value("model.iou")),
            image_size=int(value("model.image_size")),
            max_detections=int(value("model.max_detections")),
            rgb_topic=str(value("input.rgb_topic")),
            depth_topic=str(value("input.depth_topic")),
            camera_info_topic=str(value("input.camera_info_topic")),
            sync_queue_size=int(value("synchronization.queue_size")),
            sync_slop=float(value("synchronization.slop")),
            minimum_depth_m=float(value("depth.minimum_m")),
            maximum_depth_m=float(value("depth.maximum_m")),
            roi_scale=float(value("depth.roi_scale")),
            minimum_valid_ratio=float(value("depth.minimum_valid_ratio")),
            surface_percentile=float(value("depth.surface_percentile")),
            surface_band_m=float(value("depth.surface_band_m")),
            endpoint_radius_px=int(value("depth.endpoint_radius_px")),
            target_frame=str(value("tf.target_frame")),
            tf_lookup_timeout=float(value("tf.lookup_timeout")),
            enforce_horizontal_axis=bool(
                value("localization.enforce_horizontal_axis")
            ),
            detections_2d_topic=str(value("output.detections_2d_topic")),
            detections_3d_camera_topic=str(
                value("output.detections_3d_camera_topic")
            ),
            detections_3d_world_topic=str(
                value("output.detections_3d_world_topic")
            ),
            debug_image_topic=str(value("output.debug_image_topic")),
            markers_topic=str(value("output.markers_topic")),
            publish_debug_image=bool(value("output.publish_debug_image")),
            initially_enabled=bool(value("inference.enabled")),
            enable_service=str(value("control.enable_service")),
            inference_verbose=bool(value("inference.verbose")),
            summary_interval=int(value("logging.summary_interval")),
        )
        if config.maximum_depth_m <= config.minimum_depth_m:
            raise ValueError("depth.maximum_m 必须大于 depth.minimum_m")
        if not config.target_frame:
            raise ValueError("tf.target_frame 不能为空")
        return config

    def _load_model(self):
        """检查环境并加载 YOLO-OBB 权重，同时验证任务类型。."""
        if YOLO is None:
            raise RuntimeError(
                "未安装 ultralytics，无法启动检测节点。请在当前 Python "
                "环境安装 ultralytics。"
            ) from ULTRALYTICS_IMPORT_ERROR
        if not self._config.model_path.is_file():
            raise RuntimeError(
                f"YOLO-OBB 权重文件不存在: {self._config.model_path}"
            )
        try:
            model = YOLO(str(self._config.model_path))
        except Exception as error:
            raise RuntimeError(
                f"加载 YOLO-OBB 权重失败: {self._config.model_path}"
            ) from error
        if getattr(model, "task", None) != "obb":
            raise RuntimeError(
                f"模型任务类型不是 OBB: task={getattr(model, 'task', None)}"
            )
        return model

    def _create_interfaces(self) -> None:
        """创建 ROS 发布器、同步订阅器和 TF 查询接口。."""
        self._bridge = CvBridge()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(
            self._tf_buffer,
            self,
            spin_thread=False,
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
        self._detections_camera_publisher = self.create_publisher(
            Detection3DArray,
            self._config.detections_3d_camera_topic,
            10,
        )
        self._detections_world_publisher = self.create_publisher(
            Detection3DArray,
            self._config.detections_3d_world_topic,
            10,
        )
        self._debug_image_publisher = self.create_publisher(
            Image,
            self._config.debug_image_topic,
            debug_image_qos,
        )
        self._markers_publisher = self.create_publisher(
            MarkerArray,
            self._config.markers_topic,
            10,
        )
        self._enable_service = self.create_service(
            SetBool,
            self._config.enable_service,
            self._on_set_enabled,
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

    def _start_worker(self) -> None:
        """启动单独的推理线程，避免阻塞 ROS 回调。."""
        self._frame_condition = threading.Condition()
        self._latest_rgbd = None
        self._stop_requested = False
        self._processed_frames = 0
        self._last_error_text = ""
        self._last_error_time = 0.0
        self._localization_warning_times = {}
        self._worker = threading.Thread(
            target=self._inference_loop,
            name="yolo_obb_rgbd_inference",
            daemon=True,
        )
        self._worker.start()

    def _rgbd_callback(
        self,
        rgb_message: Image,
        depth_message: Image,
        camera_info: CameraInfo,
    ) -> None:
        """缓存最新同步 RGB-D 数据，覆盖尚未处理的旧数据。."""
        with self._frame_condition:
            if not self._enabled:
                return
            self._latest_rgbd = (rgb_message, depth_message, camera_info)
            self._frame_condition.notify()

    def _on_set_enabled(self, request, response):
        """启停 OBB 推理，禁用时丢弃尚未处理的旧帧。."""
        with self._frame_condition:
            self._enabled = bool(request.data)
            self._latest_rgbd = None
            self._frame_condition.notify_all()
        response.success = True
        response.message = (
            "OBB inference enabled"
            if self._enabled
            else "OBB inference disabled"
        )
        self.get_logger().info(response.message)
        return response

    def _inference_loop(self) -> None:
        """在后台线程中处理最新同步数据，避免队列积压。."""
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: (
                        self._stop_requested
                        or (self._enabled and self._latest_rgbd is not None)
                    )
                )
                if self._stop_requested:
                    return
                rgbd_messages = self._latest_rgbd
                self._latest_rgbd = None
            try:
                self._process_rgbd(*rgbd_messages)
            except Exception as error:  # noqa: B902
                self._log_inference_error(error)

    def _process_rgbd(
        self,
        rgb_message: Image,
        depth_message: Image,
        camera_info: CameraInfo,
    ) -> None:
        """线性完成消息转换、检测、定位、TF 和结果发布。."""
        frame, depth_m, intrinsics = self._convert_inputs(
            rgb_message,
            depth_message,
            camera_info,
        )
        result = self._predict(frame)
        with self._frame_condition:
            if not self._enabled:
                return
        detections = self._parse_detections(result)

        camera_targets = []
        for detection in detections:
            endpoints = obb_axis_endpoints(
                detection.center,
                detection.size,
                detection.angle,
            )
            try:
                depth = estimate_obb_depth(
                    depth_m,
                    detection.corners,
                    endpoints,
                    roi_scale=self._config.roi_scale,
                    minimum_depth_m=self._config.minimum_depth_m,
                    maximum_depth_m=self._config.maximum_depth_m,
                    minimum_valid_ratio=self._config.minimum_valid_ratio,
                    surface_percentile=self._config.surface_percentile,
                    surface_band_m=self._config.surface_band_m,
                    endpoint_radius_px=self._config.endpoint_radius_px,
                    center_depth_offset_m=0.0,
                )
                pose_camera = calculate_obb_pose_3d(
                    detection.center,
                    endpoints,
                    depth,
                    intrinsics,
                )
            except ValueError as error:
                self._log_localization_warning(detection.class_name, error)
                continue
            camera_targets.append(
                LocalizedTarget(
                    detection=detection,
                    geometry=_estimate_object_geometry(
                        detection, depth.center_depth_m, intrinsics
                    ),
                    endpoint_pixels=endpoints,
                    depth=depth,
                    pose_camera=pose_camera,
                )
            )

        world_targets = self._transform_targets(
            rgb_message.header,
            camera_targets,
        )
        self._publish_results(
            rgb_message.header,
            frame,
            detections,
            camera_targets,
            world_targets,
        )

        self._processed_frames += 1
        # if self._processed_frames % self._config.summary_interval == 0:
        #     self.get_logger().info(
        #         f"已处理 {self._processed_frames} 帧，"
        #         f"二维目标 {len(detections)} 个，"
        #         f"有效三维目标 {len(camera_targets)} 个，"
        #         f"world 目标 {len(world_targets)} 个"
        #     )

    def _publish_results(
        self,
        header,
        frame: np.ndarray,
        detections: Sequence[DetectedObb],
        camera_targets: Sequence[LocalizedTarget],
        world_targets: Sequence[LocalizedTarget],
    ) -> None:
        """构造并发布二维、三维、Marker 和调试图像结果。."""
        self._detections_2d_publisher.publish(
            self._build_detection_2d(header, detections)
        )

        camera_detections = Detection3DArray()
        camera_detections.header = header
        for target in camera_targets:
            camera_detections.detections.append(
                self._build_detection_3d(
                    header,
                    target,
                    target.pose_camera,
                )
            )
        self._detections_camera_publisher.publish(camera_detections)

        world_header = copy.deepcopy(header)
        world_header.frame_id = self._config.target_frame
        world_detections = Detection3DArray()
        world_detections.header = world_header
        for target in world_targets:
            if target.pose_world is not None:
                world_detections.detections.append(
                    self._build_detection_3d(
                        world_header,
                        target,
                        target.pose_world,
                    )
                )
        self._detections_world_publisher.publish(world_detections)

        self._markers_publisher.publish(
            build_marker_array(world_header, world_targets)
        )

        visual_targets = self._merge_world_targets(
            camera_targets,
            world_targets,
        )
        if self._config.publish_debug_image:
            annotated = render_debug_image(frame, detections, visual_targets)
            debug_message = self._bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8",
            )
            debug_message.header = header
            self._debug_image_publisher.publish(debug_message)

    def _convert_inputs(
        self,
        rgb_message: Image,
        depth_message: Image,
        camera_info: CameraInfo,
    ) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """转换 RGB-D 和 CameraInfo，并验证三者尺寸及坐标系一致。."""
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
            raise RuntimeError("无法转换同步 RGB-D 图像") from error

        depth_m = depth_image_to_meters(depth_raw, depth_message.encoding)
        if frame.shape[:2] != depth_m.shape:
            raise RuntimeError(
                f"RGB 与深度图尺寸不一致: {frame.shape[:2]} != {depth_m.shape}"
            )
        if rgb_message.header.frame_id != depth_message.header.frame_id:
            raise RuntimeError(
                "RGB 与深度图坐标系不一致；当前节点要求深度已对齐到彩色相机"
            )
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
            raise RuntimeError("CameraInfo 尺寸与同步深度图不一致")
        return frame, depth_m, intrinsics

    def _predict(self, frame: np.ndarray):
        """对一帧 BGR 彩色图像执行 YOLO-OBB 推理。."""
        arguments = {
            "source": frame,
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "imgsz": self._config.image_size,
            "max_det": self._config.max_detections,
            "verbose": self._config.inference_verbose,
        }
        if self._config.device:
            arguments["device"] = self._config.device
        if self._class_ids is not None:
            arguments["classes"] = self._class_ids
        results = self._model.predict(**arguments)
        if not results:
            raise RuntimeError("YOLO 推理未返回结果对象")
        return results[0]

    def _parse_detections(self, result) -> list[DetectedObb]:
        """解析 Ultralytics OBB 输出为与模型库解耦的检测数据类。."""
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return []
        boxes = obb.xywhr.cpu().numpy()
        polygons = obb.xyxyxyxy.cpu().numpy()
        class_ids = obb.cls.cpu().numpy().astype(int)
        confidences = obb.conf.cpu().numpy()
        detections = []
        for index, (box, corners, class_id, confidence) in enumerate(
            zip(boxes, polygons, class_ids, confidences)
        ):
            center_x, center_y, width, height, angle = box
            detections.append(
                DetectedObb(
                    index=index,
                    class_name=self._model_names.get(
                        int(class_id),
                        str(class_id),
                    ),
                    confidence=float(confidence),
                    center=(float(center_x), float(center_y)),
                    size=(float(width), float(height)),
                    angle=float(angle),
                    corners=np.asarray(corners, dtype=np.float64),
                )
            )
        return detections

    def _build_detection_2d(
        self,
        header,
        detections: Sequence[DetectedObb],
    ) -> Detection2DArray:
        """构造带旋转中心和尺寸的 vision_msgs 二维检测数组。."""
        detection_array = Detection2DArray()
        detection_array.header = header
        for detected in detections:
            detection = Detection2D()
            detection.header = header
            detection.bbox.center.position.x = detected.center[0]
            detection.bbox.center.position.y = detected.center[1]
            detection.bbox.center.theta = detected.angle
            detection.bbox.size_x = detected.size[0]
            detection.bbox.size_y = detected.size[1]
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detected.class_name
            hypothesis.hypothesis.score = detected.confidence
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)
        return detection_array

    def _transform_targets(
        self,
        header,
        targets: Sequence[LocalizedTarget],
    ) -> list[LocalizedTarget]:
        """按图像时间戳查询 TF 并生成目标坐标系下的定位结果。."""
        if not targets:
            return []
        source_frame = header.frame_id
        if not source_frame:
            self._log_localization_warning("tf", ValueError("图像 frame_id 为空"))
            return []
        try:
            transform = self._tf_buffer.lookup_transform(
                self._config.target_frame,
                source_frame,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=self._config.tf_lookup_timeout),
            )
        except TransformException as error:
            self._log_localization_warning("tf", error)
            return []

        translation = np.array(
            (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ),
            dtype=np.float64,
        )
        rotation = np.array(
            (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ),
            dtype=np.float64,
        )
        transformed_targets = []
        for target in targets:
            pose_world = transform_obb_pose(
                target.pose_camera,
                translation,
                rotation,
            )
            if self._config.enforce_horizontal_axis:
                horizontal = np.array(
                    (pose_world.axis[0], pose_world.axis[1], 0.0),
                    dtype=np.float64,
                )
                if np.linalg.norm(horizontal) > 1.0e-9:
                    pose_world = ObbPose3D(
                        center=pose_world.center,
                        axis=normalize_vector(horizontal),
                    )
            transformed_targets.append(replace(target, pose_world=pose_world))
        return transformed_targets

    @staticmethod
    def _merge_world_targets(
        camera_targets: Sequence[LocalizedTarget],
        world_targets: Sequence[LocalizedTarget],
    ) -> list[LocalizedTarget]:
        """把成功的世界坐标结果合并回全部相机定位结果供可视化使用。."""
        world_by_index = {
            target.detection.index: target.pose_world
            for target in world_targets
        }
        return [
            replace(
                target,
                pose_world=world_by_index.get(target.detection.index),
            )
            for target in camera_targets
        ]

    def _build_detection_3d(
        self,
        header,
        target: LocalizedTarget,
        pose: ObbPose3D,
    ) -> Detection3D:
        """构造表面中心、长轴姿态和可见尺寸估计的三维检测消息。."""
        quaternion = quaternion_from_x_axis(pose.axis)
        detection = Detection3D()
        detection.header = header
        detection.id = (
            f"{target.detection.class_name}_{target.detection.index}"
        )
        detection.bbox.center.position.x = float(pose.center[0])
        detection.bbox.center.position.y = float(pose.center[1])
        detection.bbox.center.position.z = float(pose.center[2])
        detection.bbox.center.orientation.x = float(quaternion[0])
        detection.bbox.center.orientation.y = float(quaternion[1])
        detection.bbox.center.orientation.z = float(quaternion[2])
        detection.bbox.center.orientation.w = float(quaternion[3])
        detection.bbox.size.x = target.geometry.length
        detection.bbox.size.y = target.geometry.diameter
        detection.bbox.size.z = target.geometry.diameter

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = target.detection.class_name
        hypothesis.hypothesis.score = target.detection.confidence
        hypothesis.pose.pose = detection.bbox.center
        position_sigma = max(target.depth.mad_m, 0.0025)
        axis_sigma = math.radians(5.0)
        for index in (0, 7, 14):
            hypothesis.pose.covariance[index] = position_sigma**2
        for index in (21, 28, 35):
            hypothesis.pose.covariance[index] = axis_sigma**2
        detection.results.append(hypothesis)
        return detection

    def _log_localization_warning(self, key: str, error: Exception) -> None:
        """对重复的深度或 TF 警告限频。."""
        now = time.monotonic()
        previous = self._localization_warning_times.get(key, 0.0)
        if now - previous >= 5.0:
            self.get_logger().warning(f"{key} 三维定位暂不可用：{error}")
            self._localization_warning_times[key] = now

    def _log_inference_error(self, error: Exception) -> None:
        """对重复帧处理错误限频，避免日志淹没终端。."""
        error_text = f"{type(error).__name__}: {error}"
        now = time.monotonic()
        if (
            error_text != self._last_error_text
            or now - self._last_error_time >= 5.0
        ):
            self.get_logger().error(f"YOLO-OBB RGB-D 处理失败：{error_text}")
            self._last_error_text = error_text
            self._last_error_time = now

    def destroy_node(self):
        """请求推理线程退出后再销毁 ROS 节点。."""
        with self._frame_condition:
            self._stop_requested = True
            self._frame_condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            self.get_logger().warning("推理线程未在五秒内退出")
        return super().destroy_node()


def main(args=None) -> None:
    """启动 YOLO-OBB RGB-D 检测节点并进入 ROS 事件循环。."""
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorObb()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(f"检测节点启动失败：{error}")
        else:
            print(f"[yolo_obb_detector] 检测节点启动失败：{error}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
