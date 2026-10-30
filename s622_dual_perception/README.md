# s622_dual_perception

S622 双臂系统的环境感知入口，包含两条按任务阶段切换的 RGB-D 推理链：

- 全局相机 YOLO-OBB：识别 `wire` 和 `copper_terminal`，为双臂抓取提供三维位姿。
- 左臂相机 YOLO segmentation：识别 `hole` 和 `terminal_face`，计算铜端子孔中心三维坐标。

`target_acquisition` 节点在此包内筛选 `wire` 和 `copper_terminal`，按本轮请求
验证时间戳和连续可见时长，通过 `/target_acquisition/acquire_targets` Action
返回锁定快照。任务节点不再逐帧订阅原始 OBB 检测。

OBB 不再依赖共享工件尺寸配置。三维位置使用目标表面深度，不添加半径补偿；
三维框的长度和宽度由二维 OBB、深度和相机内参估算，厚度暂以可见宽度近似，
不代表已知的真实工件尺寸。Marker 长轴使用同一估算长度。
相机话题默认使用 `config/obb_detector.yaml` 中的配置，仍可通过
`rgb_topic`、`depth_topic` 和 `camera_info_topic` 启动参数临时覆盖。

左右六维力的重力补偿与滤波目前仍由 `ft_sensor_processing` 节点实现，
感知包通过 `dual_force.launch.py` 统一启动并提供处理后力数据。

节点只负责 ROS 2 数据流、模型推理和 TF；掩膜后处理、三维几何与可视化均位于
`visual_perception_utils` 中。分割结果的图像绘制与 Marker 构造统一放在
`visual_perception_utils/visualization.py`。

## 目录职责

```text
s622_dual_perception/
├── s622_dual_perception/
│   ├── nodes/
│   │   ├── target_acquisition.py
│   │   ├── yolo_detector_obb.py
│   │   └── yolo_terminal_segmenter.py
│   ├── target_confirmation.py
│   └── visual_perception_utils/
│       ├── depth_estimation.py
│       ├── obb_geometry.py
│       ├── segmentation_postprocess.py
│       ├── terminal_pose_estimation.py
│       └── visualization.py
├── config/
│   ├── obb_detector.yaml
│   └── terminal_segmenter.yaml
├── launch/
│   ├── yolo_obb_perception.launch.py
│   ├── yolo_terminal_segmentation.launch.py
│   ├── dual_force.launch.py
│   └── dual_perception.launch.py
└── models/
    ├── yolo_obb.pt
    └── yolo_seg.pt
```

## 计算方法

分割节点选择相互匹配的孔洞与端面实例，对孔洞轮廓拟合椭圆。它在完整
`terminal_face` 掩膜中提取有效深度，保留占比足够的主深度簇并取其中值，再用该
深度反投影 `hole` 椭圆中心。结果按图像时间戳转换到 `world`，只按位置范围、连续
帧数和持续时间进行稳定确认，不再计算端面平面或孔轴。

## 构建和启动

Ultralytics 需要安装在运行节点使用的 Python 环境中：

```bash
python3 -m pip install ultralytics
```

```bash
cd /home/dsl/dual_arm_mujoco_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to s622_dual_perception
source install/setup.bash
ros2 launch s622_dual_perception dual_perception.launch.py
```

组合 launch 默认只加载 OBB 和目标确认节点。当前抓取与无接触预对接任务
不使用分割；需要孔中心调试时传入 `start_segmentation:=true`，或者单独启动
下列分割 launch。两侧力处理默认不启动，可传
`start_force_processing:=true`，或单独启动 `dual_force.launch.py`。
`peg_in_hole.launch.py` 独立包含力处理 launch，保证视觉开关
不会意外关闭力数据源。

```bash
ros2 launch s622_dual_perception yolo_obb_perception.launch.py
ros2 launch s622_dual_perception yolo_terminal_segmentation.launch.py
ros2 launch s622_dual_perception dual_force.launch.py
```

深度图必须对齐到彩色相机。检测结果继承 RGB 时间戳，并用该时间戳查询 TF。

## 推理控制

```text
/s622_dual_perception/obb/set_enabled
  std_srvs/srv/SetBool

/s622_dual_perception/left_camera/set_enabled
  std_srvs/srv/SetBool

/s622_dual_perception/left_camera/reset_stability
  std_srvs/srv/Trigger
```

夹取并记录 `eef_T_object` 刚体关系后，当前任务状态机关闭 OBB。
左相机分割的启停与稳定窗口复位服务保留给后续插入阶段使用。

## 主要输出

OBB 输出维持原命名空间：

```text
/s622_dual_perception/detections_2d
/s622_dual_perception/detections_3d_camera
/s622_dual_perception/detections_3d
/s622_dual_perception/debug_image
/s622_dual_perception/markers
```

左相机分割统一使用 `left_camera` 命名空间：

```text
/s622_dual_perception/left_camera/features_2d
/s622_dual_perception/left_camera/hole_center
/s622_dual_perception/left_camera/hole_mask
/s622_dual_perception/left_camera/end_face_mask
/s622_dual_perception/left_camera/debug_image
/s622_dual_perception/left_camera/markers
```

`hole_center` 只在稳定确认后发布，类型为 `geometry_msgs/msg/PointStamped`。
同一坐标还会以直径 6 mm 的纯蓝色球点发布到 Marker 话题。该点仅供 RViz 目视检查，
不进入 MoveIt PlanningScene、没有碰撞属性，也不会改变机械臂规划结果。孔中心及 Marker
使用可靠、瞬态本地 QoS，便于稍后启动的检查工具读取最终结果。
两个 `debug_image` 使用 `RELIABLE + VOLATILE + KEEP_LAST(1)`，兼容 ROS 2
Humble 的 `rqt_image_view`，同时不会向晚加入的订阅者回放过期图像。

孔洞—端面配对参数位于 `config/terminal_segmenter.yaml` 的 `pairing` 分组。
配对依次尝试凸包含、凸包邻域、扩展端面框和中心距离回退；面积比及中心距离仍有
独立上限。debug 图中的 `pair` 行会显示实际采用的模式、距离比和面积比。

## 检查结果

```bash
ros2 topic echo /s622_dual_perception/left_camera/hole_center
rqt_image_view /s622_dual_perception/left_camera/debug_image
```

默认 MoveIt RViz 配置已启用 `Computed Hole Center` 显示；`Fixed Frame` 为 `world`。
如使用自定义 RViz 配置，请添加 `MarkerArray`，选择
`/s622_dual_perception/left_camera/markers`。蓝色球点表示稳定孔中心。
