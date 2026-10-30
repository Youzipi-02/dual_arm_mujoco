# 模型文件说明

功能包安装并默认使用两个 Ultralytics 模型：

```text
yolo_obb.pt  YOLO-OBB，类别：copper_terminal、wire
yolo_seg.pt  YOLO segmentation，类别：hole、terminal_face
```

安装后由 ROS 包索引解析为：

```text
<s622_dual_perception_share>/models/yolo_obb.pt
<s622_dual_perception_share>/models/yolo_seg.pt
```

可分别通过对应 launch 的 `model_path` 参数覆盖：

```bash
ros2 launch s622_dual_perception yolo_obb_perception.launch.py \
  model_path:=/absolute/path/to/obb.pt

ros2 launch s622_dual_perception yolo_terminal_segmentation.launch.py \
  model_path:=/absolute/path/to/seg.pt
```

类别名称必须与 `config/obb_detector.yaml` 和 `config/terminal_segmenter.yaml`
中的配置一致；节点启动时会验证任务类型与所需类别。
