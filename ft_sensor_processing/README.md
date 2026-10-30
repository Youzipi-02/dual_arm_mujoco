# ft_sensor_processing

ROS 2 Humble 六维力传感器数据处理功能包。节点订阅原始
`geometry_msgs/msg/WrenchStamped`，根据传感器实时姿态补偿下游结构重力和
标定零偏，再对六个通道执行因果 Butterworth 低通滤波。

## 数据流

```text
/left_ft_sensor_broadcaster/wrench
  -> /left_ft_sensor_processing/wrench_compensated
  -> /left_ft_sensor_processing/wrench_filtered

/right_ft_sensor_broadcaster/wrench
  -> /right_ft_sensor_processing/wrench_compensated
  -> /right_ft_sensor_processing/wrench_filtered
```

左右节点分别执行重力与标定零偏补偿、可选运行时残余清零以及二阶
Butterworth 低通滤波。输出保留原始消息的时间戳和对应传感器坐标系。

## 构建与启动

```bash
cd /home/dsl/dual_arm_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ft_sensor_processing
source install/setup.bash

ros2 launch ft_sensor_processing ft_sensor_processing.launch.py \
  use_sim_time:=true
```

`s622_dual_gazebo` 已集成本启动文件，正常启动仿真时会在六维力 broadcaster
成功加载后自动启动处理节点。可通过
`start_ft_sensor_processing:=false` 暂时关闭。

## 标定参数

左右参数分别位于 `config/left_ft_sensor.yaml` 和
`config/right_ft_sensor.yaml`。当前 Gazebo 左右末端结构相同，因此右侧先使用
左侧辨识结果作为初值；更换右侧夹具或工件后，应单独标定右侧参数。

左侧默认参数为：

- 下游结构重量：`5.240600 N`
- 传感器坐标系中的重心：
  `[-0.0023576, -0.0106920, 0.0074035] m`
- 力和力矩固定零偏：当前 Gazebo 标定均为零

设 `R_SW` 为把世界坐标系向量旋转到传感器坐标系的矩阵，处理公式为：

```text
f_g = R_SW [0, 0, -G]^T
tau_g = r x f_g
f_out = f_raw - b_f - f_g - b_runtime_force
tau_out = tau_raw - b_tau - tau_g - b_runtime_torque
```

其中 `G` 是已辨识重量，`r` 是传感器原点指向重心的向量，`b_f` 和
`b_tau` 是标定零偏。该模型只补偿静态重力，不补偿快速运动产生的惯性力。

## 服务

```bash
# 机械臂静止且末端无接触时，对最近 0.5 s 补偿残差清零
ros2 service call /left_ft_sensor_processing/tare std_srvs/srv/Trigger {}

# 清除运行时清零结果，恢复纯标定补偿
ros2 service call /left_ft_sensor_processing/clear_tare std_srvs/srv/Trigger {}

# 只清除滤波历史
ros2 service call /left_ft_sensor_processing/reset_filter std_srvs/srv/Trigger {}
```

右侧服务名称只需把上述路径中的 `left` 替换为 `right`。

`tare` 会检查样本数量、信号标准差、关节状态新鲜度和最大关节速度。稳定的
外部接触仍可能被当成零偏，因此调用前必须确认末端没有接触环境。

## 滤波与 TF

默认滤波器为 500 Hz 采样率、20 Hz 截止频率的二阶 Butterworth。Gazebo
无噪声调试时可将 `filter.enabled` 设为 `false`。节点以 250 Hz 缓存最新
TF，使 500 Hz 的六维力回调中不执行 TF 查询。

旧标定 rosbag 中 `/tf` 约为 20 Hz。回放该 bag 时，应临时把参数
`tf.max_age_sec` 调大到 `0.06`；实时仿真已把 `robot_state_publisher` 提高到
250 Hz，应保持默认的 `0.02 s`。

## 验证

```bash
colcon test --packages-select ft_sensor_processing \
  --event-handlers console_cohesion+
colcon test-result --verbose
```

还应使用未参与参数辨识的新静态姿态进行独立验证。原始标定集上的零残差
不能替代独立姿态验证。

## 预设姿态运动测试

`ft_sensor_processing/ft_sensor_processing_test.py` 会依次向左臂轨迹控制器
发布文件顶部 `PRESET_POSITIONS` 中的关节姿态。修改该列表即可增删测试姿态，
每个姿态必须按照 `left_j1` 到 `left_j6` 的顺序填写 6 个弧度值。

先启动 Gazebo，再在另一个终端执行：

```bash
source /opt/ros/humble/setup.bash
source /home/dsl/dual_arm_ws/install/setup.bash
ros2 run ft_sensor_processing ft_sensor_processing_test
```

该脚本直接控制 `joint_trajectory_controller`，不执行 MoveIt 碰撞规划，因此
只应填写已经在仿真中确认安全的姿态。
