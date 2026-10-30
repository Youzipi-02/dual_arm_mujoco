# s622_dual_bringup

系统级 launch 组合现有节点，不实现运动规划。

| 入口 | 内容 |
| --- | --- |
| `mujoco.launch.py` | MuJoCo、控制器、机器人 TF、MoveIt 和 RViz |
| `peg_in_hole.launch.py` | 仿真、感知、力处理、抓取、导纳、搜孔和总任务状态机 |
| `dual_admittance_control.launch.py` | 双侧导纳、连续位姿跟随器和内部控制权协调；每侧一个 enable 调用即可启停 |

`vision_pick` / `dual_task` 的 execute 默认 false。完整任务 autostart 默认 false。
独立抓取入口 autostart 默认 true；当 execute:=true 时提交一次抓取 Action，
抓取成功后保持夹持。传 autostart:=false 可仅启动服务并手动发送 Action。
`peg_in_hole` 的 planar_search 默认 true；设为 false 时不启动搜孔 Action，状态机会
跳过搜孔阶段。控制模式管理只由连续控制 launch 启动一个实例。

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to s622_dual_bringup
source install/setup.bash
ros2 launch s622_dual_bringup peg_in_hole.launch.py \
  execute:=true autostart:=false planar_search:=true
ros2 service call /s622_dual_task_fsm/start std_srvs/srv/Trigger '{}'
```

已有仿真时传 simulation:=false；完整任务还支持 perception、force_processing。
use_sim_time 在所有入口之间传递。robot_state_publisher 和 controller_manager
仅由 MuJoCo 启动一次。相机和工件模型沿用本工作区 MuJoCo 定义。

抓取链路：任务 → vision_pick Action → pymoveit2 → MoveIt → 控制器。
装配链路：搜孔 → 导纳目标 → s622_dual_admittance 连续跟随器 → 控制器。
规划/导纳互斥归 s622_dual_assembly/control_mode，不提供通用运动转发服务。

导纳链路启动后，调用 `/left_admittance/enable` 或 `/right_admittance/enable`
（`std_srvs/srv/SetBool`，`data: true`）即可开启对应侧，执行器自动协调，
无需另行调用 `set_compliant_mode`。最后一侧关闭（`data: false`）后自动释放控制权。
规划租约仍被占用时拒绝开启；原有任务的模式接口保留兼容。

同时运行本包的视觉抓取与导纳控制入口时，只保留导纳入口创建的
`assembly_control`（`followers_enabled=true`）。抓取入口传入
`start_control_mode:=false`，导纳入口使用 `start_control_mode:=true`。
两个入口单独运行时均默认启动各自需要的控制协调节点。

共享尺寸、夹爪和 Home/暂歇目标位于 s622_dual_descriptions/config/shared_model.yaml；
抓取参数及规划场景策略位于 vision_pick/config/vision_pick.yaml；搜孔参数位于装配包。
