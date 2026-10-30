# s622_dual_task

总状态机只编排任务。视觉目标确认、抓取几何、抓取运动、夹持状态及工件 TF 已迁入
`vision_pick`；抓取和释放分别通过 `VisionPick` / `ReleaseGrasp` Action 调用。

```text
INIT / RESET_CONTEXT → WAIT_INTERFACES
IDLE / WAIT_TASK
GRASP_TARGET / START_PICK → WAIT_PICK
RIGHT_PREDOCK / ADD_CLEARANCE_GUARD → MOVE_TO_PREDOCK
RIGHT_DOCK / START_PLANAR_SEARCH → WAIT_PLANAR_SEARCH（可跳过）
COMPLETE / RELEASE_ASSEMBLY → WAIT_RELEASE → RETREAT → RESET_SCENE → MOVE_HOME
IDLE / WAIT_TASK
```

抓取准备（Home、开爪）在 pick Action 内执行，所以独立调用抓取不需要总状态机。
右臂预对接和完成后的撤退/回 Home 由本节点直接使用 pymoveit2 规划执行。
每次运动先取得 `/assembly_control/planning_lease`，确认执行结束后归还；不经过
通用运动 Action 服务。中央净空约束直接通过 MoveIt ApplyPlanningScene 更新。

错误或取消进入 `ERROR_HOLD/CANCEL_ACTIVE_GOALS → STOP_ADMITTANCE → HOLD`。
等待子 Action、在途服务和本节点轨迹停止后才归还控制权，不自动释放、撤退或回 Home。
取消晚到的 Action 接受响应时仍会发送取消，并跟踪其终态。
`~/reset` 只在 HOLD 且操作停止后接受；vision_pick 仍可能夹持时会拒绝重置。
此时应先处理工件并显式调用 release。不会根据全局 reset 静默清除抓取关系。

使用本工作区：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to s622_dual_bringup
source install/setup.bash
ros2 launch s622_dual_bringup peg_in_hole.launch.py
```

默认 `execute=false`、`autostart=false`，只初始化接口，不发送运动或场景变更。
开启任务执行：

```bash
ros2 launch s622_dual_bringup peg_in_hole.launch.py execute:=true autostart:=true planar_search:=true
ros2 service call /s622_dual_task_fsm/start std_srvs/srv/Trigger '{}'
ros2 service call /s622_dual_task_fsm/cancel std_srvs/srv/Trigger '{}'
ros2 service call /s622_dual_task_fsm/reset std_srvs/srv/Trigger '{}'
ros2 topic echo /s622_dual_task_fsm/state
```

自动启动只执行一次，下一轮通过 start 启动。已有仿真时传 `simulation:=false`；
已有感知或力处理时分别传 `perception:=false`、`force_processing:=false`，避免重复启动。
共享物理模型及 Home/暂歇关节目标在 `s622_dual_descriptions/config/shared_model.yaml`；
本包 config/task.yaml 只保留任务开关、能力 Action 名称、超时和预对接目标。

抓取后的 TF 始终由常驻 vision_pick 节点维护。MuJoCo 接触夹持的实际效果及预对接、
撤退目标仍需仿真验收；软件测试的替身接口不证明物理抓取成功。
