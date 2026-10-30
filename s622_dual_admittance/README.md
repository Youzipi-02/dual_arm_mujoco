# s622_dual_admittance

六维力输入来自 `ft_sensor_processing/ft_sensor_processing.launch.py` 提供的
`/{arm}_ft_sensor_processing/wrench_filtered`；导纳节点计算目标位姿，
连续关节执行由本包的 `cartesian_pose_executor` 负责。质量、阻尼、刚度和
力超时参数只在本包 `config/left_admittance.yaml`、
`config/right_admittance.yaml` 中设置。

## 启动与开启

已有仿真、机器人 TF、关节控制器和力处理节点时，启动完整控制链路：

```bash
source /home/dsl/dual_arm_mujoco_ws/install/setup.bash
ros2 launch s622_dual_bringup dual_admittance_control.launch.py use_sim_time:=true
```

另一个终端加载工作区后，每侧只需一个服务调用。例如开启右臂：

```bash
ros2 service call /right_admittance/enable std_srvs/srv/SetBool "{data: true}"
ros2 topic echo /right_admittance/status
```

返回 `success: true` 表示启用请求完成；状态持续为 `RUNNING` 才表示正在输出目标。
左臂使用 `/left_admittance/enable`。关闭对应侧只需把 `data` 改为 `false`。
开启第二侧不会重置第一侧；最后一侧关闭后自动停止连续执行器并释放控制权。
使用真实机器人时，所有相关节点应统一设置 `use_sim_time:=false`。
已有完整任务启动这些节点时不重复启动；`peg_in_hole.launch.py` 需设置
`planar_search:=true` 才会包含导纳控制链路。

无需手动调用 `/assembly_control/set_compliant_mode`。现有的
`s622_dual_assembly/control_mode` 在内部协调执行器、导纳启停和规划互斥；
已有规划任务持有租约时，开启请求返回失败，需等待该任务结束并释放租约。
若启停失败或超时，控制权保持阻塞；排除 TF/服务故障后，向任一侧发送
`data: false` 确认整体停止，再重新开启。仍在途的服务请求需先等其完成。
此故障恢复会停止双侧；正常关闭一侧不影响另一侧。

MoveIt 的关节空间规划和笛卡尔路径计算通过任务代码中的调用选择；
这里协调的是 MoveIt 与连续导纳执行器对关节的控制权，不更换规划算法。
当前接触阶段使用“搜孔名义速度 → 导纳修正 → IK → 位置控制器”的串联结构；
MoveIt 直接执行抓取和自由空间轨迹。当前未将 MoveIt 规划轨迹接入导纳输入。
`/assembly_control/set_compliant_mode` 保留给现有任务兼容使用；传 `false`
会停止双侧执行器和导纳节点。

左右臂各运行一个位置型六维导纳节点。节点订阅感知链处理后的
`wrench_filtered`，在抓取坐标系计算导纳修正，向
`/left_admittance/target_pose`、`/right_admittance/target_pose` 发布
`world` 下的 `PoseStamped`；关节命令由本包的连续位姿跟随器负责。

每侧接口：

| 接口 | 类型 | 用途 |
| --- | --- | --- |
| `/{arm}_admittance/configure` | `s622_dual_interfaces/srv/ConfigureAdmittance` | 禁用状态下设置名义速度与期望力；启用后可在线更新期望力，名义速度表达在抓取坐标系 |
| `/{arm}_admittance/nominal_velocity_command` | `geometry_msgs/msg/TwistStamped` | 搜孔等任务运行时更新抓取坐标系下的名义速度；仅在配置要求动态命令时生效 |
| `/{arm}_admittance/enable` | `std_srvs/srv/SetBool` | 完整启动时自动协调执行器与导纳启停；启用时锁定当前末端位姿与传感器到抓取点 TF，并等待新力 |
| `/{arm}_admittance/status` | `s622_dual_interfaces/msg/AdmittanceStatus` | 发布 `DISABLED`、`WAIT_WRENCH`、`RUNNING`、`STALE_WRENCH`、`TF_UNAVAILABLE`、`CALC_ERROR` |

只接受坐标系匹配、数值有限、源时间戳非零且晚于本次启用的力样本；
源年龄超过 `wrench_timeout_sec` 或时间戳不递增时拒绝样本。启用后
没有新力或运行中力超时会停止发布目标并报 `STALE_WRENCH`。
每次发布目标时同步发布 `RUNNING` 状态，供任务和连续执行器检查。

任务的 `COMPLIANT_HOLD` 阶段通过 `configure` 明确传入参考输入；
默认六维名义速度和期望力全为零。节点本身不判断接触阶段是否成功，
也不发布关节轨迹。数值核心位于 `admittance_core`，可独立运行单元测试。
当配置请求设置 `require_velocity_command=true` 时，启用后必须持续接收带源时间戳的
名义速度命令；命令超时会停止目标输出并报告 `STALE_COMMAND`。

连续跟随器保留既有 IK、关节限速、目标/关节状态/导纳心跳超时门禁。
它服务于接触装配，与视觉抓取使用的 pymoveit2 路径规划分工不同。

仅调试导纳计算时，可独立运行
`ros2 launch s622_dual_admittance dual_admittance.launch.py`。
该入口不启动连续执行器，`enable` 只控制目标位姿计算。
完整启动时，计算节点的底层开关被 remap 为 `/{arm}_admittance/enable_controller`，
公共 `enable` 由控制权管理节点提供；任务和手动操作应使用公共接口。
