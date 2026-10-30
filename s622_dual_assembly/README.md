# s622_dual_assembly

该功能包负责右臂接触式平面搜孔。`planar_search_core.py` 是不依赖 ROS 2 的
等距螺旋轨迹与终止条件核心；`planar_search_node.py` 提供 `/planar_search`
Action，并协调右臂导纳、连续笛卡尔执行器和六维力输入。

坐标约定：

- 搜索平面是 `right_grasp_frame` 的 XZ 平面；
- 插入方向是 `right_grasp_frame` 的 -Y；
- 环境接触反力是 `right_grasp_frame` 的 +Fy；
- 距离、速度、力、力矩分别使用 m、m/s、N、N·m。

接触阶段先将期望 Fy 设为 0，以 `approach_speed` 沿 -Y 发送名义速度；
首次检测到接触后在线将期望 Fy 设为 10 N。若在
`max_approach_distance` 内未检测到稳定接触，则停止并报告失败。该距离只限制
名义靠近量，右臂导纳的 Y 向力修正还受其自身 `max_displacement` 限制。
搜孔节点发布名义速度作为轨迹输入；导纳节点将其积分成名义位姿，
叠加力反馈位移后仍发布位置目标 `target_pose`。右臂导纳的
`max_velocity` 只限制力反馈位移的变化速度；名义靠近速度由搜孔包的
`approach_speed` 独立设置。
首次达到 `contact_threshold` 即停止 -Y 名义速度；接触持续
`contact_stable_duration` 后，在维持 `desired_contact_force` 的同时生成 XZ 螺线。
首次接触后即使 Fy 短时跌破阈值，也不重新发送名义 -Y 靠近速度；
由导纳修正恢复接触，避免过力退让与全速重新压入反复切换。
当前接触阈值为 5 N，控制目标为 10 N。
若 +Fy 超过 15 N，搜孔暂停螺线；Y 向导纳根据实际力与 10 N 目标的误差
修正位置，不再额外叠加名义 +Y 退让速度。回到接触力范围并稳定后继续搜索。
侧向力超限等故障仍会终止搜索。导纳和执行器发布的是位置目标，接触瞬间的
峰值仍取决于靠近速度、仿真接触刚度及控制链延迟，不能保证峰值恰好为 10 N。

当前 `spacing=0.00015 m`、`tangential_speed=0.002 m/s`、`timeout=30 s`。
即使不计靠近阶段，螺线在 30 s 内的理论半径也只有约 1.7 mm；
配置的 `max_radius=20 mm` 无法在此时限内扫到。孔位偏差较大时，
应根据允许的搜索时间和工件尺寸重新设定间距、切向速度与最大半径。

独立启动节点：

```bash
ros2 launch s622_dual_assembly planar_search.launch.py use_sim_time:=true
```

节点要求力处理、右臂导纳、连续笛卡尔执行器和 `assembly_control` 已经启动。
任务层通过 `s622_dual_interfaces/action/PlanarSearch` 启动和取消搜索。
接触阶段的链路为：搜孔名义速度 → 导纳修正 → IK → 关节位置控制器。
搜孔节点只调用 `configure` 与公共 `enable`，不再显式切换控制模式。
抓取和自由空间运动继续沿用 MoveIt 直接执行，阶段交接由内部控制权协调完成。

`control_mode` 提供 `/assembly_control/set_compliant_mode` 和
`/assembly_control/planning_lease`，保证规划与连续控制互斥。规划租约按节点所有者
显式获取/归还，停止未确认时不自动超时释放。已有其他规划节点时也必须遵守此接口。

完整导纳启动时还提供 `/left_admittance/enable`、`/right_admittance/enable`。
每侧一个 `SetBool` 调用即可启停导纳并自动协调连续执行器，无需先手动切换模式。
开启第二侧不重置已运行的一侧，最后一侧关闭后自动释放控制权。
底层计算节点的开关 remap 为 `/{arm}_admittance/enable_controller`，仅供内部协调。
原有先 `set_compliant_mode` 再 `enable` 的任务流程继续兼容；全局退出模式会同时
停止双侧导纳，避免下次进入时恢复旧目标。启停结果不明确时阻塞规划，确认整体
关闭后才能恢复；此时对任一侧调用 `enable(false)` 会执行整体停止。
