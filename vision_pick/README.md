# vision_pick

使用 pymoveit2 直接调用 MoveIt 2；不经过通用运动执行服务，也不生成自定义规划算法。
`/vision_pick/pick` 执行准备、视觉确认、预抓取、下探、闭爪、抬升及暂歇位运动。
`/vision_pick/release` 张开夹爪并清理工件附着。动作返回后节点继续维护工件 TF。

构建：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to vision_pick s622_dual_task s622_dual_bringup
source install/setup.bash
```

独立运行（只需抓取，不启动总状态机）：

```bash
ros2 launch s622_dual_bringup vision_pick.launch.py execute:=true
```

此组合入口默认 `autostart=true`：等待 Action 服务后提交一次抓取请求，节点再等待运动、
相机和目标确认接口就绪。成功后停在暂歇位保持夹持，不自动释放或重复抓取。
`execute=false` 时不会提交请求。手动模式使用 `execute:=true autostart:=false`，
然后在另一个已 source 工作区的终端调用：

```bash
ros2 action send_goal /vision_pick/pick s622_dual_interfaces/action/VisionPick '{execute: true}' --feedback
ros2 action send_goal /vision_pick/release s622_dual_interfaces/action/ReleaseGrasp '{execute: true}' --feedback
```

已有仿真时传 `simulation:=false`。已有相机、MoveIt 和模式管理节点时只启动
`ros2 launch vision_pick vision_pick.launch.py execute:=true`，不要重复启动基础节点。
默认 `execute=false`；服务端参数和 Action Goal 必须同时允许执行。

预抓取通过 pymoveit2 的单臂 IK 接口求解，再将两臂关节目标交给 `both_arms` 联合规划，
避开 Humble 多末端约束采样的 dirty-state 问题；暂歇位也使用联合关节规划。
下探和抬升依次调用左、右臂的原生笛卡尔路径规划，另一臂保持；不宣称为双臂同步
笛卡尔规划。路径完成比例默认至少 0.999，碰撞检查开启。Humble 的笛卡尔服务不支持
速度缩放字段，因此只拉长返回轨迹的时间并相应缩放速度/加速度，不重新生成路径。
仅预抓取的规划失败会尝试一次反向轴候选；执行失败不自动尝试另一个候选。

物理尺寸及 Home/暂歇关节目标由 `s622_dual_descriptions/config/shared_model.yaml`
统一管理；抓取参数、规划参数、场景策略位于 `config/vision_pick.yaml`。
MuJoCo 闭爪目标分别为电线 ±0.0025 m、端子 ±0.0065 m，相对工件半径
各预压 1 mm，以建立接触夹持力；原 Gazebo 附着流程使用的略大于直径的开口不适用。
夹爪关节规划容差为 0.05 mm，避免默认 1 mm 容差抵消预压。MuJoCo 手指执行器
单独使用 `kp=1000 N/m、kv=10 N·s/m`，保留 28 N 力限幅；机械臂增益保持原值。
摩擦对照实验在 `robot_model/s622_dual.xml` 的 `s622_fingertip_contact` 中统一设置
四个指尖的 `friction="4.0 0.005 0.0001"`，第一个数为滑动摩擦系数。
当前工件滑动摩擦为 2.0，接触双方默认同优先级时逐项取较大值，因此可将指尖
第一个数依次设为 1.0（有效值 2.0）、4.0、8.0 对比；修改后需重启 MuJoCo。
指尖和两个工件使用 `solref="0.004 1"`；在 2 ms 仿真步长下，这使 1 mm
闭爪预压主要形成法向夹持力，避免默认 20 ms 软接触将预压吸收为几何穿透。
桌面和固定中间件的 MoveIt 碰撞模型与本工作区 MuJoCo 场景位姿对应。
桌面顶面为 0.750 m，底座安装面为 0.753 m，保留 3 mm 间隙；未禁用底座碰撞检查。

MuJoCo 使用手指接触抓取，没有 Gazebo DetachableJoint 附着/释放命令。
本实现使用轨迹执行结果和闭爪后末端 TF 到位检查。Action 的 `left_holding` /
`right_holding` 是“可能夹持”的保守标记，不能证明工件实际夹稳或没有滑落。
工件 TF 来自锁定视觉和夹爪刚体关系，不使用仿真物体真值，也不是滑移检测。
仿真验收可独立查看 `/object_state_monitor/free_joint_states` 的真实工件位姿；
该只读话题不参与视觉定位和抓取控制。
真实夹持效果仍需在 MuJoCo 中验收；若接触导致夹爪轨迹报错会进入保持，不假报成功。

取消/异常会取消感知与轨迹，等待终态；停止未确认时保留规划控制权，禁止装配接管。
不会自动开爪、释放、撤退或回 Home。仍可能夹持工件时拒绝新的 pick；需要显式 release。
`/vision_pick/reset`（Trigger）只在无夹持且所有操作已结束后接受。
节点重新启动后丢失的抓取关系不能自动恢复；重新允许执行前应人工处理已有夹持。

完整装配通过 `s622_dual_task` 调用这两个 Action。连续导纳跟随器归入
`s622_dual_admittance`，控制权由 `s622_dual_assembly/control_mode` 管理；不存在
`motion_executor` 运行依赖。原工作区 `/home/dsl/dual_arm_ws` 未作修改。
