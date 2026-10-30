"""启动 MuJoCo、感知、抓取、导纳搜孔和总任务状态机。"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    execute = LaunchConfiguration('execute')
    autostart = LaunchConfiguration('autostart')
    enable_planar_search = LaunchConfiguration('planar_search')

    # 仿真、机器人 TF 与底层关节控制器。
    mujoco = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('s622_dual_bringup'),
                'launch',
                'mujoco.launch.py'
            ])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # 左右六维力处理节点，只在此处启动一次。
    ft_process = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ft_sensor_processing'),
                'launch',
                'ft_sensor_processing.launch.py'
            ])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # 统一启动导纳、执行器和启停协调服务，保留内部 enable 服务重映射。
    admittance_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('s622_dual_bringup'),
                'launch',
                'dual_admittance_control.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'start_control_mode': 'true',
        }.items(),
    )

    # YOLO-OBB RGB-D 感知节点，提供 3D 检测结果。
    yolo_obb_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('s622_dual_perception'),
                'launch',
                'yolo_obb_perception.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # vision_pick 包的基础入口不创建 control_mode；全系统共用导纳入口的协调节点。
    # followers_enabled 保持 true 表示支持导纳执行，抓取阶段通过 enable 服务保持导纳关闭。
    vision_pick = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('vision_pick'),
                'launch',
                'vision_pick.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'execute': execute,
        }.items(),
    )

    # 平面搜孔是独立能力，由任务状态机通过 /planar_search Action 调用。
    planar_search = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('s622_dual_assembly'),
                'launch',
                'planar_search.launch.py'
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items(),
        condition=IfCondition(enable_planar_search),
    )

    # 总状态机只负责任务编排。MoveIt 规划和导纳连续控制仍由
    # assembly_control 的 planning lease 保证互斥。
    task_fsm = Node(
        package='s622_dual_task',
        executable='fsm',
        name='s622_dual_task_fsm',
        output='screen',
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('s622_dual_descriptions'),
                'config',
                'shared_model.yaml'
            ]),
            PathJoinSubstitution([
                FindPackageShare('s622_dual_task'),
                'config',
                'task.yaml'
            ]),
            {
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                'task.execute': ParameterValue(execute, value_type=bool),
                'task.autostart': ParameterValue(autostart, value_type=bool),
                'task.enable_planar_search': ParameterValue(
                    enable_planar_search, value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('execute', default_value='false'),
        DeclareLaunchArgument('autostart', default_value='false'),
        DeclareLaunchArgument('planar_search', default_value='true'),
        mujoco,
        ft_process,
        admittance_control,
        yolo_obb_perception,
        vision_pick,
        planar_search,
        task_fsm,
    ])
