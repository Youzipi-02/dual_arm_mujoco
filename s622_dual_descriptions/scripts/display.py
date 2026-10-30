import mujoco.viewer
 
def main():
    model = mujoco.MjModel.from_xml_path('src/s622_dual_descriptions/robot_model/scene.xml')
    data = mujoco.MjData(model)
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 只在启动时设置默认显示，运行中仍可通过面板开关。
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
 
if __name__ == "__main__":
    main()
