"""Parameters for orchestration and the two remaining joint-space motions."""
import math
from vision_pick.config import load_defaults


class TaskConfigMixin:
    def _load_parameters(self):
        defaults = load_defaults('s622_dual_descriptions', 'shared_model.yaml')
        defaults.update(load_defaults('s622_dual_task', 'task.yaml'))
        self.cfg = {key: self.declare_parameter(key, value).value
                    for key, value in defaults.items() if key != 'use_sim_time'}
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        for key, value in self.cfg.items():
            if key.endswith(('timeout', 'rate')) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be positive and finite')
        for key in ('motion.velocity_scale', 'motion.acceleration_scale'):
            if not 0 < self.cfg[key] <= 1:
                raise ValueError(f'{key} must lie in (0, 1]')
        for key in ('left_arm.home', 'right_arm.home', 'wire.stage_joints', 'terminal.stage_joints', 'terminal.predock_joints'):
            if len(self.cfg[key]) != 6 or not all(math.isfinite(v) for v in self.cfg[key]):
                raise ValueError(f'{key} must contain six finite joint positions')
        self.execute = self.cfg['task.execute']
        self.autostart = self.cfg['task.autostart']
        self.enable_planar_search = self.cfg['task.enable_planar_search']
