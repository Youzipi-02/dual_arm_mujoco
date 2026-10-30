"""ROS independent planar spiral search algorithm."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class SearchPhase(str, Enum):
    """平面搜孔算法的内部阶段."""

    IDLE = "IDLE"
    MOVE_TO_PLANE = "MOVE_TO_PLANE"
    ESTABLISH_CONTACT = "ESTABLISH_CONTACT"
    RELIEVE_FORCE = "RELIEVE_FORCE"
    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass(frozen=True)
class PlanarSearchConfig:
    """平面搜孔算法使用的物理参数和终止阈值."""

    spacing: float
    tangential_speed: float
    max_radius: float
    approach_speed: float
    max_approach_distance: float
    desired_contact_force: float
    contact_threshold: float
    hole_force_threshold: float
    contact_stable_duration: float
    hole_confirm_duration: float
    timeout: float
    force_limit: float
    torque_limit: float
    contact_offset_z: float = 0.0

    def __post_init__(self) -> None:
        """验证配置中的单位、范围和阈值关系."""
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("planar search parameters must be finite")
        if any(value <= 0.0 for value in values[:-1]):
            raise ValueError("planar search parameters must be positive")
        if self.contact_offset_z < 0.0:
            raise ValueError("contact_offset_z must be non-negative")
        if self.contact_threshold >= self.desired_contact_force:
            raise ValueError("contact_threshold must be below desired_contact_force")
        if self.hole_force_threshold >= self.contact_threshold:
            raise ValueError("hole_force_threshold must be below contact_threshold")


@dataclass(frozen=True)
class WrenchSample:
    """表达在夹爪坐标系中的六维外部作用力."""

    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float

    def __post_init__(self) -> None:
        """拒绝包含非有限数值的力传感器样本."""
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("wrench sample contains NaN or Inf")

    @property
    def values(self) -> tuple[float, ...]:
        """按 Fx、Fy、Fz、Mx、My、Mz 的顺序返回六维数据."""
        return (
            self.force_x,
            self.force_y,
            self.force_z,
            self.torque_x,
            self.torque_y,
            self.torque_z,
        )

    @property
    def force_norm(self) -> float:
        """返回合力大小，单位为牛."""
        return math.sqrt(
            self.force_x**2 + self.force_y**2 + self.force_z**2
        )

    @property
    def torque_norm(self) -> float:
        """返回合力矩大小，单位为牛米."""
        return math.sqrt(
            self.torque_x**2 + self.torque_y**2 + self.torque_z**2
        )


@dataclass(frozen=True)
class SearchCommand:
    """单个算法周期产生的夹爪坐标系速度与诊断结果."""

    velocity_x: float
    velocity_y: float
    velocity_z: float
    desired_force_y: float
    phase: SearchPhase
    radius: float
    reason: str = ""


class PlanarSearchCore:
    """在夹爪 XZ 平面生成螺旋速度并判断孔口."""

    def __init__(self, config: PlanarSearchConfig) -> None:
        """保存配置并将算法置于空闲状态."""
        self.config = config
        self.reset()

    def reset(self) -> None:
        """清除一次搜索中的计时、角度和终止结果."""
        self.phase = SearchPhase.IDLE
        self.elapsed = 0.0
        self.angle = 0.0
        self.radius = 0.0
        self.approach_distance = 0.0
        self._contact_offset_traveled = 0.0
        self._contact_duration = 0.0
        self._hole_duration = 0.0
        self._contact_seen = False
        self.reason = ""

    def start(self) -> SearchCommand:
        """开始沿 -Y 限速靠近，并在平面内保持零速度."""
        self.reset()
        self.phase = (
            SearchPhase.MOVE_TO_PLANE
            if self.config.contact_offset_z > 0.0
            else SearchPhase.ESTABLISH_CONTACT
        )
        return self._command()

    def cancel(self) -> SearchCommand:
        """取消搜索并返回零速度命令."""
        self.phase = SearchPhase.CANCELED
        self.reason = "planar search canceled"
        return self._command()

    @property
    def terminal(self) -> bool:
        """返回算法是否已经成功、失败或被取消."""
        return self.phase in {
            SearchPhase.FOUND,
            SearchPhase.FAILED,
            SearchPhase.CANCELED,
        }

    def update(self, dt: float, wrench: WrenchSample) -> SearchCommand:
        """
        使用一帧六维力更新搜孔状态并计算 XZ 平面速度.

        参数:
            dt (float): 距离上一算法周期的时间，单位为秒。
            wrench (WrenchSample): 夹爪坐标系下的外部六维力。

        返回:
            SearchCommand: 当前周期应发送给导纳控制器的命令。

        异常:
            ValueError: 当周期无效或算法尚未启动时抛出。
        """
        if self.phase == SearchPhase.IDLE:
            raise ValueError("planar search has not been started")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("search update period must be finite and positive")
        if self.terminal:
            return self._command()

        self.elapsed += dt
        if wrench.force_y >= self.config.contact_threshold:
            self._contact_seen = True
        # Positive +Fy overload is corrected by the Y admittance channel.
        # Lateral or opposite-direction overloads have no known safe retreat.
        if math.hypot(wrench.force_x, wrench.force_z) > self.config.force_limit:
            return self._fail("force limit exceeded")
        if wrench.force_y < -self.config.force_limit:
            return self._fail("force limit exceeded")
        if self.elapsed > self.config.timeout:
            return self._fail("search timeout")

        recovery_threshold = (
            2.0 * self.config.desired_contact_force
            - self.config.contact_threshold
        )
        if wrench.force_y > recovery_threshold:
            if self.phase != SearchPhase.RELIEVE_FORCE:
                self._contact_duration = 0.0
            self.phase = SearchPhase.RELIEVE_FORCE
            return self._command()

        if self.phase == SearchPhase.RELIEVE_FORCE:
            if wrench.force_y > self.config.desired_contact_force:
                self._contact_duration = 0.0
                return self._command()
            if wrench.force_y < self.config.contact_threshold:
                self.phase = SearchPhase.ESTABLISH_CONTACT
                self._contact_duration = 0.0
                return self._command()
            self._contact_duration += dt
            if self._contact_duration >= self.config.contact_stable_duration:
                self.phase = SearchPhase.SEARCHING
            return self._command()

        if wrench.force_norm > self.config.force_limit:
            return self._fail("force limit exceeded")
        if wrench.torque_norm > self.config.torque_limit:
            return self._fail("torque limit exceeded")

        if self.phase == SearchPhase.MOVE_TO_PLANE:
            self._contact_offset_traveled += self.config.tangential_speed * dt
            if self._contact_offset_traveled >= self.config.contact_offset_z:
                self.phase = SearchPhase.ESTABLISH_CONTACT
            return self._command()

        if self.phase == SearchPhase.ESTABLISH_CONTACT:
            # Once contact has occurred, let admittance restore the force;
            # reversing the nominal approach after each force dip causes chatter.
            if self._contact_duration == 0.0 and not self._contact_seen:
                self.approach_distance += self.config.approach_speed * dt
            if wrench.force_y >= self.config.contact_threshold:
                self._contact_duration += dt
            else:
                self._contact_duration = 0.0
            if self._contact_duration >= self.config.contact_stable_duration:
                self.phase = SearchPhase.SEARCHING
            elif (
                self._contact_duration == 0.0
                and not self._contact_seen
                and self.approach_distance >= self.config.max_approach_distance
            ):
                return self._fail("maximum contact approach distance exceeded")
            return self._command()

        if wrench.force_y <= self.config.hole_force_threshold:
            self._hole_duration += dt
        else:
            self._hole_duration = 0.0
        if self._hole_duration >= self.config.hole_confirm_duration:
            self.phase = SearchPhase.FOUND
            self.reason = "contact force dropped inside hole region"
            return self._command()

        spacing_scale = self.config.spacing / (2.0 * math.pi)
        angle_rate = self.config.tangential_speed / (
            spacing_scale * math.sqrt(1.0 + self.angle**2)
        )
        velocity_x = angle_rate * (
            spacing_scale * math.cos(self.angle)
            - self.radius * math.sin(self.angle)
        )
        velocity_z = angle_rate * (
            spacing_scale * math.sin(self.angle)
            + self.radius * math.cos(self.angle)
        )
        self.angle += angle_rate * dt
        self.radius = spacing_scale * self.angle
        if self.radius > self.config.max_radius:
            return self._fail("maximum search radius exceeded")
        return self._command(velocity_x, velocity_z)

    def _fail(self, reason: str) -> SearchCommand:
        """记录失败原因并生成零速度终止命令."""
        self.phase = SearchPhase.FAILED
        self.reason = reason
        return self._command()

    def _command(
        self,
        velocity_x: float = 0.0,
        velocity_z: float = 0.0,
    ) -> SearchCommand:
        """组装当前状态对应的不可变搜索命令."""
        velocity_y = 0.0
        if self.phase == SearchPhase.MOVE_TO_PLANE:
            velocity_z = self.config.tangential_speed
        if (
            self.phase == SearchPhase.ESTABLISH_CONTACT
            and self._contact_duration == 0.0
            and not self._contact_seen
        ):
            velocity_y = -self.config.approach_speed
        if self.terminal:
            velocity_x = 0.0
            velocity_z = 0.0
        return SearchCommand(
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            velocity_z=velocity_z,
            desired_force_y=(
                self.config.desired_contact_force if self._contact_seen else 0.0
            ),
            phase=self.phase,
            radius=self.radius,
            reason=self.reason,
        )
