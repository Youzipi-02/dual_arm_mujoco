import math

import pytest

from s622_dual_assembly.planar_search_core import (
    PlanarSearchConfig,
    PlanarSearchCore,
    SearchPhase,
    WrenchSample,
)


def config(**overrides):
    values = {
        "spacing": 0.00015,
        "tangential_speed": 0.002,
        "max_radius": 0.020,
        "approach_speed": 0.001,
        "max_approach_distance": 0.010,
        "desired_contact_force": 10.0,
        "contact_threshold": 9.0,
        "hole_force_threshold": 1.0,
        "contact_stable_duration": 0.06,
        "hole_confirm_duration": 0.04,
        "timeout": 30.0,
        "force_limit": 40.0,
        "torque_limit": 2.0,
    }
    values.update(overrides)
    return PlanarSearchConfig(**values)


def wrench(force_y, *, force_x=0.0, torque_z=0.0):
    return WrenchSample(force_x, force_y, 0.0, 0.0, 0.0, torque_z)


def establish_contact(core):
    core.start()
    for _ in range(3):
        command = core.update(0.02, wrench(9.5))
    assert command.phase == SearchPhase.SEARCHING


def test_contact_must_remain_above_positive_fy_threshold():
    core = PlanarSearchCore(config())
    command = core.start()
    assert command.velocity_x == 0.0
    assert command.velocity_y == pytest.approx(-0.001)
    assert command.velocity_z == 0.0
    assert command.desired_force_y == 0.0
    core.update(0.04, wrench(9.5))
    command = core.update(0.02, wrench(-8.0))
    assert command.phase == SearchPhase.ESTABLISH_CONTACT
    assert command.velocity_y == 0.0
    command = core.update(0.02, wrench(9.5))
    assert command.phase == SearchPhase.ESTABLISH_CONTACT
    assert command.velocity_y == 0.0
    assert command.desired_force_y == pytest.approx(10.0)


def test_offset_moves_laterally_before_axial_contact_approach():
    core = PlanarSearchCore(config(
        contact_offset_z=0.004, tangential_speed=0.004,
    ))
    command = core.start()
    assert command.phase == SearchPhase.MOVE_TO_PLANE
    assert command.velocity_y == 0.0
    assert command.velocity_z == pytest.approx(0.004)
    command = core.update(0.5, wrench(0.0))
    assert command.phase == SearchPhase.MOVE_TO_PLANE
    assert command.velocity_y == 0.0
    command = core.update(0.5, wrench(0.0))
    assert command.phase == SearchPhase.ESTABLISH_CONTACT
    assert command.velocity_z == 0.0
    assert command.velocity_y == pytest.approx(-core.config.approach_speed)


def test_approach_pauses_on_first_contact_sample():
    core = PlanarSearchCore(config())
    core.start()
    command = core.update(0.02, wrench(9.5))
    distance = core.approach_distance
    assert command.phase == SearchPhase.ESTABLISH_CONTACT
    assert command.velocity_y == 0.0
    command = core.update(0.02, wrench(9.5))
    assert core.approach_distance == pytest.approx(distance)
    assert command.velocity_y == 0.0
    command = core.update(0.02, wrench(9.5))
    assert command.phase == SearchPhase.SEARCHING
    assert command.velocity_y == 0.0


def test_spiral_command_stays_in_xz_plane_at_configured_speed():
    core = PlanarSearchCore(config())
    establish_contact(core)
    command = core.update(0.02, wrench(8.0))
    speed = math.hypot(command.velocity_x, command.velocity_z)
    assert command.phase == SearchPhase.SEARCHING
    assert command.velocity_y == 0.0
    assert speed == pytest.approx(core.config.tangential_speed)
    assert command.desired_force_y == pytest.approx(10.0)
    assert command.radius > 0.0


def test_low_fy_must_persist_before_hole_is_confirmed():
    core = PlanarSearchCore(config())
    establish_contact(core)
    assert core.update(0.02, wrench(0.5)).phase == SearchPhase.SEARCHING
    command = core.update(0.02, wrench(0.5))
    assert command.phase == SearchPhase.FOUND
    assert command.velocity_x == 0.0
    assert command.velocity_z == 0.0
    assert command.velocity_y == 0.0


def test_contact_approach_stops_at_bounded_distance():
    core = PlanarSearchCore(config(max_approach_distance=0.00005))
    core.start()
    command = core.update(0.05, wrench(0.0))
    assert command.phase == SearchPhase.FAILED
    assert command.velocity_y == 0.0
    assert command.reason == "maximum contact approach distance exceeded"


def test_force_and_torque_limits_stop_the_search():
    core = PlanarSearchCore(config())
    core.start()
    command = core.update(0.02, wrench(9.5, force_x=40.0))
    assert command.phase == SearchPhase.FAILED
    assert command.reason == "force limit exceeded"

    core.start()
    command = core.update(0.02, wrench(9.5, torque_z=2.1))
    assert command.phase == SearchPhase.FAILED
    assert command.reason == "torque limit exceeded"


def test_normal_overload_uses_admittance_then_resumes_spiral():
    core = PlanarSearchCore(config(approach_speed=0.005))
    establish_contact(core)
    search = core.update(0.02, wrench(9.5))
    radius_before_recovery = search.radius

    command = core.update(0.02, wrench(152.0, force_x=-10.0, torque_z=18.0))
    assert command.phase == SearchPhase.RELIEVE_FORCE
    assert command.velocity_x == 0.0
    assert command.velocity_y == 0.0
    assert command.desired_force_y == pytest.approx(10.0)
    assert command.velocity_z == 0.0
    assert command.radius == radius_before_recovery

    command = core.update(0.02, wrench(10.5))
    assert command.phase == SearchPhase.RELIEVE_FORCE
    assert command.velocity_y == 0.0

    for _ in range(3):
        command = core.update(0.02, wrench(9.5))
    assert command.phase == SearchPhase.SEARCHING
    assert command.velocity_y == 0.0
    assert command.radius == radius_before_recovery
    assert core.update(0.02, wrench(9.5)).radius > radius_before_recovery


def test_recovery_leaves_nominal_y_still_when_contact_is_lost():
    core = PlanarSearchCore(config())
    core.start()
    assert core.update(0.02, wrench(100.0)).phase == SearchPhase.RELIEVE_FORCE
    command = core.update(0.02, wrench(8.0))
    assert command.phase == SearchPhase.ESTABLISH_CONTACT
    assert command.velocity_y == 0.0
    assert command.desired_force_y == pytest.approx(10.0)


def test_contact_loss_does_not_repeat_full_speed_approach():
    core = PlanarSearchCore(config(approach_speed=0.002))
    core.start()
    assert core.update(0.02, wrench(100.0)).phase == SearchPhase.RELIEVE_FORCE
    assert core.update(0.02, wrench(0.0)).velocity_y == 0.0
    command = core.update(0.02, wrench(0.0))
    assert command.phase == SearchPhase.ESTABLISH_CONTACT
    assert command.velocity_y == 0.0
    assert core.approach_distance == 0.0


def test_radius_limit_stops_search():
    core = PlanarSearchCore(config(max_radius=0.00001))
    establish_contact(core)
    command = core.update(0.02, wrench(8.0))
    assert command.phase == SearchPhase.FAILED
    assert command.reason == "maximum search radius exceeded"


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError, match="hole_force_threshold"):
        config(hole_force_threshold=9.5)
    with pytest.raises(ValueError, match="positive"):
        config(approach_speed=0.0)
