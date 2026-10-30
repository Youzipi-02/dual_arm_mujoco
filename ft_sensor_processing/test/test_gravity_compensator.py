# Copyright 2026 dsl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for model-based gravity compensation."""

import numpy as np
import pytest

from ft_sensor_processing.gravity_compensator import GravityCompensator


def make_compensator():
    """Create a deterministic compensator used by the tests."""
    return GravityCompensator(
        payload_weight_n=10.0,
        center_of_mass_m=[0.1, -0.2, 0.3],
        force_bias_n=[1.0, 2.0, 3.0],
        torque_bias_nm=[0.1, 0.2, 0.3],
    )


def test_aligned_gravity_and_bias_compensate_to_zero():
    """An exact static model measurement must produce no external wrench."""
    compensator = make_compensator()
    gravity = compensator.gravity_wrench(np.eye(3))
    measured = gravity + np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    output = compensator.compensate(measured, gravity)

    np.testing.assert_allclose(output, np.zeros(6), atol=1.0e-12)


def test_gravity_torque_is_center_of_mass_cross_force():
    """The moment at the sensor origin must use r cross force."""
    compensator = make_compensator()

    gravity = compensator.gravity_wrench(np.eye(3))

    np.testing.assert_allclose(gravity[:3], [0.0, 0.0, -10.0])
    np.testing.assert_allclose(
        gravity[3:],
        np.cross([0.1, -0.2, 0.3], [0.0, 0.0, -10.0]),
    )


def test_rotated_sensor_changes_gravity_direction():
    """A sensor rotated around Y must observe gravity along its X axis."""
    compensator = make_compensator()
    rotation_sensor_from_world = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    gravity = compensator.gravity_wrench(rotation_sensor_from_world)

    np.testing.assert_allclose(gravity[:3], [-10.0, 0.0, 0.0])


def test_invalid_rotation_shape_is_rejected():
    """A malformed orientation must never enter the compensation path."""
    with pytest.raises(ValueError):
        make_compensator().gravity_wrench(np.eye(2))


def test_negative_payload_weight_is_rejected():
    """Payload weight is a non-negative magnitude."""
    with pytest.raises(ValueError):
        GravityCompensator(-1.0, [0.0] * 3, [0.0] * 3, [0.0] * 3)
