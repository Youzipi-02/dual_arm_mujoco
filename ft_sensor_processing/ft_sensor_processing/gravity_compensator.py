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

"""Pure numerical gravity compensation for a six-axis force sensor."""

from typing import Sequence

import numpy as np


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    """Convert a sequence to a finite one-dimensional NumPy vector."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


class GravityCompensator:
    """Remove identified payload gravity and fixed sensor biases."""

    def __init__(
        self,
        payload_weight_n: float,
        center_of_mass_m: Sequence[float],
        force_bias_n: Sequence[float],
        torque_bias_nm: Sequence[float],
    ) -> None:
        """Store validated calibration parameters in the sensor frame."""
        self.payload_weight_n = float(payload_weight_n)
        if not np.isfinite(self.payload_weight_n):
            raise ValueError("payload_weight_n must be finite")
        if self.payload_weight_n < 0.0:
            raise ValueError("payload_weight_n must be non-negative")

        self.center_of_mass_m = _finite_vector(
            center_of_mass_m, 3, "center_of_mass_m"
        )
        self.force_bias_n = _finite_vector(
            force_bias_n, 3, "force_bias_n"
        )
        self.torque_bias_nm = _finite_vector(
            torque_bias_nm, 3, "torque_bias_nm"
        )
        self._static_bias = np.concatenate(
            (self.force_bias_n, self.torque_bias_nm)
        )

    def gravity_wrench(
        self, rotation_sensor_from_world: np.ndarray
    ) -> np.ndarray:
        """Return payload gravity wrench expressed at the sensor origin."""
        rotation = np.asarray(rotation_sensor_from_world, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(
                "rotation_sensor_from_world must have shape (3, 3)"
            )
        if not np.all(np.isfinite(rotation)):
            raise ValueError("rotation matrix must contain finite values")

        gravity_world = np.array(
            [0.0, 0.0, -self.payload_weight_n], dtype=float
        )
        gravity_force_sensor = rotation @ gravity_world
        gravity_torque_sensor = np.cross(
            self.center_of_mass_m,
            gravity_force_sensor,
        )
        return np.concatenate(
            (gravity_force_sensor, gravity_torque_sensor)
        )

    def compensate(
        self,
        measured_wrench: Sequence[float],
        gravity_wrench: Sequence[float],
    ) -> np.ndarray:
        """Subtract fixed bias and the supplied gravity wrench."""
        measured = _finite_vector(measured_wrench, 6, "measured_wrench")
        gravity = _finite_vector(gravity_wrench, 6, "gravity_wrench")
        return measured - self._static_bias - gravity
