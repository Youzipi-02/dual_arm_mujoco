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

"""Stateful streaming Butterworth filter for multi-channel sensor data."""

from typing import Optional, Sequence

import numpy as np
from scipy import signal


class StreamingButterworthFilter:
    """Advance one causal low-pass filter sample at a time."""

    def __init__(
        self,
        sampling_frequency_hz: float,
        cutoff_frequency_hz: float,
        order: int,
        channels: int = 6,
    ) -> None:
        """Design the filter once and allocate state on the first sample."""
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.cutoff_frequency_hz = float(cutoff_frequency_hz)
        self.order = int(order)
        self.channels = int(channels)
        self._validate_configuration()

        self._sos = signal.butter(
            self.order,
            self.cutoff_frequency_hz,
            btype="low",
            fs=self.sampling_frequency_hz,
            output="sos",
        )
        self._state: Optional[np.ndarray] = None

    def _validate_configuration(self) -> None:
        """Reject unstable or physically meaningless filter settings."""
        values = [self.sampling_frequency_hz, self.cutoff_frequency_hz]
        if not all(np.isfinite(value) for value in values):
            raise ValueError("filter frequencies must be finite")
        if self.sampling_frequency_hz <= 0.0:
            raise ValueError("sampling frequency must be positive")
        if not 0.0 < self.cutoff_frequency_hz < 0.5 * self.sampling_frequency_hz:
            raise ValueError("cutoff must lie between zero and Nyquist")
        if self.order < 1:
            raise ValueError("filter order must be at least one")
        if self.channels < 1:
            raise ValueError("channel count must be at least one")

    def reset(self, initial_value: Optional[Sequence[float]] = None) -> None:
        """Clear state or initialize it to the steady state of a value."""
        if initial_value is None:
            self._state = None
            return

        sample = self._sample(initial_value)
        base_state = signal.sosfilt_zi(self._sos)
        self._state = (
            base_state[:, :, np.newaxis]
            * sample[np.newaxis, np.newaxis, :]
        )

    def update(self, sample: Sequence[float]) -> np.ndarray:
        """Filter and return one multi-channel sample."""
        input_sample = self._sample(sample)
        if self._state is None:
            self.reset(input_sample)

        output, self._state = signal.sosfilt(
            self._sos,
            input_sample[np.newaxis, :],
            axis=0,
            zi=self._state,
        )
        return output[0].copy()

    def _sample(self, values: Sequence[float]) -> np.ndarray:
        """Convert and validate one filter input sample."""
        sample = np.asarray(values, dtype=float)
        if sample.shape != (self.channels,):
            raise ValueError(
                f"filter sample must contain {self.channels} values"
            )
        if not np.all(np.isfinite(sample)):
            raise ValueError("filter sample must contain finite values")
        return sample.copy()
