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

"""Unit tests for the streaming Butterworth filter."""

import numpy as np
import pytest

from ft_sensor_processing.butterworth_filter import (
    StreamingButterworthFilter,
)


def make_filter():
    """Construct the filter used by the processing node."""
    return StreamingButterworthFilter(
        sampling_frequency_hz=500.0,
        cutoff_frequency_hz=20.0,
        order=2,
        channels=6,
    )


def test_first_sample_initializes_without_startup_transient():
    """A constant first sample must pass without a zero-state ramp."""
    sample = np.arange(1.0, 7.0)

    output = make_filter().update(sample)

    np.testing.assert_allclose(output, sample, atol=1.0e-12)


def test_step_response_converges_to_new_value():
    """The causal low-pass output must converge to a sustained step."""
    low_pass = make_filter()
    low_pass.update(np.zeros(6))
    output = np.zeros(6)
    for _ in range(250):
        output = low_pass.update(np.ones(6))

    np.testing.assert_allclose(output, np.ones(6), atol=1.0e-6)


def test_reset_reinitializes_from_next_sample():
    """Reset must remove all history and initialize at the following sample."""
    low_pass = make_filter()
    low_pass.update(np.zeros(6))
    low_pass.update(np.ones(6))
    low_pass.reset()

    output = low_pass.update(np.full(6, 3.0))

    np.testing.assert_allclose(output, np.full(6, 3.0), atol=1.0e-12)


def test_cutoff_at_nyquist_is_rejected():
    """Digital filter cutoff must remain strictly below Nyquist."""
    with pytest.raises(ValueError):
        StreamingButterworthFilter(500.0, 250.0, 2)


def test_wrong_sample_size_is_rejected():
    """The six-axis filter must reject incomplete wrench vectors."""
    with pytest.raises(ValueError):
        make_filter().update([1.0, 2.0, 3.0])
