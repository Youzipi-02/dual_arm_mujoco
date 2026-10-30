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

"""Check target acquisition's request-specific source-time validation."""

import threading
from types import SimpleNamespace

from vision_msgs.msg import Detection3DArray

from s622_dual_perception.nodes.target_acquisition import TargetAcquisition


class RecordingGate:
    """Record accepted samples and reset reasons."""

    def __init__(self):
        self.samples = []
        self.resets = []

    def update(self, stamp, received, targets, missing):
        self.samples.append((stamp, targets, missing))

    def reset(self, reason):
        self.resets.append(reason)


class AcquisitionContext:
    """Supply the callback with a deterministic ROS clock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gate = RecordingGate()
        self._started_ros = 90.0
        self._maximum_message_age = 1.0
        self._expected_frame = "world"

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))


def _message(stamp):
    message = Detection3DArray()
    message.header.frame_id = "world"
    message.header.stamp.sec = int(stamp)
    message.header.stamp.nanosec = int((stamp % 1) * 1_000_000_000)
    return message


def test_old_detection_cannot_confirm_new_request():
    """Reject source samples older than the new request or maximum age."""

    context = AcquisitionContext()
    TargetAcquisition._on_detections(context, _message(0.0))
    TargetAcquisition._on_detections(context, _message(89.0))
    TargetAcquisition._on_detections(context, _message(98.0))
    assert len(context._gate.resets) == 3
    assert not context._gate.samples


def test_fresh_detection_reaches_confirmation_gate():
    """Forward one fresh frame for normal class and continuity checks."""

    context = AcquisitionContext()
    TargetAcquisition._on_detections(context, _message(99.5))
    assert not context._gate.resets
    assert len(context._gate.samples) == 1
