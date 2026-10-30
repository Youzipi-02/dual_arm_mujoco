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

"""Select and continuously confirm visual target messages."""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

from vision_msgs.msg import Detection3D, Detection3DArray


REQUIRED_CLASSES = ("wire", "copper_terminal")


@dataclass(frozen=True)
class TargetPair:
    """Highest-confidence 3D targets selected from the same frame."""

    wire: Detection3D
    terminal: Detection3D


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def detection_score(detection: Detection3D) -> Optional[Tuple[str, float]]:
    """Return the highest-scoring valid task-class hypothesis."""
    candidates = []
    for result in detection.results:
        class_id = str(result.hypothesis.class_id)
        score = float(result.hypothesis.score)
        if class_id in REQUIRED_CLASSES and math.isfinite(score):
            candidates.append((class_id, score))
    return max(candidates, key=lambda item: item[1]) if candidates else None


def detection_is_valid(detection: Detection3D, expected_frame: str) -> bool:
    """Check frame, center, quaternion and bounding-box size."""
    if detection.header.frame_id != expected_frame:
        return False
    center = detection.bbox.center
    position = (center.position.x, center.position.y, center.position.z)
    quaternion = (
        center.orientation.x,
        center.orientation.y,
        center.orientation.z,
        center.orientation.w,
    )
    size = (detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z)
    if not _finite((*position, *quaternion, *size)):
        return False
    if math.sqrt(sum(value * value for value in quaternion)) < 1.0e-9:
        return False
    return all(value > 0.0 for value in size)


def extract_best_targets(
    message: Detection3DArray, expected_frame: str
) -> tuple[Optional[TargetPair], tuple[str, ...]]:
    """Keep the best detection per class and report missing classes."""
    if message.header.frame_id != expected_frame:
        return None, REQUIRED_CLASSES
    best: Dict[str, tuple[float, Detection3D]] = {}
    for detection in message.detections:
        hypothesis = detection_score(detection)
        if hypothesis is None or not detection_is_valid(detection, expected_frame):
            continue
        class_id, score = hypothesis
        previous = best.get(class_id)
        if previous is None or score > previous[0]:
            best[class_id] = (score, detection)
    missing = tuple(name for name in REQUIRED_CLASSES if name not in best)
    if missing:
        return None, missing
    return TargetPair(best["wire"][1], best["copper_terminal"][1]), ()


class ContinuousDetectionGate:
    """Lock targets after continuous visibility and sufficient frames."""

    def __init__(
        self,
        confirm_duration: float,
        maximum_gap: float,
        minimum_confirmations: int,
    ) -> None:
        if confirm_duration <= 0.0 or maximum_gap <= 0.0:
            raise ValueError("vision durations must be positive")
        if minimum_confirmations < 1:
            raise ValueError("minimum confirmations must be positive")
        self.confirm_duration = float(confirm_duration)
        self.maximum_gap = float(maximum_gap)
        self.minimum_confirmations = int(minimum_confirmations)
        self.reset("等待首帧")

    def reset(self, reason: str = "", last_streak: float = 0.0) -> None:
        self.first_stamp: Optional[float] = None
        self.last_stamp: Optional[float] = None
        self.last_received: Optional[float] = None
        self.confirmations = 0
        self.latest: Optional[TargetPair] = None
        self.locked: Optional[TargetPair] = None
        self.last_reason = reason
        self.last_streak_duration = max(0.0, float(last_streak))

    @property
    def elapsed(self) -> float:
        if self.first_stamp is None or self.last_stamp is None:
            return 0.0
        return max(0.0, self.last_stamp - self.first_stamp)

    def update(
        self,
        stamp: float,
        received: float,
        targets: Optional[TargetPair],
        missing: Sequence[str] = (),
    ) -> bool:
        """Receive a frame and deep-copy the last one when confirmed."""
        if self.locked is not None:
            return True
        if not math.isfinite(stamp) or not math.isfinite(received):
            self.reset("检测时间戳无效")
            return False
        if targets is None:
            names = ", ".join(missing) if missing else "目标"
            self.reset(f"当前帧缺少 {names}", self.elapsed)
            self.last_received = received
            return False
        discontinuous = (
            self.last_stamp is not None
            and (stamp <= self.last_stamp or stamp - self.last_stamp > self.maximum_gap)
        )
        if self.first_stamp is None or discontinuous:
            previous_elapsed = self.elapsed
            self.first_stamp = stamp
            self.confirmations = 1
            if discontinuous:
                self.last_streak_duration = previous_elapsed
                self.last_reason = "检测帧间隔超限，已重新计时"
        else:
            self.confirmations += 1
        self.last_stamp = stamp
        self.last_received = received
        self.latest = targets
        if (
            self.elapsed >= self.confirm_duration
            and self.confirmations >= self.minimum_confirmations
        ):
            self.locked = deepcopy(targets)
            self.last_reason = "两个目标连续可见，快照已锁定"
            return True
        return False

    def expire(self, received: float) -> bool:
        """Reset the streak when the detection stream stops for too long."""
        if self.locked is not None or self.last_received is None:
            return False
        if received - self.last_received <= self.maximum_gap:
            return False
        self.reset("检测消息中断，已重新计时", self.elapsed)
        return True

    def status(self) -> str:
        return (
            f"continuous={self.elapsed:.1f}/{self.confirm_duration:.1f}s "
            f"frames={self.confirmations}/{self.minimum_confirmations} "
            f"last_streak={self.last_streak_duration:.1f}s "
            f"reason={self.last_reason or '-'}"
        )


def stamp_seconds(message: Detection3DArray) -> float:
    """Convert a ROS stamp to seconds."""
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
