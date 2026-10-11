"""Image-space approximation of a forward walking corridor (not a safe path)."""
from dataclasses import dataclass
import math
from typing import Optional, Tuple

from src.contracts.detection import BoundingBox


@dataclass(frozen=True)
class WalkingCorridor:
    center_x: float = 0.5
    top_y: float = 0.35
    top_width: float = 0.20
    bottom_width: float = 0.60
    min_overlap_ratio: float = 0.10

    def __post_init__(self):
        values = (self.center_x, self.top_y, self.top_width,
                  self.bottom_width, self.min_overlap_ratio)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Walking corridor settings must be finite")
        if not (0 <= self.top_y < 1 and 0 < self.top_width <= self.bottom_width <= 1
                and 0 < self.min_overlap_ratio <= 1
                and self.bottom_width / 2 <= self.center_x <= 1 - self.bottom_width / 2):
            raise ValueError("Invalid walking corridor geometry or overlap threshold")

    def polygon(self, frame_width: float, frame_height: float) -> Tuple[Tuple[float, float], ...]:
        """Vertices in original image coordinates; also used by the letterboxed HUD."""
        if not all(math.isfinite(v) and v > 0 for v in (frame_width, frame_height)):
            return ()
        cx = self.center_x * frame_width
        top_half = self.top_width * frame_width / 2
        bottom_half = self.bottom_width * frame_width / 2
        top = self.top_y * frame_height
        return ((cx - top_half, top), (cx + top_half, top),
                (cx + bottom_half, frame_height), (cx - bottom_half, frame_height))

    def overlap_ratio(self, bbox: BoundingBox, frame_width: float,
                      frame_height: float) -> Optional[float]:
        """Exact polygon/rectangle intersection. None means invalid input geometry."""
        polygon = self.polygon(frame_width, frame_height)
        if not polygon or not all(math.isfinite(v) for v in bbox.as_tuple()) or bbox.area <= 0:
            return None
        box = bbox.clamp(frame_width, frame_height)
        if box.area <= 0:
            return None
        points = list(polygon)
        # Sutherland–Hodgman clipping against each edge of the bounding rectangle.
        for axis, boundary, sign in ((0, box.xmin, 1), (0, box.xmax, -1),
                                     (1, box.ymin, 1), (1, box.ymax, -1)):
            clipped = []
            for previous, current in zip(points[-1:] + points[:-1], points):
                previous_in = sign * (previous[axis] - boundary) >= 0
                current_in = sign * (current[axis] - boundary) >= 0
                if previous_in != current_in:
                    fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
                    clipped.append(tuple(previous[i] + fraction * (current[i] - previous[i])
                                         for i in (0, 1)))
                if current_in:
                    clipped.append(current)
            points = clipped
            if not points:
                return 0.0
        area = abs(sum(a[0] * b[1] - b[0] * a[1]
                       for a, b in zip(points, points[1:] + points[:1]))) / 2
        corridor_area = (self.top_width + self.bottom_width) * frame_width / 2 * (1 - self.top_y) * frame_height
        return min(1.0, max(0.0, area / min(box.area, corridor_area)))
