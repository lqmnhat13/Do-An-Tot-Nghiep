import numpy as np
from typing import Tuple, Optional
from src.contracts.depth_map import DepthMap, DepthRepresentation

class DepthRepresentationHelper:
    """
    Tiện ích chuẩn hóa và diễn giải giá trị độ sâu.
    Tuân thủ chặt chẽ mục 4.1 của Kế hoạch:
    - Không ngầm định metric distance từ relative depth.
    - Xử lý đúng chiều gần/xa dựa trên cờ near_is_larger.
    """

    @staticmethod
    def normalize_relative_depth(depth_values: np.ndarray, valid_mask: np.ndarray, near_is_larger: bool = True) -> np.ndarray:
        """
        Chuẩn hóa mảng độ sâu về đoạn [0.0, 1.0] trong đó 1.0 là gần nhất, 0.0 là xa nhất.
        """
        norm = np.zeros_like(depth_values, dtype=np.float32)
        if not np.any(valid_mask):
            return norm

        valid_vals = depth_values[valid_mask]
        v_min = float(np.min(valid_vals))
        v_max = float(np.max(valid_vals))
        span = v_max - v_min

        if span < 1e-6:
            norm[valid_mask] = 0.5
            return norm

        if near_is_larger:
            # Giá trị lớn hơn = gần hơn -> (val - min) / span
            norm[valid_mask] = (depth_values[valid_mask] - v_min) / span
        else:
            # Metric depth: giá trị nhỏ hơn = gần hơn -> (max - val) / span
            norm[valid_mask] = (v_max - depth_values[valid_mask]) / span

        return np.clip(norm, 0.0, 1.0)
