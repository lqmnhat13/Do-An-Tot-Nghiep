import cv2
import numpy as np
from typing import Dict, Any, Optional

from src.contracts.risk import RiskLevel, Direction

COLOR_HIGH_RISK = (0, 0, 255)    # Đỏ
COLOR_MED_RISK = (0, 215, 255)   # Vàng cam
COLOR_LOW_RISK = (0, 255, 0)     # Xanh lá
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 20, 20)

class HUDRenderer:
    """
    Hiển thị thông tin trực quan lên khung hình camera (Heads-Up Display).
    Tuân thủ mục 6 của Kế hoạch:
    - Bounding Box theo mức nguy cơ kèm nhãn chữ rõ ràng (không chỉ dựa vào màu).
    - La bàn / vạch chia 3 vùng Trái - Giữa - Phải.
    - Bản đồ độ sâu thu nhỏ (Mini Depth Map).
    - Bảng thông số kỹ thuật (FPS, Latency, Tuổi dữ liệu, Audio status).
    """

    def __init__(self, target_width: int = 800, target_height: int = 600):
        self.target_width = target_width
        self.target_height = target_height
        self._cached_depth_id: int = -1
        self._cached_depth_mini: Optional[np.ndarray] = None

    def render(self, snapshot: Dict[str, Any]) -> np.ndarray:
        packet = snapshot.get("packet")
        if packet is None:
            canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            cv2.putText(canvas, "Dang cho ket noi camera...", (50, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
            return canvas

        frame = cv2.resize(packet.image, (self.target_width, self.target_height))
        h, w = frame.shape[:2]
        orig_w, orig_h = packet.original_size

        scale_x = w / float(orig_w) if orig_w > 0 else 1.0
        scale_y = h / float(orig_h) if orig_h > 0 else 1.0

        # 1. Vẽ các vạch chia vùng không gian 35% - 30% - 35%
        x_left = int(w * 0.35)
        x_right = int(w * 0.65)
        cv2.line(frame, (x_left, 45), (x_left, h - 35), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(frame, (x_right, 45), (x_right, h - 35), (120, 120, 120), 1, cv2.LINE_AA)

        cv2.putText(frame, "TRAI", (int(x_left * 0.4), h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(frame, "GIUA (LOI DI)", (int(w * 0.43), h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "PHAI", (int(x_right + (w - x_right) * 0.4), h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # 2. Vẽ Bounding Boxes kèm nhãn và mức độ nguy cơ
        assessments = snapshot.get("assessments", [])
        detection_res = snapshot.get("detection")
        det_map = {}
        if detection_res:
            for d in detection_res.detections:
                if d.track_id is not None:
                    det_map[d.track_id] = d

        for a in assessments:
            d = det_map.get(a.track_id)
            if d is None:
                continue

            bx1 = int(round(d.bbox.xmin * scale_x))
            by1 = int(round(d.bbox.ymin * scale_y))
            bx2 = int(round(d.bbox.xmax * scale_x))
            by2 = int(round(d.bbox.ymax * scale_y))

            # Chọn màu theo mức nguy cơ
            if a.risk_level == RiskLevel.HIGH:
                color = COLOR_HIGH_RISK
                thickness = 3
                level_str = "[NGUY CO CAO]"
            elif a.risk_level == RiskLevel.MEDIUM:
                color = COLOR_MED_RISK
                thickness = 2
                level_str = "[CHU Y]"
            else:
                color = COLOR_LOW_RISK
                thickness = 1
                level_str = "[AN TOAN]"

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness)

            # Nhãn đối tượng kèm khoảng cách mét
            label = f"[{a.class_name.upper()}] {level_str} - {a.proximity_desc}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (bx1, max(0, by1 - 20)), (bx1 + tw + 6, max(20, by1)), color, -1)
            cv2.putText(frame, label, (bx1 + 3, max(15, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if color != COLOR_HIGH_RISK else COLOR_WHITE, 1, cv2.LINE_AA)

        # Hiển thị Banner cảnh báo khẩn cấp nếu có vật cản gần
        dw = 180
        urgent_items = [a for a in assessments if a.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM)]
        if urgent_items:
            top_a = max(urgent_items, key=lambda x: (2 if x.risk_level == RiskLevel.HIGH else 1, x.relative_proximity))
            b_color = COLOR_HIGH_RISK if top_a.risk_level == RiskLevel.HIGH else COLOR_MED_RISK
            dir_str = "TRUC DIEN" if top_a.direction == Direction.CENTER else ("BEN TRAI" if top_a.direction == Direction.LEFT else "BEN PHAI")
            banner_msg = f"! CANH BAO: {top_a.class_name.upper()} {top_a.proximity_desc.upper()} ({dir_str})"
            cv2.rectangle(frame, (10, 45), (w - dw - 25, 78), b_color, -1)
            cv2.putText(frame, banner_msg, (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if b_color == COLOR_MED_RISK else COLOR_WHITE, 2, cv2.LINE_AA)

        # 3. Vẽ Mini Depth Map góc phải phía trên
        depth_map = snapshot.get("depth")
        if depth_map is not None:
            dw, dh = 180, 135
            if self._cached_depth_id != depth_map.frame_id or self._cached_depth_mini is None:
                self._cached_depth_id = depth_map.frame_id
                v = depth_map.values
                # Resize trước khi colormap để tăng tốc độ xử lý gấp 16 lần
                v_small = cv2.resize(v, (dw, dh), interpolation=cv2.INTER_NEAREST)
                v_norm = cv2.normalize(v_small, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self._cached_depth_mini = cv2.applyColorMap(v_norm, cv2.COLORMAP_INFERNO)

            # Viền cho mini map
            cv2.rectangle(frame, (w - dw - 15, 45), (w - 15, 45 + dh), (255, 255, 255), 1)
            frame[45:45 + dh, w - dw - 15:w - 15] = self._cached_depth_mini
            cv2.putText(frame, "Depth Map", (w - dw - 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 4. Thanh trạng thái phía trên (Top Header Bar)
        cv2.rectangle(frame, (0, 0), (w, 40), COLOR_DARK_BG, -1)
        mode = snapshot.get("mode", "OBSERVATION")
        metrics = snapshot.get("metrics", {})
        fps = metrics.get("fps", 0.0)
        det_p50 = metrics.get("detection", {}).get("p50", 0.0)
        depth_p50 = metrics.get("depth", {}).get("p50", 0.0)
        speaking = snapshot.get("is_speaking", False)

        status_text = f"CHE DO: {mode}  |  FPS: {fps:.1f}  |  Det: {det_p50:.1f}ms  |  Depth: {depth_p50:.1f}ms"
        cv2.putText(frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        if speaking:
            cv2.putText(frame, "[DANG NOI...]", (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

        # 5. Thanh hướng dẫn phím tắt phía dưới (Bottom Footer Bar)
        cv2.rectangle(frame, (0, h - 30), (w, h), COLOR_DARK_BG, -1)
        help_text = "[SPACE] Doc chu (OCR)    [Q] Mo ta anh (VQA)    [S] Dung doc    [ESC] Thoat"
        cv2.putText(frame, help_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        return frame
