import time
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from src.contracts.risk import RiskLevel, Direction, RiskAssessment

# Bảng màu chuẩn BGR cho Dark Translucent HUD
COLOR_PANEL_BG = (28, 22, 18)        # Nền tối charcoal
COLOR_PANEL_BORDER = (55, 45, 38)    # Viền panel xám xanh tối
COLOR_TEXT_MAIN = (245, 245, 245)    # Chữ chính sáng rõ
COLOR_TEXT_MUTED = (195, 185, 175)   # Chữ phụ trung tính
COLOR_CYAN = (220, 190, 40)          # Accent cyan BGR
COLOR_HIGH_RISK = (45, 45, 230)      # Đỏ cảnh báo nguy cơ cao
COLOR_MED_RISK = (30, 180, 255)      # Amber cảnh báo chú ý
COLOR_LOW_RISK = (180, 190, 60)      # Teal nguy cơ thấp (không dùng "AN TOAN")
COLOR_ONLINE = (60, 220, 80)         # Xanh lá trạng thái camera online
COLOR_WHITE = (255, 255, 255)
COLOR_BG = (20, 16, 14)              # Nền letterbox

# Bảng chuyển đổi ký tự tiếng Việt có dấu sang ASCII an toàn cho OpenCV Hershey font
_VI_ASCII_TRANS = str.maketrans({
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
    "À": "A", "Á": "A", "Ả": "A", "Ã": "A", "Ạ": "A",
    "Ă": "A", "Ằ": "A", "Ắ": "A", "Ẳ": "A", "Ẵ": "A", "Ặ": "A",
    "Â": "A", "Ầ": "A", "Ấ": "A", "Ẩ": "A", "Ẫ": "A", "Ậ": "A",
    "È": "E", "É": "E", "Ẻ": "E", "Ẽ": "E", "Ẹ": "E",
    "Ê": "E", "Ề": "E", "Ế": "E", "Ể": "E", "Ễ": "E", "Ệ": "E",
    "Ì": "I", "Í": "I", "Ỉ": "I", "Ĩ": "I", "Ị": "I",
    "Ò": "O", "Ó": "O", "Ỏ": "O", "Õ": "O", "Ọ": "O",
    "Ô": "O", "Ồ": "O", "Ố": "O", "Ổ": "O", "Ỗ": "O", "Ộ": "O",
    "Ơ": "O", "Ờ": "O", "Ớ": "O", "Ở": "O", "Ỡ": "O", "Ợ": "O",
    "Ù": "U", "Ú": "U", "Ủ": "U", "Ũ": "U", "Ụ": "U",
    "Ư": "U", "Ừ": "U", "Ứ": "U", "Ử": "U", "Ữ": "U", "Ự": "U",
    "Ỳ": "Y", "Ý": "Y", "Ỷ": "Y", "Ỹ": "Y", "Ỵ": "Y",
    "Đ": "D"
})


class HUDRenderer:
    """
    Giao diện HUD camera chuyên nghiệp (Dark Translucent HUD) cho hệ thống hỗ trợ người khiếm thị.
    Tuân thủ nghiêm ngặt các ràng buộc:
    - Bố cục gọn gàng, không che khuất camera, thẩm mỹ cao.
    - Hiển thị mức nguy cơ bằng cả màu sắc và chữ viết.
    - Tuyệt đối không dùng từ 'AN TOAN' cho LOW_RISK (chỉ dùng 'NGUY CO THAP').
    - Vùng trung tâm mang nhãn 'TRUNG TAM', không dùng 'LOI DI'.
    - Bounding Box và Banner luôn được clamp trong khung hình, không che mini depth map.
    - Hỗ trợ letterbox bảo toàn tỷ lệ khung hình camera gốc.
    """

    def __init__(
        self,
        target_width: int = 800,
        target_height: int = 600,
        config: Optional[Dict[str, Any]] = None
    ):
        self.target_width = max(320, int(target_width))
        self.target_height = max(240, int(target_height))
        self.config = config or {}

        self._cached_depth_id: int = -1
        self._cached_depth_mini: Optional[np.ndarray] = None
        self._start_time = time.monotonic()

    @staticmethod
    def _clean_ascii(text: Any) -> str:
        """Chuẩn hóa chuỗi hiển thị sang ASCII an toàn cho OpenCV Hershey font."""
        if text is None:
            return ""
        return str(text).translate(_VI_ASCII_TRANS)

    @staticmethod
    def _fit_text(
        text: str,
        max_width: int,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        thickness: int = 1
    ) -> str:
        """Cắt ngắn chuỗi an toàn với dấu '...' nếu vượt quá chiều rộng tối đa."""
        if max_width <= 20:
            return ""
        (tw, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        if tw <= max_width:
            return text

        clipped = text
        while len(clipped) > 1:
            clipped = clipped[:-1]
            candidate = clipped + "..."
            (w_cand, _), _ = cv2.getTextSize(candidate, font_face, font_scale, thickness)
            if w_cand <= max_width:
                return candidate
        return "..."

    def _draw_translucent_panel(
        self,
        canvas: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        bg_color: Tuple[int, int, int] = COLOR_PANEL_BG,
        border_color: Optional[Tuple[int, int, int]] = COLOR_PANEL_BORDER,
        alpha: float = 0.78,
        border_thickness: int = 1
    ) -> None:
        """Vẽ panel phủ mờ bán trong suốt cực nhanh bằng ROI alpha-blending."""
        h, w = canvas.shape[:2]
        rx1 = max(0, min(x1, w))
        ry1 = max(0, min(y1, h))
        rx2 = max(0, min(x2, w))
        ry2 = max(0, min(y2, h))

        if rx2 <= rx1 or ry2 <= ry1:
            return

        roi = canvas[ry1:ry2, rx1:rx2]
        patch = np.full_like(roi, bg_color, dtype=np.uint8)
        cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0.0, roi)

        if border_color is not None and border_thickness > 0:
            cv2.rectangle(canvas, (rx1, ry1), (rx2 - 1, ry2 - 1), border_color, border_thickness, cv2.LINE_AA)

    def _draw_status_pill(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int] = COLOR_CYAN,
        bg_color: Tuple[int, int, int] = COLOR_PANEL_BG,
        font_scale: float = 0.42
    ) -> int:
        """Vẽ một status pill nhỏ bo viền và trả về tọa độ x kết thúc."""
        clean_txt = self._clean_ascii(text)
        (tw, th), baseline = cv2.getTextSize(clean_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        pill_w = tw + 14
        pill_h = th + 10
        x2 = x + pill_w
        y2 = y + pill_h

        self._draw_translucent_panel(canvas, x, y, x2, y2, bg_color=bg_color, border_color=color, alpha=0.85, border_thickness=1)
        text_y = y + th + 4
        cv2.putText(canvas, clean_txt, (x + 7, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        return x2

    def render(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """
        Public contract chính của HUDRenderer.
        Nhận snapshot runtime dict và trả về ảnh numpy BGR (target_height, target_width, 3).
        Không mutate dữ liệu đầu vào.
        """
        h, w = self.target_height, self.target_width

        packet = snapshot.get("packet")
        if packet is None or packet.image is None or packet.image.size == 0:
            return self._draw_empty_state(snapshot)

        # 1. Letterbox ảnh camera để bảo toàn tỷ lệ khung hình
        canvas, scale, pad_x, pad_y, scaled_w, scaled_h = self._apply_letterbox(packet.image, w, h)

        # 2. Phân vùng không gian (Spatial Guidance)
        if self.config.get("show_spatial_guides", True):
            self._draw_spatial_guides(canvas, pad_x, pad_y, scaled_w, scaled_h)

        # 3. Khung nhận diện vật thể & Nhãn mức nguy cơ (Bounding boxes & Labels)
        det_res = snapshot.get("detection")
        assessments = snapshot.get("assessments") or []
        self._draw_assessments(canvas, det_res, assessments, packet.original_size, scale, pad_x, pad_y)

        # 4. Banner cảnh báo ưu tiên khẩn cấp (Priority Alert Banner)
        self._draw_alert_banner(canvas, assessments, w)

        # 5. Thẻ chuẩn đoán & Bản đồ độ sâu (Right Diagnostics Card)
        if self.config.get("show_depth_map", True):
            self._draw_depth_card(canvas, snapshot, w, h)

        # 6. Thanh trạng thái đỉnh (Top Status Bar)
        self._draw_header(canvas, snapshot, w)

        # 7. Thanh phím tắt đáy (Bottom Action Bar)
        self._draw_footer(canvas, w, h)

        return canvas

    def _apply_letterbox(
        self,
        image: np.ndarray,
        target_w: int,
        target_h: int
    ) -> Tuple[np.ndarray, float, int, int, int, int]:
        """Tạo canvas letterbox nền tối và vẽ ảnh camera giữ nguyên tỷ lệ."""
        cam_h, cam_w = image.shape[:2]
        if cam_h <= 0 or cam_w <= 0:
            canvas = np.full((target_h, target_w, 3), COLOR_BG, dtype=np.uint8)
            return canvas, 1.0, 0, 0, target_w, target_h

        scale = min(target_w / float(cam_w), target_h / float(cam_h))
        scaled_w = max(1, int(round(cam_w * scale)))
        scaled_h = max(1, int(round(cam_h * scale)))
        pad_x = (target_w - scaled_w) // 2
        pad_y = (target_h - scaled_h) // 2

        canvas = np.full((target_h, target_w, 3), COLOR_BG, dtype=np.uint8)
        resized_cam = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = resized_cam

        return canvas, scale, pad_x, pad_y, scaled_w, scaled_h

    def _draw_header(self, canvas: np.ndarray, snapshot: Dict[str, Any], width: int) -> None:
        """A. Top Status Bar: Logo SECOND EYE, status pill, audio pill, FPS."""
        header_h = 40
        self._draw_translucent_panel(canvas, 0, 0, width, header_h, bg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, alpha=0.88, border_thickness=1)

        # Chấm camera online
        cv2.circle(canvas, (20, 20), 4, COLOR_ONLINE, -1, cv2.LINE_AA)

        # Tên sản phẩm
        cv2.putText(canvas, "SECOND EYE", (32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 2, cv2.LINE_AA)

        # Status Pill cho chế độ hiện tại
        mode_str = snapshot.get("mode", "OBSERVATION")
        mode_pill_map = {
            "OBSERVATION": ("QUAN SAT", COLOR_CYAN),
            "VQA": ("DANG PHAN TICH", COLOR_MED_RISK),
            "OCR": ("DANG DOC CHU", COLOR_MED_RISK),
            "DEGRADED": ("TIN HIEU YEU", COLOR_HIGH_RISK)
        }
        pill_text, pill_color = mode_pill_map.get(mode_str, (self._clean_ascii(mode_str).upper()[:14], COLOR_TEXT_MUTED))
        next_x = self._draw_status_pill(canvas, 140, 8, pill_text, color=pill_color, font_scale=0.40)

        # Pill Đang nói nếu TTS đang phát
        is_speaking = bool(snapshot.get("is_speaking", False))
        if is_speaking:
            self._draw_status_pill(canvas, next_x + 10, 8, "DANG NOI", color=(100, 235, 255), font_scale=0.40)

        # FPS hiển thị góc phải
        metrics = snapshot.get("metrics") or {}
        fps_val = metrics.get("fps", 0.0)
        fps_text = f"FPS {fps_val:.1f}" if fps_val > 0 else "FPS --"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        fps_pill_x = width - tw - 24
        self._draw_status_pill(canvas, fps_pill_x, 8, fps_text, color=COLOR_TEXT_MAIN, font_scale=0.40)

    def _draw_spatial_guides(
        self,
        canvas: np.ndarray,
        pad_x: int,
        pad_y: int,
        scaled_w: int,
        scaled_h: int
    ) -> None:
        """B. Spatial Guidance: Đường phân chia và nhãn TRAI - TRUNG TAM - PHAI."""
        x_left = pad_x + int(round(scaled_w * 0.35))
        x_right = pad_x + int(round(scaled_w * 0.65))
        y_top = 45
        y_bottom = canvas.shape[0] - 42

        # Đường kẻ phân chia vùng mờ tinh tế
        line_color = (65, 55, 45)
        cv2.line(canvas, (x_left, y_top), (x_left, y_bottom), line_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (x_right, y_top), (x_right, y_bottom), line_color, 1, cv2.LINE_AA)

        # Nhãn khu vực ở đáy (Tuyệt đối không dùng 'LOI DI')
        label_y = y_bottom - 6
        font_scale = 0.40

        left_cx = pad_x + int(round(scaled_w * 0.175))
        center_cx = pad_x + int(round(scaled_w * 0.50))
        right_cx = pad_x + int(round(scaled_w * 0.825))

        for text, cx in [("TRAI", left_cx), ("TRUNG TAM", center_cx), ("PHAI", right_cx)]:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(canvas, text, (cx - tw // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_assessments(
        self,
        canvas: np.ndarray,
        detection_res: Any,
        assessments: List[RiskAssessment],
        original_size: Tuple[int, int],
        scale: float,
        pad_x: int,
        pad_y: int
    ) -> None:
        """C. Object Bounding Boxes & Risk Labels."""
        if not detection_res or not hasattr(detection_res, "detections"):
            return

        h, w = canvas.shape[:2]
        orig_w, orig_h = original_size
        if orig_w <= 0 or orig_h <= 0:
            return

        det_map = {}
        for d in detection_res.detections:
            if getattr(d, "track_id", None) is not None:
                det_map[d.track_id] = d

        for a in assessments:
            if not isinstance(a, RiskAssessment):
                continue
            d = det_map.get(a.track_id)
            if d is None or not hasattr(d, "bbox"):
                continue

            # Quy đổi tọa độ từ ảnh gốc sang canvas letterbox
            bx1 = pad_x + int(round((d.bbox.xmin / float(orig_w)) * (orig_w * scale)))
            by1 = pad_y + int(round((d.bbox.ymin / float(orig_h)) * (orig_h * scale)))
            bx2 = pad_x + int(round((d.bbox.xmax / float(orig_w)) * (orig_w * scale)))
            by2 = pad_y + int(round((d.bbox.ymax / float(orig_h)) * (orig_h * scale)))

            # Clamp trong giới hạn frame
            bx1 = max(0, min(bx1, w - 1))
            by1 = max(0, min(by1, h - 1))
            bx2 = max(0, min(bx2, w - 1))
            by2 = max(0, min(by2, h - 1))

            if bx2 <= bx1 or by2 <= by1:
                continue

            # Phân định mức nguy cơ (Màu sắc + Chữ viết; Tuyệt đối KHÔNG dùng 'AN TOAN')
            if a.risk_level == RiskLevel.HIGH:
                color = COLOR_HIGH_RISK
                thickness = 2
                level_str = "NGUY CO CAO"
            elif a.risk_level == RiskLevel.MEDIUM:
                color = COLOR_MED_RISK
                thickness = 2
                level_str = "CHU Y"
            else:
                color = COLOR_LOW_RISK
                thickness = 1
                level_str = "NGUY CO THAP"

            # Vẽ khung chữ nhật nhận diện
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness, cv2.LINE_AA)

            # Vẽ 4 góc điểm nhấn hiện đại
            corner_len = min(12, max(4, (bx2 - bx1) // 5, (by2 - by1) // 5))
            cthick = thickness + 1
            # Top-left
            cv2.line(canvas, (bx1, by1), (bx1 + corner_len, by1), color, cthick)
            cv2.line(canvas, (bx1, by1), (bx1, by1 + corner_len), color, cthick)
            # Top-right
            cv2.line(canvas, (bx2, by1), (bx2 - corner_len, by1), color, cthick)
            cv2.line(canvas, (bx2, by1), (bx2, by1 + corner_len), color, cthick)
            # Bottom-left
            cv2.line(canvas, (bx1, by2), (bx1 + corner_len, by2), color, cthick)
            cv2.line(canvas, (bx1, by2), (bx1, by2 - corner_len), color, cthick)
            # Bottom-right
            cv2.line(canvas, (bx2, by2), (bx2 - corner_len, by2), color, cthick)
            cv2.line(canvas, (bx2, by2), (bx2, by2 - corner_len), color, cthick)

            # Chuẩn bị nội dung nhãn
            obj_name = self._clean_ascii(a.class_name).upper()
            prox_str = self._clean_ascii(a.proximity_desc)
            label_text = f"{obj_name} | {level_str} - {prox_str}"

            font_scale = 0.40
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tag_w = tw + 10
            tag_h = th + 8

            # Kiểm tra xem đặt nhãn phía trên có va chạm với Thẻ Depth (góc trên bên phải) không
            show_depth = self.config.get("show_depth_map", True)
            depth_x1 = w - 186 - 12
            depth_y2 = 48 + 195

            overlaps_depth_card = show_depth and (bx2 >= depth_x1 and (by1 - tag_h - 2) <= depth_y2)

            # Đặt nhãn phía trên box nếu đủ chỗ và không đè lên depth card, ngược lại đặt phía dưới
            if (by1 - tag_h - 2 >= 45) and not overlaps_depth_card:
                tag_x1 = bx1
                tag_y1 = by1 - tag_h - 2
            elif by2 + tag_h + 2 <= h - 42:
                tag_x1 = bx1
                tag_y1 = by2 + 2
            else:
                tag_x1 = bx1 + 2
                tag_y1 = max(45, by1 + 2)

            tag_x1 = max(2, min(tag_x1, w - tag_w - 2))
            tag_x2 = tag_x1 + tag_w
            tag_y2 = tag_y1 + tag_h

            self._draw_translucent_panel(canvas, tag_x1, tag_y1, tag_x2, tag_y2, bg_color=COLOR_PANEL_BG, border_color=color, alpha=0.90, border_thickness=1)
            cv2.putText(canvas, label_text, (tag_x1 + 5, tag_y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    def _draw_alert_banner(self, canvas: np.ndarray, assessments: List[RiskAssessment], width: int) -> None:
        """D. Priority Alert Banner: Chỉ hiển thị cho HIGH hoặc MEDIUM, không che depth card."""
        urgent = [a for a in assessments if isinstance(a, RiskAssessment) and a.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM)]
        if not urgent:
            return

        # Chọn đối tượng có rủi ro cao nhất
        top_a = max(urgent, key=lambda x: (2 if x.risk_level == RiskLevel.HIGH else 1, getattr(x, "relative_proximity", 0.0)))

        is_high = (top_a.risk_level == RiskLevel.HIGH)
        b_color = COLOR_HIGH_RISK if is_high else COLOR_MED_RISK
        prefix = "NGUY CO CAO" if is_high else "CHU Y"

        dir_map = {Direction.CENTER: "TRUNG TAM", Direction.LEFT: "TRAI", Direction.RIGHT: "PHAI"}
        dir_str = dir_map.get(top_a.direction, self._clean_ascii(getattr(top_a.direction, "value", "")))

        obj_name = self._clean_ascii(top_a.class_name).upper()
        prox_str = self._clean_ascii(top_a.proximity_desc).upper()
        raw_msg = f"! {prefix} · {obj_name} · {dir_str} · {prox_str}"

        # Đảm bảo chiều rộng không đè lên thẻ Depth bên phải (chiều rộng thẻ ~190px)
        banner_x1 = 16
        banner_y1 = 48
        max_banner_w = max(100, width - 215 - banner_x1)

        font_scale = 0.46 if is_high else 0.42
        fitted_msg = self._fit_text(raw_msg, max_banner_w - 20, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        (tw, th), _ = cv2.getTextSize(fitted_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        banner_w = min(max_banner_w, tw + 20)
        banner_h = th + 14
        banner_x2 = banner_x1 + banner_w
        banner_y2 = banner_y1 + banner_h

        self._draw_translucent_panel(canvas, banner_x1, banner_y1, banner_x2, banner_y2, bg_color=COLOR_PANEL_BG, border_color=b_color, alpha=0.88, border_thickness=1)
        # Vạch màu cảnh báo bên trái banner
        cv2.rectangle(canvas, (banner_x1, banner_y1), (banner_x1 + 4, banner_y2), b_color, -1)
        cv2.putText(canvas, fitted_msg, (banner_x1 + 10, banner_y1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    def _draw_depth_card(self, canvas: np.ndarray, snapshot: Dict[str, Any], width: int, height: int) -> None:
        """E. Right Diagnostics Card: Thẻ góc phải chứa Mini Depth Map và Latencies."""
        card_w = 186
        card_h = 195
        card_x1 = width - card_w - 12
        card_y1 = 48
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h

        # Nền card bán trong suốt
        self._draw_translucent_panel(canvas, card_x1, card_y1, card_x2, card_y2, bg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, alpha=0.82, border_thickness=1)

        # Tiêu đề Card
        cv2.putText(canvas, "DEPTH MAP", (card_x1 + 10, card_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)

        # Mini depth map
        dw, dh = 166, 115
        mx1 = card_x1 + 10
        my1 = card_y1 + 24
        mx2 = mx1 + dw
        my2 = my1 + dh

        depth_map = snapshot.get("depth")
        if depth_map is not None and getattr(depth_map, "values", None) is not None:
            frame_id = getattr(depth_map, "frame_id", -1)
            if self._cached_depth_id != frame_id or self._cached_depth_mini is None:
                self._cached_depth_id = frame_id
                v = depth_map.values
                v_small = cv2.resize(v, (dw, dh), interpolation=cv2.INTER_NEAREST)
                v_norm = cv2.normalize(v_small, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self._cached_depth_mini = cv2.applyColorMap(v_norm, cv2.COLORMAP_INFERNO)

            canvas[my1:my2, mx1:mx2] = self._cached_depth_mini
            cv2.rectangle(canvas, (mx1, my1), (mx2, my2), COLOR_PANEL_BORDER, 1)
        else:
            # Trạng thái chưa có depth
            cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (32, 26, 22), -1)
            cv2.rectangle(canvas, (mx1, my1), (mx2, my2), COLOR_PANEL_BORDER, 1)
            cv2.putText(canvas, "DEPTH --", (mx1 + dw // 2 - 30, my1 + dh // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Thống kê hiệu năng (Det / Depth latencies)
        metrics = snapshot.get("metrics") or {}
        det_p50 = metrics.get("detection", {}).get("p50", 0.0)
        depth_p50 = metrics.get("depth", {}).get("p50", 0.0)

        t_det = f"Det P50  : {det_p50:.1f}ms" if det_p50 > 0 else "Det P50  : --"
        t_depth = f"Depth P50: {depth_p50:.1f}ms" if depth_p50 > 0 else "Depth P50: --"

        stat_y = my2 + 18
        cv2.putText(canvas, t_det, (mx1, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, t_depth, (mx1, stat_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray, width: int, height: int) -> None:
        """F. Bottom Action Bar: Keycaps hiển thị phím tắt gọn gàng, vừa vặn."""
        footer_h = 36
        y1 = height - footer_h
        self._draw_translucent_panel(canvas, 0, y1, width, height, bg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, alpha=0.88, border_thickness=1)

        shortcuts = [
            ("Q", "Hoi canh vat"),
            ("SPACE", "Doc chu"),
            ("S", "Dung doc"),
            ("ESC", "Thoat")
        ]
        if width < 720:
            shortcuts = [
                ("Q", "VQA"),
                ("SPACE", "OCR"),
                ("S", "Dung"),
                ("ESC", "Thoat")
            ]

        curr_x = 16
        for key, desc in shortcuts:
            # Keycap viền nhỏ
            (kw, _), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            k_box_w = kw + 10
            k_box_h = 20
            k_y1 = y1 + 8
            k_y2 = k_y1 + k_box_h

            if curr_x + k_box_w + 80 > width:
                break

            cv2.rectangle(canvas, (curr_x, k_y1), (curr_x + k_box_w, k_y2), (55, 45, 38), -1)
            cv2.rectangle(canvas, (curr_x, k_y1), (curr_x + k_box_w, k_y2), COLOR_CYAN, 1)
            cv2.putText(canvas, key, (curr_x + 5, k_y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)

            curr_x += k_box_w + 6
            clean_desc = self._clean_ascii(desc)
            cv2.putText(canvas, clean_desc, (curr_x, k_y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            (dw, _), _ = cv2.getTextSize(clean_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            curr_x += dw + 18

    def _draw_empty_state(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """G. Empty / Loading State khi chưa có camera packet."""
        w, h = self.target_width, self.target_height
        canvas = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)

        # Thẻ trung tâm
        card_w, card_h = min(420, w - 40), 170
        cx, cy = w // 2, h // 2
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = x1 + card_w, y1 + card_h

        self._draw_translucent_panel(canvas, x1, y1, x2, y2, bg_color=COLOR_PANEL_BG, border_color=COLOR_PANEL_BORDER, alpha=0.92, border_thickness=1)

        # Chấm pulsing dựa trên thời gian
        elapsed = time.monotonic() - self._start_time
        num_dots = int((elapsed * 2.0) % 4)
        dots = "." * num_dots

        # Tiêu đề & phụ đề
        (tw1, _), _ = cv2.getTextSize("SECOND EYE", cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(canvas, "SECOND EYE", (cx - tw1 // 2, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_WHITE, 2, cv2.LINE_AA)

        sub = "He Thong AI Ho Tro Nguoi Khiem Thi"
        (tw2, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(canvas, sub, (cx - tw2 // 2, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        status_msg = f"Dang ket noi camera{dots}"
        (tw3, _), _ = cv2.getTextSize("Dang ket noi camera...", cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.putText(canvas, status_msg, (cx - tw3 // 2, cy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN, 1, cv2.LINE_AA)

        # Header và Footer vẫn hiển thị
        self._draw_header(canvas, snapshot, w)
        self._draw_footer(canvas, w, h)

        return canvas
