import time
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image

from src.contracts.frame_packet import FramePacket
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.runtime.model_loading import offline_load_error, pretrained_kwargs

class DepthEstimator:
    """
    Adapter cho mô hình ước lượng độ sâu đơn mục Depth Anything V2 Small.
    Sử dụng thư viện HuggingFace Transformers, chạy tăng tốc trên Apple Silicon MPS.
    Đầu ra tuân thủ nghiêm ngặt hợp đồng DepthMap:
    - relative_depth: giá trị tương đối, near_is_larger=True (model gốc).
    - metric_depth: mét, near_is_larger=False (model Metric-Indoor). Giá trị được nhân
      metric_scale vì thang mét của model lệch theo camera và độ phân giải đầu vào
      (đo trên NYU ở 256px: model ước lượng xa gấp ~1.66 lần) -> cần hiệu chỉnh.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "mps",
        input_size: Tuple[int, int] = (256, 256),
        representation: str = "relative_depth",
        metric_scale: float = 1.0
    ):
        self.model_name = model_name
        self.input_size = input_size
        self.representation = DepthRepresentation(representation)
        if self.representation == DepthRepresentation.INVERSE_DEPTH:
            raise ValueError("DepthEstimator chỉ hỗ trợ relative_depth hoặc metric_depth")
        self.metric_scale = float(metric_scale)
        if device == "mps" and not torch.backends.mps.is_available():
            print("[DepthEstimator] MPS không khả dụng, chuyển sang CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._processor = None
        self._load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"[DepthEstimator] Đang nạp mô hình {self.model_name} lên {self.device}...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            load_kwargs = pretrained_kwargs()
            self._processor = AutoImageProcessor.from_pretrained(self.model_name, **load_kwargs)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name, **load_kwargs)
            self._model.to(self.device)
            self._model.eval()
            self._load_error = None
            print("[DepthEstimator] Nạp mô hình Depth Anything V2 thành công.")
        except Exception as exc:
            self._processor = None
            self._model = None
            self._load_error = offline_load_error("Depth Anything V2", self.model_name, exc)
            print(f"[DepthEstimator] {self._load_error} Dùng fallback depth không khả dụng.")

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def estimate(self, packet: FramePacket) -> DepthMap:
        """
        Thực hiện ước lượng độ sâu cho FramePacket.
        Trả về DepthMap có cùng kích thước với ảnh gốc.
        """
        t0 = time.monotonic()
        w_orig, h_orig = packet.original_size
        is_metric = self.representation == DepthRepresentation.METRIC_DEPTH

        if self._model is None or self._processor is None:
            # Fallback nếu model chưa sẵn sàng
            dummy_vals = np.zeros((h_orig, w_orig), dtype=np.float32)
            dummy_mask = np.zeros((h_orig, w_orig), dtype=bool)
            return DepthMap(
                representation=self.representation,
                unit="meter" if is_metric else "relative",
                near_is_larger=not is_metric,
                values=dummy_vals,
                valid_mask=dummy_mask,
                frame_id=packet.frame_id,
                capture_timestamp=packet.timestamp_mono
            )

        # Chuyển đổi BGR sang RGB cho PIL
        rgb_img = Image.fromarray(packet.image[:, :, ::-1])
        inputs = self._processor(
            images=rgb_img,
            return_tensors="pt",
            size={"height": self.input_size[0], "width": self.input_size[1]}
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate về đúng kích thước ảnh gốc qua bilinear
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h_orig, w_orig),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth_np = prediction.cpu().numpy().astype(np.float32)

        # Kiểm tra valid mask (loại trừ NaN, Inf)
        valid_mask = np.isfinite(depth_np) & (~np.isnan(depth_np))
        if is_metric:
            depth_np = depth_np * self.metric_scale
            valid_mask &= depth_np > 0.0

        return DepthMap(
            representation=self.representation,
            unit="meter" if is_metric else "relative",
            # Relative (Depth Anything V2): lớn hơn = gần hơn; metric: mét, nhỏ hơn = gần hơn
            near_is_larger=not is_metric,
            values=depth_np,
            valid_mask=valid_mask,
            frame_id=packet.frame_id,
            capture_timestamp=packet.timestamp_mono,
            preprocess_transform={
                "original_size": (w_orig, h_orig),
                "model_name": self.model_name,
                "metric_scale": self.metric_scale if is_metric else None
            }
        )
