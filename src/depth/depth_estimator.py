import time
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image

from src.contracts.frame_packet import FramePacket
from src.contracts.depth_map import DepthMap, DepthRepresentation

class DepthEstimator:
    """
    Adapter cho mô hình ước lượng độ sâu đơn mục Depth Anything V2 Small.
    Sử dụng thư viện HuggingFace Transformers, chạy tăng tốc trên Apple Silicon MPS.
    Đầu ra tuân thủ nghiêm ngặt hợp đồng DepthMap (relative_depth, near_is_larger=True).
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "mps",
        input_size: Tuple[int, int] = (256, 256)
    ):
        self.model_name = model_name
        self.input_size = input_size
        if device == "mps" and not torch.backends.mps.is_available():
            print("[DepthEstimator] MPS không khả dụng, chuyển sang CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"[DepthEstimator] Đang nạp mô hình {self.model_name} lên {self.device}...")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print("[DepthEstimator] Nạp mô hình Depth Anything V2 thành công.")

    def estimate(self, packet: FramePacket) -> DepthMap:
        """
        Thực hiện ước lượng độ sâu cho FramePacket.
        Trả về DepthMap có cùng kích thước với ảnh gốc.
        """
        t0 = time.monotonic()
        w_orig, h_orig = packet.original_size

        if self._model is None or self._processor is None:
            # Fallback nếu model chưa sẵn sàng
            dummy_vals = np.zeros((h_orig, w_orig), dtype=np.float32)
            dummy_mask = np.zeros((h_orig, w_orig), dtype=bool)
            return DepthMap(
                representation=DepthRepresentation.RELATIVE_DEPTH,
                unit="relative",
                near_is_larger=True,
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

        return DepthMap(
            representation=DepthRepresentation.RELATIVE_DEPTH,
            unit="relative",
            near_is_larger=True, # Depth Anything V2: số lớn hơn = gần hơn
            values=depth_np,
            valid_mask=valid_mask,
            frame_id=packet.frame_id,
            capture_timestamp=packet.timestamp_mono,
            preprocess_transform={
                "original_size": (w_orig, h_orig),
                "model_name": self.model_name
            }
        )
