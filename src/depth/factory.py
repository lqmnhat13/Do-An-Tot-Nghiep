from typing import Any, Dict

from src.depth.depth_estimator import DepthEstimator
from src.depth.roi_extractor import ROIExtractor, metric_breakpoints

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def build_depth_estimator(model_cfg: Dict[str, Any], device: str) -> DepthEstimator:
    """Dựng DepthEstimator từ khối `depth` của configs/model_config.yaml."""
    depth_cfg = model_cfg.get("depth", {})
    return DepthEstimator(
        model_name=depth_cfg.get("hf_repo_id", DEFAULT_DEPTH_MODEL),
        device=device,
        input_size=(int(depth_cfg.get("input_height", 256)), int(depth_cfg.get("input_width", 256))),
        representation=depth_cfg.get("representation", "relative_depth"),
        metric_scale=float(depth_cfg.get("metric_scale", 1.0)),
    )


def build_roi_extractor(model_cfg: Dict[str, Any], fusion_cfg: Dict[str, Any]) -> ROIExtractor:
    """Dựng ROIExtractor; ngưỡng mét được neo vào ngưỡng điểm của RiskFSM."""
    depth_cfg = model_cfg.get("depth", {})
    fsm_cfg = fusion_cfg.get("risk_fsm", {})
    breakpoints = metric_breakpoints(
        high_m=float(fsm_cfg.get("distance_threshold_high_m", 1.5)),
        medium_m=float(fsm_cfg.get("distance_threshold_medium_m", 2.5)),
        low_m=float(fsm_cfg.get("distance_threshold_low_m", 3.5)),
        score_high=float(fsm_cfg.get("depth_threshold_high", 0.75)),
        score_medium=float(fsm_cfg.get("depth_threshold_medium", 0.45)),
        score_low=float(fsm_cfg.get("depth_threshold_low", 0.25)),
    )
    return ROIExtractor(
        erosion_ratio=float(depth_cfg.get("roi_erosion_ratio", 0.15)),
        percentile=float(depth_cfg.get("percentile_roi", 80.0)),
        metric_breakpoints=breakpoints,
    )
