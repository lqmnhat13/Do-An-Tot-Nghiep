#!/usr/bin/env python3
"""VizWiz-VQA (ảnh và câu hỏi thật của người khiếm thị) qua VQAService + OCRService.

Nguồn: VizWiz (Gurari et al., CVPR 2018), bản validation trên Hugging Face
`Multimodal-Fatima/VizWiz_validation` (shard 0, 864 câu). Lấy mẫu cố định (seed 7):
45 câu trả lời được + 15 câu "unanswerable"; giữ 34 câu thuộc phạm vi đề tài (thuốc,
đồ uống, sách, biển, thực đơn, mô tả vật, màn hình) và dịch câu hỏi sang tiếng Việt.
Đáp án gốc tiếng Anh được lưu để CHẤM TAY (đánh giá chéo ngôn ngữ, không chấm tự động).
Câu hỏi đọc chữ cũng được chạy qua OCR để so sánh.

Chạy: python eval_vizwiz.py --parquet $W/data/vizwiz/val0.parquet --out vizwiz.json
"""
import argparse
import ast
import json
import random
import time
from collections import Counter

import cv2
import numpy as np

from common import load_configs, save_json

VI = {
    1: "Đây là thuốc gì?", 2: "Tên sản phẩm này là gì?", 4: "Đây là cửa hàng nào?",
    8: "Đồ uống này có caffeine hay không?", 9: "Đây là thuốc gì?",
    10: "Trang giấy này trông như thế nào?", 11: "Trên vỏ gối có gì?",
    14: "Thực đơn này có món bít tết không?", 19: "Bầu trời trong ảnh trông như thế nào?",
    20: "Tên cuốn sách này là gì?", 22: "Đó là cái gì?", 25: "Khoai tây chiên giá bao nhiêu?",
    27: "Trong ảnh có cái chai nào không? Nếu có thì là chai gì?", 29: "Đây là cái gì?",
    30: "Tấm biển ghi gì?", 35: "Đây là loại nước ngọt gì?", 40: "Màn hình này ghi gì?",
    42: "Dòng chữ này ghi gì?", 43: "Đây là cái gì?", 44: "Đây là đồ uống gì?",
    45: "Chữ viết ghi gì?", 46: "Bên này của xe có chữ không?", 47: "Vé số nào có hai dãy số?",
    48: "Cái này ghi gì?", 49: "Đây là phần mềm gì? Đọc số serial nếu thấy.",
    50: "Chữ trên màn hình đã thay đổi chưa?", 51: "Những chữ cái đó là gì?",
    52: "Trong xô này có gì?", 53: "Cái này ghi gì?", 54: "Màn hình máy tính đang hiện thông báo gì?",
    55: "Số model của máy tính là gì?", 56: "Đây là loại súp gì?", 58: "Màn hình của tôi có gì?",
    59: "Cái này ghi gì?",
}
READING = {2, 20, 30, 40, 42, 45, 48, 51, 53, 59}


def sample(parquet):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet, columns=["filename", "image", "question", "answers",
                                           "answer_type"]).to_pylist()
    random.seed(7)
    ans = [r for r in rows if r["answer_type"] != "unanswerable"]
    una = [r for r in rows if r["answer_type"] == "unanswerable"]
    picked = random.sample(ans, 45) + random.sample(una, 15)
    for r in picked:
        a = r["answers"] if isinstance(r["answers"], list) else ast.literal_eval(r["answers"])
        r["top"] = Counter(a).most_common(3)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="vizwiz.json")
    args = ap.parse_args()
    cfg = load_configs()["model"]
    from src.contracts.request import OCRRequest, VQARequest
    from src.ocr.image_quality import ImageQualityChecker
    from src.ocr.ocr_service import OCRService
    from src.vqa.vqa_service import VQAService
    v, o = cfg["vqa"], cfg["ocr"]
    vqa = VQAService(device="mps", backend_name=v["backend"], mlx_vlm_config=v["mlx_vlm"])
    ocr = OCRService(languages=o["languages"], engine=o["engine"], min_confidence=o["min_confidence"],
                     quality_checker=ImageQualityChecker(o["blur_laplacian_threshold"],
                                                         o["min_brightness"], o["max_brightness"]))
    items = sample(args.parquet)
    rows = []
    for idx, q in VI.items():
        item = items[idx]
        img = cv2.imdecode(np.frombuffer(item["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        raw, orig = {}, vqa.backend.answer

        def spy(image, question):
            try:
                raw["text"] = orig(image, question)
                return raw["text"]
            except Exception as exc:
                raw["error"] = str(exc)
                raise
        vqa.backend.answer = spy
        t = time.monotonic()
        res = vqa.answer(VQARequest(request_id=str(idx), image=img, question=q))
        vqa.backend.answer = orig
        row = {"idx": idx, "file": item["filename"], "answer_type": item["answer_type"],
               "question_en": item["question"], "question_vi": q, "gt_top3": item["top"],
               "answer": res.answer, "backend_raw": raw.get("text"),
               "backend_error": raw.get("error"), "latency_sec": round(time.monotonic() - t, 2)}
        if idx in READING:
            r = ocr.process(OCRRequest(request_id=str(idx), image=img))
            row["ocr_text"], row["ocr_quality"] = r.text[:400], r.quality_status
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    path = save_json(args.out, {
        "dataset": "VizWiz-VQA validation shard 0 (Multimodal-Fatima/VizWiz_validation), seed 7",
        "note": "Câu hỏi dịch sang tiếng Việt; cột manual_grade cần chấm tay theo gt_top3.",
        "n": len(rows), "rows": rows})
    print("saved", path)


if __name__ == "__main__":
    main()
