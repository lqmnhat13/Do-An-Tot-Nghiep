#!/usr/bin/env python3
"""Sinh bộ ảnh văn bản tiếng Việt MÔ PHỎNG ảnh chụp điện thoại (dữ liệu tổng hợp).

Dùng để bổ sung cho VinText (chủ yếu là chữ cảnh/biển hiệu): kiểm tra OCR với trang
văn bản in nhiều dấu thanh. Đây KHÔNG phải ảnh chụp thật; kết quả chỉ mang tính bổ sung.

Mỗi ảnh có một điều kiện chụp: clean, perspective, shadow, blur, noise_jpeg, camera_640x480.
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["Arial.ttf", "Times New Roman.ttf", "Verdana.ttf", "Tahoma.ttf", "Georgia.ttf", "Courier New.ttf"]
FONT_DIR = "/System/Library/Fonts/Supplemental"

TEXTS = [
    "THÔNG BÁO\nKể từ ngày 15 tháng 10, thư viện mở cửa từ 7 giờ 30 đến 21 giờ.\nBạn đọc vui lòng xuất trình thẻ khi vào cửa.",
    "HƯỚNG DẪN SỬ DỤNG\nUống 1 viên sau bữa ăn, ngày 2 lần.\nKhông dùng cho phụ nữ có thai.\nBảo quản nơi khô ráo, tránh ánh nắng.",
    "Phòng khám Đa khoa Hoà Bình\nĐịa chỉ: 25 đường Nguyễn Trãi, quận Thanh Xuân, Hà Nội\nĐiện thoại: 024 3856 7890",
    "THỰC ĐƠN HÔM NAY\nPhở bò tái chín 45.000 đồng\nBún chả Hà Nội 40.000 đồng\nCơm rang dưa bò 50.000 đồng\nTrà đá miễn phí",
    "Lịch học tuần này\nThứ hai: Toán cao cấp, phòng 302\nThứ tư: Vật lý đại cương, phòng 105\nThứ sáu: Tiếng Anh chuyên ngành",
    "CẢNH BÁO\nSàn ướt, trơn trượt.\nXin vui lòng đi chậm và bám tay vịn cầu thang.",
    "Hoá đơn bán lẻ\nSữa tươi không đường 2 hộp 64.000\nBánh mì sandwich 1 gói 28.000\nTổng cộng 92.000 đồng",
    "Giấy mời họp phụ huynh\nThời gian: 8 giờ sáng chủ nhật ngày 20 tháng 10\nĐịa điểm: hội trường trường Tiểu học Kim Đồng",
    "Hạn sử dụng: 12/2026\nThành phần: nước, đường, hương cam tự nhiên.\nSản xuất tại Việt Nam.",
    "Quý khách vui lòng xếp hàng theo thứ tự.\nNhân viên sẽ hỗ trợ trong giây lát.\nXin cảm ơn!",
]
CONDITIONS = ["clean", "perspective", "shadow", "blur", "noise_jpeg", "camera_640x480"]


def render(text, font_path, size):
    font = ImageFont.truetype(font_path, size)
    lines = text.split("\n")
    pad = size
    widths = [font.getbbox(line)[2] for line in lines]
    w = max(widths) + 2 * pad
    h = int(len(lines) * size * 1.6) + 2 * pad
    img = Image.new("RGB", (w, h), (245, 242, 235))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((pad, pad + int(i * size * 1.6)), line, fill=(25, 25, 25), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def place_on_background(doc, rng, out_w=1280, out_h=960):
    bg = np.full((out_h, out_w, 3), (90, 110, 130), np.uint8)
    bg = cv2.add(bg, rng.integers(0, 25, bg.shape, dtype=np.uint8))
    scale = min(out_w * 0.85 / doc.shape[1], out_h * 0.85 / doc.shape[0])
    doc = cv2.resize(doc, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y, x = (out_h - doc.shape[0]) // 2, (out_w - doc.shape[1]) // 2
    bg[y:y + doc.shape[0], x:x + doc.shape[1]] = doc
    return bg


def apply_condition(img, cond, rng):
    h, w = img.shape[:2]
    if cond == "perspective":
        d = 0.12
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * d, h * 0.05], [w * (1 - d * 0.3), 0], [w, h], [0, h * (1 - 0.04)]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 110, 130))
    elif cond == "shadow":
        grad = np.tile(np.linspace(0.45, 1.0, w, dtype=np.float32), (h, 1))[..., None]
        img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    elif cond == "blur":
        img = cv2.GaussianBlur(img, (0, 0), 1.6)
    elif cond == "noise_jpeg":
        img = np.clip(img.astype(np.float32) + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 35])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    pyrng = random.Random(0)
    meta = []
    for ti, text in enumerate(TEXTS):
        for cond in CONDITIONS:
            font = pyrng.choice(FONTS)
            doc = render(text, os.path.join(FONT_DIR, font), 22 if cond == "camera_640x480" else 44)
            img = place_on_background(doc, rng)
            if cond == "camera_640x480":
                # Độ phân giải camera của app (configs/app_config.yaml), chữ nhỏ hơn.
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            img = apply_condition(img, cond, rng)
            name = f"doc{ti:02d}_{cond}.jpg"
            cv2.imwrite(os.path.join(args.out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.append({"image": name, "condition": cond, "font": font, "text": text})
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(meta)} images to {args.out}")


if __name__ == "__main__":
    main()
