"""Nhận diện mệnh giá tiền Việt Nam từ chữ OCR: không bao giờ đoán khi bằng chứng yếu/mâu thuẫn."""
import threading
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.tts_engine import TTSEngine
from src.contracts.frame_packet import FramePacket
from src.contracts.request import OCRResult
from src.currency.currency_service import (
    MIXED_TEXT, UNCLEAR_TEXT, CurrencyService, parse_banknote_text, spoken_text)
from src.fusion.alert_aggregator import AlertAggregator
from src.runtime.system_coordinator import MODE_ANNOUNCEMENTS, SystemCoordinator
from src.vqa.safety import SafetyGuardrail


class TestParseBanknoteText(unittest.TestCase):
    def assertReads(self, text, denomination):
        reading = parse_banknote_text(text)
        self.assertEqual(reading.status, "OK", f"{text!r} -> {reading}")
        self.assertEqual(reading.denomination, denomination)

    def assertUnclear(self, text):
        reading = parse_banknote_text(text)
        self.assertEqual(reading.status, "UNCLEAR", f"{text!r} -> {reading}")
        self.assertIsNone(reading.denomination)

    # Chuỗi Vision đọc thật từ ảnh chụp tiền (Wikimedia Commons), giữ nguyên lỗi dấu/ký tự.
    def test_real_vision_outputs(self):
        # Tờ 100.000 mặt trước, khung 640x480: chữ mệnh giá bị đọc thiếu, chỉ còn số + dấu hiệu tờ tiền.
        self.assertReads("CỘNG HOA XÃ HỘI CHỦ NGHĨA VIỆT NAM 100000 MỘI TR NGHIN WD 20123707", 100000)
        self.assertReads("302012902 CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM 100.000 MỘT TR NGHÌN L100 WD 20123707",
                         100000)
        # Tờ 200.000: chữ sai dấu ("ĐÓNG", "TRAM").
        self.assertReads("200.000 HAI TRAM NGHÌN ĐÓNG 200 00/", 200000)
        # Xấp tờ 500.000 mặt sau: có chữ in ngược "A00005", "5000m" (không được thành 5.000).
        self.assertReads("NĂM TRĂM NGHÌN ĐỒNG NĂM TRĂM NGHÌN ĐỒNG A00005 ĐNỌG NỊHON TA TAN 5000m "
                         "NĂM TRĂM NGHÌN ĐÔNG NGÂN HÀNG NHÀ NƯỚC VIỆT NAM 500000 NĂM TRĂM NGHÌN ĐỒNG", 500000)

    def test_every_denomination_by_words(self):
        cases = {"MỘT NGHÌN ĐỒNG": 1000, "HAI NGHÌN ĐỒNG": 2000, "NĂM NGHÌN ĐỒNG": 5000,
                 "MƯỜI NGHÌN ĐỒNG": 10000, "HAI MƯƠI NGHÌN ĐỒNG": 20000, "NĂM MƯƠI NGHÌN ĐỒNG": 50000,
                 "MỘT TRĂM NGHÌN ĐỒNG": 100000, "HAI TRĂM NGHÌN ĐỒNG": 200000,
                 "NĂM TRĂM NGHÌN ĐỒNG": 500000, "HAI MƯƠI NGÀN ĐỒNG": 20000}
        for text, value in cases.items():
            with self.subTest(text=text):
                self.assertReads(text, value)

    def test_numbers_need_banknote_context(self):
        self.assertReads("NGÂN HÀNG NHÀ NƯỚC VIỆT NAM 50.000", 50000)
        self.assertReads("50 000 ĐỒNG", 50000)
        # Bảng giá / số đứng một mình: không coi là tờ tiền.
        self.assertUnclear("100.000")
        self.assertUnclear("Giá 100.000đ")
        self.assertUnclear("Áo thun 200.000 VNĐ")

    def test_small_numbers_alone_need_two_sightings(self):
        # Số nhỏ dễ trùng năm in/ký hiệu: một lần thấy chưa đủ, hai lần hoặc có chữ mệnh giá thì được.
        self.assertUnclear("NGÂN HÀNG NHÀ NƯỚC VIỆT NAM 2000")
        self.assertReads("NGÂN HÀNG NHÀ NƯỚC VIỆT NAM 2000 2000", 2000)
        self.assertReads("2000 HAI NGHÌN ĐỒNG", 2000)

    def test_viet_nam_and_ngan_hang_are_not_denomination_words(self):
        self.assertUnclear("NGÂN HÀNG NHÀ NƯỚC VIỆT NAM NGHÌN ĐỒNG")
        self.assertUnclear("VIET NAM NGHIN DONG")

    def test_serials_and_glued_numbers_ignored(self):
        self.assertUnclear("QB 20000123 ĐỒNG")
        self.assertUnclear("ĐỒNG 5000m A00005 L100 200 00/")
        self.assertUnclear("ĐỒNG 1000000")  # không phải mệnh giá đang lưu hành
        self.assertUnclear("ĐỒNG 5000~ 5000- 5000/ /5000")  # mảnh số bị che, dính ký hiệu

    def test_stack_of_notes_with_truncated_numbers_is_mixed(self):
        """Chuỗi OCRService đọc thật từ ảnh xấp tờ 500.000 xoè: có mảnh số bị cắt ("50000.")."""
        text = ("A00005. 5000m NGHÌN ĐÔNG NĂM TRĂM NGHÌN ĐỒNG NĂM TRĂM. NGÂN HÀNG NHÀ NƯỚC VIỆT NAM. "
                "500000. 500000 5000~. 000S anos. 500 500. 900005. 1000002 NĂM TRĂM NGHÌN ĐỒNG. "
                "50000. NĂM TRĂM NGHÌN ĐỒNG. 5000-. 000005.")
        reading = parse_banknote_text(text)
        self.assertEqual(reading.status, "MIXED")  # an toàn: bảo đưa từng tờ, không đoán
        self.assertEqual(reading.denominations, (50000, 500000))

    def test_conflicting_denominations_are_mixed(self):
        reading = parse_banknote_text("MỘT TRĂM NGHÌN ĐỒNG HAI TRĂM NGHÌN ĐỒNG")
        self.assertEqual(reading.status, "MIXED")
        self.assertEqual(reading.denominations, (100000, 200000))
        self.assertIsNone(reading.denomination)
        self.assertEqual(parse_banknote_text("NĂM TRĂM NGHÌN ĐỒNG 5000 5000").status, "MIXED")

    def test_nothing_readable(self):
        for text in ("", "Không phát hiện thấy chữ trong ảnh.", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"):
            self.assertUnclear(text)

    def test_spoken_text(self):
        self.assertEqual(spoken_text(parse_banknote_text("MỘT TRĂM NGHÌN ĐỒNG")), "Một trăm nghìn đồng.")
        self.assertEqual(spoken_text(parse_banknote_text("")), UNCLEAR_TEXT)
        self.assertEqual(spoken_text(parse_banknote_text("MỘT NGHÌN ĐỒNG HAI NGHÌN ĐỒNG")), MIXED_TEXT)
        self.assertNotIn("100", " ".join(CurrencyService.spoken_texts()))  # đọc bằng chữ, dựng sẵn clip


class TestCurrencyService(unittest.TestCase):
    def test_uses_ocr_text_and_ignores_failed_reads(self):
        ocr = MagicMock()
        ocr.process.return_value = OCRResult("r", True, "HAI TRĂM NGHÌN ĐỒNG", latency_sec=.2)
        service = CurrencyService(ocr)
        result = service.recognize(np.zeros((8, 8, 3), np.uint8), request_id="r")
        self.assertEqual(result.text, "Hai trăm nghìn đồng.")
        self.assertEqual(result.reading.denomination, 200000)
        # OCR lỗi/chất lượng kém: thông báo của OCR không được phân tích như chữ trên tiền.
        ocr.process.return_value = OCRResult("r", False, "Ảnh quá tối 100.000 ĐỒNG", latency_sec=.1)
        self.assertEqual(service.recognize(np.zeros((8, 8, 3), np.uint8)).text, UNCLEAR_TEXT)
        ocr.process.return_value = OCRResult("r", False, "Đã huỷ đọc chữ.", latency_sec=.1,
                                             error_message="cancelled")
        self.assertTrue(service.recognize(np.zeros((8, 8, 3), np.uint8)).cancelled)


class TestMoneyMode(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock(spec=TTSEngine)
        self.engine.is_speaking.return_value = False
        self.audio = AudioCoordinator(self.engine)
        self.audio.wait_until_idle = MagicMock(return_value=True)
        self.ocr = MagicMock()
        self.release = threading.Event()
        self.entered = threading.Event()

        def ocr_process(req):
            self.entered.set()
            self.release.wait(timeout=5)
            return OCRResult(req.request_id, True, "NĂM MƯƠI NGHÌN ĐỒNG", latency_sec=.1)
        self.ocr.process.side_effect = ocr_process
        camera = MagicMock()
        camera.get_latest_frame.side_effect = lambda: FramePacket(
            1, "t", np.zeros((8, 8, 3), np.uint8), (8, 8), time.monotonic())
        self.system = SystemCoordinator(camera, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                        MagicMock(), AlertAggregator(MagicMock()), self.audio,
                                        self.ocr, MagicMock())

    def tearDown(self):
        self.release.set()
        self.system.stop()

    def texts(self):
        return [e.task.text for e in list(self.audio._queue.queue)]

    def test_money_key_reads_denomination_with_alerts_muted(self):
        self.assertTrue(self.system.trigger_currency())
        self.assertTrue(self.entered.wait(timeout=2))
        snapshot = self.system.get_runtime_snapshot()
        self.assertEqual(snapshot["selected_mode"], "MONEY")
        self.assertFalse(snapshot["safety_alerts_enabled"])
        self.release.set()
        self.system._on_demand_thread.join(timeout=2)
        texts = self.texts()
        self.assertEqual(texts[0], MODE_ANNOUNCEMENTS["MONEY"])
        self.assertIn("Năm mươi nghìn đồng.", texts)
        # Kết thúc tác vụ vẫn giữ chế độ tiền; C mới bật lại cảnh báo.
        self.assertEqual(self.system.get_runtime_snapshot()["selected_mode"], "MONEY")

    def test_c_cancels_money_read(self):
        self.assertTrue(self.system.trigger_currency())
        self.assertTrue(self.entered.wait(timeout=2))
        self.system.enter_warning_mode()
        self.ocr.cancel.assert_called()
        self.release.set()
        self.system._on_demand_thread.join(timeout=2)
        self.assertNotIn("Năm mươi nghìn đồng.", self.texts())

    def test_set_on_demand_active_accepts_money(self):
        self.system.set_on_demand_active(True, "MONEY")
        self.assertEqual(self.system.get_runtime_snapshot()["selected_mode"], "MONEY")


class TestMoneyGuardrail(unittest.TestCase):
    def test_money_questions_redirected_to_money_key(self):
        guard = SafetyGuardrail()
        for q in ("Tờ tiền này bao nhiêu?", "Đây là tờ tiền mệnh giá bao nhiêu?",
                  "Tờ này có phải tiền giả không?", "Tiền này thật hay giả?", "Mệnh giá là bao nhiêu?"):
            with self.subTest(q=q):
                self.assertEqual(guard.check(q), SafetyGuardrail.MONEY_REDIRECT)
        for q in ("Hóa đơn tiền điện bao nhiêu?", "Món này bao nhiêu tiền?", "Phía trước có gì?"):
            with self.subTest(q=q):
                self.assertIsNone(guard.check(q))


if __name__ == "__main__":
    unittest.main()
