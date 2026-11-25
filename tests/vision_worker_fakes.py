"""Reader giả cho tiến trình con Vision trong test (không phải file test: discover bỏ qua)."""
import os
import time


class EchoReader:
    def __init__(self, languages):
        self.languages = languages

    def readtext(self, image):
        h, w = image.shape[:2]
        return [([[0, 0], [w, 0], [w, h], [0, h]], f"{w}x{h} {os.getpid()}", 0.9)]


class HangReader(EchoReader):
    def readtext(self, image):
        time.sleep(60)


class HangFirstProcessReader(EchoReader):
    """Treo ở tiến trình đầu tiên (đánh dấu bằng file), các tiến trình sau đọc bình thường."""

    def readtext(self, image):
        marker = os.environ["VISION_FAKE_MARKER"]
        if not os.path.exists(marker):
            open(marker, "w").close()
            time.sleep(60)
        return super().readtext(image)


class BrokenInitReader:
    def __init__(self, languages):
        raise ImportError("No module named 'Vision'")


class ErrorReader(EchoReader):
    def readtext(self, image):
        raise ValueError("NSInvalidArgumentException - key does not exist")
