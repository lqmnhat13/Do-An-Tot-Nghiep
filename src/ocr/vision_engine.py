"""Adapter for Apple's on-device Vision text recognition."""

import cv2
import numpy as np


class VisionOCRReader:
    def __init__(self, languages):
        import Foundation
        import Vision

        self._foundation = Foundation
        self._vision = Vision
        language_codes = {"vi": "vi-VT", "en": "en-US"}
        self.languages = [language_codes.get(language, language) for language in languages]

    def readtext(self, image: np.ndarray):
        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise ValueError("Không thể mã hóa ảnh để đọc chữ bằng Vision")

        data = self._foundation.NSData.dataWithBytes_length_(
            buffer.tobytes(), len(buffer)
        )
        request = self._vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(
            self._vision.VNRequestTextRecognitionLevelAccurate
        )
        request.setRecognitionLanguages_(self.languages)
        request.setUsesLanguageCorrection_(True)
        handler = self._vision.VNImageRequestHandler.alloc().initWithData_options_(
            data, {}
        )
        success, error = handler.performRequests_error_([request], None)
        if not success:
            raise RuntimeError(f"Vision không thể nhận dạng chữ: {error}")

        height, width = image.shape[:2]
        results = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            bounds = observation.boundingBox()
            left = bounds.origin.x * width
            right = (bounds.origin.x + bounds.size.width) * width
            top = (1.0 - bounds.origin.y - bounds.size.height) * height
            bottom = (1.0 - bounds.origin.y) * height
            box = [[left, top], [right, top], [right, bottom], [left, bottom]]
            results.append((box, candidate.string(), float(candidate.confidence())))
        return results
