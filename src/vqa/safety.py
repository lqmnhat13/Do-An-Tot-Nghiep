import re
import unicodedata
from typing import Optional


class SafetyGuardrail:
    """Guardrail độc lập, luôn chạy trước backend model."""

    REFUSAL = (
        "Hệ thống không thể xác nhận đường đi có an toàn hay không. "
        "Xin hãy cẩn thận dùng gậy dẫn đường và kiểm tra xung quanh."
    )
    # Phân loại theo ý định thay vì khớp từ khoá đơn lẻ, vì từ khoá đơn ("đi thẳng",
    # "qua đường", "an toàn") chặn nhầm câu hỏi đọc biển báo hoặc đếm người
    # (9/3545 câu OpenViVQA bị chặn nhầm) mà lại bỏ lọt "Tôi có thể đi tiếp không?".
    _MOVE = r"(?:đi|bước|qua đường|sang đường|băng qua|di chuyển|tiến lên|tiến tới|rẽ)"
    _ASK = r"(?:có thể|được không|được chứ|được chưa|có được|có sao không|có ổn không|nên|có nên|liệu)"
    MOVE = re.compile(rf"\b{_MOVE}\b")
    # Xin phép di chuyển: từ hỏi phép đứng sát động từ di chuyển ("có thể đi tiếp",
    # "đi về phía trước được không"), không tính câu hỏi chỉ đường/đếm người.
    MOVE_PERMISSION = re.compile(
        rf"\b{_ASK}\s+(?:\w+\s+){{0,2}}{_MOVE}\b|\b{_MOVE}(?:\s+\w+){{0,4}}\s+{_ASK}\b")
    WH_QUESTION = re.compile(
        r"\b(?:nào|đâu|gì|ai|mấy|bao nhiêu|bao xa|tại sao|vì sao|thế nào|như thế nào)\b")
    FIRST_PERSON = re.compile(r"\b(?:tôi|mình|tớ|tao)\b")
    THIRD_PERSON = re.compile(r"\b(?:người|họ|khách|xe)\b")
    HAZARD = re.compile(
        r"\b(?:an toàn|vật cản|chướng ngại|nguy hiểm|va chạm|va vào|đâm vào|vấp ngã|té ngã|"
        r"ngã(?! (?:ba|tư|năm|sáu|bảy)))\b")
    # Hỏi sự có mặt của bậc/hố/cầu thang dạng "có ... không" (không chặn câu mô tả "bậc thang dẫn đến đâu").
    HAZARD_PRESENCE = re.compile(
        r"\bcó\b[^?]*\b(?:bậc|cầu thang|hố|ổ gà)\b[^?]*\bkhông\b"
        r"|\bcó\b[^?]*\bxe\b[^?]*\b(?:tới|đến|chạy|lại gần|đi tới)\b[^?]*\bkhông\b")
    SAFETY_TOPIC = re.compile(r"\ban toàn (?:giao thông|thực phẩm|lao động|vệ sinh|thông tin|điện|cháy nổ)\b")
    READING = re.compile(
        r"\b(?:ghi|viết|đọc|chữ|nội dung|tuyên truyền|khẩu hiệu|biển (?:báo|hiệu)|bảng (?:báo|hiệu)|"
        r"tờ giấy|nhãn|thông báo)\b")
    ENGLISH = re.compile(
        r"\b(?:safe to|is it safe|can i (?:walk|cross|go|move|proceed)|"
        r"(?:any|is there an?) (?:obstacle|hazard|danger)|"
        r"(?:path|way|road) (?:is )?clear|clear ahead|in my way)")
    # Hỏi lối đi có trống/thông thoáng/vướng/cản không: cần đủ chủ ngữ lối đi, từ chỉ trạng thái
    # lối đi và tiểu từ hỏi. "Phía trước có gì?" không có từ trạng thái nên không bị chặn.
    PATH_SUBJECT = re.compile(r"\b(?:lối đi|đường|phía trước|trước mặt)\b")
    PATH_STATE = re.compile(
        r"\b(?:trống|thông thoáng|thoáng|quang\b(?! cảnh)|vướng|cản|(?<!chắc )chắn)\b")
    YES_NO = re.compile(r"\b(?:không|chứ|chưa|đúng không|phải không)\b")
    # "Cứ đi thẳng", "đi tiếp là được": xin xác nhận di chuyển dù không có từ hỏi phép.
    MOVE_CONFIRM = re.compile(
        rf"\bcứ\s+(?:\w+\s+){{0,1}}{_MOVE}\b|\b{_MOVE}(?:\s+\w+){{0,4}}\s+là được\b")

    def check(self, question: str) -> Optional[str]:
        # NFC trước: chữ Việt dạng tổ hợp (NFD) làm mọi regex có dấu không khớp.
        normalized = " ".join(unicodedata.normalize("NFC", question).strip().lower().split())
        if self.ENGLISH.search(normalized):
            return self.REFUSAL
        reading = bool(self.READING.search(normalized))
        first_person = bool(self.FIRST_PERSON.search(normalized))
        # 1. Người dùng hỏi bản thân có đi/bước/qua đường được không.
        if self.MOVE.search(normalized) and first_person:
            return self.REFUSAL
        if self.MOVE_CONFIRM.search(normalized):
            return self.REFUSAL
        if (self.MOVE_PERMISSION.search(normalized) and not reading
                and not self.WH_QUESTION.search(normalized)
                and not self.THIRD_PERSON.search(normalized)):
            return self.REFUSAL
        # 2. Hỏi lối đi có trống/thông thoáng/vướng không (an toàn trước hết: không miễn câu đọc chữ).
        if (self.PATH_SUBJECT.search(normalized) and self.PATH_STATE.search(normalized)
                and self.YES_NO.search(normalized)):
            return self.REFUSAL
        # 3. Hỏi có vật cản/nguy hiểm/an toàn không (trừ câu hỏi nội dung chữ hoặc chủ đề chung).
        hazard = self.HAZARD.search(normalized) or self.HAZARD_PRESENCE.search(normalized)
        if hazard and (first_person or not (reading or self.SAFETY_TOPIC.search(normalized))):
            return self.REFUSAL
        return None

    # --- Kiểm tra câu trả lời của VLM: câu hỏi mặc định là "Phía trước có gì?", nên mọi câu
    # nói phía trước trống/ổn/không có gì đều bị coi là khẳng định lối đi an toàn.
    # Vế từ chối "không thể xác nhận ..." (chính là REFUSAL) được bỏ trước khi kiểm tra.
    ANSWER_UNCERTAIN = re.compile(r"\bkhông thể xác nhận[^,]{0,40}")
    # Câu đọc chữ trên biển/nhãn: "Biển ghi: ..." — chỉ khi trước động từ đọc có danh từ vật mang chữ;
    # phần trích dẫn (tới dấu phẩy đầu tiên) được kiểm tra riêng, lỏng hơn.
    ANSWER_QUOTED_TEXT = re.compile(
        r"^[^:“\"]*\b(?:biển|bảng|tờ|nhãn|áp phích|tấm|giấy|màn hình|bìa|poster|hộp|chai)\b"
        r"[^:“\"]*\b(?:ghi|viết|có chữ|nội dung)\b\s*(?:là\s*)?[:“\"]([^,]*),?")
    # Xin phép/khuyên di chuyển.
    ANSWER_MOVE = re.compile(
        r"\b(?:có thể|nên|hãy|cứ|được|yên tâm)\s+(?:đi|bước|qua đường|sang đường|băng qua|"
        r"di chuyển|tiếp tục|tiến)\b"
        r"|\b(?:đi|bước|đi tiếp|đi thẳng|đi qua) được\b"
        r"|^(?:bạn\s+)?(?:đi thẳng|đi tiếp|tiến lên|tiến tới|tiếp tục đi)\b")
    # Dấu hiệu vắng mặt/phủ định sự tồn tại ("không có gì", "không phát hiện vật cản"...).
    ANSWER_ABSENCE = re.compile(
        r"\b(?:không|chẳng|chả|không hề) có\b|\bkhông (?:còn|thấy|nhìn thấy|phát hiện|gặp)\b"
        r"|\bkhông bị (?:cản|chặn|chắn|vướng)|\bkhông nguy hiểm\b")
    # Ngoại lệ: phủ định thuộc tính của một đồ vật cụ thể ("Trên bàn không có gì",
    # "Có một chai nước, không có nắp"); chỉ áp dụng khi câu không nhắc nơi chốn/lối đi.
    ANSWER_OBJECT_PREFIX = re.compile(
        r"^(?:có\s+)?(?:một|hai|ba|vài|cái|chiếc|con|tấm|chai|cốc|ly|hộp|quyển|cuốn|trên|dưới|cạnh)\b")
    ANSWER_PLACE = re.compile(
        r"\b(?:phía|trước mặt|lối|đường|hành lang|phòng|sân|sàn|ngõ|vỉa hè|cầu thang|khu vực|"
        r"không gian|xung quanh|"
        r"ảnh|khung cảnh|mọi thứ|tất cả|bạn)\b")
    # Từ chỉ khung cảnh trống/ổn, chặn ở mọi vị trí trừ khi bị phủ định ("không an toàn").
    ANSWER_SCENE_OK = re.compile(
        r"(?<!không )(?<!chưa )(?<!chẳng )"
        r"\b(?:thông thoáng|thoáng|quang đãng|quang\b(?! cảnh)|vắng|rộng rãi|khoảng không|"
        r"không gian (?:rộng|mở|trống|lớn)|tự do|ổn|bình thường|an toàn|yên tâm|thoải mái)\b")
    # "sạch" chỉ tính khi nói về nơi chốn ("Phía trước sạch", "Đường đi sạch").
    ANSWER_CLEAN = re.compile(r"\bsạch\b")
    # "trống" là khẳng định lối đi trống, trừ khi đứng sau danh từ đồ vật ("ghế trống",
    # "chai nước trống") hoặc nơi chốn trống nhưng có đồ ("phòng trống có một cái bàn").
    ANSWER_EMPTY = re.compile(r"(?:^|(\S+)\s+)(?<!không )trống\b(?!\s+có\b)")
    _EMPTY_SCENE_PREV = frozenset(
        "trước mặt sân phòng khoảng gian lang đi đường bãi sàn nhà nền lối vực rất khá toàn "
        "là đều thì hơi vẫn đang cũng quá".split())
    # "chỉ có" + toàn sàn/tường/trần/trời (không có đồ vật nào khác).
    ANSWER_ONLY = re.compile(r"\bchỉ (?:có|là|thấy)\s+(.*)$")
    _ONLY_SURFACE = re.compile(
        r"\b(?:sàn nhà|sàn|tường|trần nhà|trần|mặt đất|nền nhà|nền|bầu trời|trời|một|bức|cái|"
        r"mặt|những|các|và|màu|trắng|xám|gạch|gỗ|phẳng)\b|[,\s]")
    ANSWER_ENGLISH = re.compile(
        r"\b(?:safe to (?:walk|cross|proceed)|you can (?:walk|cross|proceed|go)|it(?: is|'s) safe|"
        r"no obstacles?|nothing (?:ahead|in front)|(?:all )?clear)\b")

    def check_answer(self, answer: str) -> Optional[str]:
        normalized = " ".join(unicodedata.normalize("NFC", answer).strip().lower().split())
        for sentence in re.split(r"[.!?;\n]", normalized):
            sentence = self.ANSWER_UNCERTAIN.sub("", sentence)
            quoted = self.ANSWER_QUOTED_TEXT.match(sentence)
            if quoted:
                # Chữ trên biển vẫn bị chặn nếu là lời khuyên di chuyển ("đi thẳng được") hoặc
                # nhắc lối đi kèm khẳng định trống/an toàn ("Phía trước an toàn"); "Chú ý an toàn" vẫn đọc.
                text = quoted.group(1).strip(" “”\"")
                if self.ANSWER_MOVE.search(text) or (
                        self.ANSWER_PLACE.search(text) and self._claims_clear_path(text)):
                    return self.REFUSAL
                sentence = sentence[quoted.end():]
            sentence = self.SAFETY_TOPIC.sub("", sentence).strip(" ,")
            if sentence and self._claims_clear_path(sentence):
                return self.REFUSAL
        return None

    def _claims_clear_path(self, sentence: str) -> bool:
        if self.ANSWER_MOVE.search(sentence) or self.ANSWER_ENGLISH.search(sentence):
            return True
        place = bool(self.ANSWER_PLACE.search(sentence))
        absence = self.ANSWER_ABSENCE.search(sentence)
        if absence and not (absence.start() > 0 and not place
                            and self.ANSWER_OBJECT_PREFIX.search(sentence)):
            return True
        if self.ANSWER_SCENE_OK.search(sentence) or (place and self.ANSWER_CLEAN.search(sentence)):
            return True
        for match in self.ANSWER_EMPTY.finditer(sentence):
            # "trống" đứng đầu câu, ngay sau dấu phẩy ("hành lang dài, trống") hoặc sau nơi chốn/trạng từ.
            prev = match.group(1)
            if prev is None or prev.endswith(",") or prev in self._EMPTY_SCENE_PREV:
                return True
        only = self.ANSWER_ONLY.search(sentence)
        return bool(only and not self._ONLY_SURFACE.sub("", only.group(1)))
