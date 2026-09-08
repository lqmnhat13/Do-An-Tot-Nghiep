import threading
from collections import deque
from typing import Generic, TypeVar, Optional, List

T = TypeVar("T")

class BoundedBuffer(Generic[T]):
    """
    Bộ đệm thread-safe có giới hạn dung lượng với cơ chế Drop-Oldest.
    Tuân thủ mục 3.2:
    - Khi đầy, việc put() một phần tử mới sẽ tự động loại bỏ (drop) phần tử cũ nhất mà không chặn producer.
    - Consumer có thể lấy phần tử mới nhất hoặc đợi theo timeout.
    - Ghi nhận số lượng frame đã bị drop để giám sát hiệu năng.
    """

    def __init__(self, maxsize: int = 2):
        if maxsize < 1:
            raise ValueError("maxsize phải >= 1")
        self.maxsize = maxsize
        self._deque: deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped_count = 0
        self._total_pushed = 0

    def put(self, item: T) -> bool:
        """
        Đẩy phần tử mới vào bộ đệm.
        Nếu bộ đệm đầy, phần tử cũ nhất sẽ bị loại bỏ ngay lập tức (không chặn).
        Trả về True nếu một phần tử cũ bị drop.
        """
        dropped = False
        with self._lock:
            self._total_pushed += 1
            if len(self._deque) >= self.maxsize:
                self._deque.popleft() # Drop oldest
                self._dropped_count += 1
                dropped = True
            self._deque.append(item)
            self._not_empty.notify()
        return dropped

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """
        Lấy phần tử ra khỏi bộ đệm (FIFO).
        Nếu bộ đệm rỗng, đợi tối đa timeout giây.
        Trả về None nếu hết thời gian chờ.
        """
        with self._not_empty:
            if not self._deque:
                if timeout is not None and timeout <= 0:
                    return None
                if not self._not_empty.wait(timeout):
                    return None # Timeout
            if self._deque:
                return self._deque.popleft()
            return None

    def get_latest(self, clear_older: bool = True) -> Optional[T]:
        """
        Lấy trực tiếp phần tử mới nhất hiện có trong bộ đệm.
        Nếu clear_older=True, dọn sạch tất cả các phần tử cũ hơn còn lại.
        """
        with self._lock:
            if not self._deque:
                return None
            latest = self._deque.pop()
            if clear_older:
                dropped_remains = len(self._deque)
                self._dropped_count += dropped_remains
                self._deque.clear()
            return latest

    def peek(self) -> Optional[T]:
        """Xem phần tử mới nhất mà không lấy ra khỏi bộ đệm."""
        with self._lock:
            if not self._deque:
                return None
            return self._deque[-1]

    def clear(self) -> None:
        """Dọn sạch bộ đệm."""
        with self._lock:
            self._dropped_count += len(self._deque)
            self._deque.clear()

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def total_pushed(self) -> int:
        with self._lock:
            return self._total_pushed

    def size(self) -> int:
        with self._lock:
            return len(self._deque)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._deque) == 0
