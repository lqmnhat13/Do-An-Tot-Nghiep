import time
from typing import Dict, Optional, Callable

class Watchdog:
    """
    Giám sát hoạt động của các luồng xử lý và phát hiện đóng băng (freeze/stall).
    Tuân thủ mục 3.1 và 8 của Kế hoạch:
    - Theo dõi heartbeat của Camera, Detection, Depth.
    - Kích hoạt cảnh báo hệ thống khi một thành phần bị treo vượt quá timeout.
    """

    def __init__(self, timeout_sec: float = 3.0, on_stall_callback: Optional[Callable[[str], None]] = None):
        self.timeout_sec = timeout_sec
        self.on_stall_callback = on_stall_callback
        self._heartbeats: Dict[str, float] = {}

    def beat(self, component_name: str) -> None:
        """Cập nhật nhịp tim (heartbeat) cho một thành phần."""
        self._heartbeats[component_name] = time.monotonic()

    def check_health(self) -> Dict[str, bool]:
        """
        Kiểm tra sức khỏe các thành phần.
        Trả về dictionary {component_name: is_healthy}.
        """
        now = time.monotonic()
        health = {}
        for name, last_beat in self._heartbeats.items():
            is_ok = (now - last_beat) <= self.timeout_sec
            health[name] = is_ok
            if not is_ok and self.on_stall_callback:
                self.on_stall_callback(name)
        return health

    def is_all_healthy(self) -> bool:
        health = self.check_health()
        return all(health.values()) if health else True
