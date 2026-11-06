from collections.abc import Iterable, Iterator
from time import perf_counter


class Progress[T]:
    """用标准输出原地刷新进度；迭代体执行完后才计入完成数。"""
    def __init__(self, items: Iterable[T], *, total: int) -> None:
        self.items, self.total = items, total

    def __iter__(self) -> Iterator[T]:
        if self.total == 0: return
        started = updated = perf_counter()
        self._draw(0, 0.0)
        try:
            for completed, item in enumerate(self.items, 1):
                yield item
                now = perf_counter()
                if completed == self.total or now - updated >= 0.1:
                    self._draw(completed, now - started)
                    updated = now
        finally:
            print()

    def _draw(self, completed: int, elapsed: float) -> None:
        filled = 24 * completed // self.total
        bar = "▰" * filled + "▱" * (24 - filled)
        print(f"\r{bar} {completed / self.total:6.1%}  {completed}/{self.total}  {elapsed:.1f}s", end="", flush=True)
