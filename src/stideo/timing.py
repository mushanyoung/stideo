"""Time independent stages of a single stitch."""

import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

type Stage = Callable[[str], AbstractContextManager[object]]


class Timings:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.seconds = {}

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.monotonic() - started)

    def report(self) -> None:
        total = self.elapsed()
        other = max(0.0, total - sum(self.seconds.values()))
        print("\n耗时统计（各阶段累计实际用时）")
        for name, seconds in [*self.seconds.items(), ("其他开销", other)]:
            print(f"  {name}  {seconds:.3f} 秒  {100 * seconds / total:.1f}%")
        minutes, seconds = divmod(total, 60)
        print(f"总耗时 {total:.3f} 秒（{int(minutes)} 分 {seconds:.2f} 秒）", flush=True)
