"""兼容旧命令；主实现位于 examples/lm/train.py。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "lm/train.py"), run_name="__main__")
