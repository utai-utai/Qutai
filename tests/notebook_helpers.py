"""在空命名空间和临时目录中执行阶段实验，不修改 Notebook 文件。"""

import json
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory


def run_notebook(name: str, *, stop_after: str | None = None) -> dict:
    """按顺序执行，可在指定单元后取出中间模型。"""
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    original_directory = Path.cwd()
    with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
        data = path.parent / "data"
        if data.is_dir(): copytree(data, Path(directory) / "data")
        os.chdir(directory)
        try:
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"{name}:{cell['id']}", "exec"), namespace)
                if cell["id"] == stop_after: break
        finally:
            os.chdir(original_directory)
    return namespace
