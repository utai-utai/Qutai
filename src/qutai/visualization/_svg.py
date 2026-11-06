from html import escape
from pathlib import Path


def text(x: float, y: float, value: object, *, anchor: str = "start", size: int = 12, fill: str = "#334155") -> str:
    return f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" font-size="{size}" fill="{fill}">{escape(str(value))}</text>'


def svg(width: int, height: int, title: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(str(title), quote=True)}" '
            'style="font-family:system-ui,sans-serif;background:white">'
            f'<title>{escape(str(title))}</title><rect width="100%" height="100%" fill="white"/>{body}</svg>')


def save_svg(path: str | Path, content: str) -> Path:
    path = Path(path)
    path.write_text(content, encoding="utf-8")
    return path
