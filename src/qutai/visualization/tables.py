import csv
from collections.abc import Iterable
from html import escape
from pathlib import Path
from typing import Self


class Table:
    def __init__(self, rows: Iterable[Iterable[object] | dict[str, object]], columns: Iterable[object] | None = None, title: str = "") -> None:
        rows = list(rows)
        if rows and isinstance(rows[0], dict):
            columns = list(rows[0]) if columns is None else list(columns)
            rows = [[row[column] for column in columns] for row in rows]
        self.rows = [list(row) for row in rows]
        self.columns = None if columns is None else list(columns)
        self.title = title
        width = len(self.columns) if self.columns is not None else (len(self.rows[0]) if self.rows else 0)
        if any(len(row) != width for row in self.rows): raise ValueError("表格每行的列数必须相同，并与列名数量一致")

    @classmethod
    def from_csv(cls, path: str | Path, limit: int | None = None) -> Self:
        """第一行作为列名，limit 限制展示的数据行数。"""
        if limit is not None and limit < 0: raise ValueError("显示行数不能为负数")
        with Path(path).open(newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            columns = next(reader, [])
            rows = []
            for row in reader:
                if limit is not None and len(rows) >= limit: break
                rows.append(row)
        return cls(rows, columns=columns)

    def _repr_html_(self) -> str:
        body = '<div style="overflow-x:auto"><table style="border-collapse:collapse;font:14px system-ui,sans-serif;color:#1e293b;background:white">'
        if self.title: body += f'<caption style="text-align:left;font-weight:600;padding:10px">{escape(str(self.title))}</caption>'
        if self.columns is not None:
            body += '<thead><tr>' + ''.join(f'<th style="padding:8px 14px;text-align:left;border-bottom:2px solid #cbd5e1">{escape(str(column))}</th>' for column in self.columns) + '</tr></thead>'
        body += '<tbody>'
        for row in self.rows:
            body += '<tr>' + ''.join(f'<td style="padding:8px 14px;border-bottom:1px solid #e2e8f0">{escape(str(value))}</td>' for value in row) + '</tr>'
        if not self.rows: body += '<tr><td style="padding:12px;color:#64748b">暂无数据</td></tr>'
        return body + '</tbody></table></div>'

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        content = f'<!doctype html><html><head><meta charset="utf-8"><title>{escape(str(self.title))}</title></head><body>{self._repr_html_()}</body></html>'
        path.write_text(content, encoding="utf-8")
        return path
