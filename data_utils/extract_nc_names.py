#!/usr/bin/env python3
"""
掃描 LabelMe 格式的 JSON 標註檔案資料夾，收集唯一的標籤，
計算 nc（類別數量），並列出名稱。可選擇寫入 YOLO 格式的 YAML 檔案。

用法（Windows PowerShell 範例）：
    # 直接執行，彈出資料夾選擇器（Windows GUI），選擇後自動執行
    python scripts/extract_nc_names.py

    # 指定資料夾（不彈窗）
    python scripts/extract_nc_names.py --dir .
    python scripts/extract_nc_names.py --dir . --yaml dataset.yaml

注意：
- 假設每個 JSON 都有一個頂層鍵 "shapes"，其值為包含 "label" 欄位的物件列表。
- 跳過無效的 JSON 或缺少預期鍵的檔案。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set


@dataclass
class Summary:
    names: List[str]
    counts: dict[str, int]

    @property
    def nc(self) -> int:
        return len(self.names)


def read_json_labels(path: Path) -> Iterable[str]:
    """如果符合預期格式，則從單個 JSON 檔案中產生標籤。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return []
    for s in shapes:
        if isinstance(s, dict):
            label = s.get("label")
            if isinstance(label, str) and label:
                yield label


def collect_labels(folder: Path, recursive: bool = False) -> Summary:
    pattern = "**/*.json" if recursive else "*.json"
    labels: Set[str] = set()
    counts: dict[str, int] = {}
    for p in folder.glob(pattern):
        # 避免抓取名為 *.json 的目錄（不太可能發生，但作為安全防護）
        if not p.is_file():
            continue
        for label in read_json_labels(p):
            labels.add(label)
            counts[label] = counts.get(label, 0) + 1
    names = sorted(labels)
    return Summary(names=names, counts=counts)


def write_yaml(
    yaml_path: Path,
    summary: Summary,
    overwrite: bool = False,
    *,
    include_counts: bool = False,
    style: str = "list",
) -> None:
    """寫入包含 nc 和 names 的最小 YOLO 格式資料集 YAML。

    如果 include_counts 為 True，則同時附加計數映射區段。
    """
    if yaml_path.exists() and not overwrite:
        raise FileExistsError(f"拒絕覆蓋現有檔案：{yaml_path}")
    yaml_text = build_yaml_content(summary, include_counts=include_counts, style=style)
    yaml_path.write_text(yaml_text, encoding="utf-8")


def _yaml_quote_scalar(s: str) -> str:
    """使用雙引號和最少的跳脫字元安全地引用 YAML 純量。

    這避免了僅為了輸出微小的 YAML 而引入 PyYAML。
    """
    # 替換反斜線和雙引號，標準化換行符/製表符。
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    return f'"{s}"'


def _yaml_single_quoted(s: str) -> str:
    """返回 YAML 單引號純量。單引號通過重複兩次來跳脫。

    換行符/製表符被標準化為空格，以便進行緊湊的單行輸出。
    """
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace("'", "''")
    return f"'{s}'"


def build_yaml_content(summary: Summary, *, include_counts: bool = False, style: str = "list") -> str:
    """返回包含 nc 和 names 的最小 YOLO 格式 YAML 文字。

    參數：
        include_counts: 當為 True 時，也包含計數映射。
        style: 'list'（預設）將名稱輸出為 YAML 列表；'map' 輸出 name->id 映射。
    """
    content_lines = [f"nc: {summary.nc}"]

    # names 區段
    if style == "map":
        content_lines.append("names:")
        for idx, name in enumerate(summary.names):
            content_lines.append(f"  {_yaml_quote_scalar(name)}: {idx}")
    elif style == "flow":
        # 單行 YAML 流程序列：names: ['a','b']
        items = ",".join(_yaml_single_quoted(n) for n in summary.names)
        content_lines.append(f"names: [{items}]")
    else:
        content_lines.append("names:")
        for name in summary.names:
            content_lines.append(f"  - {_yaml_quote_scalar(name)}")

    # 可選的計數區段
    if include_counts and summary.counts:
        content_lines.append("counts:")
        # 通過對名稱進行排序來保留 YAML 鍵順序，以獲得確定性的輸出
        for name in sorted(summary.names):
            content_lines.append(f"  {_yaml_quote_scalar(name)}: {summary.counts.get(name, 0)}")
    return "\n".join(content_lines) + "\n"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="從 JSON 中提取唯一標籤並計算 nc。")
    parser.add_argument("--dir", dest="dir", default=None, help="包含 JSON 檔案的目錄（預設：彈出資料夾選擇器）")
    parser.add_argument("--recursive", action="store_true", help="遞迴進入子資料夾")
    parser.add_argument("--yaml", dest="yaml", default=None, help="寫入資料集 YAML 的可選路徑")
    parser.add_argument("--yaml-counts", action="store_true", help="在 YAML 輸出（列印/檔案）中包含每個類別的計數")
    parser.add_argument("--yaml-only", action="store_true", help="只輸出 YAML 到標準輸出，不列印其他資訊")
    parser.add_argument(
        "--yaml-style",
        choices=["list", "map", "flow"],
        default="list",
        help="YAML names 的輸出風格：list、map(name->id)、flow(單行 names: ['a','b'])",
    )
    parser.add_argument("--overwrite", action="store_true", help="允許覆蓋現有的 YAML 檔案")
    parser.add_argument("--counts", action="store_true", help="同時列印每個類別的出現次數")
    parser.add_argument("--print-yaml", action="store_true", help="同時將 dataset.yaml 內容列印到標準輸出")
    parser.add_argument("--nogui", action="store_true", help="禁用 GUI 資料夾選擇器並要求 --dir")
    return parser.parse_args(argv)


def _select_dir_gui() -> Path | None:
    """開啟資料夾選擇對話框 (tkinter)。如果取消或出錯則返回 Path 或 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    try:
        selected = filedialog.askdirectory(title="选择包含 JSON 标注的资料夹")
    finally:
        root.destroy()
    if not selected:
        return None
    p = Path(selected)
    return p if p.exists() and p.is_dir() else None


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    folder: Path | None

    if args.dir:
        folder = Path(args.dir).resolve()
    elif not args.nogui:
        folder = _select_dir_gui()
        if folder is None:
            print("未选择资料夹或环境不支持图形界面，请使用 --dir 指定资料夹或加入 --nogui 禁用弹窗。", file=sys.stderr)
            return 2
    else:
        print("未提供 --dir 且禁用了 GUI（--nogui），無法確定資料夾。", file=sys.stderr)
        return 2

    # 驗證資料夾
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Directory not found: {folder}", file=sys.stderr)
        return 2

    summary = collect_labels(folder, recursive=args.recursive)
    if not args.yaml_only:
        print(f"nc: {summary.nc}")
        print("names:")
        for i, name in enumerate(summary.names):
            print(f"  {i}: {name}")
        if args.counts:
            print("counts:")
            for name in sorted(summary.names, key=lambda n: summary.counts.get(n, 0), reverse=True):
                print(f"  {name}: {summary.counts.get(name, 0)}")

    if args.print_yaml or args.yaml_only:
        if not args.yaml_only:
            print("\n# YAML 格式")
        print(
            build_yaml_content(
                summary,
                include_counts=args.yaml_counts,
                style=args.yaml_style,
            ),
            end="",
        )

    if args.yaml:
        out = Path(args.yaml)
        if not out.is_absolute():
            out = (folder / out).resolve()
        try:
            write_yaml(
                out,
                summary,
                overwrite=args.overwrite,
                include_counts=args.yaml_counts,
                style=args.yaml_style,
            )
            print(f"YAML 已寫入：{out}")
        except FileExistsError as e:
            print(str(e), file=sys.stderr)
            return 3
        except Exception as e:
            print(f"寫入 YAML 失敗：{e}", file=sys.stderr)
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
