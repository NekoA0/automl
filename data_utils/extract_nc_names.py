#!/usr/bin/env python3
"""
Scan a folder of LabelMe-style JSON annotation files, collect unique labels,
compute nc (number of classes), and list names. Optionally write a YOLO-style YAML.

Usage (Windows PowerShell examples):
    # 直接运行，弹出资料夹选择器（Windows GUI），选择后自动执行
    python scripts/extract_nc_names.py

    # 指定资料夹（不弹窗）
    python scripts/extract_nc_names.py --dir .
    python scripts/extract_nc_names.py --dir . --yaml dataset.yaml

Notes:
- Assumes each JSON has a top-level key "shapes" which is a list of objects with a "label" field.
- Skips files that are invalid JSON or missing the expected keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple


@dataclass
class Summary:
    names: List[str]
    counts: dict[str, int]

    @property
    def nc(self) -> int:
        return len(self.names)


def read_json_labels(path: Path) -> Iterable[str]:
    """Yield labels from a single JSON file if it matches the expected schema."""
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
        # Avoid catching directories named *.json (unlikely but safe guard)
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
    """Write a minimal YOLO-style dataset YAML with nc and names.

    If include_counts is True, also append a counts mapping section.
    """
    if yaml_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {yaml_path}")
    yaml_text = build_yaml_content(summary, include_counts=include_counts, style=style)
    yaml_path.write_text(yaml_text, encoding="utf-8")


def _yaml_quote_scalar(s: str) -> str:
    """Quote a YAML scalar safely using double quotes and minimal escapes.

    This avoids pulling in PyYAML just for emitting a tiny YAML.
    """
    # Replace backslash and double quotes, normalize newlines/tabs.
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    return f'"{s}"'


def _yaml_single_quoted(s: str) -> str:
    """Return YAML single-quoted scalar. Single quotes are escaped by doubling.

    Newlines/tabs are normalized to spaces for compact one-line output.
    """
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace("'", "''")
    return f"'{s}'"


def build_yaml_content(summary: Summary, *, include_counts: bool = False, style: str = "list") -> str:
    """Return a minimal YOLO-style YAML text with nc and names.

    Args:
        include_counts: When True, also include a counts mapping.
        style: 'list' (default) to emit names as a YAML list; 'map' to emit name->id mapping.
    """
    content_lines = [f"nc: {summary.nc}"]

    # names section
    if style == "map":
        content_lines.append("names:")
        for idx, name in enumerate(summary.names):
            content_lines.append(f"  {_yaml_quote_scalar(name)}: {idx}")
    elif style == "flow":
        # YAML flow sequence on a single line: names: ['a','b']
        items = ",".join(_yaml_single_quoted(n) for n in summary.names)
        content_lines.append(f"names: [{items}]")
    else:
        content_lines.append("names:")
        for name in summary.names:
            content_lines.append(f"  - {_yaml_quote_scalar(name)}")

    # optional counts section
    if include_counts and summary.counts:
        content_lines.append("counts:")
        # Preserve YAML key order by sorting names for deterministic output
        for name in sorted(summary.names):
            content_lines.append(f"  {_yaml_quote_scalar(name)}: {summary.counts.get(name, 0)}")
    return "\n".join(content_lines) + "\n"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract unique labels from JSONs and compute nc.")
    parser.add_argument("--dir", dest="dir", default=None, help="Directory containing JSON files (default: prompt a folder picker)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    parser.add_argument("--yaml", dest="yaml", default=None, help="Optional path to write a dataset YAML")
    parser.add_argument("--yaml-counts", action="store_true", help="Include per-class counts in YAML outputs (print/file)")
    parser.add_argument("--yaml-only", action="store_true", help="只輸出 YAML 到標準輸出，不列印其他資訊")
    parser.add_argument(
        "--yaml-style",
        choices=["list", "map", "flow"],
        default="list",
        help="YAML names 的輸出風格：list、map(name->id)、flow(單行 names: ['a','b'])",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing YAML file")
    parser.add_argument("--counts", action="store_true", help="Also print per-class occurrence counts")
    parser.add_argument("--print-yaml", action="store_true", help="Also print the dataset.yaml content to stdout")
    parser.add_argument("--nogui", action="store_true", help="Disable GUI folder picker and require --dir")
    return parser.parse_args(argv)


def _select_dir_gui() -> Path | None:
    """Open a folder picker dialog (tkinter). Returns Path or None if canceled/errors."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()  # Hide main window
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
        print("未提供 --dir 且禁用了 GUI（--nogui），无法确定资料夹。", file=sys.stderr)
        return 2

    # Validate folder
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
            print(f"YAML written: {out}")
        except FileExistsError as e:
            print(str(e), file=sys.stderr)
            return 3
        except Exception as e:
            print(f"Failed to write YAML: {e}", file=sys.stderr)
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
