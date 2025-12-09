#!/usr/bin/env python3
"""
Convert LabelMe-style JSON annotations to YOLO detection TXT format.

Assumptions:
- Each JSON contains top-level keys: imageWidth, imageHeight (preferred).
  If missing, will try to open the linked image (imagePath) with PIL if available.
- shapes: list of objects each with 'label', 'points' and optional 'shape_type'.
- For polygon/any shape, a bounding box is computed from all points.

Output:
- For each JSON file, write a .txt file with the same stem containing lines:
    <class_id> <x_center> <y_center> <width> <height>
  where coordinates are normalized to [0,1] by image width/height.
- By default, empty annotations are skipped (no .txt created). Use --write-empty to create empty files.

Names mapping:
- Provide class names via one of:
    --yaml dataset.yaml   # parse names from YAML produced by extract_nc_names.py
    --names "a,b,c"       # comma-separated names
  If neither provided, the script scans all JSONs first to build an alphabetical list.

Usage (PowerShell):
  python .\json_to_yolo.py --src . --out .\labels_yolo --recursive --yaml .\dataset.yaml
  python .\json_to_yolo.py --src . --out .\labels_yolo --names "cat,dog,person"

"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional


@dataclass(frozen=True)
class Shape:
    label: str
    points: List[Tuple[float, float]]


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert LabelMe JSON to YOLO TXT (detection)")
    p.add_argument("--src", required=True, help="來源資料夾（包含 JSON）")
    p.add_argument("--out", default=None, help="輸出資料夾（預設 <src>/labels_yolo）")
    p.add_argument("--recursive", action="store_true", help="遞迴掃描子資料夾")
    p.add_argument("--yaml", default=None, help="dataset.yaml（讀取 names 作為類別對應）")
    p.add_argument("--names", default=None, help="逗號分隔的類別名稱，例如 'cat,dog,person'")
    p.add_argument("--precision", type=int, default=6, help="浮點數小數位數（預設 6）")
    p.add_argument("--overwrite", action="store_true", help="允許覆寫已存在的 .txt")
    p.add_argument("--write-empty", action="store_true", help="若無標註也建立空白 .txt")
    p.add_argument("--delete-json", action="store_true", help="轉換完成後刪除原始 JSON 檔")
    return p.parse_args(argv)


def _load_image_size_from_json(data: dict, json_path: Path) -> Optional[Tuple[int, int]]:
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h
    # Optional: try to open imagePath via PIL if available
    img_rel = data.get("imagePath")
    if isinstance(img_rel, str) and img_rel:
        img_path = (json_path.parent / img_rel).resolve()
        try:
            from PIL import Image  # type: ignore

            with Image.open(img_path) as im:
                return im.width, im.height
        except Exception:
            return None
    return None


def _parse_shapes(data: dict) -> Iterable[Shape]:
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return []
    out: List[Shape] = []
    for s in shapes:
        if not isinstance(s, dict):
            continue
        label = s.get("label")
        pts = s.get("points")
        if not isinstance(label, str) or not label:
            continue
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        parsed: List[Tuple[float, float]] = []
        for p in pts:
            if (
                isinstance(p, (list, tuple))
                and len(p) == 2
                and isinstance(p[0], (int, float))
                and isinstance(p[1], (int, float))
            ):
                parsed.append((float(p[0]), float(p[1])))
        if len(parsed) >= 2:
            out.append(Shape(label=label, points=parsed))
    return out


def _bbox_from_points(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return x_min, y_min, x_max, y_max


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _to_yolo_line(
    cls_id: int,
    bbox_xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    precision: int,
) -> Optional[str]:
    x1, y1, x2, y2 = bbox_xyxy
    # Clamp to image bounds and ensure valid box
    x1 = _clamp(x1, 0.0, float(img_w))
    x2 = _clamp(x2, 0.0, float(img_w))
    y1 = _clamp(y1, 0.0, float(img_h))
    y2 = _clamp(y2, 0.0, float(img_h))
    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
    bw = max(0.0, x_max - x_min)
    bh = max(0.0, y_max - y_min)
    if bw <= 0.0 or bh <= 0.0 or img_w <= 0 or img_h <= 0:
        return None
    x_c = (x_min + x_max) / 2.0 / float(img_w)
    y_c = (y_min + y_max) / 2.0 / float(img_h)
    w = bw / float(img_w)
    h = bh / float(img_h)
    fmt = f"{{:.{precision}f}}"
    return f"{cls_id} {fmt.format(x_c)} {fmt.format(y_c)} {fmt.format(w)} {fmt.format(h)}"


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _iter_json_files(src: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    for p in src.glob(pattern):
        if p.is_file():
            yield p


def _parse_names_from_yaml(yaml_path: Path) -> Optional[List[str]]:
    """Parse names from a minimal YAML produced by extract_nc_names.py.

    Supports:
      names:\n  - "a"\n  - "b"
      names: ['a','b']
      names:\n  "a": 0\n  "b": 1
    """
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except Exception:
        return None

    lines = [ln.rstrip() for ln in text.splitlines()]
    names_section_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("names:"):
            names_section_idx = i
            break
    if names_section_idx is None:
        return None

    # Try flow style on the same line: names: [ ... ]
    after = lines[names_section_idx].split(":", 1)[1].strip() if ":" in lines[names_section_idx] else ""
    if after.startswith("[") and after.endswith("]"):
        inner = after[1:-1]
        # split by commas respecting basic quotes
        out: List[str] = []
        token = ""
        quote = None
        for ch in inner:
            if quote:
                if ch == quote:
                    quote = None
                else:
                    token += ch
            else:
                if ch in ("'", '"'):
                    quote = ch
                elif ch == ',':
                    out.append(token.strip())
                    token = ""
                else:
                    token += ch
        if token.strip():
            out.append(token.strip())
        # strip surrounding quotes
        out = [s.strip().strip("'\"") for s in out]
        return [s for s in out if s]

    # Otherwise parse subsequent indented list or map
    # Detect block type by looking at next non-empty line
    i = names_section_idx + 1
    items: List[str] = []
    pairs: Dict[str, int] = {}
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "" or ln.startswith("#"):
            i += 1
            continue
        if not ln.startswith(" ") and not ln.startswith("\t"):
            break  # out of names section
        stripped = ln.strip()
        if stripped.startswith("- "):
            val = stripped[2:].strip()
            val = val.strip().strip('"').strip("'")
            if val:
                items.append(val)
        else:
            # map style: "name": idx
            if ":" in stripped:
                key, _, rest = stripped.partition(":")
                key = key.strip().strip('"').strip("'")
                try:
                    idx = int(rest.strip())
                except Exception:
                    idx = None  # type: ignore
                if key and idx is not None:
                    pairs[key] = idx
        i += 1
    if items:
        return items
    if pairs:
        # sort by idx
        return [k for k, _ in sorted(pairs.items(), key=lambda kv: kv[1])]
    return None


def _collect_all_labels(src: Path, recursive: bool) -> List[str]:
    labels: set[str] = set()
    for jp in _iter_json_files(src, recursive):
        data = _read_json(jp)
        if not data:
            continue
        for sh in _parse_shapes(data):
            labels.add(sh.label)
    return sorted(labels)


def _build_name_to_id(names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def convert_folder(
    src: Path,
    out_dir: Path,
    recursive: bool,
    names: List[str],
    precision: int,
    overwrite: bool,
    write_empty: bool,
    delete_json: bool,
) -> Tuple[int, int, int]:
    """Convert JSONs in src to YOLO TXT into out_dir.

    Returns tuple: (files_written, shapes_converted, files_skipped)
    """
    name_to_id = _build_name_to_id(names)
    files_written = 0
    shapes_total = 0
    files_skipped = 0

    json_deleted = 0
    for jp in _iter_json_files(src, recursive):
        data = _read_json(jp)
        if not data:
            files_skipped += 1
            continue
        size = _load_image_size_from_json(data, jp)
        if not size:
            print(f"警告：找不到影像尺寸，略過 {jp}")
            files_skipped += 1
            continue
        img_w, img_h = size
        shapes = list(_parse_shapes(data))
        lines: List[str] = []
        for sh in shapes:
            if sh.label not in name_to_id:
                # Unknown labels are skipped; ensure names provided cover all labels
                print(f"警告：未知類別 '{sh.label}'，檔案 {jp} 此 shape 已忽略。")
                continue
            bbox = _bbox_from_points(sh.points)
            yline = _to_yolo_line(name_to_id[sh.label], bbox, img_w, img_h, precision)
            if yline:
                lines.append(yline)
        # Decide output path (mirror relative path inside out_dir)
        rel = jp.relative_to(src)
        out_path = (out_dir / rel).with_suffix(".txt")
        wrote_or_exists = False
        if not lines and not write_empty:
            # nothing to write (and no empty requested)
            wrote_or_exists = False
        elif out_path.exists() and not overwrite:
            print(f"略過（已存在）：{out_path}")
            files_skipped += 1
            wrote_or_exists = True
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            files_written += 1
            shapes_total += len(lines)
            wrote_or_exists = True

        # Delete source JSON only when requested and conversion result exists (written or already existed)
        if delete_json and wrote_or_exists:
            try:
                jp.unlink()
                json_deleted += 1
            except Exception:
                pass
    return files_written, shapes_total, files_skipped


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    src = Path(args.src).resolve()
    if not src.exists() or not src.is_dir():
        print(f"來源資料夾不存在：{src}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).resolve() if args.out else (src / "labels_yolo").resolve()

    # Determine class names
    names: Optional[List[str]] = None
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    elif args.yaml:
        yp = Path(args.yaml)
        if not yp.is_absolute():
            yp = (src / yp).resolve()
        names = _parse_names_from_yaml(yp)
        if names is None:
            print(f"無法從 YAML 解析 names：{yp}", file=sys.stderr)
    if names is None:
        # Fallback: scan all labels alphabetically
        print("未提供 --yaml 或 --names，將先掃描所有 JSON 蒐集類別並依字母排序。")
        names = _collect_all_labels(src, args.recursive)
        if not names:
            print("找不到任何標籤，終止。", file=sys.stderr)
            return 3

    # Summary of mapping
    print("類別對應：")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")

    out_dir.mkdir(parents=True, exist_ok=True)
    files_written, shapes_total, files_skipped = convert_folder(
        src, out_dir, args.recursive, names, args.precision, args.overwrite, args.write_empty, args.delete_json
    )
    print("\n完成轉換：")
    print(f"  寫出 TXT 檔數：{files_written}")
    print(f"  轉換標註數：{shapes_total}")
    print(f"  略過檔案數：{files_skipped}")
    print(f"輸出位置：{out_dir}")
    if args.delete_json:
        # 簡單再次統計剩餘 JSON 數量（僅作回報，非必要）
        remaining = sum(1 for _ in _iter_json_files(src, args.recursive))
        print(f"  JSON 清理完成（剩餘 {remaining} 個 JSON）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
