#!/usr/bin/env python3
"""
Split images and matching labels into train/val/test folders.

- Supports labels as YOLO TXT (preferred) under <src>/labels_yolo mirroring image paths, or JSON next to images.
- Only pairs with both image and label (YOLO or JSON) are included.
- Default split ratio is 8:2:2 (train:val:test).
- Supports recursion while preserving subdirectory structure.
- By default copies files; can move with --move.

Example (Windows PowerShell):
    python .\split_dataset.py --src C:\data --recursive --use-yolo --labels-dir labels_yolo --ratio 8,1,1

Notes:
- Common image extensions are matched case-insensitively: jpg, jpeg, png, bmp, tif, tiff, webp
- When --recursive, subdirectory structure is preserved under each split/images and split/labels.
- If --out is inside --src, scanning will ignore the output folder to avoid self-inclusion.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Pair:
    img_rel: Path  # relative path from src
    lbl_rel: Path  # relative path for label (mirrors image rel path with .txt/.json)
    lbl_src: Path  # absolute source path to label file (in src for JSON, in labels_dir for YOLO)
    is_yolo: bool


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split images and labels into train/val/test.")
    p.add_argument("--src", required=True, help="來源資料夾（包含影像；YOLO 標註位於 <src>/labels_yolo）")
    p.add_argument("--out", default=None, help="輸出資料夾（預設為 <src>/dataset_split）")
    p.add_argument("--recursive", action="store_true", help="遞迴掃描子資料夾")
    p.add_argument("--ratio", default="8,2,2", help="分割比例 train,val,test（例如: 8,2,2）")
    p.add_argument("--seed", type=int, default=42, help="隨機種子")
    p.add_argument("--move", action="store_true", help="以移動檔案取代複製")
    p.add_argument("--dry-run", action="store_true", help="僅顯示將要執行的操作，不實際複製/移動")
    p.add_argument("--extensions", default=None, help="自訂影像副檔名，逗號分隔（預設: jpg,jpeg,png,bmp,tif,tiff,webp）")
    p.add_argument("--use-yolo", action="store_true", help="優先使用 YOLO TXT 標註（來源 <src>/labels_yolo）")
    p.add_argument("--labels-dir", default="labels_yolo", help="YOLO TXT 標註來源資料夾名稱（位於 <src> 下，預設 labels_yolo）")
    return p.parse_args(argv)


def parse_ratio(s: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"--ratio 需要三段，以逗號分隔，例如 8,2,2；取得: {s}")
    try:
        a, b, c = (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        raise ValueError(f"--ratio 僅接受整數，例如 8,2,2；取得: {s}")
    if a < 0 or b < 0 or c < 0 or (a + b + c) <= 0:
        raise ValueError("--ratio 無效：必須為非負整數且總和 > 0")
    return a, b, c


def iter_image_files(src: Path, recursive: bool, ignore: Path | None, exts: set[str]) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in src.glob(pattern):
        if p.is_dir():
            continue
        if ignore is not None:
            try:
                if p.resolve().is_relative_to(ignore):
                    continue
            except AttributeError:
                try:
                    p.resolve().relative_to(ignore)
                    continue
                except Exception:
                    pass
        if p.suffix.lower() in exts:
            yield p


def collect_pairs(src: Path, recursive: bool, out_dir: Path | None, exts: set[str], *, use_yolo: bool, labels_dir_name: str) -> List[Pair]:
    ignore = out_dir if (out_dir is not None and out_dir.exists()) else None
    pairs: List[Pair] = []
    missing_json = 0
    missing_txt = 0

    labels_dir = src / labels_dir_name
    have_yolo = labels_dir.exists()

    for img_path in iter_image_files(src, recursive, ignore, exts):
        rel = img_path.relative_to(src)
        used_yolo = False
        if use_yolo and have_yolo:
            txt_rel = rel.with_suffix(".txt")
            txt_src = labels_dir / txt_rel
            if not (txt_src.exists() and txt_src.is_file()):
                # 兼容：若影像位於 images/... 而標註存於 labels/...（不含 images 前綴）
                parts = list(rel.parts)
                if parts and parts[0].lower() == "images":
                    alt_rel = Path(*parts[1:]).with_suffix(".txt")
                    alt_src = labels_dir / alt_rel
                    if alt_src.exists() and alt_src.is_file():
                        txt_rel = alt_rel
                        txt_src = alt_src
            if txt_src.exists() and txt_src.is_file():
                pairs.append(Pair(img_rel=rel, lbl_rel=txt_rel, lbl_src=txt_src, is_yolo=True))
                used_yolo = True
            else:
                missing_txt += 1
        if not used_yolo:
            json_rel = rel.with_suffix(".json")
            json_src = src / json_rel
            if json_src.exists() and json_src.is_file():
                pairs.append(Pair(img_rel=rel, lbl_rel=json_rel, lbl_src=json_src, is_yolo=False))
            else:
                missing_json += 1

    if use_yolo and have_yolo and missing_txt:
        print(f"警告：有 {missing_txt} 張影像找不到對應 YOLO TXT（已忽略或回退 JSON）。")
    if missing_json:
        print(f"警告：有 {missing_json} 張影像找不到對應 JSON（若無 YOLO 亦無 JSON 則忽略）。")
    pairs.sort(key=lambda pr: (str(pr.img_rel).lower()))
    return pairs


def split_indices(n: int, ratio: Tuple[int, int, int], seed: int) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    a, b, c = ratio
    s = a + b + c
    base_train = (n * a) // s
    base_val = (n * b) // s
    base_test = (n * c) // s
    used = base_train + base_val + base_test
    rem = n - used
    extras = [0, 0, 0]
    for i in range(rem):
        extras[i % 3] += 1
    tN = base_train + extras[0]
    vN = base_val + extras[1]
    train_idx = idxs[:tN]
    val_idx = idxs[tN:tN + vN]
    test_idx = idxs[tN + vN:]
    return train_idx, val_idx, test_idx


def ensure_dir(path: Path, dry: bool) -> None:
    if dry:
        print(f"MKDIR {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool, dry: bool) -> None:
    if dry:
        op = "MOVE" if move else "COPY"
        print(f"{op} {src} -> {dst}")
        return
    ensure_dir(dst.parent, dry=False)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def strip_leading_component(rel: Path, name: str) -> Path:
    """Remove leading path component 'name' (case-insensitive) from a relative Path.

    Example: strip_leading_component(Path('images/foo/bar.jpg'), 'images') -> Path('foo/bar.jpg')
    """
    parts = list(rel.parts)
    while parts and parts[0].lower() == name.lower():
        parts = parts[1:]
    return Path(*parts) if parts else Path("")


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    src = Path(args.src).resolve()
    if not src.exists() or not src.is_dir():
        print(f"來源資料夾不存在：{src}", file=sys.stderr)
        return 2

    out = Path(args.out).resolve() if args.out else (src / "dataset_split").resolve()

    exts = IMAGE_EXTS
    if args.extensions:
        exts = {("." + e.strip().lstrip(".")).lower() for e in args.extensions.split(",") if e.strip()}
        if not exts:
            exts = IMAGE_EXTS

    ratio = parse_ratio(args.ratio)

    pairs = collect_pairs(src, recursive=args.recursive, out_dir=out, exts=exts, use_yolo=args.use_yolo, labels_dir_name=args.labels_dir)
    if not pairs:
        print("找不到任何成對的 影像+標註（YOLO 或 JSON）。請檢查來源資料夾與副檔名設定。")
        return 0

    train_idx, val_idx, test_idx = split_indices(len(pairs), ratio, seed=args.seed)

    for split in ("train", "val", "test"):
        for leaf in ("images", "labels"):
            ensure_dir(out / split / leaf, dry=args.dry_run)

    def dst_path(split: str, leaf: str, rel: Path) -> Path:
        return out / split / leaf / rel

    def handle_indices(indices: Sequence[int], split: str) -> None:
        for i in indices:
            pr = pairs[i]
            # Avoid duplicating leading 'images' in destination under split/images
            img_rel_out = strip_leading_component(pr.img_rel, "images")

            # For labels, avoid leading 'labels' or 'images' to keep mirror structure clean
            if pr.is_yolo:
                lbl_rel_out = strip_leading_component(pr.lbl_rel, "labels")
                lbl_rel_out = strip_leading_component(lbl_rel_out, "images")
            else:
                # JSON labels are relative to src; typically mirror image path -> may start with 'images'
                lbl_rel_out = strip_leading_component(pr.lbl_rel, "images")

            copy_or_move(src / pr.img_rel, dst_path(split, "images", img_rel_out), move=args.move, dry=args.dry_run)
            copy_or_move(pr.lbl_src, dst_path(split, "labels", lbl_rel_out), move=args.move, dry=args.dry_run)

    handle_indices(train_idx, "train")
    handle_indices(val_idx, "val")
    handle_indices(test_idx, "test")

    print("\n完成分割：")
    print(f"  train: {len(train_idx)} 組")
    print(f"  val  : {len(val_idx)} 組")
    print(f"  test : {len(test_idx)} 組")
    print(f"輸出位置：{out}")
    if args.dry_run:
        print("（dry-run 僅顯示操作，未實際複製/移動）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
