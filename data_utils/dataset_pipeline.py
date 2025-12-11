from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IMAGE_EXTS_SET = {ext.lower() for ext in IMAGE_EXTS}
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ReorgResult:
    images_dir: str
    labels_dir: str
    ok: bool


@dataclass
class MergeOutcome:
    ok: bool
    added_images: list[str]


def _strip_leading_component(rel_path: str, name: str) -> str:
    """
    去掉路徑開頭重複的資料夾名稱

    """
    rel_norm = rel_path.replace("\\", "/").lstrip("/")
    parts = [p for p in rel_norm.split("/") if p]
    name_lower = name.lower()
    while parts and parts[0].lower() == name_lower:
        parts = parts[1:]
    return "/".join(parts)


def _unique_path(path: str) -> str:
    """
    「讓檔案路徑變成唯一的名字」。
    如果那個檔案已經存在，它會在檔名後面加上隨機的代碼，避免重覆
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    return f"{base}-{uuid.uuid4().hex[:8]}{ext}"


def _collapse_nested_dir(parent_dir: str, child_name: str) -> bool:
    """
    把 parent_dir/child_name 裡的所有東西搬到 parent_dir 裡，然後刪掉 child_name 資料夾
    """
    nested = os.path.join(parent_dir, child_name)
    if not os.path.isdir(nested):
        return True
    ok = True
    try:
        for root, dirs, files in os.walk(nested):
            rel = os.path.relpath(root, nested)
            dst_root = parent_dir if rel == "." else os.path.join(parent_dir, rel)
            os.makedirs(dst_root, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(dst_root, d), exist_ok=True)
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(dst_root, f)
                if os.path.exists(dst):
                    dst = _unique_path(dst)
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)
    except Exception:
        ok = False
    return ok


def _flatten_subfolder(base_dir: str, sub_name: str) -> bool:
    sub_dir = os.path.join(base_dir, sub_name)
    if not os.path.isdir(sub_dir):
        return True
    ok = True
    try:
        for root, dirs, files in os.walk(sub_dir):
            rel = os.path.relpath(root, sub_dir)
            dst_root = base_dir if rel == "." else os.path.join(base_dir, rel)
            os.makedirs(dst_root, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(dst_root, d), exist_ok=True)
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(dst_root, f)
                if os.path.exists(dst):
                    dst = _unique_path(dst)
                shutil.move(src, dst)
        shutil.rmtree(sub_dir, ignore_errors=True)
    except Exception:
        ok = False
    return ok


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


async def save_upload_zip(upload_file: Any, upload_dir: str, base_filename: str) -> str:
    _ensure_dir(upload_dir)
    dest_path = _unique_path(os.path.join(upload_dir, base_filename))
    try:
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(upload_file.file, out)
    finally:
        await upload_file.close()
    return dest_path


def create_processing_dir(project_root: str) -> str:
    _ensure_dir(project_root)
    tmp_root = os.path.join(project_root, "__tmp__")
    _ensure_dir(tmp_root)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]
    proc_dir = os.path.join(tmp_root, suffix)
    _ensure_dir(proc_dir)
    return proc_dir


def safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    _ensure_dir(dest_dir)
    dest_dir_abs = os.path.abspath(dest_dir) + os.sep
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = os.path.abspath(os.path.join(dest_dir, member.filename))
            if not target.startswith(dest_dir_abs):
                continue
            if member.is_dir():
                _ensure_dir(target)
            else:
                _ensure_dir(os.path.dirname(target))
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def extract_nc_names(base_dir: str, scan_root: str) -> str:
    cmd = [
        sys.executable,
        os.path.join(base_dir, "extract_nc_names.py"),
        "--dir",
        scan_root,
        "--recursive",
        "--yaml-only",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode == 0:
        return proc.stdout or ""
    return ""


def json_to_yolo(base_dir: str, src_root: str, out_dir: str) -> int:
    _ensure_dir(out_dir)
    cmd = [
        sys.executable,
        os.path.join(base_dir, "json_to_yolo.py"),
        "--src",
        src_root,
        "--out",
        out_dir,
        "--recursive",
        "--overwrite",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return proc.returncode
    except Exception:
        return -1


def _is_under(path: str, base: str) -> bool:
    path_abs = os.path.abspath(path) + os.sep
    base_abs = os.path.abspath(base) + os.sep
    return path_abs.startswith(base_abs)


def reorganize_converted_tree(temp_root: str) -> ReorgResult:
    labels_yolo = os.path.join(temp_root, "labels_yolo")
    images_dir = os.path.join(temp_root, "images")
    labels_dir = os.path.join(temp_root, "labels")
    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)
    ok = True

    try:
        for root, _dirs, files in os.walk(labels_yolo):
            for fn in files:
                if not fn.lower().endswith(".txt"):
                    continue
                src_txt = os.path.join(root, fn)
                rel = os.path.relpath(src_txt, labels_yolo)
                rel_clean = _strip_leading_component(_strip_leading_component(rel, "labels"), "images")
                dst_txt = os.path.join(labels_dir, rel_clean)
                _ensure_dir(os.path.dirname(dst_txt))
                try:
                    shutil.move(src_txt, dst_txt)
                except Exception:
                    ok = False
                    continue
                rel_dir = os.path.dirname(rel_clean)
                rel_dir = _strip_leading_component(rel_dir, "images")
                stem = os.path.splitext(fn)[0]
                src_img_dir = os.path.join(temp_root, rel_dir)
                for ext in IMAGE_EXTS:
                    candidate = os.path.join(src_img_dir, stem + ext)
                    if os.path.exists(candidate):
                        dst_img = os.path.join(images_dir, rel_dir, stem + ext)
                        _ensure_dir(os.path.dirname(dst_img))
                        try:
                            shutil.move(candidate, dst_img)
                        except Exception:
                            ok = False
                        break
    except Exception:
        ok = False

    try:
        skip_dirs = {images_dir, labels_dir, labels_yolo}
        for root, _dirs, files in os.walk(temp_root):
            if any(_is_under(root, skip) for skip in skip_dirs):
                continue
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in IMAGE_EXTS_SET:
                    continue
                src_img = os.path.join(root, fn)
                rel = os.path.relpath(src_img, temp_root)
                rel = _strip_leading_component(rel, "images")
                dst_img = os.path.join(images_dir, rel)
                _ensure_dir(os.path.dirname(dst_img))
                try:
                    shutil.move(src_img, dst_img)
                except Exception:
                    ok = False
    except Exception:
        ok = False

    try:
        shutil.rmtree(labels_yolo, ignore_errors=True)
    except Exception:
        ok = False

    if not _flatten_subfolder(images_dir, "test_data"):
        ok = False
    if not _flatten_subfolder(labels_dir, "test_data"):
        ok = False
    if not _collapse_nested_dir(images_dir, "images"):
        ok = False
    if not _collapse_nested_dir(labels_dir, "labels"):
        ok = False

    return ReorgResult(images_dir=images_dir, labels_dir=labels_dir, ok=ok)


def _resolve_pair_destinations(img_dir: str, lbl_dir: str, stem: str, ext: str) -> tuple[str, str]:
    dst_img = os.path.join(img_dir, stem + ext)
    dst_lbl = os.path.join(lbl_dir, stem + ".txt")
    if not os.path.exists(dst_img) and not os.path.exists(dst_lbl):
        return dst_img, dst_lbl
    for _ in range(5):
        suffix = "-" + uuid.uuid4().hex[:6]
        cand_img = os.path.join(img_dir, stem + suffix + ext)
        cand_lbl = os.path.join(lbl_dir, stem + suffix + ".txt")
        if not os.path.exists(cand_img) and not os.path.exists(cand_lbl):
            return cand_img, cand_lbl
    suffix = "-" + uuid.uuid4().hex[:10]
    return (
        os.path.join(img_dir, stem + suffix + ext),
        os.path.join(lbl_dir, stem + suffix + ".txt"),
    )


def merge_into_dataset(
    images_dir: str,
    labels_dir: str,
    dst_images_root: str,
    dst_labels_root: str,
) -> MergeOutcome:
    _ensure_dir(dst_images_root)
    _ensure_dir(dst_labels_root)
    ok = True
    added: list[str] = []

    for root, _dirs, files in os.walk(images_dir):
        rel_dir = os.path.relpath(root, images_dir)
        if rel_dir == ".":
            rel_dir = ""
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMAGE_EXTS_SET:
                continue
            src_img = os.path.join(root, fn)
            dst_img_dir = os.path.join(dst_images_root, rel_dir)
            dst_lbl_dir = os.path.join(dst_labels_root, rel_dir)
            _ensure_dir(dst_img_dir)
            _ensure_dir(dst_lbl_dir)
            stem = os.path.splitext(fn)[0]
            dst_img, dst_lbl = _resolve_pair_destinations(dst_img_dir, dst_lbl_dir, stem, os.path.splitext(fn)[1])
            src_lbl = os.path.join(labels_dir, rel_dir, stem + ".txt")
            try:
                shutil.move(src_img, dst_img)
                if os.path.exists(src_lbl):
                    _ensure_dir(os.path.dirname(dst_lbl))
                    shutil.move(src_lbl, dst_lbl)
                added.append(os.path.abspath(dst_img).replace("\\", "/"))
            except Exception:
                ok = False
    return MergeOutcome(ok=ok, added_images=added)


def move_jsons(temp_root: str, jsons_root: str) -> bool:
    _ensure_dir(jsons_root)
    ok = True
    try:
        for root, _dirs, files in os.walk(temp_root):
            for fn in files:
                if not fn.lower().endswith(".json"):
                    continue
                src_json = os.path.join(root, fn)
                rel = os.path.relpath(src_json, temp_root)
                rel = _strip_leading_component(_strip_leading_component(rel, "images"), "labels")
                dst_json = os.path.join(jsons_root, rel)
                _ensure_dir(os.path.dirname(dst_json))
                try:
                    shutil.move(src_json, dst_json)
                except Exception:
                    ok = False
    except Exception:
        ok = False
    return ok


def cleanup_dir(path: str) -> bool:
    try:
        shutil.rmtree(path, ignore_errors=True)
        return True
    except Exception:
        return False


def remove_file(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except Exception:
        return False
