from fastapi import APIRouter, HTTPException
import os, shutil, subprocess
from pathlib import Path
from pydantic import BaseModel
from utils.user_utils import ensure_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["UPLOAD_DATA"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_HOME_DIR  = BASE_DIR / "dataset_home"
THUMBS_BASE_DIR   = Path("/shared/thumbs").resolve()


def _next_snapshot_version(versions_root: str) -> int:
    """回傳 versions_root 底下快照目錄 v01、v02 等的下一個版本序號。"""
    try:
        versions_root = Path(versions_root)
        versions_root.mkdir(parents=True, exist_ok=True)

        mx = 0
        for p in versions_root.iterdir():  # 列出所有子目錄
            if not p.is_dir():
                continue
            name = p.name
            if name.lower().startswith("v") and len(name) >= 3:
                try:
                    n = int(name[1:])
                    mx = max(mx, n)
                except Exception:
                    continue
        return mx + 1
    except Exception:
        return 1


def _fmt_ver(n: int) -> str:
    return f"v{n:02d}"

def _snapshot_txts(split_root: str, version_int: int | None = None) -> str:
    """將 split_root 目前的 train.txt/val.txt/test.txt 複製到 versions/vXX/，並回傳版本名稱。

    若 version_int 為 None，會自動遞增下一個版本號；回傳值為類似 'v01' 的資料夾名稱。
    """
    versions_dir = os.path.join(split_root, 'versions')
    if version_int is None:
        version_int = _next_snapshot_version(versions_dir)
    ver_name = _fmt_ver(version_int)
    ver_dir = os.path.join(versions_dir, ver_name)
    os.makedirs(ver_dir, exist_ok=True)
    for part in ('train', 'val', 'test'):
        src = os.path.join(split_root, f'{part}.txt')
        dst = os.path.join(ver_dir, f'{part}.txt')
        try:
            if os.path.isfile(src):
                with open(src, 'r', encoding='utf-8') as fs, open(dst, 'w', encoding='utf-8') as fd:
                    fd.write(fs.read())
            else:
                # create empty file if missing
                open(dst, 'w', encoding='utf-8').close()
        except Exception:
            pass
    return ver_name

def _run_gen_thumbs(base_dir: str, src_path: str, dest_path: str, txt_path: str | None, log_path: str):
    """以絕對路徑呼叫 GenThumbs.exe 的 --src、--dest，以及選擇性的 --txt 參數。
    - src_path：影像根目錄的絕對路徑（例如 dataset_home/<user>/<project>/images 或專案根）
    - dest_path：/shared/Thumbs/<user>/<project> 的絕對路徑
    - txt_path：包含 train/val/test.txt 的 versions/<vXX> 目錄絕對路徑，或 None
    """
    try:
        exe_path = os.path.join(base_dir, "GenThumbs")
        cmd = [exe_path, "--src", src_path, "--dest", dest_path]
        if txt_path:
            cmd += ["--txt", txt_path]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                log_file.write(f"CMD: {' '.join(cmd)}\nCWD: {base_dir}\n\n")
            except Exception:
                pass
            if not os.path.isfile(exe_path):
                log_file.write("ERROR: GenThumbs.exe not found.\n")
                return
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=base_dir)
    except Exception:
        pass

class RemoveRequest(BaseModel):
    user_name: str
    dataset_name: str
    train: list[str] | None = None
    val: list[str] | None = None
    test: list[str] | None = None

@router.post("/dataset/remove")
def remove_by_part(req: RemoveRequest):
    """依分割別刪除清單中的影像（僅從 txt 移除，不刪實體檔案），並建立新版本快照。"""
    # 正規化使用者/專案
    safe_user = ensure_user_name(req.user_name)
    safe_dataset = "".join(ch for ch in (req.dataset_name or "") if ch.isalnum() or ch in ("_", "-"))
    if not safe_dataset:
        raise HTTPException(status_code=400, detail="dataset_name 不合法")

    split_root = os.path.join(DATASET_HOME_DIR, safe_user, safe_dataset)
    if not os.path.isdir(split_root):
        raise HTTPException(status_code=404, detail="找不到資料夾：dataset_home/<user>/<project>")

    targets: dict[str, set[str]] = {}
    for part in ("train", "val", "test"):
        arr = getattr(req, part)
        if arr:
            targets[part] = set(os.path.basename(x).strip().lower() for x in arr if isinstance(x, str) and x.strip())

    if not targets:
        return {"updated": False, "reason": "無有效刪除清單"}

    def _read(path: str) -> list[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [ln.rstrip("\n") for ln in f]
        except Exception:
            return []

    def _write(path: str, lines: list[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    summary: dict[str, dict] = {}
    total_removed = 0
    total_kept = 0
    sample_removed: list[str] = []

    for part, names in targets.items():
        txt_path = os.path.join(split_root, f"{part}.txt")
        lines = _read(txt_path)
        if not lines:
            summary[part] = {"found": 0, "removed": 0, "kept": 0}
            continue
        keep: list[str] = []
        removed: list[str] = []
        for ln in lines:
            bn = os.path.basename(ln.strip()).lower()
            if bn in names:
                removed.append(ln)
            else:
                keep.append(ln)
        summary[part] = {"found": len(lines), "removed": len(removed), "kept": len(keep)}
        total_removed += len(removed)
        total_kept += len(keep)
        sample_removed.extend(removed[:5])
        _write(txt_path, keep)

    # 每次刪除後皆建立新版本快照
    version_name: str | None = None
    try:
        version_name = _snapshot_txts(split_root)
    except Exception:
        version_name = None

    # 產生版本化縮圖到 /shared/Thumbs/<user>/<project>/versions/<vXX>
    try:
        if version_name:
            src_dir = os.path.join(split_root, 'images')
            if not os.path.isdir(src_dir):
                src_dir = split_root
            src_abs = os.path.abspath(src_dir)
            txt_abs = os.path.abspath(os.path.join(split_root, 'versions', version_name))
            dest_abs = os.path.join(THUMBS_BASE_DIR, safe_user, safe_dataset)
            os.makedirs(dest_abs, exist_ok=True)
            log_path = os.path.join(dest_abs, 'thumbs.log')
            _run_gen_thumbs(BASE_DIR, src_abs, dest_abs, txt_abs, log_path)
            # 複製 thumbs.json 至 dataset_home 根與版本資料夾
            try:
                thumbs_json_src = os.path.join(dest_abs, 'thumbs.json')
                if os.path.isfile(thumbs_json_src):
                    ver_dir = os.path.join(split_root, 'versions', version_name)
                    os.makedirs(ver_dir, exist_ok=True)
                    shutil.copyfile(thumbs_json_src, os.path.join(ver_dir, 'thumbs.json'))
                    shutil.copyfile(thumbs_json_src, os.path.join(split_root, 'thumbs.json'))
            except Exception:
                pass
    except Exception:
        pass

    return {
        "updated": True,
        "version_name": version_name,
        "version": (int(version_name[1:]) if version_name and version_name[1:].isdigit() else None)
    }