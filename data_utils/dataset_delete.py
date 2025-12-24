from fastapi import APIRouter, Form, HTTPException
import os, shutil, json, threading
from pathlib import Path
from utils.user_utils import ensure_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["UPLOAD_DATA"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DATA_DIR   = BASE_DIR / "upload_data_folder"
DATASET_DIR       = BASE_DIR / "Dataset"
DATASET_HOME_DIR  = BASE_DIR / "dataset_home"
THUMBS_BASE_DIR   = Path("/shared/thumbs").resolve()

DS_STATUS_FILE = BASE_DIR / "dataset_status_map.json"
ds_status_lock = threading.Lock()

def ds_save_status_map(status_map: dict | list):
    with ds_status_lock:
        tmp_path     = DS_STATUS_FILE + ".tmp"
        bak_path     = DS_STATUS_FILE + ".bak"
        ds_file_path = DS_STATUS_FILE

        try:
            # 寫入暫存檔
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(status_map, f, ensure_ascii=False, indent=2)
            
            # 原子性覆蓋正式檔
            tmp_path.replace(ds_file_path)

            # 嘗試寫備份檔
            try:
                with bak_path.open("w", encoding="utf-8") as fb:
                    json.dump(status_map, fb, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception:# 嘗試刪除暫存檔
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

def ds_load_status_map() -> dict | list:
    with ds_status_lock:
        def _read(path: str):
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            if not data.strip():
                raise json.JSONDecodeError("empty", data, 0)
            return json.loads(data)
        
        if os.path.exists(DS_STATUS_FILE):
            try:
                return _read(DS_STATUS_FILE)
            except json.JSONDecodeError:
                bak = DS_STATUS_FILE + ".bak"
                if os.path.exists(bak):
                    try:
                        return _read(bak)
                    except Exception:
                        return {}
                return {}
            except Exception:
                return {}
        return {}

def _status_list_delete(user: str, name: str) -> bool:
    """在以使用者分組的清單結構中刪除符合 (user, name) 的資料，同時清除轉換時遺留的舊格式。"""
    try:
        cur = ds_load_status_map()
        grouped = _to_by_user(cur)
        if user not in grouped:
            return False
        before = len(grouped[user])
        grouped[user] = [e for e in grouped[user] if not (isinstance(e, dict) and e.get("name") == name)]
        if len(grouped[user]) != before:
            ds_save_status_map(grouped)
            return True
    except Exception:
        pass
    return False

def _to_by_user(obj: dict | list) -> dict:
    """將狀態資料轉換為 { user: [entries...] } 結構，支援 list 或 dict 形式的輸入。"""
    grouped: dict[str, list[dict]] = {}
    try:
        if isinstance(obj, list):
            for e in obj:
                if not isinstance(e, dict):
                    continue
                user = (e.get("user") or "").strip()
                # store entry without 'user'
                ent = {k: v for k, v in e.items() if k != 'user'}
                grouped.setdefault(user, []).append(ent)
            return grouped
        if isinstance(obj, dict):
            for user_key, v in obj.items():
                if isinstance(v, list):
                    # ensure 'user' removed in each entry
                    grouped[user_key] = [{k: val for k, val in e.items() if k != 'user'} for e in v if isinstance(e, dict)]
                elif isinstance(v, dict):
                    lst: list[dict] = []
                    for proj_key, entry in v.items():
                        if isinstance(entry, dict):
                            ent = dict(entry)
                            # enforce correct name; do not store user
                            ent.setdefault('name', proj_key)
                            lst.append(ent)
                    grouped[user_key] = lst
                else:
                    grouped.setdefault(user_key, [])
            return grouped
    except Exception:
        return {}
    return {}

# 真的刪除 Dataset：移除資料夾與狀態
@router.post("/dataset/delete")
def delete_dataset(
    user_name: str     = Form("", description="使用者名稱（自動轉小寫）"),
    dataset_name: str  = Form("", description="資料集名稱"),
):
    """刪除指定使用者的資料集（包含縮圖與暫存）：
    - 刪除 dataset_home/<user>/<dataset>
    - 刪除 Dataset/<user>/<dataset>
    - 刪除 /shared/Thumbs/<user>/<dataset>
    - 刪除 upload_data_folder/<user>/<dataset>
    - 從 dataset_status_map.json 移除對應 name 的條目
    """
    # 正規化
    safe_user = ensure_user_name(user_name)
    safe_dataset = "".join(ch for ch in (dataset_name or "") if ch.isalnum() or ch in ("_", "-"))
    if not safe_dataset:
        raise HTTPException(status_code=400, detail="dataset_name 不合法")

    # 目標路徑
    home_target     = os.path.join(DATASET_HOME_DIR, safe_user, safe_dataset)
    ds_target       = os.path.join(DATASET_DIR, safe_user, safe_dataset)
    thumbs_target   = os.path.join(THUMBS_BASE_DIR, safe_user, safe_dataset)
    upload_proj_dir = os.path.join(UPLOAD_DATA_DIR, safe_user, safe_dataset)

    def _safe_remove_dir(target: str, base: str | None = None) -> bool:
        try:
            targ_abs = os.path.abspath(target)
            if not os.path.isdir(targ_abs):
                return False
            if base:
                base_abs = os.path.abspath(base)
                try:
                    if os.path.commonpath([base_abs, targ_abs]) != base_abs:
                        return False
                except ValueError:
                    return False
            shutil.rmtree(targ_abs, ignore_errors=True)
            return True
        except Exception:
            return False

    removed = {
        "dataset_home": _safe_remove_dir(home_target, DATASET_HOME_DIR),
        "Dataset":      _safe_remove_dir(ds_target, DATASET_DIR),
        "Thumbs":       _safe_remove_dir(thumbs_target, THUMBS_BASE_DIR),
        "upload_tmp":   _safe_remove_dir(upload_proj_dir, UPLOAD_DATA_DIR),
    }

    status_removed = _status_list_delete(safe_user, safe_dataset)

    return {
        "done": True,
        "removed": removed,
        "status_removed": status_removed,
    }