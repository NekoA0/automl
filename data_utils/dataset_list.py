from fastapi import APIRouter, Query
import os, json, datetime
from utils.user_utils import normalize_user_name

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_STATUS_FILE = os.path.join(BASE_DIR, "dataset_status_map.json")
DATASET_HOME_DIR = os.path.join(BASE_DIR, "dataset_home")  # 相對於專案根目錄調整可視需要

router = APIRouter(
    prefix="/yolov9",
    tags=["GET_DATA_NAME"],
)


def _normalize_user(name: str | None) -> str:
    return normalize_user_name(name)


def _format_ts(ts: float | None) -> str:
    try:
        if ts is None:
            return ""
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _safe_load_cache() -> dict:
    try:
        if not os.path.exists(DATASET_STATUS_FILE):
            return {}
        with open(DATASET_STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        try:
            bak = DATASET_STATUS_FILE + '.bak'
            if os.path.exists(bak):
                with open(bak, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
    return {}


def _safe_save_cache(data: dict):
    try:
        tmp = DATASET_STATUS_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, DATASET_STATUS_FILE)
        try:
            with open(DATASET_STATUS_FILE + '.bak', 'w', encoding='utf-8') as fb:
                json.dump(data, fb, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception:
        pass


def _collect_versions(dataset_dir: str) -> list[str]:
    versions_dir = os.path.join(dataset_dir, 'versions')
    out: list[str] = []
    if not os.path.isdir(versions_dir):
        return out
    for nm in os.listdir(versions_dir):
        p = os.path.join(versions_dir, nm)
        if not os.path.isdir(p):
            continue
        if nm.lower().startswith('v') and len(nm) > 1:
            suffix = nm[1:]
            # 僅保留數字部分當排序與輸出（例：v01 -> 1）
            if suffix.isdigit():
                out.append(str(int(suffix)))
            else:
                # 非純數字則原樣（去掉 v）
                out.append(suffix)
    # 數字優先排序（轉 int 失敗時放最後）
    def _key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)
    return sorted(out, key=_key)


def _count_samples(dataset_dir: str) -> int:
    """估算資料量：優先統計 train/val/test.txt 總行數（去重非空）。"""
    txt_names = ['train.txt', 'val.txt', 'test.txt']
    lines_set = set()
    for tname in txt_names:
        p = os.path.join(dataset_dir, tname)
        if os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        ln = line.strip()
                        if ln:
                            lines_set.add(ln)
            except Exception:
                continue
    if lines_set:
        return len(lines_set)
    # fallback：計算 images 子目錄內的檔案數
    images_dir = os.path.join(dataset_dir, 'images')
    cnt = 0
    for root, dirs, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')):
                cnt += 1
    return cnt

def _gather_user_datasets(user: str) -> list[dict]:
    user_dir = os.path.join(DATASET_HOME_DIR, user)
    if not os.path.isdir(user_dir):
        return []
    items: list[dict] = []
    for dataset_name in sorted(os.listdir(user_dir)):
        dpath = os.path.join(user_dir, dataset_name)
        if not os.path.isdir(dpath):
            continue
        # 基本時間
        try:
            ctime = os.path.getctime(dpath)
        except Exception:
            ctime = None
        try:
            # updated 取資料夾本身與其 versions 內所有檔案/資料夾最大 mtime
            max_mtime = os.path.getmtime(dpath)
            ver_root = os.path.join(dpath, 'versions')
            if os.path.isdir(ver_root):
                for r, dirs, files in os.walk(ver_root):
                    for nm in dirs + files:
                        p = os.path.join(r, nm)
                        try:
                            mt = os.path.getmtime(p)
                            if mt > max_mtime:
                                max_mtime = mt
                        except Exception:
                            pass
        except Exception:
            max_mtime = None

        versions = _collect_versions(dpath)
        count = _count_samples(dpath)

        items.append({
            "name":         dataset_name,
            "count":        count,
            "created_date": _format_ts(ctime),
            "updated_date": _format_ts(max_mtime),
            "version":      versions
        })
    return items

def build_dataset_map(target_user: str | None = None) -> dict:
    result: dict = {}
    if target_user:
        user_norm = _normalize_user(target_user)
        result[user_norm] = _gather_user_datasets(user_norm)
        return result
    # 全部使用者：掃描 dataset_home 底下一階層
    if not os.path.isdir(DATASET_HOME_DIR):
        return {}
    for user in sorted(os.listdir(DATASET_HOME_DIR)):
        udir = os.path.join(DATASET_HOME_DIR, user)
        if not os.path.isdir(udir):
            continue
        user_norm = _normalize_user(user)
        result[user_norm] = _gather_user_datasets(user_norm)
    return result


@router.get("/Get_data_name")
def Get_data_name(user_name: str = Query("", description="使用者名稱（預設全部）")):

    single_user    = bool(user_name.strip())
    data_map       = build_dataset_map(user_name or None)
    # 寫入快取檔（覆蓋更新對應 user 或全部）
    cache = _safe_load_cache()
    for k, v in data_map.items():
        cache[k] = v
    _safe_save_cache(cache)

    if single_user:
        # 只回傳陣列
        key = _normalize_user(user_name)
        return data_map.get(key, [])
    return data_map