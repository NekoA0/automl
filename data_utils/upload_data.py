from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os, uuid, zipfile, shutil, subprocess, json, threading, random
from datetime import datetime

import dataset_pipeline as dp
from user_utils import ensure_user_name


router = APIRouter(
    prefix="/yolov9",
    tags=["UPLOAD_DATA"],
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DATA_DIR = os.path.join(BASE_DIR, "upload_data_folder")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DATASET_HOME_DIR = os.path.join(BASE_DIR, "dataset_home")
THUMBS_BASE_DIR = os.path.abspath(os.path.join(os.path.sep, "shared", "Thumbs"))

# Dataset 狀態檔與鎖
DS_STATUS_FILE = os.path.join(BASE_DIR, "dataset_status_map.json")
ds_status_lock = threading.Lock()


def ds_save_status_map(status_map: dict | list):
    with ds_status_lock:
        tmp_path = DS_STATUS_FILE + ".tmp"
        bak_path = DS_STATUS_FILE + ".bak"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(status_map, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, DS_STATUS_FILE)
            try:
                with open(bak_path, "w", encoding="utf-8") as fb:
                    json.dump(status_map, fb, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
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


def _status_list_upsert(user: str, name: str, count: int, now_str: str):
    """依 (user, name) 更新或插入一筆資料，並保留既有的 created_date。"""
    try:
        cur = ds_load_status_map()
        grouped = _to_by_user(cur)
        lst = list(grouped.get(user, []))
        idx = None
        created_date = now_str
        for i, e in enumerate(lst):
            if isinstance(e, dict) and e.get("name") == name:
                idx = i
                cd = e.get("created_date")
                if isinstance(cd, str) and cd.strip():
                    created_date = cd
                break
        new_entry = {
            "name":         name,
            "count":        int(count),
            "created_date": created_date,
            "updated_date": now_str,
        }
        if idx is None:
            lst.append(new_entry)
        else:
            lst[idx] = new_entry
        grouped[user] = lst
        ds_save_status_map(grouped)
    except Exception:
        pass


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


def _dataset_name_exists_for_user(user: str, name: str) -> bool:
    """檢查指定使用者命名空間下是否已存在資料集（僅檔案系統層面）。"""
    try:
        p1 = os.path.join(DATASET_HOME_DIR, user, name)
        p2 = os.path.join(DATASET_DIR, user, name)
        return os.path.isdir(p1) or os.path.isdir(p2)
    except Exception:
        return False




def _count_images_in_dataset(base_dir: str, user: str, project: str) -> int:
    """遞迴統計 dataset_home/<user>/<project>/images 內的影像檔案數量。"""
    try:
        images_root = os.path.join(base_dir, "dataset_home", user, project, "images")
        if not os.path.isdir(images_root):
            # fallback: count under project root if images/ doesn't exist yet
            images_root = os.path.join(base_dir, "dataset_home", user, project)
            if not os.path.isdir(images_root):
                return 0
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        cnt = 0
        for root, _dirs, files in os.walk(images_root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    cnt += 1
        return cnt
    except Exception:
        return 0


def _next_snapshot_version(versions_root: str) -> int:
    """回傳 versions_root 底下快照目錄 v01、v02 等的下一個版本序號。"""
    try:
        os.makedirs(versions_root, exist_ok=True)
        mx = 0
        for name in os.listdir(versions_root):
            if not os.path.isdir(os.path.join(versions_root, name)):
                continue
            if name.lower().startswith('v') and len(name) >= 3:
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


# ---- Helpers for updating Dataset/<user>/<project>/data.yaml ----
def _yaml_single_quote(s: str) -> str:
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace("'", "''")
    return f"'{s}'"


def _parse_names_from_yaml_text(text: str) -> list[str] | None:
    try:
        lines = [ln.rstrip() for ln in text.splitlines()]
        idx = None
        for i, ln in enumerate(lines):
            if ln.strip().startswith("names:"):
                idx = i
                break
        if idx is None:
            return None
        # flow style on same line
        after = lines[idx].split(":", 1)[1].strip() if ":" in lines[idx] else ""
        if after.startswith("[") and after.endswith("]"):
            inner = after[1:-1]
            out, token, quote = [], "", None
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
            out = [s.strip().strip("'\"") for s in out]
            return [s for s in out if s]
        # block style list/map
        i = idx + 1
        items: list[str] = []
        pairs: dict[str, int] = {}
        while i < len(lines):
            ln = lines[i]
            if ln.strip() == "" or ln.lstrip().startswith("#"):
                i += 1
                continue
            if not (ln.startswith(" ") or ln.startswith("\t")):
                break
            stripped = ln.strip()
            if stripped.startswith("- "):
                val = stripped[2:].strip().strip("'\"")
                if val:
                    items.append(val)
            else:
                if ":" in stripped:
                    key, _, rest = stripped.partition(":")
                    key = key.strip().strip("'\"")
                    try:
                        idxv = int(rest.strip())
                    except Exception:
                        idxv = None
                    if key and idxv is not None:
                        pairs[key] = idxv
            i += 1
        if items:
            return items
        if pairs:
            return [k for k, _ in sorted(pairs.items(), key=lambda kv: kv[1])]
    except Exception:
        return None
    return None


def _scan_label_max_id(labels_root: str) -> int:
    """遞迴掃描 labels_root 下的 YOLO 標註檔，回傳最大類別編號；若不存在則回傳 -1。"""
    max_id = -1
    for root, _dirs, files in os.walk(labels_root):
        for fn in files:
            if not fn.lower().endswith('.txt'):
                continue
            fpath = os.path.join(root, fn)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            cid = int(parts[0])
                            if cid > max_id:
                                max_id = cid
                        except Exception:
                            continue
            except Exception:
                continue
    return max_id


def _write_dataset_yaml(base_dir: str, user: str, project: str, names: list[str]) -> str:
    """將設定寫入 Dataset/<user>/<project>/data.yaml 並回傳檔案路徑。"""
    data_yaml_path = os.path.join(base_dir, 'Dataset', user, project, 'data.yaml')
    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)
    # path 指向 dataset_home/<user>/<project>（扁平化結構）
    ds_path = os.path.join(base_dir, 'dataset_home', user, project)
    ds_path_norm = os.path.abspath(ds_path).replace('\\', '/')
    lines = [
        f"path: {ds_path_norm}",
        "test:  test.txt",
        "train: train.txt",
        "val:   val.txt",
        f"nc:   {len(names)}",
        "names:",
    ]
    for n in names:
        lines.append(f"  - {_yaml_single_quote(n)}")
    with open(data_yaml_path, 'w', encoding='utf-8') as yf:
        yf.write("\n".join(lines) + "\n")
    return data_yaml_path


async def _upload_data_zip_impl(
    file:           UploadFile,
    user_name:      str,
    dataset_name:   str | None = None,
    task:           str | None = None,
    delete_zip:     bool = False,
):
    time_fmt = "%Y-%m-%d %H:%M:%S"
    # 僅允許 zip
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="只接受 .zip 檔")

    os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(DATASET_HOME_DIR, exist_ok=True)

    safe_user = ensure_user_name(user_name)

    # 檢查/標準化 dataset_name（若新參數空，落回舊 task）
    ds_in = dataset_name or task or ""
    safe_dataset = "".join(ch for ch in ds_in if ch.isalnum() or ch in ("_", "-"))
    if not safe_dataset:
        raise HTTPException(status_code=400, detail="dataset_name 不可為空或包含非法字元")

    # 檢查重複（user 範圍內唯一）
    if _dataset_name_exists_for_user(safe_user, safe_dataset):
        raise HTTPException(status_code=409, detail="dataset_name 已存在")

    # 先將上傳 zip 改名為 <dataset_name>.zip（避免重名再補唯一字尾）
    filename = f"{safe_dataset}.zip"

    # 儲存 zip（避免重名再補唯一字尾）
    save_path = await dp.save_upload_zip(file, UPLOAD_DATA_DIR, filename)

    # 初始化時間字串供後續流程使用
    now_str = datetime.now().strftime(time_fmt)

    # 更新資料集狀態（扁平清單）：初始化資料 {name, count, created_date, updated_date}
    try:
        # 初始階段先記 0（處理完會再更新）
        _status_list_upsert(safe_user, safe_dataset, 0, now_str)
    except Exception:
        # 狀態檔失敗不阻斷主要流程
        pass

    # 立即在 upload_data_folder/<user>/<dataset_name>/__tmp__/<unique> 解壓縮（保留既有未分割資料，最終合併到固定路徑）
    extract_base = os.path.join(UPLOAD_DATA_DIR, safe_user, safe_dataset)  # 固定專案根
    os.makedirs(extract_base, exist_ok=True)

    unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]
    extract_root = os.path.join(extract_base, "__tmp__", unique_suffix)  # 暫存處理資料夾
    os.makedirs(extract_root, exist_ok=True)

    try:
        dp.safe_extract_zip(save_path, extract_root)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="zip 檔損毀，無法解壓縮")

    nc_names_yaml = dp.extract_nc_names(BASE_DIR, extract_root) or ""

    labels_yolo_dir = os.path.join(extract_root, "labels_yolo")
    rc = dp.json_to_yolo(BASE_DIR, extract_root, labels_yolo_dir)

    reorg_result = dp.reorganize_converted_tree(extract_root)
    dst_images_root = os.path.join(extract_base, 'images')
    dst_labels_root = os.path.join(extract_base, 'labels')
    merge_result = dp.merge_into_dataset(
        reorg_result.images_dir,
        reorg_result.labels_dir,
        dst_images_root,
        dst_labels_root,
    )
    jsons_root = os.path.join(extract_base, 'jsons')
    move_jsons_ok = dp.move_jsons(extract_root, jsons_root)
    reorg_ok = reorg_result.ok and merge_result.ok and move_jsons_ok

    dp.cleanup_dir(extract_root)

    # 先複製一份完整處理後的資料到 dataset_home/<user>/<dataset_name>
    full_copy_ok = True
    dataset_home_target = os.path.join(DATASET_HOME_DIR, safe_user, safe_dataset)
    try:
        # 若先前已有完整備份，整個覆蓋以確保與最新處理一致
        if os.path.exists(dataset_home_target):
            shutil.rmtree(dataset_home_target)
        os.makedirs(os.path.dirname(dataset_home_target), exist_ok=True)
        # 從固定專案根複製（已合併過的完整資料），排除暫存資料夾
        shutil.copytree(
            extract_base,
            dataset_home_target,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns('__tmp__')
        )
    except Exception:
        full_copy_ok = False

    # 僅產生 8:1:1 的 txt 清單（扁平化，直接放在 dataset_home/<user>/<dataset_name>）
    dataset_home_dir = os.path.join(DATASET_HOME_DIR, safe_user, safe_dataset)
    dataset_home_split_dir = dataset_home_dir
    txt_ok = True
    try:
        os.makedirs(dataset_home_split_dir, exist_ok=True)
        # 掃描 dataset_root 影像
        ds_root_images = os.path.join(dataset_home_target, 'images')
        all_images: list[str] = []
        img_exts_set2 = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        if os.path.isdir(ds_root_images):
            for root, _dirs, files in os.walk(ds_root_images):
                for fn in files:
                    if os.path.splitext(fn)[1].lower() in img_exts_set2:
                        img_abs = os.path.abspath(os.path.join(root, fn)).replace('\\','/')
                        all_images.append(img_abs)
        # 隨機打散
        random.shuffle(all_images)
        n = len(all_images)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_list = all_images[:n_train]
        val_list = all_images[n_train:n_train+n_val]
        test_list = all_images[n_train+n_val:]
        parts_map = {
            'train': train_list,
            'val':   val_list,
            'test':  test_list,
        }
        for part, lines in parts_map.items():
            out_txt = os.path.join(dataset_home_split_dir, f"{part}.txt")
            with open(out_txt, 'w', encoding='utf-8') as ftxt:
                ftxt.write("\n".join(sorted(lines)))
    except Exception:
        txt_ok = False

    # 以目前 txt 建立快照 vXX（初次），快照放在 dataset_home/<user>/<project>/versions
    ver_name: str | None = None
    try:
        ver_name = _snapshot_txts(dataset_home_split_dir)
    except Exception:
        ver_name = None

    # 產生 Dataset/<user>/<dataset_name>/data.yaml，並把 nc/names 也寫入（path 指向扁平化路徑）
    data_yaml_path = os.path.join(DATASET_DIR, safe_user, safe_dataset, "data.yaml")
    try:
        os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)
        ds_path = os.path.join(DATASET_HOME_DIR, safe_user, safe_dataset)
        # 轉為 Windows-friendly forward slashes（YOLO 通常可接受）
        ds_path_norm = os.path.abspath(ds_path).replace('\\', '/')
        base_yaml = [
            f"path: {ds_path_norm}",
            "test:  test.txt",
            "train: train.txt",
            "val:   val.txt",
        ]
        with open(data_yaml_path, 'w', encoding='utf-8') as yf:
            yf.write("\n".join(base_yaml) + "\n")
            if nc_names_yaml.strip():
                # 附加 nc/names 內容
                yf.write(nc_names_yaml if nc_names_yaml.endswith("\n") else nc_names_yaml + "\n")
    except Exception:
        data_yaml_path = ""

    # 嘗試從 data.yaml 掃描 names 清單（若有），用於產生 Dataset/<user>/<project>/data.yaml
    gen_thumbs_result = {}
    try:
        # src: dataset_home/<user>/<project>/images（若不存在則用專案根）
        src_dir = os.path.join(dataset_home_split_dir, 'images')
        if not os.path.isdir(src_dir):
            src_dir = dataset_home_split_dir
        src_abs = os.path.abspath(src_dir)
        # txt: 使用剛建立的版本 vXX（若有）
        txt_abs = None
        if ver_name:
            txt_abs = os.path.abspath(os.path.join(dataset_home_split_dir, 'versions', ver_name))
        # 縮圖圖片固定輸出到 /shared/Thumbs/<user>/<dataset>
        dest_abs = os.path.join(THUMBS_BASE_DIR, safe_user, safe_dataset)
        os.makedirs(dest_abs, exist_ok=True)
        gen_thumbs_log = os.path.join(dest_abs, "thumbs.log")
        _run_gen_thumbs(BASE_DIR, src_abs, dest_abs, txt_abs, gen_thumbs_log)
    # 將產生於 /shared/Thumbs 的 thumbs.json 搬/複製到 dataset_home 目前版本與版本目錄
        thumbs_json_src = os.path.join(dest_abs, 'thumbs.json')
        active_json = None
        versioned_json = None
        try:
            if os.path.isfile(thumbs_json_src):
                # 版本化 thumbs.json
                if ver_name:
                    versioned_dir = os.path.join(dataset_home_split_dir, 'versions', ver_name)
                    os.makedirs(versioned_dir, exist_ok=True)
                    versioned_json = os.path.join(versioned_dir, 'thumbs.json')
                    shutil.copyfile(thumbs_json_src, versioned_json)
                # 啟用版本 thumbs.json 寫在 dataset_home 根目錄
                active_json = os.path.join(dataset_home_split_dir, 'thumbs.json')
                shutil.copyfile(thumbs_json_src, active_json)
        except Exception:
            active_json = active_json or None
            versioned_json = versioned_json or None
        gen_thumbs_result = {
            "dest":                 dest_abs.replace('\\','/'),
            "log":                  gen_thumbs_log.replace('\\','/'),
            "version_name":         ver_name,
            "thumbs_json":          (active_json.replace('\\','/') if active_json else None),
            "thumbs_json_version":  (versioned_json.replace('\\','/') if versioned_json else None),
        }
    except Exception:
        gen_thumbs_result = {}

    # 決定回傳狀態碼（不寫入 dataset 狀態檔）
    rc2 = rc if rc != 0 or reorg_ok else 0
    if rc == 0 and not reorg_ok:
        rc2 = 1
    # 綜合結果（轉換與整理、複製到 dataset_home、產生 txt）
    overall_ok = (rc2 == 0 and full_copy_ok and txt_ok)

    # 可選：處理完成後刪除 zip
    zip_removed = dp.remove_file(save_path) if delete_zip else False

    # 兩個步驟完成後，刪除 upload_data_folder/<user> 下的所有內容
    upload_user_dir = os.path.join(UPLOAD_DATA_DIR, safe_user)
    upload_user_removed = False
    try:
        # 保護性檢查，避免誤刪
        base_abs = os.path.abspath(UPLOAD_DATA_DIR) + os.sep
        target_abs = os.path.abspath(upload_user_dir)
        if os.path.isdir(upload_user_dir) and (target_abs + os.sep).startswith(base_abs):
            shutil.rmtree(upload_user_dir)
            upload_user_removed = True
    except Exception:
        upload_user_removed = False

    # 最終更新資料集狀態（扁平清單）：寫入最終 {name, count, created_date, updated_date}
    try:
        img_count = _count_images_in_dataset(BASE_DIR, safe_user, safe_dataset)
        _status_list_upsert(safe_user, safe_dataset, int(img_count), now_str)
    except Exception:
        pass

    return {
        "done":         overall_ok,
        "version_name": ver_name,
        "version":      (int(ver_name[1:]) if ver_name and ver_name[1:].isdigit() else None)
    }

async def _append_to_specific_split_impl(
    *,
    file:       UploadFile,
    user_name:  str,
    task:       str,
    target_part:str,  # 'train' | 'val' | 'test'
    delete_zip: bool = True,
):
    """將資料直接加入指定分割（train/val/test），不實際移動到分割資料夾。

    - 真實檔案維持在 dataset_home/<user>/<project>/{images,labels}（扁平化結構）。
    - 僅更新對應的 txt 清單並建立 versions/vXX 快照。
    - 盡力更新 Dataset/<user>/<project>/data.yaml 的 names 與 nc。
    """

    part_map = {"train", "val", "test"}
    target_part = (target_part or "").strip().lower()
    
    if target_part not in part_map:
        raise HTTPException(status_code=400, detail="target_part 需為 'train'|'val'|'test'")

    # 基本檢查與正規化
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="只接受 .zip 檔")

    safe_user = ensure_user_name(user_name)
    safe_project = "".join(ch for ch in (task or "") if ch.isalnum() or ch in ("_", "-"))
    if not safe_project:
        raise HTTPException(status_code=400, detail="user_name 或 task 不合法")

    # 重要路徑（扁平化，直接維持完整資料在 dataset_root）
    dataset_root = os.path.join(DATASET_HOME_DIR, safe_user, safe_project)
    split_root = dataset_root
    dst_images_root = os.path.join(dataset_root, 'images')
    dst_labels_root = os.path.join(dataset_root, 'labels')
    os.makedirs(dst_images_root, exist_ok=True)
    os.makedirs(dst_labels_root, exist_ok=True)

    # 先存 zip
    os.makedirs(UPLOAD_DATA_DIR, exist_ok=True)
    save_path = await dp.save_upload_zip(file, UPLOAD_DATA_DIR, f"{safe_project}.zip")

    # 解壓縮到暫存
    unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]
    tmp_root = os.path.join(UPLOAD_DATA_DIR, safe_user, safe_project, "__tmp__", unique_suffix)
    os.makedirs(tmp_root, exist_ok=True)
    try:
        dp.safe_extract_zip(save_path, tmp_root)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="zip 檔損毀，無法解壓縮")

    names_from_new: list[str] = []
    names_yaml_text = dp.extract_nc_names(BASE_DIR, tmp_root)
    if names_yaml_text:
        parsed = _parse_names_from_yaml_text(names_yaml_text)
        if parsed:
            names_from_new = parsed

    labels_yolo = os.path.join(tmp_root, 'labels_yolo')
    dp.json_to_yolo(BASE_DIR, tmp_root, labels_yolo)

    reorg_result = dp.reorganize_converted_tree(tmp_root)
    merge_result = dp.merge_into_dataset(
        reorg_result.images_dir,
        reorg_result.labels_dir,
        dst_images_root,
        dst_labels_root,
    )

    new_abs_list = list(merge_result.added_images)

    jsons_root = os.path.join(dataset_root, 'jsons')
    dp.move_jsons(tmp_root, jsons_root)

    # 重新生成 txt（僅目標 part）：
    # 維持其他 part 清單不變，只對目標 part 追加本次新增影像（以 new_abs_list 為準）
    try:
        # 1) 讀取現有 txts
        txt_paths = {p: os.path.join(split_root, f"{p}.txt") for p in ('train','val','test')}
        lists: dict[str, list[str]] = {}
        for p, pth in txt_paths.items():
            try:
                with open(pth, 'r', encoding='utf-8') as f:
                    lists[p] = [ln.rstrip('\n') for ln in f if ln.strip()]
            except Exception:
                lists[p] = []
        # 2) 加入目標 part（避免重複）
        exists = set(lists.get(target_part, []))
        for p in new_abs_list:
            if p not in exists:
                lists.setdefault(target_part, []).append(p)
        # 3) 寫回目標 part
        os.makedirs(split_root, exist_ok=True)
        out_txt = {}
        for p in (target_part,):
            pth = txt_paths[p]
            with open(pth, 'w', encoding='utf-8') as f:
                f.write("\n".join(sorted(lists[p])))
            out_txt[p] = pth
    except Exception:
        out_txt = {}

    try:
        snap = _snapshot_txts(split_root)
    except Exception:
        snap = None

    # 產生縮圖（--src + --txt 版本），將照片與 JSON 產出到 /shared/Thumbs/<user>/<project>
    try:
        src_dir = os.path.join(split_root, 'images')
        if not os.path.isdir(src_dir):
            src_dir = split_root
        src_abs = os.path.abspath(src_dir)
        txt_abs = None
        if snap:
            txt_abs = os.path.abspath(os.path.join(split_root, 'versions', snap))
        dest_abs = os.path.join(THUMBS_BASE_DIR, safe_user, safe_project)
        os.makedirs(dest_abs, exist_ok=True)
        log_path = os.path.join(dest_abs, "thumbs.log")
        _run_gen_thumbs(BASE_DIR, src_abs, dest_abs, txt_abs, log_path)
        # 同步 thumbs.json 至目前啟用與版本資料夾
        try:
            thumbs_json_src = os.path.join(dest_abs, 'thumbs.json')
            if os.path.isfile(thumbs_json_src):
                if snap:
                    ver_dir = os.path.join(split_root, 'versions', snap)
                    os.makedirs(ver_dir, exist_ok=True)
                    shutil.copyfile(thumbs_json_src, os.path.join(ver_dir, 'thumbs.json'))
                shutil.copyfile(thumbs_json_src, os.path.join(split_root, 'thumbs.json'))
        except Exception:
            pass
    except Exception:
        pass

    if delete_zip:
        dp.remove_file(save_path)
    dp.cleanup_dir(tmp_root)

    # 更新 data.yaml 的 names 與 nc（合併）
    try:
        data_yaml_path = os.path.join(DATASET_DIR, safe_user, safe_project, 'data.yaml')
        existing_names: list[str] = []
        if os.path.isfile(data_yaml_path):
            try:
                with open(data_yaml_path, 'r', encoding='utf-8') as yf:
                    text = yf.read()
                parsed = _parse_names_from_yaml_text(text)
                if parsed:
                    existing_names = parsed
            except Exception:
                existing_names = []
        merged_names = list(existing_names)
        for n in names_from_new:
            if n not in merged_names:
                merged_names.append(n)
        # 以整個 dataset_root 下 labels 的最大 id 校正 nc
        max_id = _scan_label_max_id(os.path.join(dataset_root, 'labels'))
        desired_nc = max((max_id + 1) if max_id >= 0 else 0, len(merged_names))
        if desired_nc > len(merged_names):
            for i in range(len(merged_names), desired_nc):
                merged_names.append(f"cls_{i}")
        _write_dataset_yaml(BASE_DIR, safe_user, safe_project, merged_names)
    except Exception:
        pass

    ver_num = int(snap[1:]) if snap and len(snap) > 1 and snap[1:].isdigit() else None
    # 更新資料集狀態清單（計數與時間）
    try:
        time_fmt = "%Y-%m-%d %H:%M:%S"
        now_str2 = datetime.now().strftime(time_fmt)
        img_count2 = _count_images_in_dataset(BASE_DIR, safe_user, safe_project)
        _status_list_upsert(safe_user, safe_project, int(img_count2), now_str2)
    except Exception:
        pass

    return {
        "appended": True,
        "part": target_part,
        "version": ver_num,
        "version_name": snap,
    }


@router.post("/upload-data")
async def upload_or_append_zip(
    file: UploadFile    = File(..., description=".zip 資料集或新增資料 zip"),
    user_name: str      = Form("", description="使用者名稱（允許 . 並自動轉小寫）"),
    task: str           = Form("", description="資料集專案名稱"),
    mode: int           = Form(0, description="0=全新建立並自動分割, 1=加入到 train, 2=加入到 val, 3=加入到 test"),
    delete_zip: bool    = Form(True, description="處理完成後刪除上傳 zip")
):
    if mode == 0:
        return await _upload_data_zip_impl(file=file, user_name=user_name, task=task, delete_zip=delete_zip)
    elif mode in (1, 2, 3):
        part = {1: 'train', 2: 'val', 3: 'test'}[mode]
        return await _append_to_specific_split_impl(file=file, user_name=user_name, task=task, target_part=part, delete_zip=delete_zip)
    else:
        raise HTTPException(status_code=400, detail="mode 僅支援 0/1/2/3")

# JSON Body 版本：依分割別刪除 txt 條目（不刪實體檔案），並建立新版本快照
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