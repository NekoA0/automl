from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import uuid, zipfile, shutil, subprocess, json, threading, yaml
from datetime import datetime
from pathlib import Path
import data_utils.dataset_pipeline as dp
from utils.user_utils import ensure_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["UPLOAD_DATA"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_UTILS_DIR = BASE_DIR / "data_utils"

UPLOAD_DATA_DIR   = BASE_DIR / "upload_data_folder"
DATASET_DIR       = BASE_DIR / "Dataset"
DATASET_HOME_DIR  = BASE_DIR / "dataset_home"
THUMBS_BASE_DIR   = Path("/shared/thumbs").resolve()

# Dataset 狀態檔與鎖（移到專案根）
DS_STATUS_FILE = BASE_DIR / "dataset_status_map.json"
ds_status_lock = threading.Lock()


def ds_save_status_map(status_map: dict | list):
    with ds_status_lock:
        tmp_path     = DS_STATUS_FILE.with_name(DS_STATUS_FILE.name + ".tmp")
        bak_path     = DS_STATUS_FILE.with_name(DS_STATUS_FILE.name + ".bak")
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
        def _read(path: Path):
            with path.open("r", encoding="utf-8") as f:
                data = f.read()
            if not data.strip():
                raise json.JSONDecodeError("empty", data, 0)
            return json.loads(data)
        
        if DS_STATUS_FILE.exists():
            try:
                return _read(DS_STATUS_FILE)
            except json.JSONDecodeError:
                bak = DS_STATUS_FILE.with_name(DS_STATUS_FILE.name + ".bak")
                if bak.exists():
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


def _dataset_name_exists_for_user(user: str, name: str) -> bool:
    """檢查指定使用者命名空間下是否已存在資料集（僅檔案系統層面）。"""
    try:
        p1 = DATASET_HOME_DIR / user / name
        p2 = DATASET_DIR / user / name
        return p1.is_dir() or p2.is_dir()
    except Exception:
        return False


def _count_images_in_dataset(base_dir: str | Path, user: str, project: str) -> int:
    """遞迴統計 dataset_home/<user>/<project>/images 內的影像檔案數量。"""
    try:
        base_path = Path(base_dir)
        images_root = base_path / "dataset_home" / user / project / "images"
        if not images_root.is_dir():
            # fallback: count under project root if images/ doesn't exist yet
            images_root = base_path / "dataset_home" / user / project
            if not images_root.is_dir():
                return 0
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        cnt = 0
        for p in images_root.rglob("*"):  # 遞迴列出所有檔案
            if p.is_file() and p.suffix.lower() in exts:
                cnt += 1
        return cnt
    except Exception:
        return 0


def _next_snapshot_version(versions_root: str | Path) -> int:
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


def _snapshot_txts(split_root: str | Path, version_int: int | None = None) -> str:
    """將 split_root 目前的 train.txt/val.txt/test.txt 複製到 versions/vXX/，並回傳版本名稱。

    若 version_int 為 None，會自動遞增下一個版本號；回傳值為類似 'v01' 的資料夾名稱。
    """
    split_root = Path(split_root)
    versions_dir = split_root / 'versions'
    if version_int is None:
        version_int = _next_snapshot_version(versions_dir)
    ver_name = _fmt_ver(version_int)
    ver_dir = versions_dir / ver_name
    ver_dir.mkdir(parents=True, exist_ok=True)
    for part in ('train', 'val', 'test'):
        src = split_root / f'{part}.txt'
        dst = ver_dir / f'{part}.txt'
        try:
            if src.is_file():
                dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
            else:
                # create empty file if missing
                dst.touch()
        except Exception:
            pass
    return ver_name


def _run_gen_thumbs(base_dir: str | Path, src_path: str | Path, dest_path: str | Path, txt_path: str | Path | None, log_path: str | Path):
    """以絕對路徑呼叫 GenThumbs.exe 的 --src、--dest，以及選擇性的 --txt 參數。
    - src_path：影像根目錄的絕對路徑（例如 dataset_home/<user>/<project>/images 或專案根）
    - dest_path：/shared/Thumbs/<user>/<project> 的絕對路徑
    - txt_path：包含 train/val/test.txt 的 versions/<vXX> 目錄絕對路徑，或 None
    """
    try:
        base_dir = Path(base_dir)
        exe_path = base_dir / "GenThumbs"
        cmd = [str(exe_path), "--src", str(src_path), "--dest", str(dest_path)]
        if txt_path:
            cmd += ["--txt", str(txt_path)]
        
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                log_file.write(f"CMD: {' '.join(cmd)}\nCWD: {base_dir}\n\n")
            except Exception:
                pass
            if not exe_path.is_file():
                log_file.write("ERROR: GenThumbs.exe not found.\n")
                return
            subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(base_dir))
    except Exception:
        pass


def _generate_split_txt_balanced(images_dir: str, labels_dir: str, out_dir: str, rare_obj_thresh: int = 30) -> bool:
    """
    使用 new_split_dataset.py (subprocess) 生成 out_dir 下的 train/val/test.txt。
    """
    try:
        script_path = BASE_DIR / "data_utils" / "new_split_dataset.py"
        
        # 建構指令
        # 注意：不使用 --summary-only，因為需要產生 txt 分割檔
        # 修正: 使用 Path 操作來正確指向 class.txt
        class_txt_path = Path(labels_dir).parent / "class.txt"

        cmd = [
            "python",   str(script_path),
            "--images", str(images_dir),
            "--labels", str(labels_dir),
            "--out",    str(out_dir),
            "--rare-obj-thresh", str(rare_obj_thresh),
            "--class-txt",       str(class_txt_path),
            "--seed", "1"
        ]
        
        
        # 執行
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
        return True
    except Exception as e:
        print(f"[ERROR] _generate_split_txt_balanced failed: {e}")
        return False


def _parse_names_from_yaml_text(yaml_text: str) -> list[str]:
    try:
        data = yaml.safe_load(yaml_text)
        if not data:
            return []
        names = data.get('names')
        if isinstance(names, list):
            return names
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return []
    except Exception:
        return []


async def _upload_data_zip_impl(
    file:           UploadFile,
    user_name:      str,
    dataset_name:   str | None = None,
    task:           str | None = None,
    delete_zip:     bool = False,
    split:          bool = True,):

    time_fmt = "%Y-%m-%d %H:%M:%S"
    # 1. 驗證檔案格式：僅允許 zip
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="只接受 .zip 檔")
    
    # 建立基礎目錄結構
    UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_HOME_DIR.mkdir(parents=True, exist_ok=True)

    safe_user = ensure_user_name(user_name)

    # 2. 檢查/標準化 dataset_name（若新參數空，落回舊 task）
    ds_in = dataset_name or task or ""
    safe_dataset = "".join(ch for ch in ds_in if ch.isalnum() or ch in ("_", "-"))
    if not safe_dataset:
        raise HTTPException(status_code=400, detail="dataset_name 不可為空或包含非法字元")

    # 3. 檢查重複（user 範圍內唯一）
    if _dataset_name_exists_for_user(safe_user, safe_dataset):
        raise HTTPException(status_code=409, detail="dataset_name 已存在")

    # 先將上傳 zip 改名為 <dataset_name>.zip（避免重名再補唯一字尾）
    filename = f"{safe_dataset}.zip"

    # 4. 儲存上傳的 zip 檔
    save_path = await dp.save_upload_zip(file, UPLOAD_DATA_DIR, filename)

    # 初始化時間字串供後續流程使用
    now_str = datetime.now().strftime(time_fmt)

    # 5. 更新資料集狀態（初始化計數為 0）
    try:
        _status_list_upsert(safe_user, safe_dataset, 0, now_str)
    except Exception:
        pass

    # 6. 解壓縮至暫存目錄
    # 路徑: upload_data_folder/<user>/<dataset>/__tmp__/<unique>
    extract_base = UPLOAD_DATA_DIR / safe_user / safe_dataset  # 固定專案根
    extract_base.mkdir(parents=True, exist_ok=True)

    unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]
    extract_root = extract_base / "__tmp__" / unique_suffix  # 暫存處理資料夾
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        dp.safe_extract_zip(save_path, extract_root)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="zip 檔損毀，無法解壓縮")

    # 7. 處理 class.txt
    # 優先檢查是否已包含 class.txt (忽略大小寫)，若無則嘗試從 YAML 解析，再無則使用 extract_nc_names 掃描 JSON
    found_class_txt = []
    for f in extract_root.rglob("*"):
        if f.is_file() and f.name.lower() == "class.txt":
            found_class_txt.append(f)
            break

    if found_class_txt:
        try:
            shutil.copy2(found_class_txt[0], extract_base / "class.txt")
        except Exception:
            pass
    else:
        class_names = []
        # 使用 extract_nc_names.py 掃描 JSON 生成
        nc_names_yaml = dp.extract_nc_names(str(DATA_UTILS_DIR), extract_root) or ""
        if nc_names_yaml:
            class_names = _parse_names_from_yaml_text(nc_names_yaml)

        if class_names:
            (extract_base / "class.txt").write_text("\n".join(class_names), encoding="utf-8")

    # 8. 格式轉換與整理
    # JSON -> YOLO txt
    labels_yolo_dir = extract_root / "labels_yolo"
    rc = dp.json_to_yolo(str(DATA_UTILS_DIR), extract_root, labels_yolo_dir)

    # 整理資料夾結構 (images, labels)
    reorg_result = dp.reorganize_converted_tree(extract_root)
    dst_images_root = extract_base / 'images'
    dst_labels_root = extract_base / 'labels'
    
    # 合併至 upload_data_folder 下的暫存結構
    merge_result = dp.merge_into_dataset(
        reorg_result.images_dir,
        reorg_result.labels_dir,
        dst_images_root,
        dst_labels_root,
    )
    # 移動原始 JSON 檔
    jsons_root = extract_base / 'json'
    move_jsons_ok = dp.move_jsons(extract_root, jsons_root)
    reorg_ok = reorg_result.ok and merge_result.ok and move_jsons_ok

    # 清理暫存解壓區
    dp.cleanup_dir(extract_root)

    # 9. 部署至 Dataset Home
    # 將整理好的資料複製到 dataset_home/<user>/<dataset_name>
    full_copy_ok = True
    dataset_home_target = DATASET_HOME_DIR / safe_user / safe_dataset
    try:
        # 若先前已有完整備份，整個覆蓋以確保與最新處理一致
        if dataset_home_target.exists():
            shutil.rmtree(dataset_home_target)
        dataset_home_target.parent.mkdir(parents=True, exist_ok=True)
        # 從固定專案根複製（已合併過的完整資料），排除暫存資料夾
        shutil.copytree(
            extract_base,
            dataset_home_target,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns('__tmp__')
        )
    except Exception:
        full_copy_ok = False

    dataset_home_dir = DATASET_HOME_DIR / safe_user / safe_dataset
    dataset_home_split_dir = dataset_home_dir
    
    # 10. 執行資料集分割 (Split)
    # 若 split=True，呼叫 new_split_dataset.py 產生 train/val/test.txt 與 data.yaml
    txt_ok = True
    if split:
        txt_ok = _generate_split_txt_balanced(
            images_dir=dataset_home_target / 'images',
            labels_dir=dataset_home_target / 'labels',
            out_dir=dataset_home_split_dir,
            rare_obj_thresh=30
        )

    # 11. 建立版本快照 (Snapshot)
    # 將目前的 txt 檔備份至 versions/vXX
    ver_name: str | None = None
    if split and txt_ok:
        try:
            ver_name = _snapshot_txts(dataset_home_split_dir)
        except Exception:
            ver_name = None

    # 12. 生成縮圖 (Thumbnails)
    gen_thumbs_result = {}
    try:
        # src: dataset_home/<user>/<project>/images
        src_dir = dataset_home_split_dir / 'images'
        if not src_dir.is_dir():
            src_dir = dataset_home_split_dir
        src_abs = src_dir.resolve()
        
        # txt: 使用剛建立的版本 vXX（若有）
        txt_abs = None
        if ver_name:
            txt_abs = (dataset_home_split_dir / 'versions' / ver_name).resolve()
            
        # 縮圖輸出路徑
        dest_abs = THUMBS_BASE_DIR / safe_user / safe_dataset
        dest_abs.mkdir(parents=True, exist_ok=True)
        gen_thumbs_log = dest_abs / "thumbs.log"
        
        # 執行 GenThumbs
        _run_gen_thumbs(BASE_DIR, src_abs, dest_abs, txt_abs, gen_thumbs_log)
        
        # 同步 thumbs.json 至 dataset_home
        thumbs_json_src = dest_abs / 'thumbs.json'
        active_json = None
        versioned_json = None
        try:
            if thumbs_json_src.is_file():
                # 版本化 thumbs.json
                if ver_name:
                    versioned_dir = dataset_home_split_dir / 'versions' / ver_name
                    versioned_dir.mkdir(parents=True, exist_ok=True)
                    versioned_json = versioned_dir / 'thumbs.json'
                    shutil.copyfile(thumbs_json_src, versioned_json)
                # 啟用版本 thumbs.json 寫在 dataset_home 根目錄
                active_json = dataset_home_split_dir / 'thumbs.json'
                shutil.copyfile(thumbs_json_src, active_json)
        except Exception:
            active_json = active_json or None
            versioned_json = versioned_json or None
        gen_thumbs_result = {
            "dest":                 dest_abs.as_posix(),
            "log":                  gen_thumbs_log.as_posix(),
            "version_name":         ver_name,
            "thumbs_json":          (active_json.as_posix() if active_json else None),
            "thumbs_json_version":  (versioned_json.as_posix() if versioned_json else None),
        }
    except Exception:
        gen_thumbs_result = {}

    # 決定回傳狀態碼
    rc2 = rc if rc != 0 or reorg_ok else 0
    if rc == 0 and not reorg_ok:
        rc2 = 1
    overall_ok = (rc2 == 0 and full_copy_ok and (txt_ok if split else True))

    # 13. 清理工作
    # 刪除上傳的 zip (可選)
    zip_removed = dp.remove_file(save_path) if delete_zip else False

    # 刪除 upload_data_folder 下的使用者暫存目錄
    upload_user_dir = UPLOAD_DATA_DIR / safe_user
    upload_user_removed = False
    try:
        base_abs = UPLOAD_DATA_DIR.resolve()
        target_abs = upload_user_dir.resolve()
        if upload_user_dir.is_dir() and base_abs in target_abs.parents:
            shutil.rmtree(upload_user_dir)
            upload_user_removed = True
    except Exception:
        upload_user_removed = False

    # 14. 最終更新資料集狀態 (更新圖片數量)
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
    split:      bool = True,
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

    # 1. 基本檢查與正規化
    filename = Path(file.filename or "").name
    if not filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="只接受 .zip 檔")

    safe_user = ensure_user_name(user_name)
    safe_project = "".join(ch for ch in (task or "") if ch.isalnum() or ch in ("_", "-"))
    if not safe_project:
        raise HTTPException(status_code=400, detail="user_name 或 task 不合法")

    # 重要路徑（扁平化，直接維持完整資料在 dataset_root）
    dataset_root = DATASET_HOME_DIR / safe_user / safe_project
    split_root = dataset_root
    dst_images_root = dataset_root / 'images'
    dst_labels_root = dataset_root / 'labels'
    dst_images_root.mkdir(parents=True, exist_ok=True)
    dst_labels_root.mkdir(parents=True, exist_ok=True)

    # 2. 儲存上傳的 zip
    UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = await dp.save_upload_zip(file, UPLOAD_DATA_DIR, f"{safe_project}.zip")

    # 3. 解壓縮到暫存
    unique_suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]
    tmp_root = UPLOAD_DATA_DIR / safe_user / safe_project / "__tmp__" / unique_suffix
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        dp.safe_extract_zip(save_path, tmp_root)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="zip 檔損毀，無法解壓縮")

    # 4. 嘗試解析類別名稱 (僅供參考，不強制覆蓋)
    names_from_new: list[str] = []
    names_yaml_text = dp.extract_nc_names(BASE_DIR, tmp_root)
    if names_yaml_text:
        parsed = _parse_names_from_yaml_text(names_yaml_text)
        if parsed:
            names_from_new = parsed

    # 5. 格式轉換與整理
    # JSON -> YOLO
    labels_yolo = tmp_root / 'labels_yolo'
    dp.json_to_yolo(BASE_DIR, tmp_root, labels_yolo)

    # 整理與合併
    reorg_result = dp.reorganize_converted_tree(tmp_root)
    merge_result = dp.merge_into_dataset(
        reorg_result.images_dir,
        reorg_result.labels_dir,
        dst_images_root,
        dst_labels_root,
    )

    new_abs_list = list(merge_result.added_images)

    # 移動 JSON
    jsons_root = dataset_root / 'json'
    dp.move_jsons(tmp_root, jsons_root)

    # 6. 更新分割清單 (Split Txt)
    # 僅更新目標 part (train/val/test)，維持其他 part 不變
    snap = None
    if split:
        try:
            # 1) 讀取現有 txts
            txt_paths = {p: split_root / f"{p}.txt" for p in ('train','val','test')}
            lists: dict[str, list[str]] = {}
            for p, pth in txt_paths.items():
                try:
                    with pth.open('r', encoding='utf-8') as f:
                        lists[p] = [ln.rstrip('\n') for ln in f if ln.strip()]
                except Exception:
                    lists[p] = []
            # 2) 加入目標 part（避免重複）
            exists = set(lists.get(target_part, []))
            for p in new_abs_list:
                if str(p) not in exists:
                    lists.setdefault(target_part, []).append(str(p))
            # 3) 寫回目標 part
            split_root.mkdir(parents=True, exist_ok=True)
            out_txt = {}
            for p in (target_part,):
                pth = txt_paths[p]
                with pth.open('w', encoding='utf-8') as f:
                    f.write("\n".join(sorted(lists[p])))
                out_txt[p] = pth
        except Exception:
            out_txt = {}

        # 7. 建立版本快照
        try:
            snap = _snapshot_txts(split_root)
        except Exception:
            snap = None

    # 8. 產生縮圖與同步
    try:
        src_dir = split_root / 'images'
        if not src_dir.is_dir():
            src_dir = split_root
        src_abs = src_dir.resolve()
        txt_abs = None
        if snap:
            txt_abs = (split_root / 'versions' / snap).resolve()
        dest_abs = THUMBS_BASE_DIR / safe_user / safe_project
        dest_abs.mkdir(parents=True, exist_ok=True)
        log_path = dest_abs / "thumbs.log"
        _run_gen_thumbs(BASE_DIR, src_abs, dest_abs, txt_abs, log_path)
        # 同步 thumbs.json 至目前啟用與版本資料夾
        try:
            thumbs_json_src = dest_abs / 'thumbs.json'
            if thumbs_json_src.is_file():
                if snap:
                    ver_dir = split_root / 'versions' / snap
                    ver_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(thumbs_json_src, ver_dir / 'thumbs.json')
                shutil.copyfile(thumbs_json_src, split_root / 'thumbs.json')
        except Exception:
            pass
    except Exception:
        pass

    # 9. 清理與狀態更新
    if delete_zip:
        dp.remove_file(save_path)
    dp.cleanup_dir(tmp_root)

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
    split: bool         = Form(True, description="是否立即進行分割並產生 txt"),
    delete_zip: bool    = Form(True, description="處理完成後刪除上傳 zip")
):
    if mode == 0:
        return await _upload_data_zip_impl(file=file, user_name=user_name, task=task, delete_zip=delete_zip, split=split)
    elif mode in (1, 2, 3):
        part = {1: 'train', 2: 'val', 3: 'test'}[mode]
        return await _append_to_specific_split_impl(file=file, user_name=user_name, task=task, target_part=part, delete_zip=delete_zip, split=split)
    else:
        raise HTTPException(status_code=400, detail="mode 僅支援 0/1/2/3")



