from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Form
import os, uuid, zipfile, subprocess, sys, time, asyncio, json
from pathlib import Path
from fastapi.responses import FileResponse
from utils.user_utils import runs_root, ensure_user_name
from deployment.yolo_to_xanylabeling import batch_convert

router = APIRouter(
    prefix="/yolov9",
    tags=["DETECT"],
)

_runs_root = runs_root
_ensure_user_name = ensure_user_name

ROOT_DIR = Path(__file__).resolve().parent.parent
YOLO_DETECT_PATH = ROOT_DIR / "yolov9" / "detect.py"

DETECT_DOWNLOAD_BASE = Path("/shared/download").resolve()
THUMBS_BASE_DIR = Path("/shared/thumbs").resolve()

TRAIN_SUBPATH_BEST = Path("exp") / "weights" / "best.pt"
TRAIN_SUBPATH_LAST = Path("exp") / "weights" / "last.pt"

def _validate_task(name: str):
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="名稱不能為空")
    if any(ch in name for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']):
        raise HTTPException(status_code=400, detail='名稱不可包含路徑或特殊字元(\\/:*?"<>|)')

def _latest_version(user: str, project: str, task: str) -> str | None:
    base = Path(_runs_root(user)) / project / task
    if not base.is_dir():
        print(f"找不到資料夾：{base}")
        return None
    mx = -1
    latest = None
    
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if d.name.isdigit():
            n = int(d.name)
            if n > mx:
                mx = n
                latest = d
    return latest

def _resolve_weights(user: str, project: str, task: str, version: str | None):

    if version is None or not str(version).strip():
        version = _latest_version(user, project, task)
        if version is None:
            raise HTTPException(status_code=404, detail="找不到任何版本資料夾")
        
    base_version_dir = Path(_runs_root(user)) / project / task / str(version)
    if not base_version_dir.is_dir():
        raise HTTPException(status_code=404, detail="指定版本不存在")
    
    w_best = base_version_dir / TRAIN_SUBPATH_BEST
    w_last = base_version_dir / TRAIN_SUBPATH_LAST

    if w_best.is_file():
        return w_best, version
    if w_last.is_file():
        return w_last, version
    raise HTTPException(status_code=404, detail="找不到 best.pt 或 last.pt")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def find_imag(base_dir):
    base_dir = Path(base_dir)

    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p.parent

    return None

def run_detect(run_id: str, source_folder: str, weights_path: str, user_name: str | None = None, save_txt: bool = False, project: str = "", task: str = ""):
    user_norm = _ensure_user_name(user_name)

    shared_run_dir  = DETECT_DOWNLOAD_BASE / user_norm / "detect_folder" / run_id
    shared_run_dir.mkdir(parents=True, exist_ok=True)
    
    detect_cmd = [
        sys.executable,  YOLO_DETECT_PATH,
        "--project",     shared_run_dir,
        "--source",      source_folder,
        "--weights",     weights_path,
        "--exist-ok",
    ]
    
    if save_txt:
        detect_cmd.append("--save-txt")
        detect_cmd.append("--save-conf")

    detect_log_path = shared_run_dir / "detect.log"
    with open(detect_log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(detect_cmd, stdout=log_file, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return {"state": "failed", "error": "Detect process failed"}
    time.sleep(0.5)

    # 產生縮圖，並將縮圖日誌記錄在 /shared/Thumbs/<user>/<run_id>/thumbs.log
    thumbs_run_dir  = THUMBS_BASE_DIR / user_norm / run_id
    thumbs_run_dir.mkdir(parents=True, exist_ok=True)
    thumbs_log_path = thumbs_run_dir / "thumbs.log"

    gen_thumbs_candidates = [
        ROOT_DIR / "GenThumbs.exe",
        ROOT_DIR / "GenThumbs",
    ]
    gen_thumbs_exec = next((p for p in gen_thumbs_candidates if p.is_file()), None)

    with open(thumbs_log_path, "w", encoding="utf-8") as log_file:
            if gen_thumbs_exec is None:
                log_file.write("GenThumbs executable not found. Skipping thumbnail generation.\n")
            else:
                thumbs_cmd = [
                    str(gen_thumbs_exec),
                    "--src",  str(shared_run_dir / "exp"),
                    "--dest", str(thumbs_run_dir),
                ]
                subprocess.run(thumbs_cmd, stdout=log_file, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")

    # 壓縮結果資料夾為 zip 檔，放在 /shared/download/detect_zips/<user>/<run_id>.zip
    try:
        zip_output_dir = DETECT_DOWNLOAD_BASE / user_norm / 'detect_zips'
        zip_output_dir.mkdir(parents=True, exist_ok=True)
        zip_output_path = zip_output_dir / f"{run_id}.zip"

        target_dir = shared_run_dir
        
        if save_txt:
            # 嘗試轉換為 xanylabeling 格式
            ct_file = ROOT_DIR / "create_train_status_map.json"
            dataset_name = None
            if ct_file.exists():
                try:
                    with open(ct_file, 'r', encoding='utf-8') as f:
                        ct_map = json.load(f)
                    key = f"{user_norm}|{project}|{task}"
                    if key in ct_map:
                        dataset_name = ct_map[key].get("dataset")
                except Exception as e:
                    print(f"Error reading status map: {e}")
            
            if dataset_name:
                # 修正: 使用 ROOT_DIR / "Dataset"
                yaml_path = ROOT_DIR / "Dataset" / user_norm / dataset_name / "data.yaml"
                
                # 檢查 exp 目錄
                exp_dir = shared_run_dir / "exp"
                if not exp_dir.exists():
                     exp_dir = shared_run_dir
                
                labels_dir = exp_dir / "labels"
                
                if yaml_path.exists() and labels_dir.exists():
                    print(f"Converting labels... {labels_dir} -> {source_folder}")
                    batch_convert(
                        images_dir=str(source_folder),
                        labels_dir=str(labels_dir),
                        yaml_path=str(yaml_path)
                    )
                    # 如果轉換成功，目標目錄改為 source_folder (包含原圖和 json)
                    target_dir = source_folder
                else:
                    print(f"Missing yaml or labels: {yaml_path}, {labels_dir}")

        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(target_dir)
                    zipf.write(file_path, arcname)
        
        return {"state": "done"}
    except Exception as e:
        return {"state": "failed", "error": str(e)}
    
@router.post("/upload-zip-detect")
async def upload_zip_detect(
    request: Request,
    USER_NAME: str      = Form("", description="使用者名稱"),
    file: UploadFile    = File(...,description="上傳zip資料集"),
    PROJECT: str        = Form("", description="專案(上層)名稱"),
    TASK: str           = Form("", description="訓練名稱"),
    VERSION: str        = Form("", description="版本，可省略使用最新"),
    SAVE_TXT: bool      = Form(True, description="是否輸出 txt 標註")):
    
    if not file.filename.endswith(".zip"):
        return {"error": "請上傳 zip 檔"}

    run_id = str(uuid.uuid4())[:8]
    # 儲存 zip 並解壓
    USER_NAME       = _ensure_user_name(USER_NAME)

    uploads_root    = Path(_runs_root(USER_NAME)) / "uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)

    zip_path        = uploads_root / f"{run_id}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    extract_dir     = uploads_root / run_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    with open(zip_path, "wb") as f:
        f.write(await file.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 解壓完成後自動刪除上傳的 zip 檔
    try:
        os.remove(zip_path)
    except Exception:
        pass

    real_source_dir = find_imag(extract_dir)

    # 若 zip 內容沒有影像檔，直接回傳 code 999
    if not real_source_dir:
        return {"code": 999, "error": "zip 內容沒有影像檔"}

    # 執行偵測並產生縮圖到 /shared/Thumbs/<USER_NAME>
    _validate_task(PROJECT)
    _validate_task(TASK)
    weights_path, _ = _resolve_weights(USER_NAME, PROJECT.strip(), TASK.strip(), VERSION.strip() or None)

    status = await asyncio.to_thread(run_detect, run_id, real_source_dir, weights_path, USER_NAME, SAVE_TXT, PROJECT, TASK)

    if status.get("state") != "done":
        print(status.get("error", "偵測失敗"))
        raise HTTPException(status_code=500, detail=status.get("error", "偵測失敗"))

    # 回傳下載連結
    download_url = str(request.url_for("download_result_zip", user_name=USER_NAME, zip_name=run_id))
    
    return {
        "download_url": download_url,
    }

@router.get("/download/{user_name}/detect_zips/{zip_name}")
def download_result_zip(user_name: str, zip_name: str):
    user_name       = _ensure_user_name(user_name)
    zip_filename    = f"{zip_name}.zip"
    zip_path        = DETECT_DOWNLOAD_BASE / user_name / 'detect_zips' / zip_filename

    if not zip_path.exists():
        print("檔案不存在",zip_path)
        return {"error": "檔案不存在"}
    
    return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)