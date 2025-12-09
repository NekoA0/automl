from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Form
import os, uuid, zipfile, subprocess, sys, time, asyncio
from fastapi.responses import FileResponse
from user_utils import runs_root, ensure_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["DETECT"],
)

_runs_root = runs_root
_ensure_user_name = ensure_user_name

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DETECT_DOWNLOAD_BASE = os.path.abspath(os.path.join(os.path.sep, "shared", "download"))
THUMBS_BASE_DIR = os.path.abspath(os.path.join(os.path.sep, "shared", "Thumbs"))
TRAIN_SUBPATH_BEST = os.path.join("exp", "weights", "best.pt")
TRAIN_SUBPATH_LAST = os.path.join("exp", "weights", "last.pt")

def _validate_task(name: str):
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="名稱不能為空")
    if any(ch in name for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']):
        raise HTTPException(status_code=400, detail='名稱不可包含路徑或特殊字元(\\/:*?"<>|)')

def _latest_version(user: str, project: str, task: str) -> str | None:
    base = os.path.join(_runs_root(user), project, task)
    if not os.path.isdir(base):
        return None
    mx = -1
    latest = None
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if not os.path.isdir(p):
            continue
        if d.isdigit():
            n = int(d)
            if n > mx:
                mx = n
                latest = d
    return latest

def _resolve_weights(user: str, project: str, task: str, version: str | None):
    if version is None or not str(version).strip():
        version = _latest_version(user, project, task)
        if version is None:
            raise HTTPException(status_code=404, detail="找不到任何版本資料夾")
        
    base_version_dir = os.path.join(_runs_root(user), project, task, version)
    if not os.path.isdir(base_version_dir):
        raise HTTPException(status_code=404, detail="指定版本不存在")
    
    w_best = os.path.join(base_version_dir, TRAIN_SUBPATH_BEST)
    w_last = os.path.join(base_version_dir, TRAIN_SUBPATH_LAST)

    if os.path.isfile(w_best):
        return w_best, version
    if os.path.isfile(w_last):
        return w_last, version
    raise HTTPException(status_code=404, detail="找不到 best.pt 或 last.pt")

def find_imag(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']:
                return root
    return None

def run_detect(run_id: str, source_folder: str, weights_path: str, user_name: str | None = None):
    user_norm = _ensure_user_name(user_name)

    shared_run_dir = os.path.join(DETECT_DOWNLOAD_BASE, user_norm,'detect_folder', run_id)
    os.makedirs(shared_run_dir, exist_ok=True)
    
    detect_cmd = [
        sys.executable, "./yolov9/detect.py",
        "--project",     shared_run_dir,
        "--source",      source_folder,
        "--weights",     weights_path,
        "--exist-ok",
    ]

    detect_log_path = os.path.join(shared_run_dir, "detect.log")
    with open(detect_log_path, "w") as log_file:
        result = subprocess.run(detect_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return
    time.sleep(0.5)

    # 產生縮圖，並將縮圖日誌記錄在 /shared/Thumbs/<user>/<run_id>/thumbs.log
    thumbs_run_dir = os.path.join(THUMBS_BASE_DIR, user_norm, run_id)
    os.makedirs(thumbs_run_dir, exist_ok=True)
    thumbs_log_path = os.path.join(thumbs_run_dir, "thumbs.log")

    gen_thumbs_candidates = [
        os.path.join(BASE_DIR, "GenThumbs.exe"),
        os.path.join(BASE_DIR, "GenThumbs"),
    ]
    gen_thumbs_exec = next((p for p in gen_thumbs_candidates if os.path.isfile(p)), None)

    with open(thumbs_log_path, "w") as log_file:
            if gen_thumbs_exec is None:
                log_file.write("GenThumbs executable not found. Skipping thumbnail generation.\n")
            else:
                thumbs_cmd = [
                    gen_thumbs_exec,
                    "--src",  os.path.join(shared_run_dir,"exp"),
                    "--dest", thumbs_run_dir,
                ]
                subprocess.run(thumbs_cmd, stdout=log_file, stderr=subprocess.STDOUT)

    time.sleep(3)
    # 壓縮結果資料夾為 zip 檔，放在 /shared/download/detect_zips/<user>/<run_id>.zip
    try:
        result_dir = shared_run_dir
        zip_output_dir = os.path.join(DETECT_DOWNLOAD_BASE, user_norm, 'detect_zips')
        os.makedirs(zip_output_dir, exist_ok=True)

        zip_output_path = os.path.join(zip_output_dir, f"{run_id}.zip")
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(result_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    arcname = os.path.relpath(file_path, result_dir)
                    zipf.write(file_path, arcname)
    except Exception:
        pass
    
@router.post("/upload-zip-detect")
async def upload_zip_detect(
    request: Request,
    USER_NAME: str      = Form("", description="使用者名稱"),
    file: UploadFile    = File(..., description="上傳zip資料集"),
    PROJECT: str        = Form("", description="專案(上層)名稱"),
    TASK: str           = Form("", description="訓練名稱"),
    VERSION: str        = Form("", description="版本，可省略使用最新")
):
    
    if not file.filename.endswith(".zip"):
        return {"error": "請上傳 zip 檔"}

    run_id = str(uuid.uuid4())[:8]
    # 儲存 zip 並解壓
    USER_NAME       = _ensure_user_name(USER_NAME)
    uploads_root    = os.path.join(_runs_root(USER_NAME), 'uploads_folder')
    zip_path        = os.path.join(uploads_root, f"{run_id}.zip")
    extract_dir     = os.path.join(uploads_root, run_id)
    os.makedirs(extract_dir, exist_ok=True)

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

    await asyncio.to_thread(run_detect, run_id, real_source_dir, weights_path, USER_NAME)

    # 回傳下載連結
    download_url = str(request.url_for("download_result_zip", user_name=USER_NAME, zip_name=run_id))
    
    return {
        "download_url": download_url,
    }

@router.get("/download/{user_name}/detect_zips/{zip_name}")
def download_result_zip(user_name: str, zip_name: str):
    user_name = _ensure_user_name(user_name)
    zip_filename = f"{zip_name}.zip"
    zip_path = os.path.join(DETECT_DOWNLOAD_BASE, user_name, 'detect_zips', zip_filename)

    if not os.path.exists(zip_path):
        return {"error": "檔案不存在"}
    
    return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)