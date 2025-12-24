from fastapi import APIRouter, UploadFile, Form, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
import subprocess, shutil, uuid, sys, os, time, asyncio
from pathlib import Path
from utils.user_utils import runs_root, ensure_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["DETECT"],
)

_ensure_user_name = ensure_user_name
_runs_root = runs_root

ROOT_DIR = Path(__file__).resolve().parent.parent
YOLO_DETECT_PATH = ROOT_DIR / "yolov9" / "detect.py"
DETECT_DOWNLOAD_BASE = Path("/shared") /"download"
TRAIN_SUBPATH_BEST   = Path("exp") /"weights" /"best.pt"
TRAIN_SUBPATH_LAST   = Path("exp") /"weights" /"last.pt"

# In-memory job registry (per-process). For cross-process, persist small JSON if needed.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".webp",
}

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
    """Return (weights_path, version_used). Prefer best.pt then last.pt"""
    
    if version is None or not str(version).strip():
        version = _latest_version(user, project, task)

        if version is None:
            print(f"找不到任何版本資料夾：{base}")
            raise HTTPException(status_code=404, detail="找不到任何版本資料夾")
    # 版本資料夾必須存在
    base = Path(_runs_root(user)) / project / task
    base_version_dir = base / version

    if not base_version_dir.is_dir():
        print(f"指定版本資料夾不存在：{base_version_dir}")
        raise HTTPException(status_code=404, detail="指定版本不存在")
    
    w_best = base_version_dir / TRAIN_SUBPATH_BEST
    w_last = base_version_dir / TRAIN_SUBPATH_LAST

    if w_best.is_file():
        return w_best, version
    
    if w_last.is_file():
        return w_last, version
    
    print(f"找不到 best.pt 或 last.pt")
    raise HTTPException(status_code=404, detail="找不到 best.pt 或 last.pt")



def run_detect(run_id: str, image_path: str, weights_path: Path | str, user_name: str, save_txt: bool = False):

    detect_dir = DETECT_DOWNLOAD_BASE / user_name / "detect"
    output_dir = detect_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, str(YOLO_DETECT_PATH),
        "--project",    str(detect_dir),
        "--source",     str(image_path),
        "--weights",    str(weights_path),
        "--name",       run_id,
        "--exist-ok",
    ]
    if save_txt:
        command.append("--save-txt")

    log_path = output_dir / "log.txt"
    start_ts = time.time()

    download_log = "/" + str(Path("download") / user_name / "detect" / run_id / "log.txt").replace("\\","/")
    download_dir = "/" + str(Path("download") / user_name / "detect" / run_id).replace("\\", "/")

    with open(log_path, "w") as log_file:
        try:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                status = {
                    "state": "error",
                    "error": f"偵測程序返回代碼 {result.returncode}",
                    "log": download_log,
                }
            else:
                status = {
                    "state": "done",
                    "log": download_log,
                    "output_dir": os.path.join(detect_dir, run_id),
                    "output_url": download_dir,
                }
        except Exception as e:
            status = {"state": "error", "error": str(e), "log": download_log}
    status["elapsed_sec"] = round(time.time() - start_ts, 3)
    return status



@router.post("/detect")
async def Detectation(
    request: Request,
    USER_NAME: str      = Form("", description="使用者名稱"),
    file: UploadFile    = File(..., description="上傳圖片"),
    PROJECT: str        = Form("", description="專案(上層)名稱"),
    TASK: str           = Form("", description="訓練名稱"),
    VERSION: str        = Form("", description="版本，可留空使用最新"),
    SAVE_TXT: bool      = Form(False, description="是否輸出 txt 標註")):

    # 基礎驗證與正規化
    USER_NAME = _ensure_user_name(USER_NAME)
    _validate_task(PROJECT)
    _validate_task(TASK)
    ver = VERSION.strip() or None
    weights_path, _ = _resolve_weights(USER_NAME, PROJECT.strip(), TASK.strip(), ver)

    # 保存圖片
    upload_dir = Path(_runs_root(USER_NAME)) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / file.filename

    with open(image_path, "wb") as img:
        shutil.copyfileobj(file.file, img)

    run_id = str(uuid.uuid4())[:8]
    status = await asyncio.to_thread(run_detect, run_id, image_path, weights_path, USER_NAME, SAVE_TXT)
    
    if status.get("state") != "done":
        print(status.get("error", "偵測失敗"))
        raise HTTPException(status_code=500, detail=status.get("error", "偵測失敗"))

    output_dir_value = status.get("output_dir") or DETECT_DOWNLOAD_BASE / USER_NAME / "detect" / run_id
    output_dir = Path(output_dir_value)

    result_path = None
    if output_dir.is_dir():
        for file_path in sorted(output_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.name.lower() == "log.txt":
                continue
            ext = file_path.suffix.lower()
            if not ext or ext not in IMAGE_EXTENSIONS:
                continue
            result_path = file_path
            break

    if not result_path:
        print(f"偵測結果影像不存在: {output_dir}")
        raise HTTPException(status_code=404, detail="偵測結果影像不存在")

    media_type = "image/jpeg"
    ext = result_path.suffix.lower()

    if ext == ".png":
        media_type = "image/png"
    elif ext == ".bmp":
        media_type = "image/bmp"
    elif ext in {".tif", ".tiff"}:
        media_type = "image/tiff"
    elif ext == ".webp":
        media_type = "image/webp"
        
    download_name = result_path.name
    download_url = str(request.url_for("download_detect_result_file", user_name=USER_NAME, run_id=run_id, file_name=download_name))


    return StreamingResponse(
        open(result_path, "rb"),
        media_type=media_type,
        headers={
            "download_url":download_url
        }
    )


@router.get("/detect/download/{user_name}/detect/{run_id}/{file_name}", name="download_detect_result_file")
def download_result_file(user_name: str, run_id: str, file_name: str):
    user_name = _ensure_user_name(user_name)
    file_path = DETECT_DOWNLOAD_BASE / user_name / "detect" / run_id / file_name

    if not file_path.is_file():
        print("error: 檔案不存在")
        return {"error": "檔案不存在"}

    return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)