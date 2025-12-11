from fastapi import APIRouter, Form, HTTPException, Request

try:
    from fastapi.responses import FileResponse  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    from starlette.responses import FileResponse  # type: ignore[import]
import os, re, shutil
from glob import glob
from utils.user_utils import runs_root, validate_user_name
from utils.thumbs import THUMBS_BASE_DIR
_validate_user_name = validate_user_name
_runs_root = runs_root

router = APIRouter(
    prefix="/yolov9",
    tags=["VAL"],
)


@router.post("/get_val")
async def get_val(
    request: Request,
    USER_NAME: str  = Form("", description="使用者名稱"),
    PROJECT: str    = Form("", description="專案名稱 (project)"),
    TASK: str       = Form("", description="訓練名稱 (task)"),
    VERSION: str    = Form("", description="版本號 (version)")
):
    _validate_user_name(USER_NAME)
    runs_base = _runs_root(USER_NAME)
    log_path = os.path.join(runs_base, "val-test", PROJECT, TASK, VERSION, "log", "log.txt")

    if not os.path.isfile(log_path):
        return {"error": "log not found"}

    def strip_ansi(text: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [strip_ansi(line.rstrip("\n")) for line in f if line.strip()]

    headers = ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    number_token = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)"
    row_pattern = re.compile(
        rf"^\s*(\S+)\s+(\d+)\s+(\d+)\s+{number_token}\s+{number_token}\s+{number_token}\s+{number_token}\s*$",
        re.IGNORECASE,
    )

    rows = []
    summary_started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Speed") or stripped.startswith("Results saved"):
            break
        if not summary_started:
            if stripped.startswith("Class"):
                summary_started = True
            continue
        match = row_pattern.match(stripped)
        if not match:
            continue
        rows.append(dict(zip(headers, match.groups())))

    if not rows:
        return {"error": "summary not found"}

    user_dir = os.path.basename(runs_base)
    version_root = os.path.join(runs_base, "val-test", PROJECT, TASK, VERSION)
    shared_matrix_dir = os.path.join(THUMBS_BASE_DIR, user_dir, "val-test", PROJECT, TASK, VERSION)
    matrix_url = None

    try:
        os.makedirs(shared_matrix_dir, exist_ok=True)
    except Exception:
        pass

    try:
        matrix_candidates = glob(os.path.join(version_root, "**", "confusion_matrix.png"), recursive=True)
    except Exception:
        matrix_candidates = []

    valid_matrices = []
    for candidate in matrix_candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            valid_matrices.append((os.path.getmtime(candidate), candidate))
        except Exception:
            continue

    if valid_matrices:
        _, source_matrix = max(valid_matrices, key=lambda item: item[0])
        try:
            target_matrix = os.path.join(shared_matrix_dir, "confusion_matrix.png")
            shutil.copy2(source_matrix, target_matrix)
            matrix_url = str(request.url_for("get_val_imgs", user_name=user_dir, project=PROJECT, task=TASK, version=VERSION, file_name="confusion_matrix.png"))
        except Exception:
            matrix_url = None

    return {"summary": rows, "matrix_pic": {"url": matrix_url}}

@router.get("/imgs/{user_name}/{project}/{task}/{version}/{file_name}")
async def get_val_imgs(
    user_name: str,
    project: str,
    task: str,
    version: str,
    file_name: str,
):
    _validate_user_name(user_name)
    user_dir = os.path.basename(_runs_root(user_name))
    if not all(isinstance(p, str) and p.strip() for p in (project, task, version, file_name)):
        raise HTTPException(status_code=400, detail="缺少必要參數")

    if os.path.basename(file_name) != file_name:
        raise HTTPException(status_code=400, detail="檔名不合法")

    thumbs_root = os.path.abspath(THUMBS_BASE_DIR)
    target_path = os.path.abspath(
        os.path.join(thumbs_root, user_dir, "val-test", project, task, version, file_name)
    )

    # 防止路徑穿越 shared/Thumbs 目錄。
    if os.path.commonpath([thumbs_root, target_path]) != thumbs_root:
        raise HTTPException(status_code=400, detail="路徑不合法")

    if not os.path.isfile(target_path):
        raise HTTPException(status_code=404, detail="檔案不存在")

    return FileResponse(target_path)