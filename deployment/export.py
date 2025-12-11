from fastapi import APIRouter, Request, HTTPException
import subprocess, uuid, sys, os, argparse, shutil
from utils.user_utils import ensure_user_name,runs_root
from pathlib import Path


router = APIRouter(
    prefix="/yolov9",
    tags=["EXPORT"],
)
_ensure_user_name = ensure_user_name
_runs_root = runs_root

ROOT_DIR = Path(__file__).resolve().parent
YOLO_EXPORT = (ROOT_DIR.parent / "yolov9" / "export.py").resolve()
MODEL_DIR = ROOT_DIR / "yolov9" / "models" / "detect"
SHARED_DIR = Path("/shared")
DOWNLOAD_ROOT = SHARED_DIR / "download"

# 名稱驗證：task 前面不能有空格（違反時以統一結構回傳）
def _validate_task(task: str):
    try:
        if isinstance(task, str) and len(task) > 0 and task[0].isspace():
            return {"code": {"code": "999", "msg": "名子前不能有空格"}}
    except Exception:
        pass
    return None


def run_export_tiny(id1: str, task: str, user_name: str | None = None):
    user_norm = _ensure_user_name(user_name)

    save_dir = os.path.join(_runs_root(user_norm), "reparmeater", id1)
    os.makedirs(save_dir, exist_ok=True)

    norm = task.replace('\\', '/').strip('/')
    parts = [p for p in norm.split('/') if p]

    if len(parts) < 3:
        raise ValueError("project 參數需為 'project/task/version'")
    
    project, task, version_name = parts[-3], parts[-2], parts[-1]
    exp_dir = Path(runs_root(user_norm)) / project / task / version_name / "exp"
    pt_path = exp_dir / "weights" / "best.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"找不到模型權重：{pt_path}")
    
    download_dir = Path(DOWNLOAD_ROOT) / user_norm / project / task / version_name
    download_dir.mkdir(parents=True, exist_ok=True)
    

    command = [
        sys.executable, YOLO_EXPORT,
        "--weights", str(pt_path),
        "--include", "onnx",
        "--batch-size", "1",
        "--simplify",
        "--device", "0",
    ]
    with open(os.path.join(save_dir, "log.txt"), "w", encoding="utf-8") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

    onnx_path = exp_dir / "weights" / "best.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"找不到模型權重：{onnx_path}")
    shutil.copy2(onnx_path, download_dir / f"{version_name}.onnx")
    

@router.get("/download/{user_name}/{project}/{task}/{version}")
def download_file_nested_v2(user_name: str, project: str, task: str, version: str, request: Request):
    # 名稱前面不能有空格
    err = _validate_task(project) or _validate_task(task) or _validate_task(version)
    if err:
        return err
    # Prevent path traversal
    ensured_user = _ensure_user_name(user_name)
    if ensured_user != os.path.basename(ensured_user):
        raise HTTPException(status_code=400, detail="Invalid path parameters")
    for seg in (project, task, version):
        if seg != os.path.basename(seg):
            raise HTTPException(status_code=400, detail="Invalid path parameters")

    dir_path = os.path.join(DOWNLOAD_ROOT, ensured_user, project, task, version)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail=f"Project folder not found: {dir_path}")

    preferred = os.path.join(dir_path, f"{version}.onnx")
    chosen_path = None
    if os.path.isfile(preferred):
        chosen_path = preferred
    else:
        try:
            onnx_files = [
                f for f in os.listdir(dir_path)
                if f.lower().endswith(".onnx") and os.path.isfile(os.path.join(dir_path, f))
            ]
        except OSError:
            onnx_files = []

        if not onnx_files:
            raise HTTPException(status_code=404, detail="No .onnx file found in project folder")

        onnx_files.sort(key=lambda f: os.path.getmtime(os.path.join(dir_path, f)), reverse=True)
        chosen_path = os.path.join(dir_path, onnx_files[0])

    chosen_filename = os.path.basename(chosen_path)
    base_url = str(request.base_url)
    return {"download_url": f"{base_url}download/{ensured_user}/{project}/{task}/{version}/{chosen_filename}"}


def _cli_main_tiny():
    parser = argparse.ArgumentParser(description="Export ONNX")
    parser.add_argument("--cfg", "--CFG", dest="cfg", default="gelan-s.yaml", help="模型 cfg (例如 gelan-s.yaml)")
    parser.add_argument("--project", required=True, help="必填：'project/task/version' (對應 exp/weights/best.pt 所在版本)")
    parser.add_argument("--user", dest="user", default="public", help="使用者名稱")
    args = parser.parse_args()

    run_id1 = str(uuid.uuid4())[:8]
    print(f"[export.py] Export run id:       {run_id1}")
    try:
        run_export_tiny(run_id1, args.project, args.user)
        parts = [p for p in args.project.replace('\\', '/').strip('/').split('/') if p]
        if len(parts) >= 3:
            project, task, version_name = parts[-3], parts[-2], parts[-1]
            onnx_user = _ensure_user_name(args.user)
            onnx_path = os.path.join(DOWNLOAD_ROOT, onnx_user, project, task, version_name, f"{version_name}.onnx")
            if os.path.isfile(onnx_path):
                print(f"[export.py] ONNX 完成: {onnx_path}")
            else:
                print("[export.py] 未找到輸出的 ONNX 檔。")
    except (FileNotFoundError, ValueError) as e:
        print(f"[export.py] 失敗: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[export.py] 未預期錯誤: {e}")
        sys.exit(2)


if __name__ == "__main__":
    _cli_main_tiny()


