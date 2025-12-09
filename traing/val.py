from fastapi import APIRouter, Form
import subprocess, sys, os, yaml, asyncio
from utils.user_utils import runs_root, validate_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["VAL"],
)

_runs_root = runs_root
_validate_user_name = validate_user_name

train_e = "exp/weights/best.pt"


def run_val(USER_NAME: str, PROJECT: str, TASK: str, VERSION: str):
    save_dir = os.path.join(_runs_root(USER_NAME), 'val-test', PROJECT, TASK, VERSION, "log")
    os.makedirs(save_dir, exist_ok=True)

    base_path = os.path.join(_runs_root(USER_NAME), PROJECT, TASK, VERSION, 'exp')
    yaml_path = os.path.join(base_path, "opt.yaml")

    with open(yaml_path, "r", encoding="utf-8") as f:
        opt_data = yaml.safe_load(f)
    data = opt_data.get("data", None)

    path = os.path.join(_runs_root(USER_NAME), PROJECT, TASK, VERSION, train_e)

    command = [
            sys.executable, "./yolov9/val.py",
            "--weights",    path,
            "--data",       data,
            "--name",       VERSION,
            "--save-json",
            "--device",     "0",
            "--batch-size", "16",
            "--imgsz",      "320",
            "--task",       "test",
            "--project",    os.path.join(_runs_root(USER_NAME), 'val-test', PROJECT, TASK, VERSION),
            ]

    log_path = os.path.join(save_dir, "log.txt")
    with open(log_path, "w") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)


@router.post("/val")
async def Validation(
               USER_NAME: str   = Form("", description="使用者名稱"),
               PROJECT: str     = Form("", description="專案名稱 (project)"),
               TASK:    str     = Form("", description="訓練名稱 (task)"),
               VERSION: str     = Form("", description="版本號 (version)")):
    
    _validate_user_name(USER_NAME)
    await asyncio.to_thread(run_val, USER_NAME, PROJECT, TASK, VERSION)
    log_path = os.path.join(_runs_root(USER_NAME), 'val-test', PROJECT, TASK, VERSION, "log", "log.txt")
    status = "completed"

    box = [{
        "user_name":USER_NAME,
        "project":  PROJECT,
        "task":     TASK,
        "version":  VERSION,
        "status":   status
    }]
        
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        last = lines[-1].strip() if lines else ""
        if "error" in last.lower():
            status = "failed"
            box[-1] = status


    return {"box": box}

