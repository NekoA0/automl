from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import FileResponse
from utils.user_utils import ensure_user_name
from pathlib import Path
import subprocess, asyncio

from tflite_to_dla import check_sdk_layout, build_tflite2dla_shell

router = APIRouter(
    prefix="/yolov9",
    tags=["ONNX_TO_DLA"],
)

SHARED_DIR = Path("/shared")
DOWNLOAD_ROOT = SHARED_DIR / "download"
_ensure_user_name = ensure_user_name


@router.post("/onnx2tf")
async def run_onnx2tf(request: Request,
                user_name: str  = Form("", description="使用者名稱"), 
                project: str    = Form("", description="專案名稱"), 
                task: str       = Form("", description="任務名稱"), 
                version: str    = Form("", description="版本號")):
    
    user_norm   = _ensure_user_name(user_name)
    base_dir    = DOWNLOAD_ROOT / user_norm / project / task / version
    input_path  = base_dir / f"{version}.onnx"

    if not input_path.is_file():
        print(f"找不到 ONNX：{input_path}")
        raise HTTPException(status_code=404, detail=f"找不到 ONNX：{input_path}")

    check_sdk_layout()

    output_dir = base_dir / "tflite"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dla_dir = base_dir / "dla"
    dla_dir.mkdir(parents=True, exist_ok=True)

    onnx_cmd = (
        f"onnx2tf -i '{input_path.as_posix()}' -o '{output_dir.as_posix()}' -v debug"
    )

    tflite_input = output_dir / "best_float32.tflite"
    dla_output = dla_dir / f"{version}.dla"

    prep_tflite_cmd = (
        f"TF_SRC=$(find '{output_dir.as_posix()}' -maxdepth 1 -name '*float32.tflite' | head -n 1)"
        f" && if [ -z \"$TF_SRC\" ]; then echo '找不到 float32 TFLite 輸出'; exit 1; fi"
        f" && cp \"$TF_SRC\" '{tflite_input.as_posix()}'"
    )

    full_shell_cmd = (
        f"{onnx_cmd}"
        f" && {prep_tflite_cmd}"
        f" && {build_tflite2dla_shell(tflite_input, dla_output)}"
    )
    
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["bash", "-lc", full_shell_cmd],
            check=True,
        )
    except FileNotFoundError as exc:
        print("找不到 bash 執行檔")
        raise HTTPException(status_code=500, detail=f"無法找到 bash：{exc}") from exc
    except subprocess.CalledProcessError as exc:
        print(f"onnx2tf → tflite → dla 失敗，exit code {exc.returncode}")
        raise HTTPException(status_code=500,detail=f"onnx2tf → tflite → dla 失敗，exit code {exc.returncode}") from exc
    except Exception as exc:
        print(f"啟動 onnx2tf → tflite → dla 失敗：{exc}")
        raise HTTPException(status_code=500, detail=f"啟動 onnx2tf → tflite → dla 失敗：{exc}") from exc
    
    download_name = dla_output.name
    download_url  = str(request.url_for("download_dla", user_name=user_norm, project=project, task=task, version=version, file_name=download_name))

    return {
        "msg": "onnx2tf 與 tflite→dla 已啟動",
        "download_url": download_url,
    }


@router.get("/onnx2tf/download/{user_name}/{project}/{task}/{version}/{file_name}")
def download_dla(user_name: str, project: str, task: str, version: str, file_name:str):

    user        = _ensure_user_name(user_name)
    base_dir    = DOWNLOAD_ROOT / user / project / task / version
    file_path   = base_dir / "dla" / file_name

    if not file_path.exists():
        print("dla下載檔案不存在")
        raise HTTPException(status_code=404, detail="dla下載檔案不存在")
    
    return FileResponse(file_path, media_type="application/octet-stream", filename=file_name)
    