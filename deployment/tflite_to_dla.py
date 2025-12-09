from fastapi import HTTPException
from pathlib import Path
import subprocess

# 統一常數設定，若 SDK 路徑異動只需改這裡
SDK_BIN_DIR = Path("20240108_Neuron_SDK_v1.2402.01_neuron-6-0-release/host/bin")
INIT_SCRIPT = SDK_BIN_DIR / "init.sh"
NCC_TFLITE = SDK_BIN_DIR / "ncc-tflite"


def check_sdk_layout():
    if not SDK_BIN_DIR.is_dir():
        print(f"找不到 SDK 目錄：{SDK_BIN_DIR}")
        raise HTTPException(status_code=500, detail=f"找不到 SDK 目錄：{SDK_BIN_DIR}")
    if not INIT_SCRIPT.is_file():
        print(f"找不到 init.sh：{INIT_SCRIPT}")
        raise HTTPException(status_code=500, detail=f"找不到 init.sh：{INIT_SCRIPT}")
    if not NCC_TFLITE.is_file():
        print(f"找不到 ncc-tflite：{NCC_TFLITE}")
        raise HTTPException(status_code=500, detail=f"找不到 ncc-tflite：{NCC_TFLITE}")


def build_tflite2dla_shell(input_path: Path, output_path: Path) -> str:
    """產生執行 ncc-tflite 的 shell 指令字串。"""
    in_posix  = input_path.as_posix()
    out_posix = output_path.as_posix()
    sdk_posix = SDK_BIN_DIR.as_posix()
    
    return (
        f" . {SDK_BIN_DIR}/init.sh"
        f" && {SDK_BIN_DIR}/ncc-tflite '{in_posix}' -arch mdla3.0 -d '{out_posix}' --relax-fp32"
    )


