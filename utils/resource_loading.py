import psutil, GPUtil, cpuinfo
from fastapi import APIRouter

info = cpuinfo.get_cpu_info()
router = APIRouter(
    prefix="/resource",
    tags=["RESOURCE_LOADING"],
)

def _collect_gpu_stats():
    devices: list[dict] = []
    try:
        for gpu in GPUtil.getGPUs():
            memory_used = getattr(gpu, "memoryUsed", None)
            memory_total = getattr(gpu, "memoryTotal", None)
            if memory_used is not None and memory_total is not None:
                memory_str = f"{memory_used}MB / {memory_total}MB"
            elif memory_used is not None:
                memory_str = f"{memory_used}MB"
            else:
                memory_str = "N/A"

            load_val = getattr(gpu, "load", None)
            load_str = f"{round(load_val * 100, 2)}%" if load_val is not None else "N/A"

            temp = getattr(gpu, "temperature", None)
            devices.append({
                "Gpu_name":            gpu.name,
                "Gpu_id":              gpu.id,
                "Gpu_Memory 使用率":   memory_str,
                "Gpu 使用率":          load_str,
                "Gpu Temperature":     f"{temp}°C" if temp is not None else "N/A",
            })
    except Exception:
        pass
    return devices



@router.post("/loading")
def Get_computer():
    gpu_devices = _collect_gpu_stats()
    cpu_freq = psutil.cpu_freq()
    cpu_freq_str = f"{cpu_freq.current / 1000:.2f} GHz" if cpu_freq and cpu_freq.current else "N/A"

    return {
        "cpu": {
            "cpu 核心數": f"{psutil.cpu_count()}",
            "cpu 使用率": f"{psutil.cpu_percent(interval=0.5, percpu=False)} %",
            "cpu 頻率":   cpu_freq_str,
            "CPU 型號":   f"{info.get('brand_raw')}"
        },
        "memory": {
            "memory 使用率": f"{psutil.virtual_memory().percent} %"
        },
        "Gpu": {
            "count": len(gpu_devices),
            "devices": gpu_devices
        }
    }