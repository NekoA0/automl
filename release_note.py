from fastapi import APIRouter,Form
import threading, json, os, time, psutil
from contextlib import contextmanager 

router = APIRouter(
    prefix="/yolov9",
    tags=["release_note"],
)

STATUS_FILE = "release_note.json"
status_lock = threading.Lock()
_LOCK_DIR = STATUS_FILE + ".lockdir"

def save_status_map(status_map):
    """Thread-safe & more crash-safe JSON save.
    寫入步驟：
    1. 先寫到暫存檔 (.tmp)
    2. os.replace 原子性覆蓋正式檔 (避免讀到半截檔案)
    3. 同步寫一份 .bak 備份
    """
    with status_lock:
        tmp_path = STATUS_FILE + ".tmp"
        bak_path = STATUS_FILE + ".bak"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(status_map, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, STATUS_FILE)  # 原子替換（同分區）
            # 建立/更新備份
            try:
                with open(bak_path, "w", encoding="utf-8") as fb:
                    json.dump(status_map, fb, ensure_ascii=False, indent=2)
            except Exception:
                pass  # 備份失敗不阻斷主流程
        except Exception:
            # 若 tmp 寫失敗，確保 tmp 檔不殘留為空
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

# 載入狀態 JSON，如果沒有就回傳空字典
def load_status_map():
    """Thread-safe JSON load with tolerance for empty/partial files.
    若主檔損毀或為空，嘗試使用備份 (.bak)。讀不到則回傳 {}。
    """
    with status_lock:
        def _read(path: str):
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            if not data.strip():
                raise json.JSONDecodeError("empty", data, 0)
            return json.loads(data)
        if os.path.exists(STATUS_FILE):
            try:
                return _read(STATUS_FILE)
            except json.JSONDecodeError:
                # 嘗試備份
                bak_path = STATUS_FILE + ".bak"
                if os.path.exists(bak_path):
                    try:
                        return _read(bak_path)
                    except Exception:
                        return {}
                return {}
            except Exception:
                return {}
        return {}

def _status_key(user_name: str, project: str, task: str, version: str) -> str:
    return f"{user_name.strip().lower()}|{project}|{task}|{version}"

def _is_process_alive(pid: int) -> bool:
    try:
        if psutil is None:
            return True  # 無法檢查，視為活著以避免誤刪
        p = psutil.Process(pid)
        return p.is_running()
    except Exception:
        return False
    
@contextmanager
def _status_file_lock(timeout: float = 10.0, poll_interval: float = 0.1):
    """A simple cross-process lock using an atomic lock directory.
    Ensures one writer at a time across processes.
    """
    start = time.time()
    pid = os.getpid()
    while True:
        try:
            os.mkdir(_LOCK_DIR)
            # write lock info
            try:
                with open(os.path.join(_LOCK_DIR, "owner.json"), "w", encoding="utf-8") as f:
                    json.dump({"pid": pid, "ts": time.time()}, f)
            except Exception:
                pass
            break
        except FileExistsError:
            # check stale
            info_path = os.path.join(_LOCK_DIR, "owner.json")
            stale = False
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                owner_pid = int(info.get("pid", -1))
                owner_ts = float(info.get("ts", 0))
                if not _is_process_alive(owner_pid) and (time.time() - owner_ts) > 5:
                    stale = True
            except Exception:
                # 無法讀取，若鎖存在超過 10 秒則視為過期
                try:
                    st = os.stat(_LOCK_DIR)
                    if (time.time() - st.st_mtime) > 10:
                        stale = True
                except Exception:
                    pass
            if stale:
                try:
                    # best-effort cleanup
                    for fn in os.listdir(_LOCK_DIR):
                        try:
                            os.remove(os.path.join(_LOCK_DIR, fn))
                        except Exception:
                            pass
                    os.rmdir(_LOCK_DIR)
                    continue
                except Exception:
                    pass
            if time.time() - start > timeout:
                raise TimeoutError("status file lock timeout")
            time.sleep(poll_interval)
        except Exception:
            if time.time() - start > timeout:
                raise
            time.sleep(poll_interval)
    try:
        yield
    finally:
        try:
            for fn in os.listdir(_LOCK_DIR):
                try:
                    os.remove(os.path.join(_LOCK_DIR, fn))
                except Exception:
                    pass
            os.rmdir(_LOCK_DIR)
        except Exception:
            pass

def set_status_entry(status_key, status_obj):
    def _upd(m):
        m[status_key] = status_obj
        return m
    _with_status_map(_upd)

def _with_status_map(update_fn):
    with _status_file_lock():         # 跨行程目錄鎖
        m = load_status_map()         # 讀 JSON -> dict
        new_m = update_fn(m) or m
        save_status_map(new_m)        # json.dump(..., ensure_ascii=False)
        return new_m


@router.post("/release_note")
async def create_release_note(
    user_name: str       = Form(""),
    project: str    = Form(""),
    task: str   = Form(""),
    version: str         = Form(""),
    note: str            = Form("")
):
    status_key = _status_key(user_name, project, task, version)
    release_note = {"note": note}

    set_status_entry(status_key, release_note)


    return {"message": "Release note created successfully"}