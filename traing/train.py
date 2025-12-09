"""
狀態碼對照表（train_status_map.json -> code.code）

- 000: 閒置中（尚未開始或無進程）
- 001: 初始中（掃描/快取標籤、載入資料等）
- 002: 等待中（重新啟動排程中，restart_lock 期間）
- 010: 訓練中
- 011: 暫停中（以 psutil suspend/resume 控制）
- 015: 轉換中（匯出 ONNX 等）
- 019: 已完成（可能包含「未自動匯出」的說明）
- 980: 查無此專案狀態（查詢預設回覆）
- 999: 失敗/錯誤（包含讀取進度錯誤、轉換失敗等）
"""

from fastapi import APIRouter,Form,Query,HTTPException
import subprocess, yaml, sys, os, train_config, time, json, threading, shutil
from datetime import datetime
from threading import Thread
from val import run_val 
import pandas as pd
from typing import Optional
from contextlib import contextmanager
from pathlib import Path

from utils.user_utils import (
    ensure_user_name,
    extract_user_from_dataset_name,
    normalize_user_name,
    runs_root,
    validate_user_name,
)

try:
    import psutil
except Exception:
    psutil = None

router = APIRouter(
    prefix="/yolov9",
    tags=["TRAIN"],
)                                      

MODEL_FAMILY = "yolov9"
STATUS_FILE = "train_status_map.json"
RELEASE_NOTE_FILE = "release_note.json"
status_lock = threading.Lock()
AUTO_EXPORT_ENABLED = True 
AUTO_EXPORT_MIN_EPOCHS = 0
project_generations: dict[str, int] = {}
generation_lock = threading.Lock()

_normalize_user_name = normalize_user_name
_validate_user_name = validate_user_name
_runs_root = runs_root
_extract_user_from_dataset_name = extract_user_from_dataset_name
_ensure_user_name = ensure_user_name

_OPT_KEYS =["batch_size","imgsz","workers","close_mosaic"]


def _summarize_opt(opt_data: dict) -> dict:
    """過濾 opt.yaml 內常用欄位，避免整份塞進狀態檔。"""
    if not isinstance(opt_data, dict):
        return {}
    out = {}
    for k in _OPT_KEYS:
        if k in opt_data:
            out[k] = opt_data[k]
    return out

# 從資料集名稱推斷 user（格式可能為 "<user>/<folder>" 或僅 "<folder>"）
# 名稱驗證：task 前面不能有空格
def _validate_task(task: str):
    try:
        if isinstance(task, str) and len(task) > 0 and task[0].isspace():
            return {"code": {"code": "999", "msg": "名子前不能有空格"}}
        # 基本安全檢查：避免包含路徑符號或可疑片段
        if any(ch in task for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']):
            return {"code": {"code": "999", "msg": "名稱不可包含路徑或特殊字元(\\/:*?\"<>|)"}}
        lowered = task.lower()
        if ".." in task or lowered in {"con", "prn", "aux", "nul"} or any(lowered == f"com{i}" for i in range(1,10)) or any(lowered == f"lpt{i}" for i in range(1,10)):
            return {"code": {"code": "999", "msg": "不合法的名稱"}}
    except Exception:
        pass
    return None

# 儲存狀態到 JSON
def save_status_map(status_map):
    """執行緒安全且更能防止當機造成損毀的 JSON 寫入流程。
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
    """執行緒安全的 JSON 讀取流程，能容忍空白或部分內容。
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


def _load_release_note_map():
    """載入獨立的 release_note.json（若存在）。
    結構：{ "<user>|<project>|<task>|<version>": {"note": "..."} }
    """
    try:
        if not os.path.exists(RELEASE_NOTE_FILE):
            return {}
        with open(RELEASE_NOTE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------- Cross-process lock and atomic update helpers ----------------
_LOCK_DIR = STATUS_FILE + ".lockdir"

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
    """使用原子化鎖定資料夾的跨程序鎖。
    保證同一時間僅有一個程序可以寫入。
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


def _with_status_map(update_fn):
    """取得跨程序鎖後讀取現有狀態、呼叫 update_fn(map) -> map、儲存並回傳結果。"""
    with _status_file_lock():
        m = load_status_map()
        new_m = update_fn(m) or m
        save_status_map(new_m)
        return new_m

def set_status_entry(status_key: str, status_obj: dict):
    def _upd(m: dict):
        m[status_key] = status_obj
        return m
    _with_status_map(_upd)

def mutate_status_entry(status_key: str, mutator):
    """mutator(cur_dict) -> new_dict；回傳更新後的字典。"""
    res_container = {"res": None}
    def _upd(m: dict):
        cur = m.get(status_key) or {}
        new = mutator(cur) or cur
        m[status_key] = new
        res_container["res"] = new
        return m
    _with_status_map(_upd)
    return res_container["res"]


# 新結構輔助：runs/<user>/<project>/<task>/<version>
def _train_version_dir(user_name: Optional[str], project: str, task: str, version: str) -> str:
    """組合訓練輸出目錄：runs/<user>/<project>/<task>/<version>"""
    return os.path.join(_runs_root(user_name), project, task, version)

def _status_key(user_name: Optional[str], project: str, task: str, version: str) -> str:
    """唯一狀態鍵：user|project|task|version，避免不同訓練分支衝突。"""
    usr = _ensure_user_name(user_name)
    return f"{usr}|{project}|{task}|{version}"


def get_progress(version: str, ep: str = "epoch", user_name: Optional[str] = None, project: Optional[str] = None, task_name: Optional[str] = None, dataset_name: Optional[str] = None):
    if not project or not task_name:
        raise HTTPException(status_code=400, detail="缺少 project 或 task")
    
    base_path = os.path.join(_train_version_dir(user_name, project, task_name, version), "exp")
    yaml_path = os.path.join(base_path, "opt.yaml")
    csv_path = os.path.join(base_path, "results.csv")
  
    # 讀取總 epochs（若尚未產生，預設為 0）
    total_epochs = 0
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                opt_data = yaml.safe_load(f) or {}
            total_epochs = int(opt_data.get("epochs", 0))
        except Exception:
            total_epochs = 0

    # 讀取進度（當 CSV 尚未存在或尚未寫入時，提供穩健預設）
    current_epoch = 0
    data_list = []

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if not df.empty:
                if "epoch" in df.columns:
                    try:
                        epoch_series = pd.to_numeric(df["epoch"], errors="coerce").dropna()
                        if not epoch_series.empty:
                            current_epoch = int(epoch_series.max()) + 1
                        elif not df.empty:
                            current_epoch = int(len(df))
                    except Exception:
                        current_epoch = 0
                        
                if ep != "epoch":                                                       # 動態擷取常見度量欄位（避免欄位名稱差異導致空資料）
                    exclude_metrics = {"val/box_loss", "val/cls_loss", "val/dfl_loss"}  # 收集需要的度量欄位，排除不想顯示的 val 損失
                    metric_cols = [
                        c for c in df.columns
                        if c != "epoch" and (c.startswith("train/") or c.startswith("val/") or c.startswith("metrics/")) and c not in exclude_metrics
                    ]
                    selected_columns = ["epoch"] + metric_cols
                    existing_columns = [col for col in selected_columns if col in df.columns]
                    if existing_columns:
                        data_list = df[existing_columns].to_dict(orient="records")
        except Exception:
            pass  # 略過 CSV 讀取期間的中途寫入錯誤

    if ep == "epoch":
        return {"epochs": total_epochs, "data": int(current_epoch)}
    else:
        return {"epochs": total_epochs, "data": data_list}

def run_training(dataset_name: str, 
                 version: str, 
                 epochs: int, 
                 batch_size: int, 
                 img_size: int, 
                 workers: int, 
                 close_mosaic: int, 
                 initial_code: str | None = None, 
                 initial_msg: str | None = None, 
                 start_generation: int | None = None, 
                 user_name: Optional[str] = None, 
                 project: Optional[str] = None, 
                 task: Optional[str] = None, 
                 data_yaml_path: Optional[str] = None, 
                 dataset_version: Optional[str] = None,
                 device: Optional[str] = None):
    """訓練主流程
    加入 generation 概念：每次 restart 會遞增 project_generations[task]；舊執行緒偵測到不符即停止，
    並且在 finally 階段避免覆寫新的狀態。
    """
    
    user_scoped = _normalize_user_name(user_name)
    if not user_scoped:
        raise HTTPException(status_code=400, detail="缺少或不合法的 user_name")
    
    train_out_dir = _train_version_dir(user_scoped, project, task, version)
    os.makedirs(train_out_dir, exist_ok=True)
    status_key = _status_key(user_scoped, project , task , version)
    save_dir = os.path.join(_runs_root(user_scoped), "train_log", project, (task), f"{version}_log")
    os.makedirs(save_dir, exist_ok=True)

    # data_yaml_path 由呼叫端解析（支援 dataset_home 版本）
    if not data_yaml_path:
        data_yaml_rel = os.path.join(*dataset_name.split('/')) if '/' in dataset_name else dataset_name
        data_yaml_path = os.path.join(".", "Dataset", data_yaml_rel, "data.yaml")
    log_path = os.path.join(save_dir, "log.txt")
    time_format = "%Y-%m-%d %H:%M:%S"

    if start_generation is None:
        with generation_lock:
            start_generation = project_generations.get(status_key, 0)

    ds_base = dataset_name.replace('\\','/').split('/')[-1]
    train_status = {
        "time":     {"create_time": datetime.now().strftime(time_format), "start_time": "None", "end_time": "None"},
        "code":     {"code": "000", "msg": "閒置中"},
        "epoch":    {"current": 0, "total": 0},
        "model":    {"name": MODEL_FAMILY},
        "dataset":  ({"name": ds_base, "version": dataset_version} if dataset_version else {"name": ds_base}),
        "user":     {"name": user_scoped} if user_scoped else {},
        "project":  {"name": project or ''},
        "task":     {"name": task or ''},
        "version":  {"name": version},
        "generation": start_generation
    }
    if initial_code and initial_msg:
        train_status["code"] = {"code": initial_code, "msg": initial_msg}

    # 前置檢查
    if not os.path.isfile(data_yaml_path):
        train_status["code"] = {"code": "999", "msg": f"找不到資料集設定檔: {data_yaml_path}"}
        train_status["time"]["end_time"] = datetime.now().strftime(time_format)
        set_status_entry(status_key, train_status)
        return train_status
    
    train_script = os.path.join(".", "yolov9", "train_dual.py")
    if not os.path.isfile(train_script):
        train_status["code"] = {"code": "999", "msg": f"找不到訓練腳本: {train_script}"}
        train_status["time"]["end_time"] = datetime.now().strftime(time_format)
        set_status_entry(status_key, train_status)
        return train_status

    # 目標：所有訓練與 restart 都固定使用同一個 exp 目錄（不產生 exp2/exp3）
    # 如該版本下存在舊 exp 目錄且為 restart 流程，可選擇清空後重建
    exp_dir = os.path.join(train_out_dir, 'exp')
    try:
        # 若為 restart (initial_code == '001' 且 initial_msg == '初始中' 且 exp 存在)，清空舊資料
        if initial_code == '001' and initial_msg == '初始中' and os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)
    except Exception:
        pass
    device = (device or "").strip()
    requested_devices = [d.strip() for d in device.split(',') if d.strip()]
    if not requested_devices:
        requested_devices = ["0"]
    multi_gpu = len(requested_devices) > 1
    device_arg = ",".join(requested_devices)
    HYP = os.path.join(runs_root(user_scoped), project, task, "hyp.scratch-high.yaml")

    base_args = [
        "./yolov9/train.py",
        "--data",           data_yaml_path,
        "--cfg",            train_config.CFG,
        "--img",            str(img_size),
        "--hyp",            HYP,
        "--epochs",         str(epochs),
        "--batch-size",     str(batch_size),
        "--weights",        train_config.WEIGHTS,
        "--device",         device_arg,
        "--close-mosaic",   str(close_mosaic),
        "--seed",           str(train_config.SEED),
        "--workers",        str(workers),
        "--project",        train_out_dir,
        "--name", "exp",
        "--exist-ok"
    ]

    command = [sys.executable]
    if multi_gpu:
        # 使用 torch.distributed.run 進行多 GPU 訓練，避免 DataParallel 造成 shape 問題
        master_port = str(29500 + (os.getpid() % 1000))
        command += [
            "-m", "torch.distributed.run",
            "--nproc_per_node", str(len(requested_devices)),
            "--master_port", master_port
        ]
    command += base_args

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

        # 寫入 PID 與初始狀態
        train_status["proc"] = {"pid": proc.pid}
        set_status_entry(status_key, train_status)
        if train_status.get("code", {}).get("code") not in {"001", "010"}:
            train_status["code"] = {"code": "001", "msg": "初始中"}
            set_status_entry(status_key, train_status)


        hyp_dir=os.path.join(train_out_dir,"hyp.scratch-high.yaml")
        try:
            shutil.copyfile(HYP,hyp_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"複製(version) hyp.scratch-high.yaml 失敗：{e}")

        exp_path = os.path.join(train_out_dir, "exp")
        while proc.poll() is None:
            with generation_lock:
                if project_generations.get(status_key, -1) != start_generation:
                    break  # 舊世代終止
            current_status_map = load_status_map()
            existing_code = (current_status_map.get(status_key) or {}).get('code', {}).get('code') or train_status.get('code', {}).get('code')
            if existing_code == '011':  # 暫停狀態保持
                time.sleep(5)
                continue

            flags = (current_status_map.get(status_key) or {}).get('flags', {})
            if flags.get('restart_lock'):
                train_status.setdefault('flags', {})['restart_lock'] = False
                def _unset(cur: dict):
                    cur = cur or {}
                    cur.setdefault('flags', {})['restart_lock'] = False
                    return cur
                mutate_status_entry(status_key, _unset)

            # epoch 進度更新
            yaml_path = os.path.join(exp_path, "opt.yaml")
            csv_path = os.path.join(exp_path, "results.csv")
            total_epochs = 0
            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        opt_data = yaml.safe_load(f) or {}
                    total_epochs = int(opt_data.get("epochs", 0))
                    opt_summary = _summarize_opt(opt_data)
                    if opt_summary:
                        # 只在第一次或內容有變動時更新（避免每輪都寫檔）
                        if train_status.get("opt") != opt_summary:
                            train_status["opt"] = opt_summary
                except Exception:
                    total_epochs = 0
            current_epoch = 0
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    if not df.empty and "epoch" in df.columns:
                        epoch_series = pd.to_numeric(df["epoch"], errors="coerce").dropna()
                        if not epoch_series.empty:
                            current_epoch = int(epoch_series.max()) + 1
                        else:
                            current_epoch = int(len(df))
                except Exception:
                    current_epoch = 0
            train_status["epoch"] = {"current": current_epoch, "total": total_epochs}

            if current_epoch > 0:
                desired_code = "010"
                desired_msg = "訓練中"
            else:
                desired_code = "001"
                desired_msg = "初始中"

            if train_status["code"].get("code") != desired_code:
                train_status["code"] = {"code": desired_code, "msg": desired_msg}
            if desired_code in {"001", "010"} and train_status["time"].get("start_time", "None") == "None":
                train_status["time"]["start_time"] = datetime.now().strftime(time_format)
            set_status_entry(status_key, train_status)
            time.sleep(5)
        return_code = proc.wait()

        with generation_lock:
            if project_generations.get(status_key, -1) != start_generation:
                return train_status  # 舊世代不再寫入結束 / 匯出

        if return_code in (0, 15):
            # 行程結束後再讀一次 epoch（避免 loop 間隔遺漏最後一筆）
            try:
                exp_path_final = os.path.join(train_out_dir, 'exp')
                csv_path_final = os.path.join(exp_path_final, 'results.csv')
                if os.path.exists(csv_path_final):
                    df_final = pd.read_csv(csv_path_final)
                    if not df_final.empty and 'epoch' in df_final.columns:
                        epoch_series_final = pd.to_numeric(df_final['epoch'], errors='coerce').dropna()
                        if not epoch_series_final.empty:
                            train_status['epoch']['current'] = int(epoch_series_final.max()) + 1
                        else:
                            train_status['epoch']['current'] = int(len(df_final))
            except Exception:
                pass
            if train_status['epoch'].get('total'):
                train_status['epoch']['current'] = max(
                    train_status['epoch']['current'],
                    train_status['epoch']['total']
                )
            # 若關閉自動匯出或尚未達到最少匯出 epoch，直接標記完成跳過匯出
            if (not AUTO_EXPORT_ENABLED) or (AUTO_EXPORT_MIN_EPOCHS > 0 and train_status['epoch'].get('current',0) < AUTO_EXPORT_MIN_EPOCHS):
                train_status["code"] = {"code": "019", "msg": "已完成(未自動匯出)"}
                set_status_entry(status_key, train_status)
                return train_status
            # 只有在訓練輸出存在時才嘗試轉換，避免『等待中 -> 轉換中 -> 轉換失敗』的誤導
            best_pt = os.path.join(train_out_dir, 'exp', 'weights', 'best.pt')
            last_pt = os.path.join(train_out_dir, 'exp', 'weights', 'last.pt')
            if not (os.path.isfile(best_pt) or os.path.isfile(last_pt)):
                # 訓練輸出缺失，直接標記狀態並略過轉換
                train_status["code"] = {"code": "019", "msg": "已完成(未自動匯出)"}
                set_status_entry(status_key, train_status)
            else:
                train_status["code"] = {"code": "015", "msg": "轉換中"}
                set_status_entry(status_key, train_status)
                try:
                    raw_cfg_path = getattr(train_config, 'CFG', '')
                    raw_cfg_name = os.path.basename(raw_cfg_path)
                    size_token = None
                    for tk in ['t','s','m','e','c']:
                        if raw_cfg_name.endswith(f"-{tk}.yaml"): size_token = tk; break

                    gelan_cfg_name = f"gelan-{size_token}.yaml" if size_token else "gelan-s.yaml"
                    train_status.setdefault('model', {})['cfg'] = raw_cfg_path
                    train_status['model']['gelan_cfg'] = gelan_cfg_name

                    composite_project = os.path.join(project, (task), version).replace('\\', '/')
                    export_cmd = [
                        sys.executable, 
                        "./export.py", 
                        "--cfg", 
                        gelan_cfg_name, 
                        "--project", 
                        composite_project, 
                        "--user", 
                        user_scoped]

                    env = os.environ.copy()
                    # 強制 UTF-8，避免 Windows 主控台編碼造成 UnicodeEncodeError
                    env['PYTHONIOENCODING'] = 'utf-8'
                    res = subprocess.run(
                        export_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        encoding='utf-8',
                        errors='replace',
                        env=env
                    )
                    stdout_full	  = res.stdout or ''
                    stdout_lines = stdout_full.splitlines()
                    try:
                        onnx_path = Path("/shared") / "download" / user_scoped / project / task / version / f"{version}.onnx"
                        onnx_ok = onnx_path.is_file() and (onnx_path.stat().st_size > 0)
                    except Exception:
                        onnx_ok = False
                        onnx_path = Path("/shared") / "download" / user_scoped / project / task / version / f"{version}.onnx"

                    unicode_error = 'UnicodeEncodeError' in stdout_full
                    interpreted_success = (res.returncode == 0 and onnx_ok) or (onnx_ok and unicode_error)
                    export_info = {
                        'returncode':   res.returncode,
                        'onnx_exists':  onnx_ok,
                        'onnx_path':    str(onnx_path),
                        'log_tail':     stdout_lines[-20:],
                        'unicode_encode_error': unicode_error,
                        'interpreted_success':  interpreted_success
                    }
                    # 若存在可忽略的 UnicodeEncodeError 且模型輸出存在，視為成功但附帶警告
                    if interpreted_success:
                        train_status['export'] = export_info
                        val_info: dict[str, object] = {}
                        try:
                            train_status['code'] = {'code': '010', 'msg': '訓練中'}
                            set_status_entry(status_key, train_status)

                            run_val(user_scoped, project, task, version)

                            val_log = (
                                Path(_runs_root(user_scoped))
                                / 'val-test'
                                / project
                                / task
                                / version
                                / 'log'
                                / 'log.txt'
                            )
                            if val_log.is_file():
                                try:
                                    tail_lines = val_log.read_text(encoding='utf-8', errors='ignore').splitlines()
                                except Exception:
                                    tail_lines = []
                                val_info['log_tail'] = tail_lines[-20:]

                            msg = '已完成' if res.returncode == 0 else '已完成(輸出成功但終端編碼警告)'
                            train_status['code'] = {'code': '019', 'msg': msg}
                        except Exception as e_val:
                            val_info['exception'] = f'val 失敗: {e_val}'
                            train_status['code'] = {'code': '999', 'msg': '驗證失敗'}

                        if val_info:
                            train_status['val'] = val_info
                    elif res.returncode == 0 and not onnx_ok:
                        export_info['error'] = 'ONNX 檔案缺失'
                        train_status['export'] = export_info
                        train_status['code'] = {'code': '999', 'msg': '轉換結果檔缺失'}
                    else:
                        train_status['export'] = export_info
                        train_status['code'] = {'code': '999', 'msg': '轉換失敗'}
                except Exception as e_exp:
                    train_status.setdefault('export', {})['exception'] = f'export 失敗: {e_exp}'
                    train_status["code"] = {"code": "999", "msg": "轉換失敗"}
        else:
            train_status["code"] = {"code": "999", "msg": f"失敗（退出碼 {return_code}）"}
        set_status_entry(status_key, train_status)
    except Exception as e:
        train_status["code"] = {"code": "999", "msg": f"讀取進度錯誤：{e}"}
        set_status_entry(status_key, train_status)
    finally:
        with generation_lock:
            cur_gen = project_generations.get(status_key, start_generation)
        if cur_gen == start_generation:
            train_status["time"]["end_time"] = datetime.now().strftime(time_format)
            set_status_entry(status_key, train_status)
    return train_status


def _next_generation_name(user_name: str, project: str, task: str) -> str:
    """在 runs/<user>/<project>/<task>/ 下尋找下一個可用的數字資料夾名稱（1, 2, 3, ...）。"""
    try:
        base = os.path.join(_runs_root(user_name), project, task)
        os.makedirs(base, exist_ok=True)
        mx = 0
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if not os.path.isdir(p):
                continue
            try:
                n = int(name)
                if n > mx:
                    mx = n
            except Exception:
                continue
        return str(mx + 1)
    except Exception:
        return "1"


@router.post("/train")
def train_model(
    user_name: str       = Form("", description="使用者名稱"),
    dataset_name: str    = Form("", description="資料集"),
    dataset_version: str = Form("", description="資料集版本"),
    project: str         = Form("", description="專案名稱"),
    task: str            = Form("", description="訓練名稱"),
    epochs: int          = Form(10, description="訓練的 epoch 次數"),
    batch_size: int      = Form(16, description="訓練的 batch size"),
    img_size: int        = Form(320, description="訓練的圖片大小"),
    close_mosaic: int    = Form(0, description="馬賽克的閾值"),
    device: str          = Form("", description="使用的 GPU 編號 (例如 0 或 0,1)")):

    # 統一改用 train_config.WORKERS
    workers_cfg = getattr(train_config, 'WORKERS', 0)
    print(dataset_name, epochs, batch_size, img_size, workers_cfg, close_mosaic)
    # 強制 project / task 不能為空或僅空白
    if not isinstance(project, str) or not project.strip():
        raise HTTPException(status_code=400, detail="project 不能為空")
    if not isinstance(task, str) or not task.strip():
        raise HTTPException(status_code=400, detail="task 不能為空")
    # 檢查：project / task 名稱合法
    err = _validate_task(project)
    if err:
        return err
    # 驗證並正規化 user_name
    user_name = _ensure_user_name(user_name)
    # 驗證 task 必填與合法
    err_pt = _validate_task(task)
    if err_pt:
        return err_pt
    
    device_value = (device or "").strip()
    f_key = f"{user_name}|{project}|{task}"
    _ct_map = None

    # 若 dataset_name 或 dataset_version 為空，嘗試從 create_train_status_map.json 取回預設
    # key 形式： user|project|task （對應 create_train 時的 user|project|task）
    fallback_map_file = "create_train_status_map.json"
    if (not dataset_name.strip()) or (not dataset_version.strip()):
        try:
            if os.path.exists(fallback_map_file):
                with open(fallback_map_file, 'r', encoding='utf-8') as f:
                    _ct_map = json.load(f) or {}
            else:
                _ct_map = {}
        except Exception:
            _ct_map = {}
        fb_entry = _ct_map.get(f_key, {}) if isinstance(_ct_map, dict) else {}
        if not dataset_name.strip():
            dataset_name = fb_entry.get('dataset', '').strip()
        if not dataset_version.strip():
            dataset_version = str(fb_entry.get('dataset_version', '') or '').strip()
        if not device_value:
            device_value = str(fb_entry.get('device', '') or '').strip()

    if not device_value:
        if _ct_map is None:
            try:
                if os.path.exists(fallback_map_file):
                    with open(fallback_map_file, 'r', encoding='utf-8') as f:
                        _ct_map = json.load(f) or {}
                else:
                    _ct_map = {}
            except Exception:
                _ct_map = {}

        fb_entry = _ct_map.get(f_key, {}) if isinstance(_ct_map, dict) else {}
        device_value = str(fb_entry.get('device', '') or '').strip()
    # 再檢查是否仍為空
    if not dataset_name.strip():
        raise HTTPException(status_code=400, detail="dataset_name 不能為空，且在 create_train_status_map.json 中也找不到對應預設")
    if not dataset_version.strip():
        raise HTTPException(status_code=400, detail="dataset_version 不能為空，且在 create_train_status_map.json 中也找不到對應預設")
    if not device_value.strip():
        raise HTTPException(status_code=400, detail="device 不能為空，且在 create_train_status_map.json 中也找不到對應預設")

    # 正規化 dataset_name 與 user_name 的對應：
    # - 若 dataset_name 含分隔符，檢查其 user 是否等於 user_name
    # - 若不含，組合為 <user>/<dataset>
    norm_dn = dataset_name.replace('\\', '/').strip().strip('/')
    if '/' in norm_dn:
        # 防止使用者傳遞尾端為空的路徑（例如 myds/ ）導致 split 後空字串
        left, right = norm_dn.split('/', 1)
        if not right.strip():
            raise HTTPException(status_code=400, detail="dataset_name 不可以 '/' 結尾或缺少資料集資料夾名稱")
        dn_user = left
        if _normalize_user_name(dn_user) != user_name:
            raise HTTPException(status_code=400, detail=f"dataset_name 的使用者 '{dn_user}' 與指定的 user_name '{user_name}' 不一致")
        dataset_name = f"{user_name}/{right}"
    else:
        dataset_name = f"{user_name}/{norm_dn}"
    # 路徑存在性預檢（資料集版本切換）：
    # - 一律使用 Dataset/<user>/<dataset>/data.yaml（不改動、不使用 dataset_home 的 data.yaml）
    # - 若指定 dataset_version：將 dataset_home/<user>/<dataset>/versions/vXX 下的 train/val/test.txt
    #   覆蓋至 dataset_home/<user>/<dataset>/ 對應的 txt（僅覆蓋 txt，data.yaml 不變）
    vname = None
    dv = (dataset_version or '').strip()
    if dv:
        vstr = dv.lower()
        try:
            vname = f"v{int(vstr):02d}" if vstr.isdigit() else (vstr if vstr.startswith('v') else ('v' + vstr))
        except Exception:
            vname = vstr if vstr.startswith('v') else ('v' + vstr)
    base_home_dir = os.path.join('.', 'dataset_home', *dataset_name.split('/'))
    base_dataset_dir = os.path.join('.', 'Dataset', *dataset_name.split('/'))
    if vname:
        versions_dir = os.path.join(base_home_dir, 'versions')
        target_ver_dir = os.path.join(versions_dir, vname)
        # 基本安全檢查：確保版本目錄存在且位於 dataset_home 範圍內
        base_abs = os.path.normcase(os.path.abspath(base_home_dir))
        ver_abs = os.path.normcase(os.path.abspath(target_ver_dir))

        if not (ver_abs == base_abs or ver_abs.startswith(base_abs + os.sep)):
            raise HTTPException(status_code=400, detail='版本目錄位置不合法')
        
        if not os.path.isdir(target_ver_dir):
            raise HTTPException(status_code=404, detail=f"找不到資料集版本目錄：{os.path.normpath(target_ver_dir)}")
        
        # 覆蓋 train/val/test.txt
        for fname in ['train.txt', 'val.txt', 'test.txt']:
            src = os.path.join(target_ver_dir, fname)
            dst = os.path.join(base_home_dir, fname)
            if os.path.isfile(src):
                try:
                    shutil.copyfile(src, dst)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"複製 {fname} 失敗：{e}")
        data_yaml_path = os.path.join(base_dataset_dir, 'data.yaml')
    else:
        # 未指定版本：使用 Dataset/<user>/<dataset>/data.yaml
        data_yaml_path = os.path.join(base_dataset_dir, 'data.yaml')
    if not os.path.isfile(data_yaml_path):
        raise HTTPException(status_code=404, detail=f"找不到資料集設定檔：{os.path.normpath(data_yaml_path)}")
    # 自動指派世代資料夾（1,2,3,...）
    version = _next_generation_name(user_name, project, task)

    # 設定或取得目前世代
    with generation_lock:
        skey = _status_key(user_name, project, task, version)
        if skey not in project_generations:
            project_generations[skey] = 0
        start_gen = project_generations[skey]
    # digits-only version for status/response
    dv_digits = (vname[1:] if vname and vname.lower().startswith('v') else (vname or ''))
    # 先寫入一筆初始狀態（001 初始中），讓前端立即可見
    status_key = _status_key(user_name, project, task, version)
    init_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    initial_entry = {
        "time":     {"create_time": init_time, "start_time": init_time, "end_time": "None"},
        "code":     {"code": "001", "msg": "初始中"},
        "epoch":    {"current": 0, "total": 0},
        "model":    {"name": MODEL_FAMILY},
        "dataset":  {"name": dataset_name.replace('\\','/').split('/')[-1], "version": (dv_digits or "")},
        "user":     {"name": user_name},
        "project":  {"name": project},
        "task":     {"name": task},
        "version":  {"name": version},
        "generation": start_gen,
    }
    set_status_entry(status_key, initial_entry)

    # 啟動執行緒
    device_value = device_value or "0"
    t = Thread(
        target=run_training,
        args=(
            dataset_name,
            version,
            epochs,
            batch_size,
            img_size,
            workers_cfg,
            close_mosaic,
        ),
        kwargs={
            "initial_code": "001",
            "initial_msg": "初始中",
            "start_generation": start_gen,
            "user_name": user_name,
            "project": project,
            "task": task,
            "data_yaml_path": data_yaml_path,
            "dataset_version": dv_digits or None,
            "device": device_value,
        },
    )
    t.start()

    return {
        "version": version,
        "project": f"{project}/{task}",
        "dataset": dataset_name.replace('\\','/').split('/')[-1],
        "dataset_version": dv_digits or ""
    }


@router.get("/get_progress")
def get_progress_endpoint(
    user_name: str  = Query("",description="使用者名稱"),
    project: str    = Query("",description="專案名稱"),
    task: str       = Query("",description="訓練名稱"),
    version: str    = Query("",description="版本")
):
    # 基本名稱檢查
    err = _validate_task(project)
    if err:
        return err
    err_v = _validate_task(version)
    if err_v:
        return err_v
    ensured_user = _ensure_user_name(user_name)
    return get_progress(version, ep="", user_name=ensured_user, project=project, task_name=task)


def _terminate_tree(pid: int):
    if psutil is None:
        raise HTTPException(status_code=400, detail="缺少 psutil，請先安裝：pip install psutil")
    try:
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            try:
                c.terminate()
            except Exception:
                pass
        try:
            p.terminate()
        except Exception:
            pass
        gone, alive = psutil.wait_procs([p], timeout=5)
        # 強制殺掉殘留
        for a in alive:
            try:
                a.kill()
            except Exception:
                pass
        return True
    except psutil.NoSuchProcess:
        return False


def _default_status(task: str) -> dict:
    """查無狀態時回傳的預設物件（對應代碼 980）。
    重新定義（若前方未定義）供 get_training_status 使用。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "time":     {"create_time": now_str, "start_time": "None", "end_time": "None"},
        "code":     {"code": "980", "msg": "查無此專案狀態"},
        "epoch":    {"current": 0, "total": 0},
        "model":    {"name": MODEL_FAMILY},
        "dataset":  {"name": "unknown"},
    }


@router.get("/train/status")
def get_training_status(
    user_name: str          = Query("", description="使用者名稱"),
    project: str            = Query("", description="專案名稱"),
    task: str               = Query("", description="訓練名稱"),
    version: Optional[str]  = Query(None, description="版本")):

    """查詢訓練狀態（新結構）
    - 僅支援單一版本查詢：必須提供 version，回傳 runs/<user>/<project>/<task>/<version> 的狀態。
    - 不支援多版本彙整查詢。
    """
    status_map = load_status_map()
    user_name = _ensure_user_name(user_name)
    # 驗證名稱
    for nm in (project, task):
        err = _validate_task(nm)
        if err:
            return err
    # 若前端有帶 version 參數但為空字串，視為不合法，不回傳全部版本
    if version is not None and (not str(version).strip()):
        raise HTTPException(status_code=400, detail="version 不能為空")
    # 未帶 version -> 視為不合法（不提供多版本彙整）
    if version is None:
        raise HTTPException(status_code=400, detail="缺少 version 參數")
    if version:
        err_v = _validate_task(version)
        if err_v:
            return err_v

    # 抽取欄位的容錯讀取
    def _get(d: dict, path: list[str]) -> Optional[str]:
        cur = d
        try:
            for key in path:
                cur = cur.get(key, {}) if isinstance(cur, dict) else {}
            # 最後一層若不是 dict，嘗試 name 欄位
        except Exception:
            return None
        return cur if isinstance(cur, str) else (cur.get('name') if isinstance(cur, dict) else None)

    def _user_of(st: dict) -> Optional[str]:
        u = _get(st, ['user'])
        if u:
            return str(u).strip().lower()
        ds = _get(st, ['dataset']) or ''
        return _extract_user_from_dataset_name(str(ds))

    def _project_of(st: dict) -> Optional[str]:
        p = _get(st, ['project'])
        return p

    def _task_of(st: dict) -> Optional[str]:
        pt = _get(st, ['task'])
        return pt

    # 單一版本
    if version:
        # 先以複合鍵直接查詢
        skey = _status_key(user_name, project, task, version)
        st = status_map.get(skey)
        if not st:
            # 再退一步：在 map 中尋找符合欄位的項
            for k, v in (status_map or {}).items():
                try:
                    u = (v.get('user', {}) or {}).get('name')
                    p = (v.get('project', {}) or {}).get('name')
                    pt = (v.get('task', {}) or {}).get('name')
                    ver = (v.get('version', {}) or {}).get('name') or (k.split('|')[-1] if '|' in k else k)
                    if (u or '').strip().lower() == user_name and p == project and (pt or 'default') == (task or 'default') and ver == version:
                        st = v
                        break
                except Exception:
                    continue
        if not st:
            # 不存在則回傳預設 980
            st = _default_status(version)
            st['user'] = {'name': user_name}
            st['project'] = {'name': project}
            st['version'] = {'name': version}
            return st
        if _user_of(st) != user_name or _project_of(st) != project or (_task_of(st) or 'default') != (task or 'default'):
            # 範圍不符也視為不存在
            st2 = _default_status(version)
            st2['user'] = {'name': user_name}
            st2['project'] = {'name': project}
            st2['task'] = {'name': task}
            st2['version'] = {'name': version}
            return st2
        # 補全缺漏欄位
        st.setdefault('user', {'name': user_name})
        st.setdefault('project', {'name': project})
        st.setdefault('task', {'name': task})
        st.setdefault('version', {'name': version})
        try:
            # 若狀態本身已有 release_note（可能已同步），則優先使用現有格式
            if 'release_note' not in st:
                rel_map = _load_release_note_map()
                skey_release = _status_key(user_name, project, task, version)
                rel_entry = rel_map.get(skey_release)
                if isinstance(rel_entry, dict) and 'note' in rel_entry:
                    # 統一包裝格式
                    st['release_note'] = {
                        "note": rel_entry.get('note')
                    }
        except Exception:
            pass
        return st
    # 不支援多版本彙整
    raise HTTPException(status_code=400, detail="不支援多版本查詢，請提供 version")

@router.post("/project/create")
def create_project(
    user_name: str  = Form("", description="使用者名稱"), 
    project: str    = Form("", description="專案名稱")):
    
    _validate_user_name(user_name)
    user_name = _normalize_user_name(user_name) or user_name
    err = _validate_task(project)
    if err:
        return err
    base = _runs_root(user_name)
    # 禁止同使用者下專案名稱重複（專案=訓練集合上層）
    project_dir = os.path.join(base, project)
    if os.path.exists(project_dir):
        raise HTTPException(status_code=409, detail="同一使用者下的專案名稱已存在")
    os.makedirs(project_dir, exist_ok=True)
    # 同步建立 train_log 對應樹（與專案同名的層級）
    os.makedirs(os.path.join(base, "train_log", project), exist_ok=True)
    return {"message": "專案已建立", "name": project, "path": "/" + project_dir.replace('\\', '/')}


@router.post("/task/create")
def create_train(user_name: str        = Form("", description="使用者名稱"), 
                 project: str          = Form("", description="專案名稱"),
                 task: str             = Form("", description="訓練名稱"),
                 dataset_name: str     = Form("", description="資料集名稱"),
                 dataset_version: str  = Form("", description="資料集版本"),
                 device: str           = Form("0", description="使用的 GPU 編號 (例如 0 或 0,1)")):
    
    _validate_user_name(user_name)
    user_name = _normalize_user_name(user_name) or user_name
    # 驗證 project 和 task
    err = _validate_task(project)
    if err:
        return err
    err = _validate_task(task)
    if err:
        return err
    base = _runs_root(user_name)
    # 禁止同專案下訓練名稱重複
    train_root = os.path.join(base, project)
    os.makedirs(train_root, exist_ok=True)

    project_dir = os.path.join(train_root, task)
    if os.path.exists(project_dir):
        raise HTTPException(status_code=409, detail="同一專案下的訓練名稱已存在")
    os.makedirs(project_dir, exist_ok=True)
    # 同步建立 train_log 對應樹，保持相同的層級結構
    os.makedirs(os.path.join(base, "train_log", project, task), exist_ok=True)

    hyp = os.path.join(".","yolov9","data","hyps","hyp.scratch-high.yaml")
    hyp_scr = os.path.join(project_dir, "hyp.scratch-high.yaml")
    
    if not os.path.isfile(hyp):
        raise HTTPException(status_code=500, detail=f"找不到預設的 hyp.scratch-high.yaml 檔案：{os.path.normpath(hyp)}")
    try:
        shutil.copyfile(hyp, hyp_scr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"複製 hyp.scratch-high.yaml 失敗：{e}")

    # 寫入 create_train_status_map.json（訓練列表需讀取 device 與資料集資訊）
    ct_file = "create_train_status_map.json"
    def _load(path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    def _save(path: str, data: dict):
        try:
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            try:
                with open(path + '.bak', 'w', encoding='utf-8') as fb:
                    json.dump(data, fb, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception:
            pass
    ct_map = _load(ct_file)
    key = f"{user_name}|{project}|{task}"
    ct_map[key] = {
        "dataset":          dataset_name.strip(),
        "dataset_version":  (dataset_version.strip().lstrip('v') if dataset_version else ""),
        "created_date":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device":           device.strip() or "0"
   }
    _save(ct_file, ct_map)

    # 更新 projectlist_status_map.json 的 updated_date（新增訓練任務時）
    pl_file = "projectlist_status_map.json"
    try:
        pl_map = _load(pl_file)
        user_projects = pl_map.get(user_name, [])
        # 尋找現有專案項目
        found = False
        now_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for it in user_projects:
            if isinstance(it, dict) and it.get('name') == project:
                it['updated_date'] = now_date
                found = True
                break
        if found:
            _save(pl_file, pl_map)
    except Exception:
        pass

    return {"message": "訓練專案已建立", "path": "/" + project_dir.replace('\\', '/'), "Dataset_name": dataset_name, "Dataset_version": dataset_version}

# ================ delete / project / task / version  ==================

def _collect_status_keys_and_pids(prefix: str) -> tuple[list[str], list[int]]:
    status_map = load_status_map()
    matched_keys: list[str] = []
    pids: list[int] = []
    for key, value in (status_map or {}).items():
        if isinstance(key, str) and key.startswith(prefix):
            matched_keys.append(key)
            if isinstance(value, dict):
                pid = (value.get("proc") or {}).get("pid")
                if pid:
                    pids.append(int(pid))
    # 移除重複 PID，保持原順序
    unique_pids = list(dict.fromkeys(pids))
    return matched_keys, unique_pids


def _remove_status_keys(keys: list[str]) -> bool:
    if not keys:
        return False

    removed = {"flag": False}

    def _del(mm: dict):
        for key in keys:
            if key in mm:
                del mm[key]
                removed["flag"] = True
        return mm

    _with_status_map(_del)
    return removed["flag"]


def _safe_rmtree(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _status_key_project(user_name: Optional[str], project: str) -> str:
    """唯一狀態鍵：user|project，避免不同訓練分支衝突。"""
    usr = _ensure_user_name(user_name)
    return f"{usr}|{project}"


def _status_key_task(user_name: Optional[str], project: str, task: str) -> str:
    """唯一狀態鍵：user|project|task，避免不同訓練分支衝突。"""
    usr = _ensure_user_name(user_name)
    return f"{usr}|{project}|{task}"


@router.delete("/delete/project")
def delete_training_project(
    user_name: str  = Query("", description="使用者名稱"),
    project: str    = Query("", description="專案名稱")):

    """刪除專案下所有版本的輸出與狀態：
    - runs/<user>/<project>/<task>
    - runs/<user>/train_log/<project>/<task>
    - download/<user>/<project>/<task>
    - train_status_map.json 中該專案的所有相關狀態
    """
    # 驗證
    user_n = _ensure_user_name(user_name)
    err = _validate_task(project)
    if err:
        return err
    
    # 停止所有相關進程
    skey_prefix = _status_key_project(user_n, project)
    matched_keys, pids = _collect_status_keys_and_pids(skey_prefix)

    for pid in pids:
        try:
            _terminate_tree(pid)
        except Exception:
            pass
    
    # 刪除 runs 專案資料夾
    project_dir = os.path.join(_runs_root(user_n), project)
    _safe_rmtree(project_dir)
    
    # 刪除 train_log 對應
    log_dir = os.path.join(_runs_root(user_n), "train_log", project)
    _safe_rmtree(log_dir)
    
    # 刪除 download 對應
    dl_dir = os.path.join("download", user_n, project)
    _safe_rmtree(dl_dir)

    # 移除所有相關狀態
    removed_status = _remove_status_keys(matched_keys)
    
    return {
        "deleted": True,
        "paths": {
            "runs_version": "/" + project_dir.replace('\\', '/') ,
            "train_log":    "/" + log_dir.replace('\\', '/'),
            "download":     "/" + dl_dir.replace('\\', '/')
        },
        "status_removed": removed_status
    }


@router.delete("/delete/task")
def delete_training_task(
    user_name: str  = Query("", description="使用者名稱"),
    project: str    = Query("", description="專案名稱"),
    task: str       = Query("", description="訓練名稱")):

    """刪除專案下所有版本的輸出與狀態：
    - runs/<user>/<project>/<task>
    - runs/<user>/train_log/<project>/<task>
    - download/<user>/<project>/<task>
    - train_status_map.json 中該專案的所有相關狀態
    """
    # 驗證
    user_n = _ensure_user_name(user_name)
    for nm in (project, task):
        err = _validate_task(nm)
        if err:
            return err
    
    # 停止所有相關進程
    skey_prefix = _status_key_task(user_n, project, task)
    matched_keys, pids = _collect_status_keys_and_pids(skey_prefix)

    for pid in pids:
        try:
            _terminate_tree(pid)
        except Exception:
            pass
    
    # 刪除 runs 專案資料夾
    project_dir = os.path.join(_runs_root(user_n), project, task)
    _safe_rmtree(project_dir)
    
    # 刪除 train_log 對應
    log_dir = os.path.join(_runs_root(user_n), "train_log", project, task)
    _safe_rmtree(log_dir)
    
    # 刪除 download 對應
    dl_dir = os.path.join("download", user_n, project, task)
    _safe_rmtree(dl_dir)

    # 移除所有相關狀態
    removed_status = _remove_status_keys(matched_keys)
    
    # 更新 projectlist_status_map.json 中對應專案的 updated_date（任務刪除也視為更新）
    try:
        pl_file = "projectlist_status_map.json"
        def _pl_load(path: str):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        def _pl_save(path: str, data: dict):
            try:
                tmp = path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            except Exception:
                pass
        pl_map = _pl_load(pl_file)
        user_key = user_n
        projects_list = pl_map.get(user_key, [])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changed = False
        for it in projects_list:
            if isinstance(it, dict) and it.get('name') == project:
                it['updated_date'] = ts
                changed = True
                break
        if changed:
            pl_map[user_key] = projects_list
            _pl_save(pl_file, pl_map)
    except Exception:
        pass

    return {
        "deleted": True,
        "paths": {
            "runs_version": "/" + project_dir.replace('\\', '/') ,
            "train_log":    "/" + log_dir.replace('\\', '/'),
            "download":     "/" + dl_dir.replace('\\', '/')
        },
        "status_removed": removed_status
    }


@router.delete("/delete/version")
def delete_training_version(
    user_name: str  = Query("", description="使用者名稱"),
    project: str    = Query("", description="專案名稱"),
    task: str       = Query("", description="訓練名稱"),
    version: str    = Query("", description="版本")):

    """刪除單一版本的輸出與狀態：
    - runs/<user>/<project>/<task>/<version>
    - runs/<user>/train_log/<project>/<task>/<version>_log
    - download/<user>/<project>/<task>/<version>
    - train_status_map.json 中該複合鍵的狀態
    """
    # 驗證
    user_n = _ensure_user_name(user_name)
    for nm in (project, version):
        err = _validate_task(nm)
        if err:
            return err
    # 停止進程（若該版本仍在跑）
    skey = _status_key(user_n, project, task, version)
    matched_keys, pids = _collect_status_keys_and_pids(skey)

    for pid in pids:
        try:
            _terminate_tree(pid)
        except Exception:
            pass

    # 刪除 runs 版本資料夾
    vdir = os.path.join(_runs_root(user_n), project, task, version)
    _safe_rmtree(vdir)

    # 刪除 train_log 對應
    log_dir = os.path.join(_runs_root(user_n), "train_log", project, task, f"{version}_log")
    _safe_rmtree(log_dir)

    # 刪除 download 對應
    dl_dir = os.path.join("download", user_n, project, task, version)
    _safe_rmtree(dl_dir)

    # 移除狀態
    removed_status = _remove_status_keys(matched_keys)

    return {
        "deleted": True,
        "paths": {
            "runs_version": "/" + vdir.replace('\\', '/') ,
            "train_log":    "/" + log_dir.replace('\\', '/'),
            "download":     "/" + dl_dir.replace('\\', '/')
        },
        "status_removed": removed_status
    }

# ================ Pause / Resume / Restart  ==================

def _locate_status_entry(user_name: str, project: str, task: str, version: str):
    """根據複合鍵直接取得狀態與 map（呼叫方可修改後 save）。"""
    skey = _status_key(_ensure_user_name(user_name), project, task, version)
    m = load_status_map()
    st = m.get(skey)
    return skey, m, st

def _locate_latest_status_entry(user_name: str, project: str, task: str):
    """不帶 version 時，自動鎖定目前正在執行的版本，若無執行中則選最新版本。
    回傳 (skey, m, st)。找不到則拋 404。
    """
    m = load_status_map()
    user_n = _ensure_user_name(user_name)
    prefix = f"{user_n}|{project}|{task}|"
    candidates = []  # (version, skey, st)
    for k, st in (m or {}).items():
        if isinstance(k, str) and k.startswith(prefix) and isinstance(st, dict):
            ver = k.split('|')[-1]
            candidates.append((ver, k, st))
    if not candidates:
        raise HTTPException(status_code=404, detail="找不到訓練狀態")
    # 先找執行中的（有 PID 且 process 存在），多個則取版本號最新
    running = []
    if psutil is not None:
        for ver, k, st in candidates:
            pid = ((st.get('proc') or {}).get('pid')) if isinstance(st, dict) else None
            if pid:
                try:
                    p = psutil.Process(pid)
                    if p.is_running():
                        running.append((ver, k, st))
                except Exception:
                    continue
    def ver_key(x: tuple[str, str, dict]):
        v = x[0]
        try:
            return (0, int(v))  # 數字優先，較大較新
        except Exception:
            return (1, len(v), v)
    target = None
    if running:
        running.sort(key=ver_key)
        target = running[-1]
    else:
        candidates.sort(key=ver_key)
        target = candidates[-1]
    ver, skey, st = target
    return skey, m, st


@router.post("/train/pause")
def pause_training(
    user_name: str  = Form("", description="使用者名稱"),
    project: str    = Form("", description="專案名稱"),
    task: str       = Form("", description="訓練名稱")):

    """
    暫停訓練（不需指定版本）：
    - 優先暫停目前執行中的版本；若無執行中則暫停最新版本。
    - 狀態碼 -> 011。
    """
    if psutil is None:
        raise HTTPException(status_code=400, detail="缺少 psutil，無法暫停")
    skey, m, st = _locate_latest_status_entry(user_name, project, task)
    if not st:
        raise HTTPException(status_code=404, detail="找不到訓練狀態")
    pid = ((st.get('proc') or {}).get('pid')) if isinstance(st, dict) else None
    if not pid:
        raise HTTPException(status_code=400, detail="無 PID 可暫停")
    try:
        p = psutil.Process(pid)
        p.suspend()
    except psutil.NoSuchProcess:
        raise HTTPException(status_code=404, detail="進程不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"暫停失敗: {e}")
    # 更新狀態
    st.setdefault('code', {})
    st['code']['code'] = '011'
    st['code']['msg'] = '暫停中'
    m[skey] = st
    save_status_map(m)
    return {"message": "已暫停", "pid": pid, "status": st['code']}


@router.post("/train/resume")
def resume_training(
    user_name: str  = Form("", description="使用者名稱"),
    project: str    = Form("", description="專案名稱"),
    task: str       = Form("", description="訓練名稱")):
    
    """恢復訓練（不需指定版本）：
    - 優先恢復目前執行中的版本；若無執行中則針對最新版本動作。
    - 恢復後狀態碼若原為 011 -> 010。
    """
    if psutil is None:
        raise HTTPException(status_code=400, detail="缺少 psutil，無法恢復")
    skey, m, st = _locate_latest_status_entry(user_name, project, task)
    if not st:
        raise HTTPException(status_code=404, detail="找不到訓練狀態")
    pid = ((st.get('proc') or {}).get('pid')) if isinstance(st, dict) else None
    if not pid:
        raise HTTPException(status_code=400, detail="無 PID 可恢復")
    try:
        p = psutil.Process(pid)
        p.resume()
    except psutil.NoSuchProcess:
        raise HTTPException(status_code=404, detail="進程不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢復失敗: {e}")
    # 更新狀態：僅當前為 011 時調整為 010
    cur_code = (st.get('code') or {}).get('code')
    if cur_code == '011':
        st['code']['code'] = '010'
        st['code']['msg'] = '訓練中'
    m[skey] = st
    save_status_map(m)
    return {"message": "已恢復", "pid": pid, "status": st['code']}


def _recover_train_parameters(user_name: str, project: str, task: str, version: str):
    """從 opt.yaml 與 create_train_status_map.json 取得重新啟動所需參數。
    回傳 (dataset_name, dataset_version, epochs, batch_size, img_size, close_mosaic, device)
    若缺失則使用預設或 0。"""
    dataset_name = ''
    dataset_version = ''
    epochs = 0
    batch_size = 16
    img_size = 320
    close_mosaic = 0
    device_value = "0"
    try:
        exp_dir = os.path.join(_runs_root(user_name), project, task, version, 'exp')
        opt_path = os.path.join(exp_dir, 'opt.yaml')

        if os.path.isfile(opt_path):
            with open(opt_path, 'r', encoding='utf-8') as f:
                opt_data = yaml.safe_load(f) or {}

            epochs = int(opt_data.get('epochs', epochs))
            batch_size = int(opt_data.get('batch_size', opt_data.get('batch', batch_size)))
            img_size = int(opt_data.get('imgsz', opt_data.get('img', img_size)))
            close_mosaic = int(opt_data.get('close_mosaic', close_mosaic))
            opt_device = str(opt_data.get('device', "")).strip()
            if opt_device:
                device_value = opt_device
            data_path = opt_data.get('data') or ''

            if data_path:
                # 抽取 dataset 名稱
                parts = os.path.normpath(data_path).split(os.sep)
                if 'Dataset' in parts:
                    i = parts.index('Dataset')
                    sub = parts[i+1:]
                    if sub and sub[-1].lower() == 'data.yaml':
                        sub = sub[:-1]
                    if sub:
                        dataset_name = '/'.join(sub)
                # dataset_version 由狀態 map 或 create map 補
    except Exception:
        pass
    # 從 create_train_status_map.json 補 dataset、版本與裝置
    try:
        ct_file = 'create_train_status_map.json'
        if os.path.exists(ct_file):
            with open(ct_file, 'r', encoding='utf-8') as f:
                ct_map = json.load(f) or {}
        else:
            ct_map = {}
        if isinstance(ct_map, dict):
            ckey = f"{_normalize_user_name(user_name)}|{project}|{task}"
            centry = ct_map.get(ckey) or {}
            if not dataset_name and centry.get('dataset'):
                dataset_raw = str(centry.get('dataset')).strip()
                if dataset_raw:
                    dataset_name = f"{user_name}/{dataset_raw}" if '/' not in dataset_raw else dataset_raw
            if centry.get('dataset_version'):
                dataset_version = str(centry.get('dataset_version'))
            if centry.get('device'):
                device_candidate = str(centry.get('device')).strip()
                if device_candidate:
                    device_value = device_candidate
    except Exception:
        pass

    # dataset_name 正規化：若沒有 user prefix 則加上
    if dataset_name and '/' not in dataset_name:
        dataset_name = f"{user_name}/{dataset_name}"
    return dataset_name, dataset_version, epochs, batch_size, img_size, close_mosaic, device_value


@router.post("/train/restart")
def restart_training(
    user_name: str  = Form("", description="使用者名稱"),
    project: str    = Form("", description="專案名稱"),
    task: str       = Form("", description="訓練名稱"),
    version: str    = Form("", description="版本")):

    """重新啟動指定 version 訓練：
    1. 標記狀態為 002 (等待中)，設定 restart_lock。
    2. 終止舊進程。
    3. 讀取之前 opt.yaml 與 create map 參數重新啟動（世代 generation +1）。
    4. 新執行緒啟動後會寫入新的 PID 與狀態。
    """
    _validate_user_name(user_name)
    user_name = _normalize_user_name(user_name) or user_name
    skey, m, st = _locate_status_entry(user_name, project, task, version)
    if not st:
        raise HTTPException(status_code=404, detail="找不到訓練狀態")
    
    pid = ((st.get('proc') or {}).get('pid')) if isinstance(st, dict) else None
    # 標記等待中 & restart_lock
    st.setdefault('code', {})
    st['code']['code'] = '002'
    st['code']['msg'] = '等待重新啟動'
    st.setdefault('flags', {})['restart_lock'] = True
    # 增加 generation
    with generation_lock:
        cur_gen = project_generations.get(skey, 0) + 1
        project_generations[skey] = cur_gen
        st['generation'] = cur_gen
    m[skey] = st
    save_status_map(m)
    # 終止舊進程
    if pid:
        try:
            _terminate_tree(pid)
        except Exception:
            pass
    # 取得原參數
    dataset_name, dataset_version, epochs, batch_size, img_size, close_mosaic, device_value = _recover_train_parameters(user_name, project, task, version)
    if not dataset_name:
        raise HTTPException(status_code=400, detail="無法恢復 dataset 參數，取消重啟")
    # 立即寫入一筆新的初始狀態（001），更新 start_time，並清理舊 exp 目錄（若存在）
    init_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st['time'] = {"create_time": st.get('time', {}).get('create_time', init_time), "start_time": init_time, "end_time": "None"}
    st['code'] = {"code": "001", "msg": "初始中"}
    m[skey] = st
    save_status_map(m)
    # 清除舊 exp 內容（確保不生成 exp2）
    try:
        version_dir = os.path.join(_runs_root(user_name), project, task, version)
        exp_dir = os.path.join(version_dir, 'exp')
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)
    except Exception:
        pass
    # 啟動新執行緒（保持同一 version）
    t = Thread(
        target=run_training,
        args=(
            dataset_name,
            version,
            epochs,
            batch_size,
            img_size,
            getattr(train_config, 'WORKERS', 0),
            close_mosaic,
        ),
        kwargs={
            "initial_code": "001",
            "initial_msg": "初始中",
            "start_generation": project_generations.get(skey),
            "user_name": user_name,
            "project": project,
            "task": task,
            "data_yaml_path": None,
            "dataset_version": dataset_version or None,
            "device": device_value,
        },
    )
    t.start()
    return {"message": "重新啟動中", "version": version, "generation": project_generations.get(skey), "project": f"{project}/{task}", "status": {"code": "001", "msg": "初始中"}}

