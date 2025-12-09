from fastapi import APIRouter,Form,Query,HTTPException
import os, json, datetime
from user_utils import runs_root, validate_user_name, ensure_user_name


router = APIRouter(
    prefix="/yolov9",
    tags=["GET_PROJECTS"],
)

_runs_root = runs_root
_validate_user_name = validate_user_name
_ensure_user_name = ensure_user_name

STATUS_FILE = "train_status_map.json"
PL_STATUS_FILE = "projectlist_status_map.json"
CT_STATUS_FILE = "create_train_status_map.json"  


def _parse_hyp_yaml(hyp_path: str) -> dict:
    """Extract training hyperparameters from hyp.scratch-high.yaml if available."""
    try:
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyp = __import__('yaml').safe_load(f) or {}
    except Exception:
        hyp = {}

    def _pick(h: dict, *keys):
        for key in keys:
            if key in h:
                return h.get(key)
        return ''

    return {
        'batch_size':   str(_pick(hyp, 'batch_size', 'batch', 'train_batch') or ''),
        'img_size':     str(_pick(hyp, 'img_size', 'imgsz', 'img') or ''),
        'close_mosaic': str(_pick(hyp, 'close_mosaic') or ''),
        'epochs':       str(_pick(hyp, 'epochs') or ''),
    }

def _load_status_map(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_status_map(path: str, data: dict):
    try:
        tmp = path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        try:
            with open(path + ".bak", 'w', encoding='utf-8') as fb:
                json.dump(data, fb, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception:
        pass

def _load_create_train_map() -> dict:
    return _load_status_map(CT_STATUS_FILE)

def _status_key(user_name: str, project: str, task: str, version: str) -> str:
    u = _ensure_user_name(user_name)
    return f"{u}|{project}|{task}|{version}"


@router.get("/Get_projects")
def list_projects( 
    user_name: str = Query("", description="使用者名稱")):

    """列出 runs/<user_name> 底下的專案資料夾清單（僅資料夾），並排除以下保留目錄：
    - reparmeater、train_log、val
    回傳欄位：
    - name：專案資料夾名稱
    - create_time：建立日期（YYYY-MM-DD）
    - count：該專案底下的子資料夾數量（同樣排除保留目錄）
    另外，會將結果以 user_name 為鍵寫入 projectlist_status_map.json，但不回傳此鍵給前端。
    """
    _validate_user_name(user_name)
    user_norm = _ensure_user_name(user_name)
    base = _runs_root(user_norm)
    if not os.path.isdir(base):
        # 即便空，也更新快取檔
        pl_map = _load_status_map(PL_STATUS_FILE)
        pl_map[user_norm] = []
        _save_status_map(PL_STATUS_FILE, pl_map)
        return {"data": []}
    exclude = {"reparmeater", "train_log", "val-test","detect","uploads","uploads_folder","detect_folder","detect_zips"}
    try:
        items: list[dict] = []
        # 先載入舊的 projectlist 狀態，以保留 updated_date
        old_map = _load_status_map(PL_STATUS_FILE).get(user_norm, [])
        updated_date_lookup = {it.get('name'): it.get('updated_date') for it in old_map if isinstance(it, dict)}
        for nm in os.listdir(base):
            if nm.lower() in exclude:
                continue
            p = os.path.join(base, nm)
            if os.path.isdir(p):
                # 目錄建立時間（日期）
                try:
                    ctime = os.path.getctime(p)
                    create_date = datetime.datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    create_date = ""
                # 子資料夾數（排除 reparmeater/train_log/val）
                cnt = 0
                try:
                    for child in os.listdir(p):
                        if child.lower() in exclude:
                            continue
                        cp = os.path.join(p, child)
                        if os.path.isdir(cp):
                            cnt += 1
                except Exception:
                    cnt = 0
                # 若已有 updated_date 則保留，否則以 create_date 初始化
                items.append({
                    "name": nm,
                    "create_time": create_date,
                    "updated_date": updated_date_lookup.get(nm, create_date),
                    "count": str(cnt)
                })
        # 名稱排序
        items.sort(key=lambda x: x.get("name", ""))
        # 更新快取檔（key 為使用者名稱，不回傳給前端）
        pl_map = _load_status_map(PL_STATUS_FILE)
        pl_map[user_norm] = items
        _save_status_map(PL_STATUS_FILE, pl_map)
        return {"data": items}
    except Exception:
        return {"data": []}

def _collect_project_versions(user_name: str, project: str) -> list[dict]:
    user_name = _ensure_user_name(user_name)
    base = os.path.join(_runs_root(user_name), project)
    results: list[dict] = []
    if not os.path.isdir(base):
        return results
    status_map = _load_status_map(STATUS_FILE)
    # project 下每個訓練資料夾
    for train_name in sorted(os.listdir(base)):
        tpath = os.path.join(base, train_name)
        if not os.path.isdir(tpath):
            continue
        # 每個 version 資料夾
        for version in sorted(os.listdir(tpath), key=lambda x: (len(x), x)):
            vpath = os.path.join(tpath, version)
            if not os.path.isdir(vpath):
                continue
            key = _status_key(user_name, project, train_name, version)
            st = status_map.get(key) or {}
            code_obj = st.get('code') or {}
            time_obj = st.get('time') or {}
            epoch_obj = st.get('epoch') or {}
            model_obj = st.get('model') or {}
            dataset_obj = st.get('dataset') or {}
            ds_name = (dataset_obj.get('name') if isinstance(dataset_obj, dict) else dataset_obj) or ''
            ds_name = str(ds_name).replace('\\','/').split('/')[-1]

            item = {
                "version":          version,
                "model":            (model_obj.get('name') if isinstance(model_obj, dict) else model_obj) or 'yolov9',
                "dataset":          ds_name,
                "epoch":            str(epoch_obj.get('total', '')),
                "training_start":   time_obj.get('start_time', 'None'),
                "training_end":     time_obj.get('end_time', 'None'),
                "status":           f"{code_obj.get('code', '')} {code_obj.get('msg', '')}".strip()
            }
            results.append(item)
    return results

def get_project_list(user_name: str, project: str):
    user_name = _ensure_user_name(user_name)
    parent_path = os.path.join(_runs_root(user_name), project)
    # 驗證路徑存在
    if not os.path.isdir(parent_path):
        data = []
    else:
        data = _collect_project_versions(user_name, project)
    # 更新/覆寫 projectlist_status_map.json
    pl_map = _load_status_map(PL_STATUS_FILE)
    pl_key = f"{user_name}|{project}"
    pl_map[pl_key] = data
    _save_status_map(PL_STATUS_FILE, pl_map)
    # 僅回傳列表，不含 key
    return {"data": data}


# 解析 opt.yaml 取得參數與資料集資訊
def _parse_opt_yaml(opt_path: str) -> dict:
    try:
        with open(opt_path, 'r', encoding='utf-8') as f:
            opt = json.load(f) if opt_path.lower().endswith('.json') else __import__('yaml').safe_load(f)
    except Exception:
        opt = {}
    data_path = (opt or {}).get('data') or ''
    dataset = ''
    dataset_version = ''
    try:
        norm = os.path.normpath(str(data_path))
        parts = norm.split(os.sep)
        # 支援 Dataset/<user>/<dataset>(/versions/vXX)?/data.yaml
        if 'Dataset' in parts:
            i = parts.index('Dataset')
            sub = parts[i+1:]
            if sub and sub[-1].lower() == 'data.yaml':
                sub = sub[:-1]
            # 檢查是否有 versions/vXX
            if len(sub) >= 3 and sub[-2].lower() == 'versions' and sub[-1].lower().startswith('v'):
                dataset_version = sub[-1][1:] if sub[-1][1:].isdigit() else sub[-1]
                dataset = os.path.join(*sub[:-2])
            else:
                dataset = os.path.join(*sub)
        elif 'dataset_home' in parts:
            i = parts.index('dataset_home')
            sub = parts[i+1:]
            if sub and sub[-1].lower() == 'data.yaml':
                sub = sub[:-1]
            # <user>/<dataset>/versions/vXX
            if len(sub) >= 3 and sub[-2].lower() == 'versions' and sub[-1].lower().startswith('v'):
                dataset_version = sub[-1][1:] if sub[-1][1:].isdigit() else sub[-1]
                dataset = os.path.join(*sub[:-2])
            else:
                dataset = os.path.join(*sub)
    except Exception:
        pass
    return {
        'dataset':          dataset.replace('\\', '/').split('/')[-1],
        'dataset_version':  str(dataset_version) if dataset_version is not None else '',
        'batch_size':       '',
        'img_size':         '',
        'close_mosaic':     '',
        'epochs':           ''
    }


def _list_train_names(user_name: str, project: str) -> list[str]:
    base = os.path.join(_runs_root(user_name), project)
    if not os.path.isdir(base):
        return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])


def _list_versions(user_name: str, project: str, train_name: str) -> list[str]:
    tpath = os.path.join(_runs_root(user_name), project, train_name)
    if not os.path.isdir(tpath):
        return []
    vers = [v for v in os.listdir(tpath) if os.path.isdir(os.path.join(tpath, v))]
    # 數字優先排序
    def _key(x: str):
        return (len(x), x)
    return sorted(vers, key=_key)


def _collect_train_summary_item(user_name: str, project: str, train_name: str) -> dict:
    user_name = _ensure_user_name(user_name)
    status_map = _load_status_map(STATUS_FILE)
    try:
        ct_map = _load_create_train_map()
        ct_key = f"{user_name}|{project}|{train_name}"
        ct_entry = ct_map.get(ct_key) if isinstance(ct_map, dict) else {}
        if not isinstance(ct_entry, dict):
            ct_entry = {}
    except Exception:
        ct_entry = {}
    device_value = str(ct_entry.get('device') or '').strip()
    versions = _list_versions(user_name, project, train_name)
    version_list = []
    latest_st = None
    latest_ver = None
    seen_versions: set[str] = set()

    for v in versions:
        key = _status_key(user_name, project, train_name, v)
        st = status_map.get(key) or {}
        t = (st.get('time') or {}).get('end_time', 'None')
        version_list.append({'version': v, 'endtime': t})
        seen_versions.add(v)
        # 以最大版本號為最新（字串數字化）
        try:
            if latest_ver is None or int(v) >= int(latest_ver):
                latest_ver = v
                latest_st = st
        except Exception:
            if latest_ver is None:
                latest_ver = v
                latest_st = st
    # 若檔案系統無版本資料夾，嘗試從狀態檔補齊（訓練剛建立但資料夾尚未產生時）
    if latest_ver is None:
        try:
            prefix = f"{user_name}|{project}|{train_name}|"
            for k, st in status_map.items():
                if not isinstance(k, str) or not k.startswith(prefix):
                    continue
                v = k.split('|')[-1]
                # 建立版本清單項目
                t = (st.get('time') or {}).get('end_time', 'None')
                if v not in seen_versions:
                    version_list.append({'version': v, 'endtime': t})
                    seen_versions.add(v)
                # 選最新版本（數字優先）
                try:
                    if latest_ver is None or int(v) >= int(latest_ver):
                        latest_ver = v
                        latest_st = st
                except Exception:
                    if latest_ver is None:
                        latest_ver = v
                        latest_st = st
        except Exception:
            pass
    # 讀取參數：使用最新版本的 opt.yaml
    params = {'dataset': '', 'dataset_version': '', 'batch_size': '', 'img_size': '', 'close_mosaic': '', 'epochs': ''}
    if latest_ver is not None:
        opt_path = os.path.join(_runs_root(user_name), project, train_name, latest_ver, 'exp', 'opt.yaml')
        if os.path.isfile(opt_path):
            params = _parse_opt_yaml(opt_path)
    hyp_path = os.path.join(_runs_root(user_name), project, train_name, 'hyp.scratch-high.yaml')
    hyp_params = _parse_hyp_yaml(hyp_path)
    for key in ('batch_size', 'img_size', 'close_mosaic', 'epochs'):
        if hyp_params.get(key):
            params[key] = hyp_params[key]
    # 若 opt.yaml 未提供 dataset 或版本，從狀態快取補上（我們在訓練時會寫入 dataset.name 與 dataset.version）
    if latest_st:
        ds_obj = latest_st.get('dataset') or {}
        if not params.get('dataset'):
            name = (ds_obj.get('name') if isinstance(ds_obj, dict) else (ds_obj if isinstance(ds_obj, str) else ''))
            params['dataset'] = str(name or '').replace('\\','/').split('/')[-1]
        if not params.get('dataset_version'):
            ver = ds_obj.get('version') if isinstance(ds_obj, dict) else ''
            v = str(ver or '')
            params['dataset_version'] = (v[1:] if v.lower().startswith('v') else v)
    # 狀態取最新版本 (加入 start_time / end_time 供摘要使用)
    code_obj = (latest_st or {}).get('code') or {}
    epoch_obj = (latest_st or {}).get('epoch') or {}
    time_obj_latest = (latest_st or {}).get('time') or {}
    status_obj = {
        'epoch': str(epoch_obj.get('current', '0')),
        'total_epoch': str(epoch_obj.get('total', '0')),
        'code': str(code_obj.get('code', '000'))
    }
    # 補充：若 dataset / dataset_version 仍為空，嘗試從 create_train_status_map.json 取得
    if (not params.get('dataset')) or (not params.get('dataset_version')):
        if not params.get('dataset'):
            params['dataset'] = str(ct_entry.get('dataset') or '')
        if not params.get('dataset_version'):
            params['dataset_version'] = str(ct_entry.get('dataset_version') or '')
    # 取得 created_date：來自 create_train_status_map.json，若無則以資料夾建立時間或空字串
    created_date = str(ct_entry.get('created_date', '') or '')
    if not created_date:
        try:
            train_dir = os.path.join(_runs_root(user_name), project, train_name)
            if os.path.isdir(train_dir):
                created_date = datetime.datetime.fromtimestamp(os.path.getctime(train_dir)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            created_date = ''
    return {
        'name':         train_name,
        'model':        'yolov9',
        'device':       device_value,
        'created_date': created_date,
        'status':       status_obj,
        'version':      version_list,
        'parameter': {
            **params,
            'latest_start_time':  time_obj_latest.get('start_time', ''),
            'latest_end_time':    time_obj_latest.get('end_time', '')
        }
    }


def _collect_train_versions_detail(user_name: str, project: str, train_name: str) -> list[dict]:
    user_name = _ensure_user_name(user_name)
    status_map = _load_status_map(STATUS_FILE)
    items: list[dict] = []
    # 合併資料夾版本與狀態檔版本，避免剛開始訓練時版本資料夾尚未建立而看不到狀態
    fs_versions = set(_list_versions(user_name, project, train_name))
    map_versions: set[str] = set()
    try:
        prefix = f"{user_name}|{project}|{train_name}|"
        for k in status_map.keys():
            if isinstance(k, str) and k.startswith(prefix):
                map_versions.add(k.split('|')[-1])
    except Exception:
        pass
    all_versions = sorted(fs_versions.union(map_versions), key=lambda x: (len(x), x))
    for v in all_versions:
        key = _status_key(user_name, project, train_name, v)
        st = status_map.get(key) or {}
        time_obj =  st.get('time') or {}
        epoch_obj = st.get('epoch') or {}
        code_obj =  st.get('code') or {}
        # 參數讀取
        opt_path = os.path.join(_runs_root(user_name), project, train_name, v, 'exp', 'opt.yaml')
        params = _parse_opt_yaml(opt_path) if os.path.isfile(opt_path) else {
            'dataset': '', 'dataset_version': '', 'batch_size': '', 'img_size': '', 'close_mosaic': '', 'epochs': ''}
        hyp_path = os.path.join(_runs_root(user_name), project, train_name, 'hyp.scratch-high.yaml')
        hyp_params = _parse_hyp_yaml(hyp_path)
        for key in ('batch_size', 'img_size', 'close_mosaic', 'epochs'):
            if hyp_params.get(key):
                params[key] = hyp_params[key]
        # 最終強制 dataset 為 basename
        if params.get('dataset'):
            params['dataset'] = str(params['dataset']).replace('\\','/').split('/')[-1]
        # 補上狀態中紀錄的 dataset 與版本（若 opt.yaml 無法提供）
        ds_obj = st.get('dataset') or {}
        if not params.get('dataset'):
            name = (ds_obj.get('name') if isinstance(ds_obj, dict) else (ds_obj if isinstance(ds_obj, str) else ''))
            params['dataset'] = str(name or '').replace('\\','/').split('/')[-1]
        if not params.get('dataset_version'):
            ver = ds_obj.get('version') if isinstance(ds_obj, dict) else ''
            v = str(ver or '')
            params['dataset_version'] = (v[1:] if v.lower().startswith('v') else v)
        # 若參數中的 dataset/dataset_version 仍為空，嘗試 create_train map
        if (not params.get('dataset')) or (not params.get('dataset_version')):
            ct_map = _load_create_train_map()
            ct_key = f"{user_name}|{project}|{train_name}"
            ct_entry = ct_map.get(ct_key) or {}
            if not params.get('dataset'):
                params['dataset'] = (ct_entry.get('dataset') or '')
            if not params.get('dataset_version'):
                params['dataset_version'] = (ct_entry.get('dataset_version') or '')
                
        items.append({
            'version': v,
            'status': {
                'code':         str(code_obj.get('code', '')),
                'epoch':        str(epoch_obj.get('current', '')),
                'start_time':   time_obj.get('start_time', ''),
                'end_time':     time_obj.get('end_time', '')
            },
            'parameter': params
        })
    return items


@router.get("/Get_projects_task")
def project_summary(
    user_name: str  = Query("", description="使用者名稱"),
    project: str    = Query("", description="專案名稱")):
    
    """取得單一專案 (project) 下的訓練清單與概要資訊。
    - 回傳各訓練的最新狀態、參數摘要與版本清單。
    - 並更新快取檔 projectlist_status_map.json（僅記錄，不回傳 key）。
    """
    _validate_user_name(user_name)
    user_name = _ensure_user_name(user_name)

    if not project.strip():
        raise HTTPException(status_code=400, detail="缺少或不合法的 project")
    
    trains = _list_train_names(user_name, project)
    data = [_collect_train_summary_item(user_name, project, t) for t in trains]
    # 更新快取檔（含 key）
    pl_map = _load_status_map(PL_STATUS_FILE)
    pl_key = f"{user_name}|{project}"
    pl_map[pl_key] = data
    _save_status_map(PL_STATUS_FILE, pl_map)
    return {"data": data}


@router.get("/Get_project_versions")
def project_versions(
    user_name: str  = Query("", description="使用者名稱"),
    project: str    = Query("", description="專案名稱"),
    task: str       = Query("", description="訓練名稱")):

    """取得單一訓練 (task) 的所有版本詳細資訊。
    - 每個版本回傳狀態、起迄時間與訓練參數（dataset、dataset_version、batch_size、img_size、close_mosaic）。
    - 並更新快取檔 projectlist_status_map.json（僅記錄，不回傳 key）。
    """
    _validate_user_name(user_name)
    user_name = _ensure_user_name(user_name)
    if not project.strip() or not task.strip():
        raise HTTPException(status_code=400, detail="缺少或不合法的 project 或 task")
    data = _collect_train_versions_detail(user_name, project, task)
    # 更新快取檔（含 key）
    pl_map = _load_status_map(PL_STATUS_FILE)
    pl_key = f"{user_name}|{project}|{task}"
    pl_map[pl_key] = data
    _save_status_map(PL_STATUS_FILE, pl_map)
    return {"data": data}


@router.get("/GET_ALL")
def get_all_projects(
    user_name: str = Query("", description="使用者名稱")):

    """回傳使用者全部專案/訓練/版本結構 (不在最外層回傳 user_name)。
    規則：忽略保留資料夾 reparmeater / train_log / val。
    版本清單為資料夾名稱的排序（數字型以長度+字典序排序）。
    若某訓練底下沒有任何版本資料夾，version 回傳 []。
    若專案底下沒有任何訓練資料夾則省略或回傳空 task 陣列。
    """
    _validate_user_name(user_name)
    user_norm = _ensure_user_name(user_name)
    base = _runs_root(user_norm)
    exclude = {"reparmeater", "train_log", "val-test","detect","uploads","uploads_folder","detect_folder","detect_zips"}
    if not os.path.isdir(base):
        return {"data": []}
    projects: list[dict] = []
    try:
        for proj in sorted(os.listdir(base)):
            if proj.lower() in exclude:
                continue
            proj_path = os.path.join(base, proj)
            if not os.path.isdir(proj_path):
                continue
            tasks: list[dict] = []
            for train_name in sorted(os.listdir(proj_path)):
                train_path = os.path.join(proj_path, train_name)
                if not os.path.isdir(train_path):
                    continue
                # 收集版本資料夾
                versions = [v for v in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, v))]
                def _vkey(x: str):
                    return (len(x), x)
                versions = sorted(versions, key=_vkey)
                tasks.append({"name": train_name, "version": versions})
            projects.append({"project": proj, "task": tasks})
    except Exception:
        pass
    return {"data": projects}