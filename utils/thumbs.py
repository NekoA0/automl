from fastapi import APIRouter, Form, HTTPException
from user_utils import ensure_user_name
from typing import Optional
import os, json

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
THUMBS_BASE_DIR = os.path.abspath(os.path.join(os.path.sep, "shared", "Thumbs"))
DATASET_HOME_DIR = os.path.join(BASE_DIR, "dataset_home")

_ensure_user_name = ensure_user_name

router = APIRouter(
    prefix="/yolov9",
    tags=["THUMBS"],
)

@router.post("/dataset/thumbs")
async def list_thumbs(
    user_name: str         = Form("", description="使用者名稱"),
    data_name: str         = Form("", description="資料名稱"),
    version: Optional[str] = Form("", description="版本（可傳 1 或 v01；省略則取最新）")):
    
    """讀取 thumbs.json（支援 /shared/Thumbs 與 dataset_home/versions）。"""
    print( user_name, data_name, version)
    try:
        base_path = THUMBS_BASE_DIR
        base_home = DATASET_HOME_DIR
        print("1##############")
        # 嚴格要求同時提供 user_name 與 dataset_name
        if not user_name or not data_name:
            raise HTTPException(status_code=400, detail="缺少必要參數：必須同時提供 user_name 與 folderPath")

        # 驗證並正規化 user_name（小寫）
        user_name = _ensure_user_name(user_name)

        print("2##############")
        thumbs_json = None
        # 收集所有版本 (versions 資料夾下 vXX 轉成數字字串) 供回傳
        all_versions: list[str] = []
        versions_root_global = os.path.join(base_home, user_name, data_name, "versions")
        if os.path.isdir(versions_root_global):
            try:
                for nm in os.listdir(versions_root_global):
                    p = os.path.join(versions_root_global, nm)
                    if not os.path.isdir(p):
                        continue
                    if nm.lower().startswith('v') and nm[1:].isdigit():
                        all_versions.append(str(int(nm[1:])))  # 去掉前導 0
                all_versions.sort(key=lambda x: int(x))
            except Exception:
                all_versions = []
        print("3##############")
        # 候選 1：/shared/Thumbs/<user>/<folderPath>/thumbs.json
        target1 = os.path.join(base_path, user_name, data_name)
        folder1 = os.path.normpath(target1)
        base1 = os.path.normcase(os.path.abspath(base_path))
        targ1 = os.path.normcase(os.path.abspath(folder1))
        print(target1)
        print(targ1)
        print(base1)
        print(folder1)
        print(os.path.isdir(folder1))
        #if targ1 == base1 or targ1.startswith(base1 + os.sep):
        #if os.path.isdir(folder1):
        cand = os.path.join(folder1, "thumbs.json")
        print(cand)
        if os.path.exists(cand):
            thumbs_json = cand

        print(folder1)
        print(thumbs_json)

        

        # 候選 2：dataset_home/<user>/<data_name>/versions/<version>/thumbs.json（version 可省略為最新）
        if thumbs_json is None:
            versions_root = os.path.join(base_home, user_name, data_name, "versions")
            base2 = os.path.normcase(os.path.abspath(base_home))
            vr_abs = os.path.normcase(os.path.abspath(versions_root))
            if vr_abs == base2 or vr_abs.startswith(base2 + os.sep):
                if os.path.isdir(versions_root):
                    # 正規化版本
                    vname = None
                    if isinstance(version, str) and version.strip():
                        vstr = version.strip().lower()
                        vname = f"v{int(vstr):02d}" if vstr.isdigit() else (vstr if vstr.startswith('v') else ('v' + vstr))
                    else:
                        # 掃最新 vXX
                        mx = -1
                        best = None
                        for nm in os.listdir(versions_root):
                            p = os.path.join(versions_root, nm)
                            if not os.path.isdir(p):
                                continue
                            if nm.lower().startswith('v') and nm[1:].isdigit():
                                n = int(nm[1:])
                                if n > mx:
                                    mx = n
                                    best = nm
                        vname = best
                    if vname:
                        folder2 = os.path.join(versions_root, vname)
                        cand2 = os.path.join(folder2, "thumbs.json")
                        if os.path.exists(cand2):
                            thumbs_json = cand2

        if thumbs_json is None:
            raise HTTPException(status_code=404, detail="thumbs.json 不存在於 /shared/Thumbs 或 dataset_home 版本資料夾")

        with open(thumbs_json, "r", encoding="utf-8") as f:
            thumbs = json.load(f)
        return {"thumbs": thumbs,"version": all_versions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read thumbs.json: {e}")
