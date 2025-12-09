import os, yaml
from fastapi import APIRouter, HTTPException, Form
from user_utils import normalize_user_name,runs_root

router = APIRouter(
    prefix="/yolov9",
    tags=["HYP"],
)      

@router.post("/get_hyp")
async def get_hyp(  user_name: str  =Form("", description="使用者名稱"),
                    project:str     =Form("", description="專案名稱"), 
                    task:str        =Form("", description="訓練名稱"),
                    version:str     =Form("", description="版本名稱")):
     
    user_scope = normalize_user_name(user_name)
    if not user_scope:
        raise HTTPException(status_code=400, detail="缺少或不合法的 user_name")
    
    hyp_folder = os.path.join(runs_root(user_scope),project,task)
    hyp_version = os.path.join(runs_root(user_scope),project,task,version)

    if not os.path.exists(hyp_folder):
        raise HTTPException(status_code=404, detail="指定的使用者專案任務目錄不存在")
    
    if not os.path.exists(hyp_version):
        raise HTTPException(status_code=404, detail="指定的使用者專案任務版本不存在")
    
    hyp = os.path.join(hyp_folder, "hyp.scratch-high.yaml")
    hyp_v = os.path.join(hyp_version, "hyp.scratch-high.yaml")

    if not os.path.isfile(hyp) or not os.path.isfile(hyp_v):
        raise HTTPException(status_code=404, detail="超參數檔案不存在")

    try:
        if version=="":
            with open(hyp ,"r", encoding="utf-8") as f:
                hyp_data = yaml.safe_load(f)
        
        else:
            with open(hyp_v ,"r", encoding="utf-8") as f:
                hyp_data = yaml.safe_load(f)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"讀取超參數檔案失敗：{e}")

    return hyp_data