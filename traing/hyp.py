import os, yaml
from fastapi import APIRouter, HTTPException, Form
from user_utils import normalize_user_name,runs_root

router = APIRouter(
    prefix="/yolov9",
    tags=["HYP"],
)      

@router.post("/hyp")
async def hyp(user_name: str        = Form("", description="使用者名稱"),
              project:str           = Form("", description="專案名稱"),
              task:str              = Form("", description="任務名稱"),
              lr0: float            = Form(0.01, description="初始學習率"),
              weight_decay: float   = Form(0.0005, description="權重衰減"),
              warmup_epochs: float  = Form(3.0, description="預熱週期"),
              dfl: float            = Form(0.0, description="DFL"),
              hsv_h: float          = Form(0.015, description="HSV H"),
              hsv_s: float          = Form(0.7, description="HSV S"),
              hsv_v: float      = Form(0.4, description="HSV V"),
              degrees: float    = Form(0.0, description="旋轉角度"),
              translate: float  = Form(0.1, description="平移比例"),
              epochs: int       = Form(10, description="訓練週期"),
              batch_size: int    = Form(16, description="批次大小"),
              img_size: int      = Form(640, description="影像大小"),
              close_mosaic: int  = Form(15, description="關閉馬賽克的週期")):

    user_scope = normalize_user_name(user_name)
    if not user_scope:
        raise HTTPException(status_code=400, detail="缺少或不合法的 user_name")
    
    hyp_folder = os.path.join(runs_root(user_scope),project,task)
    if not os.path.exists(hyp_folder):
        raise HTTPException(status_code=404, detail="指定的使用者專案任務目錄不存在")
    
    hyp = os.path.join(hyp_folder, "hyp.scratch-high.yaml")

    hyp_data = {
        'lr0': lr0,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 0.7,
        'obj_pw': 1.0,
        'dfl' : dfl,
        'iou_t': 0.20,
        'anchor_t': 5.0,
        'fl_gamma': 0.0,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'degrees': degrees,
        'translate': translate,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,
        'copy_paste': 0.3,
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'close_mosaic': close_mosaic,
    }
    try:
        with open(hyp, "w", encoding="utf-8") as f:
            yaml.safe_dump(hyp_data, f, sort_keys=False, allow_unicode=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"寫入超參數檔案失敗：{e}")

    return {"message": "超參數已更新", "path": hyp, "hyp": hyp}
