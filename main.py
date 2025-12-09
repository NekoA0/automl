from fastapi import FastAPI
import train , val , export, detect, detect_folder, project_list, dataset_list
import resource_loading, upload_data, release_note, thumbs, login, hyp, hyp_find, val_find, onnx_to_tflite
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

origins = ["http://10.1.13.129:5173", "http://10.1.13.130:5173","http://10.1.13.230:5173"]
#origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 或 ["*"]（不推薦正式環境）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(train.router)
app.include_router(val.router)
app.include_router(val_find.router)
app.include_router(export.router)
app.include_router(detect.router)
app.include_router(detect_folder.router)
app.include_router(project_list.router)
app.include_router(dataset_list.router)
app.include_router(resource_loading.router)
app.include_router(upload_data.router)
app.include_router(release_note.router)
app.include_router(thumbs.router)
app.include_router(login.router)
app.include_router(hyp.router)
app.include_router(hyp_find.router)
app.include_router(onnx_to_tflite.router)

thumbs_path = r"/shared/Thumbs"

if not os.path.exists(thumbs_path):
    raise Exception(f"❌ 資料夾不存在: {thumbs_path}")

app.mount("/thumbs", StaticFiles(directory=thumbs_path), name="thumbs")

download_dir = r"/shared/download"
if os.path.isdir(download_dir):
    app.mount("/download", StaticFiles(directory=download_dir), name="download")


@app.get("/")
def home():
    return {"msg": "API is up"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)
