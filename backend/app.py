from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os, json, shutil

VIDEO_DIR = os.path.join(os.path.dirname(__file__), 'videos')
STATUS_FILE = '/tmp/slowmovie_status.json'
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

os.makedirs(VIDEO_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/status")
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"file": "", "percent": 0, "temperature": 0}

@app.get("/api/files")
def list_files():
    return {"files": [{"name": f} for f in os.listdir(VIDEO_DIR) if os.path.isfile(os.path.join(VIDEO_DIR, f))]}

@app.post("/api/upload")
def upload_video(video: UploadFile = File(...)):
    dest = os.path.join(VIDEO_DIR, video.filename)
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    return {"status": "ok"}

@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    path = os.path.join(VIDEO_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return {"status": "deleted"}

@app.post("/api/control/{cmd}")
def control(cmd: str):
    # 这里可以写入一个控制文件 slowmovie_mod.py 读取执行
    with open('/tmp/slowmovie_cmd', 'w') as f:
        f.write(cmd)
    return {"status": "ok"}

@app.post("/api/control/select")
def select_file(file: dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(file, f)
    return {"status": "ok"}
