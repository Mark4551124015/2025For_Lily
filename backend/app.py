from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import shutil
import cv2

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(BASE_DIR, 'videos')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STATUS_FILE = '/tmp/slowmovie_status.json'
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
CMD_FILE = '/tmp/slowmovie_cmd'
SEEK_FILE = '/tmp/slowmovie_seek'

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== 全局变量（加载配置后更新） =====
delay = 100
increment = 1
loop_mode = False
random_mode = False

current_video = ""

def load_config():
    global delay, increment, loop_mode, random_mode, current_video
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
                print("Loaded config:", cfg)
            delay = int(cfg.get("delay", delay))
            increment = int(cfg.get("increment", increment))
            loop_mode = cfg.get("loop", loop_mode)
            random_mode = cfg.get("random", random_mode)
            file_name = cfg.get("file", "")
            if file_name:
                current_video = os.path.join(VIDEO_DIR, file_name)
        except Exception as e:
            print("Config load error:", e)

# 启动时提前加载配置
load_config()

# ===== 数据模型 =====
class ConfigData(BaseModel):
    delay: int
    increment: int
    loop: bool
    random: bool
    file: str = ""  # 文件名，默认为空字符串表示未选择文件

class FileSelectRequest(BaseModel):
    name: str  # 文件名

class SeekRequest(BaseModel):
    frame: int


# ===== 工具函数 =====
def safe_filename(filename: str) -> str:
    """防止路径遍历"""
    return os.path.basename(filename)




def save_config(data: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f)


# ===== API 路由 =====

@app.get("/api/status")
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {
        "file": "",
        "percent": 0,
        "temperature": 0,
        "frame": 0,
        "total_frames": 0
    }

@app.get("/api/files")
def list_files():
    files = [
        {"name": f}
        for f in os.listdir(VIDEO_DIR)
        if os.path.isfile(os.path.join(VIDEO_DIR, f))
    ]
    return {"files": files}

@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    """直接上传完整文件（浏览器自动分片传输）"""
    filename = safe_filename(video.filename)
    dest_path = os.path.join(VIDEO_DIR, filename)
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = await video.read(1024 * 1024)  # 每次读1MB
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    return {"status": "ok", "filename": filename}


@app.delete("/api/files/{name}")
def delete_file(name: str):
    current_name = os.path.basename(current_video) if current_video else ""
    if name == current_name:
        raise HTTPException(status_code=400, detail="正在播放的文件不能删除")
    file_path = os.path.join(VIDEO_DIR, safe_filename(name))
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="文件不存在")

@app.get("/api/config")
def get_config():
    file_name = os.path.basename(current_video) if current_video else ""
    return {
        "delay": delay,
        "increment": increment,
        "loop": loop_mode,
        "random": random_mode,
        "file": file_name
    }

@app.post("/api/config")
def update_config(cfg: ConfigData):
    global delay, increment, loop_mode, random_mode, current_video
    delay = cfg.delay
    increment = cfg.increment
    loop_mode = cfg.loop
    random_mode = cfg.random
    # current_video = os.path.join(VIDEO_DIR, cfg.file) if cfg.file else ""
    cfg = {
        "delay": delay,
        "increment": increment,
        "loop": loop_mode,
        "random": random_mode
    }
    cfg["file"] = os.path.basename(current_video) if current_video else ""
    print("Updating config:", cfg)
    save_config(cfg)
    with open(CMD_FILE, 'w') as f:
        f.write('config')
    return {"status": "ok"}

@app.post("/api/control/select")
def select_file(file: FileSelectRequest):
    """选择要播放的视频"""
    global current_video
    file_name = safe_filename(file.name)
    current_video = os.path.join(VIDEO_DIR, file_name)
    save_config({
        "delay": delay,
        "increment": increment,
        "loop": loop_mode,
        "random": random_mode,
        "file": file_name
    })
    with open(CMD_FILE, 'w') as f:
        f.write('select')
    return {"status": "ok"}

@app.get("/api/video_info/{filename}")
def video_info(filename: str):
    filename = safe_filename(filename)
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video file")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"total_frames": total_frames}

@app.post("/api/control/seek")
def seek_frame(data: SeekRequest):
    try:
        with open(SEEK_FILE, 'w') as f:
            f.write(str(data.frame))
        with open(CMD_FILE, 'w') as f:
            f.write('seek')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seek write failed: {e}")
    return {"status": "ok"}

@app.post("/api/control/{cmd}")
def control(cmd: str):
    try:
        with open(CMD_FILE, 'w') as f:
            f.write(cmd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command write failed: {e}")
    return {"status": "ok"}
