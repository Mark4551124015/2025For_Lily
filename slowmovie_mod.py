#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys, time, json, random
from PIL import Image
import ffmpeg
import signal

# ==== 路径配置 ====
# VIDEO_DIR     = "/home/pi/framebox/backend/videos"
# CONFIG_FILE   = "/home/pi/framebox/backend/config.json"
# PROGRESS_FILE = "/home/pi/framebox/backend/progress.json"
STATUS_FILE   = "/tmp/slowmovie_status.json"
CMD_FILE      = "/tmp/slowmovie_cmd"
SEEK_FILE     = "/tmp/slowmovie_seek"

VIDEO_DIR     = "./backend/videos"
CONFIG_FILE   = "./backend/config.json"
PROGRESS_FILE = "./backend/progress.json"

DRIVER = "waveshare_epd.epd7in5_V2"

# ==== 默认播放参数 ====
delay        = 60
increment    = 4
loop_mode    = True
random_mode  = False
paused       = False
current_video = ""
current_frame = 0
total_frames = 0
fps = 24.0


use_epd = True
try:
    from omni_epd import displayfactory, EPDNotFoundError
    epd = displayfactory.load_display_driver(DRIVER)
    width, height = epd.width, epd.height

except Exception:
    print("No EPD found, using direct image display mode")
    use_epd = False
    epd = None
    width, height = 800, 480  # 调试分辨率


print(f"Using display: {DRIVER}, resolution: {width}x{height}")

# ==== 工具函数 ====
def get_cpu_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except:
        return 0.0

def list_videos():
    return [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".m4v"))]

def load_config():
    global delay, increment, loop_mode, random_mode, current_video
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            delay        = int(cfg.get("delay", delay))
            increment    = int(cfg.get("increment", increment))
            loop_mode    = cfg.get("loop", loop_mode)
            random_mode  = cfg.get("random", random_mode)
            file_name    = cfg.get("file", "")
            if file_name:
                current_video = os.path.join(VIDEO_DIR, file_name)
        except Exception as e:
            print("Config load error:", e)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(video, frame):
    progress = load_progress()
    progress[os.path.basename(video)] = frame
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def save_status():
    percent = (current_frame / total_frames) * 100 if total_frames else 0
    elapsed_sec = current_frame / fps if fps else 0
    total_sec = total_frames / fps if fps else 0
    remaining_sec = (total_frames - current_frame) / fps if fps else 0
    status = {
        "file": os.path.basename(current_video),
        "frame": current_frame,
        "total_frames": total_frames,
        "percent": percent,
        "elapsed": time.strftime('%H:%M:%S', time.gmtime(elapsed_sec)),
        "remaining": time.strftime('%H:%M:%S', time.gmtime(remaining_sec)),
        "total": time.strftime('%H:%M:%S', time.gmtime(total_sec)),
        "temperature": get_cpu_temp(),
        "mode": "loop" if loop_mode else ("random" if random_mode else "order"),
        "paused": paused,
        "fps": fps,
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)
    save_progress(current_video, current_frame)


def handle_seek():
    """处理前端发送的跳转帧请求"""
    global current_frame
    if not os.path.exists(SEEK_FILE):
        print("[SEEK] No seek command found")
        return False
    try:
        with open(SEEK_FILE) as f:
            frame_str = f.read().strip()
        if frame_str.isdigit():
            frame_val = int(frame_str)
            # 限制范围
            current_frame = max(0, min(frame_val, total_frames - 1))
            print(f"[SEEK] 跳转到帧 {current_frame}/{total_frames}")
            save_progress(current_video, current_frame)
        else:
            print(f"[SEEK] 无效的帧号: {frame_str}")
    except Exception as e:
        print(f"[SEEK] 处理失败: {e}")
    finally:
        try:
            # os.remove(SEEK_FILE)
            pass
        except:
            pass
    return True

def handle_command():
    """处理控制命令"""
    global paused, current_frame
    if not os.path.exists(CMD_FILE):
        return False
    with open(CMD_FILE) as f:
        cmd = f.read().strip()
    if not cmd:
        return False
    print("Received command:", cmd)
    executed = True
    if cmd == "pause":
        paused = True
        executed = False
    elif cmd == "resume":
        paused = False
        executed = False
    elif cmd == "next":
        current_frame += increment
    elif cmd == "prev":
        current_frame = max(0, current_frame - increment)

    elif cmd == "config":
        load_config()
        print("Loaded config:", {
            "delay": delay,
            "increment": increment,
            "loop": loop_mode,
            "random": random_mode,
            "file": os.path.basename(current_video) if current_video else ""
        })
        save_status()
    elif cmd == "select":
        load_config()
        print("Selected video:", current_video)
        progress_map = load_progress()
        current_frame = progress_map.get(os.path.basename(current_video), 0)
        probe_video(current_video)
    elif cmd == "seek":
        if not handle_seek():
            print("No valid seek command found")
        else:
            print("Seeked")
    else:
        print(f"Unknown command: {cmd}")
        executed = False
    os.remove(CMD_FILE)
    return executed

def probe_video(video_path):
    global total_frames, fps
    try:
        probe = ffmpeg.probe(video_path, select_streams="v:0")
        stream = probe['streams'][0]
        fps_str = stream['avg_frame_rate']
        fps = eval(fps_str) if fps_str != "0/0" else 24.0
        total_frames = int(stream.get('nb_frames') or int(float(probe['format']['duration']) * fps))
    except Exception as e:
        print("Probe error:", e)
        total_frames, fps = 1000, 24.0

def extract_frame(video_path, frame_time, out_path):
    try:
        (
            ffmpeg
            # .input(video_path, ss=frame_time, hwaccel="v4l2m2m")  # 硬件解码（Pi 上可用）
            .input(video_path, ss=frame_time)  # 硬件解码（Pi 上可用）
            .filter("scale", width, height, force_original_aspect_ratio=1)
            .filter("pad", width, height, -1, -1)
            # .filter("format", "gray")  # 如需灰度可取消注释
            .output(out_path, vframes=1, copyts=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("FFmpeg stdout:\n", e.stdout.decode('utf8', errors='ignore'))
        print("FFmpeg stderr:\n", e.stderr.decode('utf8', errors='ignore'))
        raise



# def fullscreen_filter(self):

#     if args.fullscreen:
#         if videoInfo["aspect_ratio"] > width / height:
#             return self.filter("crop", f"ih*{width / height}", "ih")
#         elif videoInfo["aspect_ratio"] < width / height:
#             return self.filter("crop", "iw", f"iw*{height / width}")
#     return self


# ffmpeg.Stream.overlay_filter = overlay_filter
# ffmpeg.Stream.fullscreen_filter = fullscreen_filter


# ==== 初始化 ====
videos = list_videos()
if not videos:
    print("No videos found in", VIDEO_DIR)
    sys.exit(1)

current_video = os.path.join(VIDEO_DIR, videos[0])
load_config()
probe_video(current_video)

progress_map = load_progress()
current_frame = progress_map.get(os.path.basename(current_video), 0)


def exithandler(signum, frame):
    epd.prepare()
    epd.clear()
    try:
        epd.close()
    finally:
        sys.exit()

if None:
    # Add hooks for interrupt signal
    signal.signal(signal.SIGTERM, exithandler)
    signal.signal(signal.SIGINT, exithandler)




# ==== 主循环 ====
while True:
    frame_time_sec = current_frame / fps if fps else 0
    tmp_frame = "/dev/shm/frame.bmp"
    if use_epd:
        extract_frame(current_video, frame_time_sec, tmp_frame)
        # extract_frame_picture(current_video, frame_time_sec, tmp_frame)
    else:
        extract_frame(current_video, frame_time_sec, tmp_frame)

    img = Image.open(tmp_frame)
    if use_epd:
        epd.prepare()
        epd.display(img)
        epd.sleep()
    else:
        img.show()  # 调试模式直接弹出图片

    current_frame += increment
    if current_frame >= total_frames:
        if loop_mode:
            current_frame = 0
        elif random_mode:
            current_video = os.path.join(VIDEO_DIR, random.choice(videos))
            probe_video(current_video)
            current_frame = load_progress().get(os.path.basename(current_video), 0)
        else:
            idx = videos.index(os.path.basename(current_video))
            idx = (idx + 1) % len(videos)
            current_video = os.path.join(VIDEO_DIR, videos[idx])
            probe_video(current_video)
            current_frame = load_progress().get(os.path.basename(current_video), 0)

    # time.sleep(delay)
    sleep_step = 0.5  # 每 0.5 秒检查一次
    slept = 0
    print("Sleep for", delay, "seconds")
    while slept < delay:
        time.sleep(sleep_step)
        if not paused:
            slept += sleep_step

        executed = handle_command()
        
        save_status()
        if executed:
            break