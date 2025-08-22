import epd7in5_V2
import time
from PIL import Image,ImageDraw,ImageFont
import traceback

from PIL import Image
import numpy as np
from numba import njit
import cv2

# 电子纸 4 阶灰度硬件值
# GRAY_LEVELS = np.array([0x00, 0x40, 0xC0, 0xFF], dtype=np.float32)
GRAY_LEVELS = np.array([0xFF, 0x80, 0x40, 0x00], dtype=np.float32)

@njit
def floyd_steinberg_4gray_hw(arr, palette):
    """Floyd–Steinberg 抖动"""
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            new_val = palette[np.argmin(np.abs(palette - old))]
            arr[y, x] = new_val
            err = old - new_val
            if x + 1 < w:
                arr[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    arr[y + 1, x - 1] += err * 3 / 16
                arr[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    arr[y + 1, x + 1] += err * 1 / 16
    arr = np.where(arr == 0x80, 0xC0, arr)
    arr = np.where(arr == 0x40, 0x80, arr)
    return arr

def dither_fs_4gray_hw(img):
    """大面积暗部提亮 + FS 抖动"""
    arr = np.array(img.convert("L"), dtype=np.float32)

    # Step 1: 对比度拉伸
    min_v, max_v = np.percentile(arr, (1, 99))
    arr = (arr - min_v) * (255.0 / (max_v - min_v))
    arr = np.clip(arr, 0, 255)

    arr = floyd_steinberg_4gray_hw(arr, GRAY_LEVELS)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# 生成四个等高横条：白-浅灰-深灰-黑（硬件4灰）
from PIL import Image, ImageDraw

def make_4gray_bars(w=800, h=480):
    bars = [0x00, 0x40, 0xC0, 0xFF]  # 硬件预期的四档：白、浅灰、深灰、黑
    # bars = [0xFF, 0xC0, 0x80, 0x00]  # 硬件预期的四档：白、浅灰、深灰、黑
    img = Image.new("L", (w, h), 0xFF)
    draw = ImageDraw.Draw(img)
    bh = h // 4
    for i, v in enumerate(bars):
        draw.rectangle([0, i*bh, w, (i+1)*bh-1], fill=int(v))
    return img




current_frame = 73266
fps = 24
epd = epd7in5_V2.EPD()
# epd.init()

current_video = "backend/videos/Hawl.mp4"

frame_time_sec = current_frame / fps if fps else 0
tmp_frame = "/dev/shm/frame.bmp"
tmp_frame_e = "/dev/shm/frame_edited.bmp"
start = time.time()
# extract_frame(current_video, frame_time_sec, tmp_frame)
extracted_time = time.time()

img = Image.open(tmp_frame).convert('L')
img_dithered = dither_fs_4gray_hw(img)
img_dithered.save(tmp_frame_e)
# img_dithered = make_4gray_bars()
finish_time = time.time()



epd.init_4Gray()
# epd.Clear()
epd.display_4Gray(epd.getbuffer_4Gray(img_dithered))
epd.sleep()
flashed_time = time.time()

print(
    f"FFmpeg: {(extracted_time - start)*1000:.1f} ms | "
    f"Dither: {(finish_time - extracted_time)*1000:.1f} ms | "
    f"E-Paper: {(flashed_time - finish_time):.2f} s | "
    f"Total: {(flashed_time - start):.2f} s"
)
