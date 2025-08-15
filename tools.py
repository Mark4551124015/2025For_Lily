from PIL import Image
import numpy as np
from numba import njit

# 电子纸 4 阶灰度硬件值
GRAY_LEVELS = np.array([0xFF, 0xC0, 0x80, 0x00], dtype=np.float32)

@njit
def floyd_steinberg_4gray_hw(arr, palette):
    """Floyd–Steinberg 抖动，直接映射到电子纸硬件灰阶值"""
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
    return arr

def dither_fs_4gray_hw(img):
    """自适应亮度 + Gamma 校正 + FS 4 阶灰度抖动"""
    arr = np.array(img.convert("L"), dtype=np.float32)

    # === Step 1: 归一化到 0-255 ===
    min_v, max_v = arr.min(), arr.max()
    if max_v > min_v:
        arr = (arr - min_v) * (255.0 / (max_v - min_v))

    # === Step 2: 判断是否偏暗，自动抬亮 ===
    mean_val = arr.mean()
    if mean_val < 128:  # 偏暗图
        gamma = max(0.35, mean_val / 128)  # 平均值越低，gamma 越小，越亮
        arr = np.power(arr / 255.0, gamma) * 255.0

    # === Step 3: FS 抖动 ===
    arr = floyd_steinberg_4gray_hw(arr, GRAY_LEVELS)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
