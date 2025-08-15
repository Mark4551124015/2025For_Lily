from PIL import Image
import numpy as np
from numba import njit
import cv2  # CLAHE 需要 OpenCV

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
    """CLAHE + 自适应 Gamma + Floyd–Steinberg 抖动"""
    # 转灰度
    arr = np.array(img.convert("L"), dtype=np.uint8)

    # Step 1: CLAHE 提升局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    # Step 2: 自适应 Gamma 提亮暗部
    arr_f = arr.astype(np.float32)
    mean_val = arr_f.mean()
    if mean_val < 150:  # 画面偏暗时才提亮
        gamma = 0.35 + 0.65 * (mean_val / 150)
        arr_f = np.power(arr_f / 255.0, gamma) * 255.0

    # Step 3: Floyd–Steinberg 抖动到电子纸灰阶
    arr_f = floyd_steinberg_4gray_hw(arr_f, GRAY_LEVELS)

    return Image.fromarray(np.clip(arr_f, 0, 255).astype(np.uint8))
