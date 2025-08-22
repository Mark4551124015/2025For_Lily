import time
from PIL import Image, ImageDraw
import numpy as np
from numba import njit
import cv2

# =========================
# 灰度常量
# =========================
BLACK = 0x00
DEEP  = 0x40  # 问题灰阶（与黑高频切换）
LIGHT = 0x80
WHITE = 0xFF

PALETTE4 = np.array([WHITE, LIGHT, DEEP, BLACK], dtype=np.float32)  # 任意顺序都行
PALETTE3 = np.array([WHITE, LIGHT, BLACK], dtype=np.float32)        # 三阶：0xFF,0xC0,0x00

# =========================
# 基础：分位数线性拉伸（增强暗部）
# =========================
def contrast_stretch_percentile(arr_u8, lo=1, hi=99):
    arr = arr_u8.astype(np.float32)
    vmin, vmax = np.percentile(arr, (lo, hi))
    if vmax <= vmin:
        return arr_u8
    out = (arr - vmin) * (255.0 / (vmax - vmin))
    return np.clip(out, 0, 255).astype(np.uint8)

# =========================
# 标准 FS 抖动到 4 阶
# =========================
@njit
def fs_dither_full(arr_u8, palette):
    h, w = arr_u8.shape
    work = arr_u8.astype(np.float32)
    for y in range(h):
        for x in range(w):
            old = work[y, x]
            # 选最近灰度
            idx = np.argmin(np.abs(palette - old))
            new_val = palette[idx]
            work[y, x] = new_val
            err = old - new_val
            if x + 1 < w:
                work[y, x + 1] += err * 7.0 / 16.0
            if y + 1 < h:
                if x > 0:
                    work[y + 1, x - 1] += err * 3.0 / 16.0
                work[y + 1, x] += err * 5.0 / 16.0
                if x + 1 < w:
                    work[y + 1, x + 1] += err * 1.0 / 16.0
    out = np.clip(work, 0, 255).astype(np.uint8)
    

    return out

# =========================
# 局部（mask 内）FS 三阶重抖动
# 误差只在 mask 内扩散，mask 外不影响
# =========================
@njit
def fs_dither_masked_3gray(src_u8, mask_bool, palette3):
    h, w = src_u8.shape
    work = src_u8.astype(np.float32)
    out  = src_u8.copy()
    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x]:
                continue
            old = work[y, x]
            idx = np.argmin(np.abs(palette3 - old))
            new_val = palette3[idx]
            out[y, x] = int(new_val)
            err = old - new_val
            # 误差仅传播到 mask 内的邻居
            if x + 1 < w and mask_bool[y, x + 1]:
                work[y, x + 1] += err * 7.0 / 16.0
            if y + 1 < h:
                if x > 0 and mask_bool[y + 1, x - 1]:
                    work[y + 1, x - 1] += err * 3.0 / 16.0
                if mask_bool[y + 1, x]:
                    work[y + 1, x] += err * 5.0 / 16.0
                if x + 1 < w and mask_bool[y + 1, x + 1]:
                    work[y + 1, x + 1] += err * 1.0 / 16.0
    return out


def enforce_black_clusters_in_mask(out, mask, ksize=2, it_open=1, it_close=1):
    """
    在 mask 内让黑色(0x00)更成团：
      - 先开运算：去掉孤立1px黑点(降成背景保持原值)
      - 再闭运算：把相邻黑点连成2x2小团，提升“下墨”成功率
    只改黑色，其他灰阶不动。
    """
    kernel = np.ones((ksize, ksize), np.uint8)

    black = ((out == 0x00) & mask).astype(np.uint8)
    if it_open > 0:
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel, iterations=it_open)
    if it_close > 0:
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel, iterations=it_close)

    # 在 mask 内：闭后为1的地方强制设黑；其余保持原值（可能是0x40/0x80/0xC0/0xFF）
    out[mask] = np.where(black[mask] == 1, 0x00, out[mask])
    return out



# =========================
# 检测 0x00 ↔ 0x80 高频切换区域
# 方法：3x3 邻域“对立类”计数 >= 阈值
# =========================
def detect_toggle_mask_00_80(arr_u8, neighbor_thresh=3, window_size=3,
                             restrict_upper_half=True, dilate=1):
    h, w = arr_u8.shape
    is_black = (arr_u8 == BLACK).astype(np.uint8)
    is_deep  = (arr_u8 == DEEP ).astype(np.uint8)

    k = np.ones((window_size, window_size), dtype=np.uint8)
    k[window_size // 2, window_size // 2] = 0

    nb_black = cv2.filter2D(is_black, -1, k, borderType=cv2.BORDER_REPLICATE)
    nb_deep  = cv2.filter2D(is_deep , -1, k, borderType=cv2.BORDER_REPLICATE)

    opposite = np.zeros_like(arr_u8, dtype=np.uint8)
    black_mask = is_black.astype(bool)
    deep_mask  = is_deep.astype(bool)
    opposite[black_mask] = nb_deep[black_mask]
    opposite[deep_mask]  = nb_black[deep_mask]

    toggle = ((is_black | is_deep).astype(bool)) & (opposite >= neighbor_thresh)

    if restrict_upper_half:
        m = np.zeros_like(toggle, dtype=bool)
        m[:h // 2, :] = toggle[:h // 2, :]
        toggle = m

    if dilate > 0:
        kernel = np.ones((3, 3), np.uint8)
        toggle = cv2.dilate(toggle.astype(np.uint8), kernel, iterations=dilate).astype(bool)

    return toggle

# =========================
# 主处理：先 4 阶抖动，再在“0x00↔0x80 高频区”用 3 阶重抖动
# =========================
def process_image(img_pil,
                  edge_lo=1, edge_hi=99,
                  neighbor_thresh=3,
                  restrict_upper_half=True,
                  dilate_iter=1):
    # 原灰度与增强
    orig_u8 = np.array(img_pil.convert("L"), dtype=np.uint8)
    stretch = contrast_stretch_percentile(orig_u8, lo=edge_lo, hi=edge_hi)

    # 全屏 4 阶 FS 抖动
    d4 = fs_dither_full(stretch, PALETTE4)

    toggle_mask = detect_toggle_mask_00_80(
        d4,
        neighbor_thresh=neighbor_thresh,
        window_size=3,
        restrict_upper_half=restrict_upper_half,
        dilate=dilate_iter
    )

    # ↓↓↓ 仅此处改为使用原图灰度（orig_u8）作为第二次抖动源 ↓↓↓
    d3_local = fs_dither_masked_3gray(stretch, toggle_mask, PALETTE3)

    # 将 mask 内的像素替换回去，非 mask 区保持 4 阶抖动结果
    out = d4.copy()
    out[toggle_mask] = d3_local[toggle_mask]
    out[out == 0x80]  = 0xC0
    out[out == 0x40] = 0x80
    out = enforce_black_clusters_in_mask(out, toggle_mask, ksize=2, it_open=1, it_close=1)
    return Image.fromarray(out, mode="L"), toggle_mask


def make_img(img):
    out, _ = process_image(
        img,
        edge_lo=1, edge_hi=99,
        neighbor_thresh=3,
        restrict_upper_half=False,
        dilate_iter=1
    )
    return out
