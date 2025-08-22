import time
import traceback
import numpy as np
import cv2
from numba import njit
from PIL import Image, ImageDraw, ImageFont

import epd7in5_V2  # 你的驱动

# =========================
# 常量
# =========================
# 电子纸 4 阶灰度（保持与 getbuffer_4Gray 映射一致的顺序）
GRAY_LEVELS = np.array([0xFF, 0x80, 0x40, 0x00], dtype=np.float32)

# 4×4 Bayer（团簇式）
BAYER4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32)

def _sheared_bayer4(h: int, w: int, shift_per_row: int = 3) -> np.ndarray:
    """
    生成“行位移”的 4×4 阈值图：
    第 y 行相位右移 (y*shift_per_row) mod 4，打散竖向重复条纹。
    shift_per_row 推荐 3（若仍有细竖纹可试 2 或 1）
    """
    ri = (np.arange(h) % 4)[:, None]  # 行相位 0..3
    ci = (np.arange(w)[None, :] + ((np.arange(h) % 4)[:, None] * shift_per_row)) % 4
    return BAYER4[ri, ci]

# =========================
# 蛇形扫描的“锁带+高频改浅灰”版 FS
# =========================
@njit
def fs_4gray_bandaware_v2_lightHF_serp(arr, palette,
                                       lock_mask,
                                       thr_map_deep, thr_map_light,
                                       use_light_mask,
                                       err_scale_in_band):
    """
    带“黑↔深灰锁带 + 高频处黑↔浅灰”的 FS，蛇形扫描：
    - 偶数行 L→R，奇数行 R→L；
    - 锁带内误差扩散按 err_scale_in_band 打折；
    - 锁带外：普通 FS 到四档。
    """
    h, w = arr.shape
    for y in range(h):
        left_to_right = (y % 2 == 0)
        if left_to_right:
            x0, x1, step = 0, w, 1
        else:
            x0, x1, step = w - 1, -1, -1

        x = x0
        while x != x1:
            old = arr[y, x]
            if lock_mask[y, x]:
                if use_light_mask[y, x]:
                    thr = thr_map_light[y, x]
                    new_val = 0.0 if old < thr else 128.0  # 高频：黑/浅灰
                else:
                    thr = thr_map_deep[y, x]
                    new_val = 0.0 if old < thr else 64.0   # 普通：黑/深灰
                arr[y, x] = new_val
                err = (old - new_val) * err_scale_in_band
            else:
                # 四档就近量化
                idx = 0
                best = 1e9
                for k in range(palette.size):
                    d = abs(palette[k] - old)
                    if d < best:
                        best = d
                        idx = k
                new_val = palette[idx]
                arr[y, x] = new_val
                err = old - new_val

            # 误差扩散（根据方向镜像邻居）
            if left_to_right:
                if x + 1 < w:
                    arr[y, x + 1] += err * 7.0 / 16.0
                if y + 1 < h:
                    if x > 0:
                        arr[y + 1, x - 1] += err * 3.0 / 16.0
                    arr[y + 1, x]     += err * 5.0 / 16.0
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * 1.0 / 16.0
            else:
                if x - 1 >= 0:
                    arr[y, x - 1] += err * 7.0 / 16.0
                if y + 1 < h:
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * 3.0 / 16.0
                    arr[y + 1, x]     += err * 5.0 / 16.0
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * 1.0 / 16.0

            x += step
    return arr

def dither_hybrid_black_deepgray_v2_lightHF_fix(
    img: Image.Image,
    top_only: bool = True,
    band_center: float = 32.0,
    band_halfwidth: float = 16.0,
    bias_black: float = 4.0,
    err_scale_in_band: float = 0.10,
    hf_thresh: float = 90.0,
    light_bias: float = 10.0,
    shear: int = 3
) -> Image.Image:
    # ===== 灰度 + 轻对比度拉伸（避免两端挤压）=====
    arr = np.array(img.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo + 1e-3:
        arr = (arr - lo) * (np.float32(255.0) / np.float32(hi - lo))
        arr = np.clip(arr, 0, 255)
    # 关键：确保仍为 float32，避免被 upcast 成 float64
    arr = arr.astype(np.float32, copy=False)

    h, w = arr.shape

    # ===== 锁带：黑(0x00) ↔ 深灰(0x40) =====
    band_lo = band_center - band_halfwidth
    band_hi = band_center + band_halfwidth
    lock_mask = (arr >= band_lo) & (arr <= band_hi)
    if top_only:
        lock_mask[h // 2 :, :] = False
    lock_mask = lock_mask.astype(np.bool_)

    # ===== 行位移 Bayer（防竖纹）=====
    by = _sheared_bayer4(h, w, shift_per_row=shear)
    thr_base = band_center + (by - 7.5) / 15.0 * band_halfwidth
    thr_map_deep  = (thr_base - bias_black).astype(np.float32)  # 黑/深灰
    thr_map_light = (thr_base + light_bias).astype(np.float32)  # 高频：黑/浅灰

    # ===== 高频掩码（Sobel 输入用 uint8，输出 CV_32F）=====
    arr8 = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    gx = cv2.Sobel(arr8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (3, 3), 0)
    highfreq = (mag >= hf_thresh)
    if top_only:
        highfreq[h // 2 :, :] = False
    use_light_mask = (highfreq & lock_mask).astype(np.bool_)

    # ===== 蛇形扫描的 v2_lightHF =====
    work = arr.copy()
    out = fs_4gray_bandaware_v2_lightHF_serp(work, GRAY_LEVELS,
                                             lock_mask,
                                             thr_map_deep, thr_map_light,
                                             use_light_mask,
                                             err_scale_in_band)

    # ===== 与驱动映射对齐：0x80→0xC0，0x40→0x80 =====
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    out_u8[out_u8 == 128] = 192
    out_u8[out_u8 == 64]  = 128
    return Image.fromarray(out_u8, mode="L")

# =========================
# （测试）生成四档条纹
# =========================
def make_4gray_bars(w=800, h=480):
    bars = [0x00, 0x40, 0xC0, 0xFF]  # 黑、深灰、浅灰、白（注意从暗到亮）
    img = Image.new("L", (w, h), 0xFF)
    draw = ImageDraw.Draw(img)
    bh = h // 4
    for i, v in enumerate(bars):
        draw.rectangle([0, i * bh, w, (i + 1) * bh - 1], fill=int(v))
    return img

# =========================
# 主流程（与你保持一致）
# =========================
current_frame = 73266
fps = 24
epd = epd7in5_V2.EPD()

current_video = "backend/videos/Hawl.mp4"
frame_time_sec = current_frame / fps if fps else 0

tmp_frame    = "/dev/shm/frame.bmp"
tmp_frame_e  = "/dev/shm/frame_edited.bmp"

start = time.time()
# 如果需要，从视频抽帧 → 保存到 tmp_frame
# extract_frame(current_video, frame_time_sec, tmp_frame)
extracted_time = time.time()

# 读取源图并抖动
img = Image.open(tmp_frame).convert('L')

img_dithered = dither_hybrid_black_deepgray_v2_lightHF_fix(
    img,
    top_only=True,        # 只修上半屏
    band_halfwidth=16.0,  # 黑↔深灰锁带宽度
    bias_black=4.0,       # 向黑轻偏
    err_scale_in_band=0.10,
    hf_thresh=90.0,       # 高频阈值（越小“改浅灰”的范围越广）
    light_bias=10.0,      # 高频处更偏浅灰
    shear=3               # 行位移步长（防竖纹）
)

img_dithered.save(tmp_frame_e)
# 如需测试条纹：img_dithered = make_4gray_bars()

finish_time = time.time()

# 刷屏
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
