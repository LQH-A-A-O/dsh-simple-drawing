# -*- coding: utf-8 -*-
"""把 115.png 做活：挥手（平滑变形）+ 眨眼（局部贴片）

坐标是从带网格的叠加图里**读出来的**（见 ana/115_头部.png 与 115_叠加.png）：
    左眼（红瞳，睁着）外框约  x 440~508   y 172~272
    右眼（已在眨）     约  x 662~758   y 208~248
    手掌（比 V）      约  x 200~420   y 180~300
    手肘（袖口转折处） 约  x 330     y 320
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import puppet as PP       # noqa: E402


def _out_dir(name):
    import tempfile
    base = os.environ.get('LAYERED_ART_OUT') or os.path.join(
        tempfile.gettempdir(), 'layered-art')
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    return d


OUT = _out_dir('puppet')
SRC = os.environ.get('PUPPET_SRC')
if not SRC:
    raise SystemExit('请用 PUPPET_SRC=<图片路径> 指定要做木偶的平面图')
FPS = 14
N = 42                                   # 3 秒

im = Image.open(SRC).convert('RGB')
a0 = np.asarray(im).astype(np.uint8)
H, W = a0.shape[:2]
bg = PP.bg_of(a0)
print('源图 %dx%d   背景色 %s' % (W, H, bg))

# ---- 关键坐标（相对原图）
EYE_L = (447, 174, 503, 269)             # 睁着的那只（画面左）—— 紧贴眼睛
HAND = (300, 232)                        # 手掌中心
ELBOW = (332, 322)                       # 旋转轴（袖口转折处）
# ⚠️ 睫毛色不能靠固定点采样 —— 我从 (474,173) 采到的是 (252,237,234)，
#    那是额头/头发（白的），画出来的闭眼线等于没画。
#    正解：**取眼框内最暗的颜色**，那才一定是眼线/睫毛。
_box = a0[EYE_L[1]:EYE_L[3], EYE_L[0]:EYE_L[2], :3].reshape(-1, 3)
_lum = _box.mean(1)
LASH = tuple(int(v) for v in _box[_lum.argmin()])
SKIN = PP.sample(a0, (EYE_L[0] + EYE_L[2]) / 2.0, EYE_L[3] + 22)
print('睫毛色 %s（眼框内最暗）  眼周肤色 %s' % (LASH, SKIN))

# ================================================================ 时间轴
# 挥手：绕肘旋转，幅度随离手掌的距离衰减 -> 读起来就是挥手
# 眨眼：0.9s 处眨一次，1.6s 处再眨一次（中间带过渡）
ARM_SEQ = [(0, 0), (0.10, -13), (0.22, 5), (0.34, -11), (0.46, 4),
           (0.58, -9), (0.70, 3), (0.82, -6), (1.0, 0)]
BLINKS = [(0.26, 0.34), (0.62, 0.72)]    # (闭的起点, 终点) 归一化时间


def arm_angle(t):
    for i in range(len(ARM_SEQ) - 1):
        t0, v0 = ARM_SEQ[i]
        t1, v1 = ARM_SEQ[i + 1]
        if t0 <= t <= t1:
            k = (t - t0) / max(1e-9, t1 - t0)
            k = k * k * (3 - 2 * k)                   # 缓动
            return v0 + (v1 - v0) * k
    return 0.0


def eye_close(t):
    """0=睁 1=闭。眨眼做成"快闭慢睁"，比对称的三角波自然。"""
    for (c0, c1) in BLINKS:
        if c0 <= t <= c1:
            k = (t - c0) / max(1e-9, c1 - c0)
            return math.sin(math.pi * k) if False else (
                min(1.0, k / 0.35) if k < 0.35 else max(0.0, 1 - (k - 0.35) / 0.65))
    return 0.0


import math  # noqa: E402

# ---- 手的可动区域：一个**边界外严格为 0** 的平滑区块。
#    ⚠️ 这是"精度"的关键。之前位移场全局施加，衰减到 0 也仍会被浮点重采样
#       轻微改动 —— 看起来就是"全身都在扭"。现在只替换 mask 内的像素。
HAND_MASK = PP.blob_mask((H, W), HAND, rx=W * 0.150, ry=H * 0.110,
                         power=1.35, feather=0.40)
print('可动区域占画面 %.1f%%（其余部分逐字节不动）'
      % (100.0 * (HAND_MASK > 0).mean()))

frames = []
max_outside = 0
# ⚠️ 允许改动的区域 = **手 ∪ 眼睛**。只拿手部当判据会把眨眼算成"渗漏"
#    （实测报了 754 的假警 —— 那个差值其实来自眼睛，不是变形渗出去的）。
ALLOW = HAND_MASK.copy()
ALLOW[max(0, EYE_L[1] - 8):EYE_L[3] + 8, max(0, EYE_L[0] - 8):EYE_L[2] + 8] = 255
for i in range(N):
    t = i / float(N)
    ang = arm_angle(t)
    dx, dy = PP.rotate_field((H, W), ELBOW, ang, center=HAND,
                             rx=W * 0.16, ry=H * 0.12, power=1.0)
    fr = PP.local_warp(a0, dx, dy, HAND_MASK)
    ec = eye_close(t)
    if ec > 0.02:
        fr = PP.blink(fr, EYE_L, lash=LASH, skin=SKIN, close=ec)
    # 精度自检：允许改动区域之外必须**逐字节等于原图**
    d = np.abs(fr.astype(np.int32) - a0.astype(np.int32)).sum(2)
    out = d[ALLOW == 0]
    if out.size:
        max_outside = max(max_outside, int(out.max()))
    frames.append(fr)

print('精度自检：可动区域外最大像素差 = %d  %s'
      % (max_outside, '✅ 逐字节一致' if max_outside == 0 else '⚠ 有渗漏'))

print('渲染 %d 帧（%.1f 秒 @ %d fps）' % (len(frames), N / float(FPS), FPS))

gif = os.path.join(OUT, '115_挥手眨眼.gif')
PP.export_gif(frames, gif, fps=FPS, max_w=760)
print('  GIF -> %s  (%.0f KB)' % (gif, os.path.getsize(gif) / 1024))
mp4 = os.path.join(OUT, '115_挥手眨眼.mp4')
got = PP.export_mp4(frames, mp4, fps=FPS)
print('  MP4 -> %s' % ('%s (%.0f KB)' % (got, os.path.getsize(got) / 1024)
                       if got else '跳过（没找到 ffmpeg）'))

# ---- 对照图：原图 / 挥手最大帧 / 闭眼帧
peak = max(range(N), key=lambda i: abs(arm_angle(i / float(N))))
shut = max(range(N), key=lambda i: eye_close(i / float(N)))
tiles = [('原图', a0), ('挥手峰值 f%d' % peak, frames[peak]),
         ('闭眼 f%d' % shut, frames[shut])]
pad = 8
tw = 380
th = int(H * tw / W)
sheet = Image.new('RGB', (len(tiles) * (tw + pad) + pad, th + 26), (22, 24, 28))
d = ImageDraw.Draw(sheet)
for i, (n, fr) in enumerate(tiles):
    x = pad + i * (tw + pad)
    sheet.paste(Image.fromarray(fr).resize((tw, th), Image.LANCZOS), (x, 20))
    d.text((x + 2, 5), n, fill=(205, 215, 230))
sheet.save(os.path.join(OUT, '对照.png'))
print('  对照 -> %s' % os.path.join(OUT, '对照.png'))

# ---- 眼睛特写对比（原 / 半闭 / 全闭）
crop_box = (EYE_L[0] - 26, EYE_L[1] - 20, EYE_L[2] + 26, EYE_L[3] + 26)
tiles2 = [('睁', a0), ('半闭', frames[shut - 2]), ('全闭', frames[shut])]
k = 3
cw = (crop_box[2] - crop_box[0]) * k
ch = (crop_box[3] - crop_box[1]) * k
s2 = Image.new('RGB', (len(tiles2) * (cw + 8) + 8, ch + 26), (22, 24, 28))
d2 = ImageDraw.Draw(s2)
for i, (n, fr) in enumerate(tiles2):
    c = Image.fromarray(fr).crop(crop_box).resize((cw, ch), Image.NEAREST)
    s2.paste(c, (8 + i * (cw + 8), 20))
    d2.text((10 + i * (cw + 8), 5), n, fill=(205, 215, 230))
s2.save(os.path.join(OUT, '眼睛对比.png'))
print('  眼睛 -> %s' % os.path.join(OUT, '眼睛对比.png'))
print('\n产物 -> %s' % OUT)
