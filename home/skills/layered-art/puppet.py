# -*- coding: utf-8 -*-
"""把**一张平面图**当提线木偶：平滑变形（挥手）+ 局部贴片（眨眼）

为什么不做"切图旋转"
--------------------
115 那条抬起的手臂和飘起来的外套**是同一种白色，袖子还盖在手臂上** ——
根本切不出"手臂"这个部件。就算切出来，转开之后背后是空的，必须生成式补图。

所以这里走两条**对单层图必然成立**的路：

  挥手  **平滑变形**（mesh warp）
        在手掌附近施加一个"随距离平滑衰减"的旋转/摆动，整图连续重采样。
        因为是连续映射，**永远不产生洞**，也不需要知道背后有什么。
        代价：是"掌部摆动"而不是"真的转肩肘"。

  眨眼  **局部贴片**
        眼睛是局部且位置固定的：取眼睛的外框，用周围肤色盖住，
        再画一条闭眼弧线。不需要补图，因为盖住的地方本来就有确定的颜色。

⚠️ 变形场的梯度必须 < 1，否则映射会折叠（画面出现撕裂）。
   这里的衰减是最多 2 次方，幅度也小，实测不折叠。
"""
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

__all__ = ['rotate_field', 'sway_field', 'warp', 'blink', 'make_frames',
           'export_gif', 'export_mp4', 'sample', 'bg_of']


# ================================================================ 采样 / 工具
def sample(a, x, y, r=3):
    """取一小块的中位数颜色（比单点稳，抗 JPEG 噪点）。"""
    h, w = a.shape[:2]
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    patch = a[y0:y1, x0:x1, :3].reshape(-1, 3)
    if not len(patch):
        return (128, 128, 128)
    return tuple(int(v) for v in np.median(patch, axis=0))


def bg_of(a, band=6):
    """背景色 = 四边各 band 像素的中位数。"""
    edge = np.concatenate([a[:band].reshape(-1, 3), a[-band:].reshape(-1, 3),
                           a[:, :band].reshape(-1, 3), a[:, -band:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(edge, axis=0))


# ================================================================ 变形场
def _falloff(shape, center, rx, ry, power=2.0):
    """以 center 为中心、半径 rx/ry 的平滑衰减权重（0..1）。"""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xs - center[0]) / max(rx, 1e-6)) ** 2 +
                ((ys - center[1]) / max(ry, 1e-6)) ** 2)
    return np.clip(1.0 - d, 0.0, 1.0) ** power


def local_warp(a, dx, dy, mask):
    """只在 mask 内变形，**mask 外逐字节保持不变**。

    ⚠️ 这是"精度"的关键。之前直接把位移场全局施加，即使衰减到 0，
       浮点重采样也会让整张图轻微变化 —— 看起来就是"全身都在扭"。
       正解：先算变形图，再**只把 mask 内的像素换掉**，mask 外直接拷贝原图。

    mask: (H,W) uint8/float，>0 才算要变形的区域
    """
    warped = warp(a.astype(np.float32), dx, dy)
    out = a.copy()
    m = mask.astype(np.float32)
    sel = m > 0
    if not sel.any():
        return out
    w = (m[sel] / 255.0)[:, None] if a.ndim == 3 else (m[sel] / 255.0)
    out[sel] = np.clip(a[sel].astype(np.float32) * (1 - w)
                       + warped[sel] * w, 0, 255).astype(np.uint8)
    return out


def blob_mask(shape, center, rx, ry, power=1.5, feather=0.10):
    """一个平滑但**在边界外严格为 0** 的区域权重（0..255）。

    power 越大越集中于中心；feather 是边缘过渡带占半径的比例。
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xs - center[0]) / max(rx, 1e-6)) ** 2 +
                ((ys - center[1]) / max(ry, 1e-6)) ** 2)
    inner = 1.0 - max(0.0, feather)
    t = np.clip((1.0 - d) / max(1e-6, inner), 0.0, 1.0)
    return (255.0 * t ** power).astype(np.uint8)


def rotate_field(shape, pivot, angle_deg, center=None, rx=None, ry=None,
                 power=2.0):
    """绕 pivot 旋转，但**旋转量随离 center 的距离衰减**。

    于是手掌附近转得多、身体几乎不动 —— 读起来就是"挥手"。
    返回 (dx, dy) 位移场。
    """
    h, w = shape
    center = center or pivot
    rx = rx or w * 0.25
    ry = ry or h * 0.25
    wt = _falloff((h, w), center, rx, ry, power)
    ang = math.radians(angle_deg) * wt
    ys, xs = np.mgrid[0:h, 0:w]
    dx0 = xs - pivot[0]
    dy0 = ys - pivot[1]
    ca, sa = np.cos(ang), np.sin(ang)
    # 目标位置 = 绕 pivot 转 ang；位移 = 目标 - 原位置
    tx = pivot[0] + dx0 * ca - dy0 * sa
    ty = pivot[1] + dx0 * sa + dy0 * ca
    return (tx - xs).astype(np.float32), (ty - ys).astype(np.float32)


def sway_field(shape, center, amp_x, amp_y, rx, ry, power=2.0):
    """纯平移式的平滑摆动（比旋转更"柔和"，适合头发/衣摆）。"""
    h, w = shape
    wt = _falloff((h, w), center, rx, ry, power)
    return (amp_x * wt).astype(np.float32), (amp_y * wt).astype(np.float32)


def warp(a, dx, dy, is_mask=False):
    """按位移场重采样。**反向映射 + 双线性**（正向映射会留空洞）。

    a: (H,W,3) 或 (H,W) 或 (H,W,4)
    """
    h, w = a.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = xs - dx
    sy = ys - dy
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0)[..., None] if a.ndim == 3 else (sx - x0)
    fy = (sy - y0)[..., None] if a.ndim == 3 else (sy - y0)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    v00 = a[y0c, x0c].astype(np.float32)
    v10 = a[y0c, x1c].astype(np.float32)
    v01 = a[y1c, x0c].astype(np.float32)
    v11 = a[y1c, x1c].astype(np.float32)
    out = (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
           + v01 * (1 - fx) * fy + v11 * fx * fy)
    if is_mask:
        out = (out > 0.5).astype(np.float32)
    return out


# ================================================================ 眨眼
def inpaint(a, mask, iters=90, radius=5.0):
    """用「反复模糊 + 每次恢复遮罩外的原始像素」填充 mask 区域。

    ⚠️ 这个迭代是必须的。单次大半径模糊会把**遮罩外的深色**（睫毛、头发）
       一起拉进来 —— 实测直接把眼白糊成灰色。迭代版只让边界颜色向内扩散，
       而且每次迭代都把已知像素钉回原值，等于解一个"边界值问题"。

    mask: (H,W) uint8，255=要填充
    """
    src = a.astype(np.float32)
    img = src.copy()
    m = (mask.astype(np.float32) / 255.0)[..., None]
    for _ in range(int(iters)):
        b = np.asarray(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
        img = b * m + src * (1 - m)
    return img


def blink(a, box, lash=None, skin=None, close=1.0, lid=0.46,
          feather=2.2, iters=90, radius=5.0):
    """把 box 里的眼睛改成"闭眼"。

    box:   (x0, y0, x1, y1) 眼睛外框（**要略大于眼睛本身**，留出余量）
    close: 0=全睁（原样） 1=全闭
    lid:   闭眼线落在盖住区域高度的百分之几处
    band:  从眼睛下方取几行皮肤来当底色

    ⚠️ 底色**不能用平均色平涂** —— 脸是有渐变的，平涂会露出一个直角矩形，
       一眼就看出来是贴上去的。正解：从眼睛正下方取几行皮肤，
       **按列平均后向上拉伸**，这样横向的明暗变化保住了，接缝才看不见。
    """
    out = a.copy().astype(np.float32)
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(a.shape[1], x1), min(a.shape[0], y1)
    if x1 <= x0 or y1 <= y0 or close <= 0.02:
        return out.astype(np.uint8)
    H, W = a.shape[:2]
    ew, eh = x1 - x0, y1 - y0
    cover = max(1, int(round(eh * close)))

    # ---- 遮罩：**内缩的椭圆**，不是矩形。
    #   矩形会把框四角的头发一起糊进来；椭圆只覆盖眼睛本体。
    #   ⚠️ 上缘必须**盖住原来那条粗睫毛线**。之前内缩 10%，结果原睫毛留着，
    #      和我画的闭眼线凑成"双层线"，看着像个空眼窝。
    #      盖掉它之后，插值的边界才落在皮肤上（而不是落在黑线上）。
    m = Image.new('L', (ew, eh), 0)
    ImageDraw.Draw(m).ellipse([int(ew * 0.04), -int(eh * 0.08),
                               int(ew * 0.96), int(eh * 1.00)], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(max(1.2, feather)))
    marr = np.asarray(m).astype(np.uint8)

    # ---- 填充：迭代插值（只让边界颜色向内扩散）
    full = np.zeros(a.shape[:2], np.uint8)
    full[y0:y0 + eh, x0:x1] = marr
    fill_all = inpaint(a, full, iters=iters, radius=radius)
    fill = fill_all[y0:y0 + eh, x0:x1, :3]

    # ---- 把填充**夹在肤色的窄范围内**。
    #   迭代插值仍会从边界带进一点暗色（睫毛/头发在边界上的残留），
    #   不夹一下，眼窝处就留一块暗晕 —— 全图尺寸看就是"一块暗斑"。
    ring = []
    for (rx0, ry0, rx1, ry1) in ((x0 - 22, y0 + 6, x0 - 4, y1 - 6),
                                 (x1 + 4, y0 + 6, x1 + 22, y1 - 6),
                                 (x0 + 6, y1 + 4, x1 - 6, min(H, y1 + 24))):
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(W, rx1), min(H, ry1)
        if rx1 > rx0 and ry1 > ry0:
            ring.append(a[ry0:ry1, rx0:rx1, :3].reshape(-1, 3))
    if ring:
        ring = np.concatenate(ring).astype(np.float32)
        # 只用亮的一半（排除混进来的睫毛/头发），取中位数当"标准肤色"
        lum = ring.mean(1)
        med = np.median(ring[lum >= np.median(lum)], axis=0)
        fill = med[None, None, :] + np.clip(fill - med[None, None, :], -15, 12)
        # 眼睑阴影：上缘略暗一档，让眼窝有体积感（不是一块平色）
        sh = np.linspace(0.90, 1.0, eh)[:, None, None]
        fill = fill * sh

    patch = Image.fromarray(np.clip(fill, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(patch)
    if lash is None:
        sub = a[y0:y1, x0:x1, :3].reshape(-1, 3)
        lash = tuple(int(v) for v in sub[sub.mean(1).argmin()])
    thick = max(2, int(eh * 0.075))
    # 弧度要够大。线宽和弧高比例不当会读成"V 字"而不是闭眼弧 ——
    # 弧幅 0.16*eh 配 0.11*eh 粗的线就是这个下场，实测踩过。
    pts = [(i, eh * (lid - 0.09 + 0.26 * math.sin(math.pi * i / float(max(1, ew - 1)))))
           for i in range(ew)]
    d.line(pts, fill=tuple(int(v) for v in lash), width=thick, joint='curve')

    ma = (marr.astype(np.float32) / 255.0)[..., None]
    pa = np.asarray(patch).astype(np.float32)
    reg = out[y0:y0 + eh, x0:x1, :3]
    out[y0:y0 + eh, x0:x1, :3] = reg * (1 - ma) + pa * ma
    return np.clip(out, 0, 255).astype(np.uint8)


# ================================================================ 导出
def find_ffmpeg():
    p = os.environ.get('FFMPEG')
    if p and os.path.exists(p):
        return p
    w = shutil.which('ffmpeg')
    if w:
        return w
    return None


def export_gif(frames, path, fps=12, loop=0, max_w=None):
    """导出 GIF。

    ⚠️ 有大片软渐变的图（照片感/厚涂）GIF 会巨大（实测 896x1152 能到 19 MB）。
       传 max_w 先把帧缩到指定宽度再编码。
    """
    d = max(20, int(1000.0 / fps))
    ims = []
    for f in frames:
        im = Image.fromarray(f)
        if max_w and im.width > max_w:
            k = max_w / float(im.width)
            im = im.resize((max_w, max(1, int(round(im.height * k)))), Image.LANCZOS)
        ims.append(im.convert('P', palette=Image.ADAPTIVE, colors=255))
    ims[0].save(path, save_all=True, append_images=ims[1:],
                duration=d, loop=loop, disposal=2, optimize=True)
    return path


def export_mp4(frames, path, fps=12, ffmpeg=None):
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        return None
    import tempfile
    tmp = tempfile.mkdtemp(prefix='puppet-')
    try:
        for i, f in enumerate(frames):
            im = Image.fromarray(f)
            if im.width % 2 or im.height % 2:
                im = im.crop((0, 0, im.width - im.width % 2, im.height - im.height % 2))
            im.save(os.path.join(tmp, 'f%05d.png' % i))
        subprocess.run([ff, '-y', '-framerate', str(fps), '-i',
                        os.path.join(tmp, 'f%05d.png'),
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                        '-pix_fmt', 'yuv420p', path], capture_output=True, check=True)
        return path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_frames(a, n, fn):
    """逐帧生成：fn(i, t) 返回该帧的图（可直接返回 a 表示不变）。"""
    out = []
    for i in range(n):
        t = i / float(n)
        out.append(fn(i, t))
    return out
