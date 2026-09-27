# -*- coding: utf-8 -*-
"""色块 + 像素细化 —— 分层出图之后的手工精修层。

为什么要有这一层
----------------
`render_regions` 只能给出**大致的色块**（这一块是蓝的、那一块是白的）：
分区是按面判定的，边界再平滑也是"统计"出来的，碰到这几件事就没辙 ——
  · 鼻梁、嘴唇：压平成纯二维之后，明暗没了，这两样必须**用线画**；
  · 两个色块的交界：想让它走一条特定的曲线；
  · 封闭图形的边缘：想手动画/抹掉一段。
这些都得在**像素级**做。

设计（就是"先色块、再像素细修"那条路线）
----------------------------------------
1. 色块阶段：`mesh.render_regions` 出块。
2. 像素阶段：本模块在一张已经出好的图上改。所有改动写成 **ops（JSON 列表）**，
   可以重放、可以只调其中一步 —— 跟 pptx-editor 的 ops 是同一套思路。

ops 一览
--------
{'op':'set',    'pts':[[x,y],...], 'color':[r,g,b], 'w':1}
{'op':'line',   'a':[x0,y0], 'b':[x1,y1], 'color':..., 'w':1}       # 两点直线
{'op':'path',   'pts':[[x,y],...], 'color':..., 'w':1, 'closed':False}
{'op':'flood',  'pt':[x,y], 'color':..., 'tol':12}                  # 有界填充（这块刷成某色）
{'op':'erase',  'pts':[...], 'w':1, 'bg':[r,g,b]}                  # 抹成背景色
{'op':'seam',   'a':[r,g,b], 'b':[r,g,b], 'color':..., 'w':1}      # 重画两块之间的交界
{'op':'smooth', 'color':[r,g,b], 'r':1.2}                          # 某色块的边界再平滑
{'op':'outline','color':[r,g,b], 'w':1, 'bg':[r,g,b]}              # 只描轮廓

线的画法：**像素块，不做抗锯齿** —— 用户要的就是"以像素块填充"。
所以 Bresenham 直线 + 方块笔刷，放大看是硬边像素。
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

__all__ = ['load', 'save', 'apply_ops', 'run_ops_file',
           'line', 'path', 'flood', 'seam_mask', 'smooth_block',
           'bbox_of_color', 'colors_in', 'edge_outline']


# ---------------------------------------------------------------- 基础

def load(path):
    return Image.open(path).convert('RGB')


def save(img, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    img.save(path)
    return path


def _stamp(a, x, y, color, w=1):
    """方块笔刷：(x,y) 为中心，边长 w（w=1 就是一个像素块）。"""
    H, W = a.shape[:2]
    r = max(0, (w - 1) // 2)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    if x0 < x1 and y0 < y1:
        a[y0:y1, x0:x1] = color


def line(a, p0, p1, color, w=1):
    """Bresenham 直线（像素块，无抗锯齿）。原地改。"""
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        _stamp(a, x0, y0, color, w)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return a


def path(a, pts, color, w=1, closed=False):
    """折线。w=1 是 1 像素硬边线；w>=3 会变成有一定粗细的描边。"""
    pts = [(int(round(p[0])), int(round(p[1]))) for p in pts]
    for i in range(len(pts) - 1):
        line(a, pts[i], pts[i + 1], color, w)
    if closed and len(pts) > 2:
        line(a, pts[-1], pts[0], color, w)
    return a


def _tol_mask(a, color, tol):
    return np.abs(a.astype(np.int16) - np.asarray(color, np.int16)).sum(2) <= tol


def flood(a, pt, color, tol=12, grow=0):
    """有界填充：从 pt 出发，把颜色相近的连通块整体刷成 color。

    就是"这一块要蓝色"那个操作。grow>0 时向外膨胀几像素（吃掉边缘的抗锯齿过渡）。
    """
    from collections import deque
    H, W = a.shape[:2]
    x, y = int(pt[0]), int(pt[1])
    if not (0 <= x < W and 0 <= y < H):
        return a, 0
    src = a[y, x].copy()
    if np.abs(src.astype(np.int16) - np.asarray(color, np.int16)).sum() <= tol:
        return a, 0
    m = _tol_mask(a, src, tol)
    seen = np.zeros((H, W), bool)
    q = deque([(x, y)])
    seen[y, x] = True
    hits = []
    while q:
        cx, cy = q.popleft()
        hits.append((cx, cy))
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < W and 0 <= ny < H and not seen[ny, nx] and m[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    reg = np.zeros((H, W), bool)
    for cx, cy in hits:
        reg[cy, cx] = True
    if grow > 0:
        from PIL import ImageFilter
        im = Image.fromarray((reg * 255).astype(np.uint8))
        reg = np.asarray(im.filter(ImageFilter.MaxFilter(grow * 2 + 1))) > 127
        reg &= ~_tol_mask(a, color, 0)          # 不吞掉已经填好的同色
    a[reg] = color
    return a, int(reg.sum())


def seam_mask(a, c1, c2, tol=18, w=1):
    """两块颜色之间的交界线掩膜。

    做法：分别取 c1 / c2 的掩膜，c1 膨胀 w 像素后与 c2 相交的部分就是交界。
    —— 这就对应"两块中间的地方再用细致的像素修改"。
    """
    from PIL import ImageFilter
    m1 = _tol_mask(a, c1, tol).astype(np.uint8) * 255
    m2 = _tol_mask(a, c2, tol).astype(np.uint8) * 255
    d1 = np.asarray(Image.fromarray(m1).filter(ImageFilter.MaxFilter(w * 2 + 1))) > 127
    d2 = np.asarray(Image.fromarray(m2).filter(ImageFilter.MaxFilter(w * 2 + 1))) > 127
    return (d1 & m2.astype(bool)) | (d2 & m1.astype(bool))


def smooth_block(a, color, tol=18, r=1.2, edge=3.0):
    """把某个纯色块的边界在 2D 上重新平滑一遍（跟 render_regions 里同一套手法）。"""
    m = _tol_mask(a, color, tol).astype(np.uint8) * 255
    if r <= 0:
        return a
    from PIL import ImageFilter
    sm = np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(float(r)))) / 255.0
    new = np.clip((sm - 0.5) * edge + 0.5, 0, 1)
    a[new > 0.5] = color
    return a


def edge_outline(a, color, w=1, bg=None, tol=24):
    """给前景轮廓描一圈（背景色由 bg 指定，默认取左上角像素）。"""
    from PIL import ImageFilter
    if bg is None:
        bg = a[1, 1]
    fg = (np.abs(a.astype(np.int16) - np.asarray(bg, np.int16)).sum(2) > tol)
    m = Image.fromarray((fg * 255).astype(np.uint8))
    dil = np.asarray(m.filter(ImageFilter.MaxFilter(w * 2 + 1))) > 127
    ring = dil & ~fg
    a[ring] = color
    return a


def cast_shadow(a, src_color, dx, dy, scale=0.80, tol=18, on_colors=None, steps=None):
    """2D 投影阴影：把 src_color 的色块沿 (dx,dy) 平移，落在别的色块上的部分压暗。

    这是"平涂阴影"的另一个来源 —— 不是法线算出来的，而是**遮挡物投下来的**。
    日漫脸里最关键的一块阴影就是它：**刘海投在额头上的那块暗部**。
    法线光照永远算不出这个（额头本身是朝前的，明明受光）。

    src_color : 遮挡物（例如刘海色）
    dx, dy    : 屏幕上的投影方向（光源在左上 -> 阴影落向右下，dx>0 dy>0）
    on_colors : 只压暗这些颜色（不给就压暗所有被投到的像素）
    """
    m = _tol_mask(a, src_color, tol)
    if not m.any():
        return a, 0
    if steps is None:
        steps = int(max(abs(dx), abs(dy))) + 1
    sh = np.zeros_like(m)
    for i in range(1, steps + 1):                 # 分步累积，保证连续覆盖不留缝
        ox = int(round(dx * i / steps))
        oy = int(round(dy * i / steps))
        sh |= np.roll(np.roll(m, oy, axis=0), ox, axis=1)
    target = sh & ~m
    if on_colors:
        tm = np.zeros_like(m)
        for c in on_colors:
            tm |= _tol_mask(a, c, tol)
        target &= tm
    if target.any():
        a[target] = np.clip(a[target].astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return a, int(target.sum())


def shade_poly(a, pts, scale=0.78, on_colors=None, tol=18, feather=0.0, to=None):
    """在多边形区域内压暗（可只压暗指定颜色）。

    这就是"以像素块填充一块阴影"：形状由调用方算好（多半来自三维几何），
    这里只负责把这块像素乘上 scale。

    日漫脸里两个最要紧的阴影都能这么来：
      · 下颌投在脖子上的那块（把下颌弧整体下移一段 -> 就是投影多边形）
      · 鼻梁/鼻底的暗面
    feather>0 时边界按高斯淡出（动漫多半是硬边，默认 0）。

    ⚠️ 传 `to` 而不是靠 `scale` 乘：一块阴影区域里往往同时有"底色"和
       "已经被 cel 压暗过的色"，同一个 scale 乘上去会得到**两档深浅**，
       看着像打了块补丁（实测脖子那块就成了围兜）。给 `to` 就统一填成一个色。
    """
    H, W = a.shape[:2]
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).polygon([(float(p[0]), float(p[1])) for p in pts], fill=255)
    w = np.asarray(m).astype(np.float32) / 255.0
    if feather > 0:
        w = _blur_u8(w * 255, feather) / 255.0
    if on_colors:
        tm = np.zeros((H, W), bool)
        for c in on_colors:
            tm |= _tol_mask(a, c, tol)
        w = w * tm
    if w.max() <= 0:
        return a, 0
    if to is not None:
        sel = w > 0.5
        a[sel] = np.asarray(to, np.uint8)
        return a, int(sel.sum())
    f = 1.0 - w * (1.0 - scale)
    a[:] = np.clip(a.astype(np.float32) * f[..., None], 0, 255).astype(np.uint8)
    return a, int((w > 0.5).sum())


def bbox_of_color(a, color, tol=18):
    """某个色块的包围盒 (x0,y0,x1,y1)，空则 None。"""
    m = _tol_mask(a, color, tol)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def colors_in(a, min_px=50):
    """图上出现过的颜色及像素数（从多到少）。"""
    flat = a.reshape(-1, 3)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-cnt)
    return [(tuple(int(v) for v in uniq[i]), int(cnt[i]))
            for i in order if cnt[i] >= min_px]


# ---------------------------------------------------------------- ops

def apply_ops(img, ops, verbose=False):
    a = np.asarray(img.convert('RGB')).copy()
    n_done = 0
    for k, op in enumerate(ops):
        kind = op.get('op')
        col = op.get('color')
        w = int(op.get('w', 1))
        if kind == 'set':
            for p in op['pts']:
                _stamp(a, int(p[0]), int(p[1]), col, w)
        elif kind == 'line':
            line(a, op['a'], op['b'], col, w)
        elif kind == 'path':
            path(a, op['pts'], col, w, op.get('closed', False))
        elif kind == 'erase':
            bg = op.get('bg', [255, 255, 255])
            for p in op['pts']:
                _stamp(a, int(p[0]), int(p[1]), bg, w)
        elif kind == 'flood':
            _, cnt = flood(a, op['pt'], col, op.get('tol', 12), op.get('grow', 0))
            if verbose:
                print('    flood%3d  %s -> %s  改了 %d px' % (k, op['pt'], tuple(col), cnt))
        elif kind == 'seam':
            m = seam_mask(a, op['a'], op['b'], op.get('tol', 18), w)
            a[m] = col
            if verbose:
                print('    seam %3d  交界 %d px' % (k, int(m.sum())))
        elif kind == 'shade':
            _, cnt = shade_poly(a, op['poly'], op.get('scale', 0.78), op.get('on'),
                                op.get('tol', 18), op.get('feather', 0.0), op.get('to'))
            if verbose:
                print('    shade %3d  %d 点多边形 -> 压暗 %d px'
                      % (k, len(op['poly']), cnt))
        elif kind == 'cast':
            _, cnt = cast_shadow(a, op['from'], op.get('dx', 6), op.get('dy', 10),
                                 op.get('scale', 0.80), op.get('tol', 18),
                                 op.get('on'), op.get('steps'))
            if verbose:
                print('    cast %3d  %s 投影 -> 压暗 %d px' % (k, tuple(op['from']), cnt))
        elif kind == 'smooth':
            smooth_block(a, col, op.get('tol', 18), op.get('r', 1.2), op.get('edge', 3.0))
        elif kind == 'outline':
            edge_outline(a, col, w, op.get('bg'), op.get('tol', 24))
        else:
            raise ValueError('未知 op: %r' % kind)
        n_done += 1
    if verbose:
        print('  共应用 %d 个 op' % n_done)
    return Image.fromarray(a)


def run_ops_file(img_path, ops_path, out_path=None, verbose=True):
    img = load(img_path)
    with open(ops_path, encoding='utf-8') as f:
        ops = json.load(f)
    out = apply_ops(img, ops, verbose=verbose)
    return save(out, out_path or img_path)
