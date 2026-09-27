# -*- coding: utf-8 -*-
"""像素画转换：把平涂色块图变成"我的世界"那种硬边像素画。

为什么适合这条路
----------------
这套管线出来的图**本来就是平涂色块**（一个区一块纯色），调色板天然很小
（实测 JK 立绘只有 25 个颜色）。像素画要的正是这个 ——
所以只需要"按粗网格取众数色 + 硬边放大"，不需要任何手绘。

⚠️ 两个关键点，做错了就不是像素画：
  1. **每格取众数，不能取平均**。平均会造出一堆中间色，出来是一张模糊的小图，
     不是像素画。像素画问的是"这一格归哪一块"，不是"这一格平均什么颜色"。
  2. **放大必须 NEAREST**，不能用 LANCZOS/BILINEAR。像素画要硬边。
"""
import numpy as np
from PIL import Image

__all__ = ['palette_of', 'index_map', 'cell_mode', 'outline_cells',
           'pixelize', 'crop_to_content']


def crop_to_content(img, bg=None, pad=0):
    """裁到内容包围盒（像素画要先裁紧，不然一堆空白格）。"""
    a = np.asarray(img.convert('RGB'))
    if bg is None:
        bg = a[2, 2]
    m = np.abs(a.astype(np.int16) - np.asarray(bg, np.int16)).sum(2) > 24
    if not m.any():
        return img
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def palette_of(img, min_px=6, bg=None):
    """图里出现过的颜色（按像素数从多到少）。返回 (colors, counts, bg_index)"""
    a = np.asarray(img.convert('RGB')).reshape(-1, 3)
    uniq, cnt = np.unique(a, axis=0, return_counts=True)
    keep = cnt >= min_px
    uniq, cnt = uniq[keep], cnt[keep]
    order = np.argsort(-cnt)
    uniq, cnt = uniq[order], cnt[order]
    if bg is None:
        bg = tuple(int(v) for v in uniq[0])
    bi = 0
    for i, c in enumerate(uniq):
        if tuple(int(v) for v in c) == tuple(int(v) for v in bg):
            bi = i
            break
    return uniq.astype(np.int16), cnt, bi


def index_map(img, palette):
    """每个像素 -> 最近调色板色的下标。

    ⚠️ 平方要用 int32 —— int16 会溢出（255^2*3 超 int16），
       实测这会让最近色判错、整图变成噪声。
    """
    a = np.asarray(img.convert('RGB')).astype(np.int32)
    pal = np.asarray(palette, np.int32)
    H, W = a.shape[:2]
    flat = a.reshape(-1, 1, 3)
    d = ((flat - pal[None, :, :]) ** 2).sum(2)          # (H*W, K)
    return d.argmin(1).reshape(H, W).astype(np.int32)


def cell_mode(idx, cell, n_colors):
    """每 cell x cell 格取**众数**索引（向量化）。返回小索引图。"""
    H, W = idx.shape
    h, w = H // cell, W // cell
    if h < 1 or w < 1:
        raise ValueError('cell 太大：图只有 %dx%d' % (W, H))
    blk = idx[:h * cell, :w * cell].reshape(h, cell, w, cell)
    blk = blk.transpose(0, 2, 1, 3).reshape(h, w, cell * cell)
    keys = ((np.arange(h)[:, None, None] * w + np.arange(w)[None, :, None]) * n_colors
            + blk)
    cnt = np.bincount(keys.ravel(), minlength=h * w * n_colors).reshape(h * w, n_colors)
    return cnt.argmax(1).reshape(h, w).astype(np.int32)


def _roll0(m, dy, dx):
    """平移但**不环绕**。

    ⚠️ np.roll 是环绕的：底部的前景会被滚到顶部，于是顶边被误判成轮廓，
       描出两根悬在头顶的横杠（实测就是这个症状，查了半天以为是水手领）。
    """
    r = np.roll(m, dy, 0) if dy else m.copy()
    if dy > 0:
        r[:dy] = False
    elif dy < 0:
        r[dy:] = False
    if dx:
        r = np.roll(r, dx, 1)
        if dx > 0:
            r[:, :dx] = False
        else:
            r[:, dx:] = False
    return r


def outline_cells(idx, bg_idx, out_idx, diag=True):
    """在小索引图上给前景外缘描一圈（4 邻域，可选 8 邻域）。"""
    fg = (idx != bg_idx)
    nb = [_roll0(fg, 1, 0), _roll0(fg, -1, 0), _roll0(fg, 0, 1), _roll0(fg, 0, -1)]
    if diag:
        nb += [_roll0(fg, 1, 1), _roll0(fg, 1, -1),
               _roll0(fg, -1, 1), _roll0(fg, -1, -1)]
    edge = np.zeros_like(fg)
    for m in nb:
        edge |= m
    edge &= ~fg
    out = idx.copy()
    out[edge] = out_idx
    return out


def pixelize(img, cell=12, max_colors=None, outline=(38, 36, 46),
             bg=None, crop=True, pad=1, diag_outline=True, verbose=False):
    """平涂图 -> 像素画。

    cell      : 一个像素格占原图多少像素（越大越像"我的世界"）
    outline   : 外缘线颜色；None 就不描
    返回 (放大回原尺寸的图, 小图, 调色板)
    """
    src = crop_to_content(img, bg=bg, pad=pad) if crop else img.convert('RGB')
    a0 = np.asarray(src.convert('RGB'))
    if bg is None:
        bg = tuple(int(v) for v in a0[2, 2])
    pal, cnt, bi = palette_of(src, min_px=max(4, cell * cell // 6), bg=bg)

    if max_colors and len(pal) > max_colors:
        pal = pal[:max_colors]
        # 背景色必须留在调色板里，否则背景会被并到别的色上
        if not (pal == np.asarray(bg, np.int16)).all(1).any():
            pal = np.vstack([np.asarray(bg, np.int16)[None, :], pal[:-1]])
        bi = int(np.argmin(np.abs(pal - np.asarray(bg, np.int16)).sum(1)))

    pal_arr = pal.astype(np.int16)
    if outline is not None:
        ol = np.asarray(outline, np.int16)[None, :]
        if not (pal_arr == ol).all(1).any():
            pal_arr = np.vstack([pal_arr, ol])
        oi = len(pal_arr) - 1
    else:
        oi = None

    idx = index_map(src, pal_arr[:len(pal)] if oi is None else pal_arr[:len(pal)])
    small_idx = cell_mode(idx, cell, len(pal_arr))
    if oi is not None:
        small_idx = outline_cells(small_idx, bi, oi, diag=diag_outline)
    small = Image.fromarray(pal_arr[small_idx].astype(np.uint8))
    big = small.resize((small.width * cell, small.height * cell), Image.NEAREST)
    if verbose:
        print('   调色板 %d 色（原图 %d 色）  小图 %dx%d  cell=%d'
              % (len(pal_arr), len(cnt), small.width, small.height, cell))
    return big, small, pal_arr


# ---------------------------------------------------------------- 腿

def split_legs(img, skin, ink, t_crotch, t_ankle, y_at, min_run=2, w=1,
               bg=None):
    """把并在一起的两条腿分开：沿每行找"腿间最窄处"，画一条中缝。

    ⚠️ 扫描范围必须**一直扫到脚踝**。只扫到大腿/膝盖的话，线只画了上半截，
       下半截两条腿还是连着的（实测就是这样：线只覆盖了 128px 中的一小段）。
    """
    from PIL import ImageDraw  # noqa: F401  (保持接口一致)
    a = np.asarray(img.convert('RGB')).copy()
    m = (np.abs(a.astype(np.int16) - np.asarray(skin, np.int16)).sum(2) <= 24)
    if bg is None:
        bg = a[2, 2]
    fg = (np.abs(a.astype(np.int16) - np.asarray(bg, np.int16)).sum(2) > 24)
    y0, y1 = int(y_at(t_crotch)), int(y_at(t_ankle))
    pts = []
    for y in range(y0, y1):
        r = np.nonzero(m[y])[0]
        if len(r) < min_run:
            continue
        # 分段
        segs, s0 = [], r[0]
        for i in range(1, len(r)):
            if r[i] != r[i - 1] + 1:
                segs.append((s0, r[i - 1]))
                s0 = r[i]
        segs.append((s0, r[-1]))
        if len(segs) >= 2:
            # ⚠️ 两腿之间**本来就是背景**的行不要再画线 ——
            #    否则会在两腿之间的空白处拖出一条悬空的墨线（实测 y=1394 中招）。
            #    只有两条腿**贴着**（皮肤连续）时才需要画中缝。
            g0, g1 = segs[0][1] + 1, segs[1][0] - 1
            if np.all(~fg[y, g0:g1 + 1]):
                continue
            pts.append((int((segs[0][1] + segs[1][0]) / 2), y))
        else:
            pts.append((int((r[0] + r[-1]) / 2), y))
    if len(pts) < 3:
        return img, 0
    arr = np.array([[p[0], p[1]] for p in pts], float)
    for _ in range(3):                     # 平滑，别跟着像素噪声抖
        arr[1:-1, 0] = (arr[:-2, 0] + 2 * arr[1:-1, 0] + arr[2:, 0]) / 4.0
    for x, y in arr:
        xi, yi = int(round(x)), int(round(y))
        a[yi, max(0, xi - w):xi + w + 1] = ink
    return Image.fromarray(a), len(arr)
