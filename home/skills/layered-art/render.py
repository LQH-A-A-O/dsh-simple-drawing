"""分层渲染：底色 → 阴影 → 高光 → 描边，以及"同一个结构换配色"。

一层一个纯函数，全局参数只有：光源方向、色带宽度、描边粗细、色阶表。
所以"换风格"是重跑，不是重画。
"""
import copy
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import shapes


# ---------------------------------------------------------------- 明度与色阶
def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def ramp_at(stops, t):
    """stops = [(位置0..1, RGB), ...]，按位置插值"""
    t = min(max(t, 0.0), 1.0)
    for k in range(len(stops) - 1):
        p0, c0 = stops[k]
        p1, c1 = stops[k + 1]
        if p0 <= t <= p1 or k == len(stops) - 2:
            u = min(max((t - p0) / max(p1 - p0, 1e-6), 0.0), 1.0)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * u) for j in range(3))
    return stops[-1][1]


# 预置配色：从亮到暗的色阶。保持明暗结构，只换色相 —— 这样骨架不变、气质全变
PRESETS = {
    '原色':      [(0.0, (253, 253, 253)), (1.0, (107, 122, 164))],
    '夜色霓虹':  [(0.0, (246, 240, 255)), (0.45, (150, 110, 200)), (1.0, (34, 16, 58))],
    '暖阳黄昏':  [(0.0, (255, 250, 238)), (0.45, (240, 176, 120)), (1.0, (96, 44, 34))],
    '深海青绿':  [(0.0, (236, 252, 250)), (0.45, (110, 190, 180)), (1.0, (16, 54, 62))],
    '樱花粉':    [(0.0, (255, 246, 248)), (0.45, (236, 168, 190)), (1.0, (110, 42, 70))],
    '硝烟灰':    [(0.0, (246, 244, 240)), (0.45, (150, 146, 140)), (1.0, (38, 36, 34))],
    '血红':      [(0.0, (255, 240, 236)), (0.45, (214, 96, 84)), (1.0, (56, 12, 16))],
}


def recolor(spec, stops, invert=False):
    """按**明度**把每个区域的颜色映射到新色阶。

    用区域自己的颜色做键（不是调色板索引）—— 因为编辑过的区域颜色可能已经
    偏离原调色板，按 ci 映射会串色。
    """
    uniq = sorted({tuple(r['color']) for r in spec['regions']}, key=lum, reverse=not invert)
    n = len(uniq)
    m = {}
    for rank, c in enumerate(uniq):
        m[c] = ramp_at(stops, rank / max(n - 1, 1))
    out = copy.deepcopy(spec)
    for r in out['regions']:
        r['color'] = m[tuple(r['color'])]
    out['pal'] = [m.get(tuple(c), tuple(c)) for c in spec['pal']]
    return out


# ---------------------------------------------------------------- 区域归属
def raster_mask(rg, w, h):
    """单个区域自己的像素（外轮廓减去内孔）"""
    x0, y0, x1, y1 = shapes.bbox(rg)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1 + 1, w), min(y1 + 1, h)
    m = Image.new('L', (max(x1 - x0, 1), max(y1 - y0, 1)), 0)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), bool)
    d = ImageDraw.Draw(m)
    d.polygon([(px - x0, py - y0) for px, py in rg['pts']], fill=255)
    for hh in rg['holes']:
        d.polygon([(px - x0, py - y0) for px, py in hh], fill=0)
    full = np.zeros((h, w), bool)
    full[y0:y1, x0:x1] = np.array(m) > 127
    return full


def owner_map(spec):
    """每个像素最终归哪个区域。按画家顺序写，孔不占像素（露出下面的）"""
    w, h = spec['w'], spec['h']
    owner = np.full((h, w), -1, np.int32)
    for i, rg in enumerate(spec['regions']):
        owner[raster_mask(rg, w, h)] = i
    return owner


def shift(a, dx, dy):
    out = np.zeros_like(a)
    if abs(dx) >= a.shape[1] or abs(dy) >= a.shape[0]:
        return out
    xs0, xs1 = max(0, dx), min(a.shape[1], a.shape[1] + dx)
    ys0, ys1 = max(0, dy), min(a.shape[0], a.shape[0] + dy)
    xd0, xd1 = max(0, -dx), min(a.shape[1], a.shape[1] - dx)
    yd0, yd1 = max(0, -dy), min(a.shape[0], a.shape[0] - dy)
    out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def darken(c, k=0.70, cool=0.10):
    r, g, b = c
    return (int(r * k * (1 - cool) + b * k * cool * 0.6),
            int(g * k * (1 - cool) + b * k * cool * 0.75), int(b * k))


def lighten(c, t=0.30):
    return tuple(int(min(v + (255 - v) * t, 255)) for v in c)


# ---------------------------------------------------------------- 分层渲染
def render_layers(spec, light=(-1, -1), band=6, outline=0, outline_color=(38, 32, 40),
                  bg=(255, 255, 255), shade=True):
    """返回 dict(layers=..., final=...)。

    light : 光**来自**的方向（屏幕坐标，y 向下）。( -1,-1 ) = 左上方
    band  : 阴影/高光色带宽度（像素）。像素画是硬边色带，band 太小看不出来
    outline : 描边粗细，0 = 不描
    """
    w, h = spec['w'], spec['h']
    base = shapes.to_image(spec, bg=bg)
    layers = {'L0 底色': base}
    arr = np.array(base)

    if shade:
        owner = owner_map(spec)
        lx, ly = light
        # ⚠️ 必须是**差集**不是交集：交集在大区域上≈整个区域，结果是"整体压暗"
        #    而不是打光（第一版就这么错的）
        sh = arr.copy()
        for i, rg in enumerate(spec['regions']):
            vis = owner == i
            if not vis.any():
                continue
            bm = vis & (~shift(vis, lx * band, ly * band))
            sh[bm] = darken(tuple(rg['color']))
        arr = sh
        layers['L1 阴影'] = Image.fromarray(arr)

        hi = arr.copy()
        for i, rg in enumerate(spec['regions']):
            vis = owner == i
            if not vis.any():
                continue
            bm = vis & (~shift(vis, -lx * band, -ly * band))
            hi[bm] = lighten(tuple(rg['color']))
        arr = hi
        layers['L2 高光'] = Image.fromarray(arr)

    if outline > 0:
        union = np.zeros((h, w), bool)
        for i, rg in enumerate(spec['regions']):
            union |= raster_mask(rg, w, h)
        sil = Image.fromarray((union * 255).astype(np.uint8))
        # 只给"最外层轮廓"描边会丢内部结构线；这里对整体剪影描，内部靠色差
        dil = sil.filter(ImageFilter.MaxFilter(outline * 2 + 1))
        ring = (np.array(dil) > 127) & (~union)
        arr = arr.copy()
        arr[ring] = tuple(outline_color)
    layers['L3 成品'] = Image.fromarray(arr)
    return dict(layers=layers, final=Image.fromarray(arr))


# ---------------------------------------------------------------- 联络图
def contact_sheet(items, path, tile_w=320, cols=3, label_font=None):
    """items = [(标签, PIL图), ...]"""
    if not items:
        return None
    th = int(tile_w * items[0][1].height / items[0][1].width)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (tile_w * cols + 20 * (cols + 1),
                              (th + 34) * rows + 20), (22, 24, 30))
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype(label_font or
                               _default_font(), 20)
    except Exception:
        f = ImageFont.load_default()
    for i, (name, im) in enumerate(items):
        x = 20 + (i % cols) * (tile_w + 20)
        y = 20 + (i // cols) * (th + 34)
        sheet.paste(im.convert('RGB').resize((tile_w, th), Image.LANCZOS), (x, y))
        d.text((x + 4, y + th + 6), str(name), font=f, fill=(235, 238, 245))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)
    return path
