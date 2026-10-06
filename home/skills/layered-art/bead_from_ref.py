#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考图驱动的渐进生长式拼豆。

把 bead_grow 的"加法"接到一张真实参考图上：
    ① 语义分区   按颜色把人设的各元素分出来（红瞳/深蓝花/浅蓝领结/中蓝裙/肤色/
                 白（发+风衣+衬衫+袜+靴）/灰（靴阴影）/背景）
    ② 染色机制   每个元素一套**从参考图采出来的**固定调色板
    ③ 细节       元素内部取参考图的局部明暗，决定在该元素调色板里取哪一色
    ④ 生长       逐层累加倍频程 -> 从大色块长到全分辨率

与"把参考图降采样"的减法路线的区别：
  · 元素**不会串色** —— 头发区域永远只从白色系取色，不会沾到肤色
  · 第 0 层是**刻意分好区**的，不是降采样碰巧得到的样子
  · 每层都有豆子清单
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bead_grow import BeadScene, fbm_octaves, _norm01        # noqa: E402

# ================================================================ 人设元素
# 颜色全部从 115.png / 116.png 采样得到（sample_persona2.py）
# 顺序 = 优先级：越靠前越优先（饱和色不能输给白/灰）
ELEMENTS = [
    # 名        参考色（采样值，粗量化过的邻近色都列进来）
    ('red', [(248, 0, 48), (248, 0, 40), (240, 0, 48), (216, 0, 64),
             (248, 40, 80), (200, 0, 72), (120, 0, 32)]),
    ('flower', [(16, 72, 224), (24, 72, 216), (16, 72, 216), (96, 144, 240),
                (88, 144, 232)]),
    ('bow', [(144, 208, 232), (144, 216, 232), (136, 208, 232), (152, 216, 232),
             (144, 200, 200), (152, 200, 200)]),
    ('skirt', [(88, 128, 160), (80, 128, 160), (88, 136, 168), (96, 136, 168),
               (144, 200, 200), (136, 200, 208)]),
    ('skin', [(248, 232, 224), (248, 232, 232), (248, 224, 224), (232, 216, 216),
              (240, 216, 216)]),
    # ⚠️ 原本这里有个独立的 'gray' 元素（靴/阴影/描边）。实测它把**白风衣的阴影
    #    全吸走**了 —— 因为 gray 的参考色里有浅灰 (192,200,208)，而白衣服的阴影
    #    也是浅灰；gray 的调色板又是深的，于是整件白风衣被涂成灰的。
    #    这个角色本来就没有独立的灰色衣物，灰全是白布料的阴影 -> 并进 white。
    ('white', [(240, 232, 232), (240, 240, 240), (232, 232, 232), (224, 224, 224),
               (240, 240, 232), (232, 232, 240), (248, 248, 248),
               (216, 216, 224), (200, 204, 214), (184, 188, 200),
               (168, 168, 184), (152, 160, 176), (120, 126, 142),
               (88, 94, 110), (64, 70, 86)]),
    ('bg', [(64, 72, 88), (216, 216, 216), (224, 224, 224)]),
]

# 每个元素的**有序调色板**（暗->亮）。细节值决定取哪一色。
PALETTES = {
    # 白（发/风衣/衬衫/袜/靴）：从灰影到高光。这是画面主角，给最多层次
    'white': [(58, 62, 76), (86, 92, 108), (118, 124, 142), (148, 154, 170),
              (176, 180, 194), (198, 202, 214), (214, 216, 226), (228, 228, 236),
              (238, 236, 238), (246, 244, 244), (252, 250, 250)],
    # 肤：腮红到高光
    'skin': [(226, 186, 178), (238, 206, 196), (246, 224, 214), (250, 236, 228),
             (252, 242, 236)],
    # 红（瞳/耳饰）
    'red': [(120, 0, 32), (176, 0, 48), (216, 0, 56), (244, 16, 64), (252, 72, 104)],
    # 深蓝花
    'flower': [(12, 40, 140), (16, 56, 188), (16, 72, 224), (64, 112, 240),
               (120, 160, 248)],
    # 浅蓝领结
    'bow': [(104, 172, 204), (120, 190, 218), (140, 208, 232), (168, 224, 240),
            (200, 240, 248)],
    # 中蓝裙
    'skirt': [(56, 88, 116), (72, 108, 140), (88, 128, 160), (112, 156, 186),
              (144, 190, 212)],
    # 灰（靴/阴影/描边）
    'gray': [(78, 84, 100), (110, 118, 134), (140, 148, 164), (168, 168, 184),
             (196, 202, 212)],
    # 背景
    'bg': [(40, 46, 60), (56, 62, 78), (72, 80, 96), (92, 100, 118), (120, 128, 146)],
}


# 位置先验：(y0, y1) 归一化的允许区间；None = 不限。
# 为什么需要：116 的裙子和领结**都是薄荷绿**，颜色分不开，只能靠位置分
# （领结在胸口、裙在腰以下）。这对应"提示词机制"里"要素位置固定"这一条。
POS_PRIOR = {
    'red': (0.00, 0.45),        # 眼 / 耳饰，在面部附近
    'flower': (0.00, 0.35),     # 头上的花
    'bow': (0.25, 0.58),        # 胸口
    'skirt': (0.36, 0.78),      # 腰以下
    'skin': (0.00, 0.85),
    'white': None,
    'bg': None,
}
POS_PENALTY = 90000.0           # 越界罚分（加到颜色距离上）


def classify(a, pos=True):
    """每个像素 -> 元素下标。按最近参考色 + 位置先验，并列时取优先级高的。"""
    H, W = a.shape[:2]
    yy = (np.arange(H, dtype=np.float64) / max(1, H - 1))[:, None]
    best_d = np.full((H, W), 1e18, np.float64)
    best_i = np.zeros((H, W), np.int32)
    flat = a.reshape(-1, 1, 3).astype(np.float64)
    for i, (name, cols) in enumerate(ELEMENTS):
        pal = np.asarray(cols, np.float64)
        d = ((flat - pal[None, :, :]) ** 2).sum(2).min(1).reshape(H, W)
        if pos:
            band = POS_PRIOR.get(name)
            if band is not None:
                y0, y1 = band
                out = (yy < y0) | (yy > y1)
                d = d + np.where(out, POS_PENALTY, 0.0)
        upd = d < best_d - 1e-9
        best_d[upd] = d[upd]
        best_i[upd] = i
    return best_i


def crop_subject(im, pad_frac=0.06, thr=0.004):
    """裁到主体（非背景）的外接框 + 一点边距。

    为什么必须裁：实测不裁的话背景占 **48.8%** —— 一半的豆子在画背景，
    对拼豆作品是纯浪费；而且人物偏在画面一侧，格阵利用率很低。
    """
    a = np.asarray(im.convert('RGB')).astype(np.int32)
    H, W = a.shape[:2]
    bgcols = np.asarray(ELEMENTS[[n for n, _ in ELEMENTS].index('bg')][1], np.float64)
    flat = a.reshape(-1, 1, 3).astype(np.float64)
    d = ((flat - bgcols[None, :, :]) ** 2).sum(2).min(1).reshape(H, W)
    fg = d > 900.0                                  # 离背景色足够远 = 主体
    # ⚠️ 不能直接用 np.where(fg) 取 min/max：背景不是纯色（有轻微渐变和抗锯齿
    #    杂点），边缘总会有零星像素被判成前景 —— 实测外接框铺满整张图
    #    （x 0~895 y 0~1151），等于没裁。改成按**行列的前景计数 profile** 定边界，
    #    对零星杂点免疫。
    rows = fg.mean(1)
    cols = fg.mean(0)
    ry = np.where(rows > thr)[0]
    rx = np.where(cols > thr)[0]
    if len(ry) < 4 or len(rx) < 4:
        return im
    y0, y1 = int(ry[0]), int(ry[-1])
    x0, x1 = int(rx[0]), int(rx[-1])
    py = int((y1 - y0) * pad_frac)
    px = int((x1 - x0) * pad_frac)
    y0 = max(0, y0 - py); y1 = min(H - 1, y1 + py)
    x0 = max(0, x0 - px); x1 = min(W - 1, x1 + px)
    return im.crop((x0, y0, x1 + 1, y1 + 1))


def _box_blur(a, k=3, iters=1):
    """k×k 均值模糊（边界用 edge 延拓）。用来在取色前抹掉线稿噪声。"""
    n = (k - 1) // 2
    for _ in range(iters):
        H, W = a.shape
        pad = np.pad(a, n, mode='edge')
        acc = np.zeros((H, W), np.float32)
        for dy in range(k):
            for dx in range(k):
                acc = acc + pad[dy:dy + H, dx:dx + W]
        a = acc / float(k * k)
    return a


def _mode_filter(cls, k=3, iters=1):
    """k×k 多数滤波，去掉孤立碎点。

    ⚠️ padding 必须是 (k-1)//2（k=3 时 =1）。第一版写成 `0 if k==3 else 1`，
       结果 k=3 时完全不 padding，切片从 H×W 缩成 (H-dy)×(W-dx)，
       np.stack 直接报 "all input arrays must have the same shape"。
    """
    n = (k - 1) // 2
    K = int(cls.max()) + 1
    for _ in range(iters):
        H, W = cls.shape
        pad = np.pad(cls, n, mode='edge')
        stack = np.stack([pad[dy:dy + H, dx:dx + W]
                          for dy in range(k) for dx in range(k)], 0)
        cnt = np.zeros((K, H, W), np.int32)
        for v in range(K):
            cnt[v] = (stack == v).sum(0)
        cls = cnt.argmax(0).astype(cls.dtype)
    return cls


def build_scene(ref_path, size=(48, 64), levels=4, seed=7, keep_bg=True,
                crop=True, smooth=True, sampled_pals=None):
    """从参考图建 BeadScene。size 是最终格数 (W, H)。"""
    im = Image.open(ref_path).convert('RGB')
    if crop:
        im = crop_subject(im)
    W, H = size

    # ① 先降到目标格数 —— 每格一个颜色，之后的分类和明暗都在格级做
    small = np.asarray(im.resize((W, H), Image.LANCZOS)).astype(np.int32)

    # ② 格级分类 + 多数滤波
    cls = classify(small)
    if smooth:
        cls = _mode_filter(cls, 3, 2)

    # ③ 明暗：直接用降采样后的格颜色（这就是"细节来自参考图"）
    lum = 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]
    lum = lum.astype(np.float32)

    sc = BeadScene(size=(W, H), levels=levels, bg=(244, 242, 246), seed=seed)
    used = []
    for i, (name, _) in enumerate(ELEMENTS):
        m = (cls == i)
        if name == 'bg' and not keep_bg:
            continue
        # ⚠️ 阈值不能太高：小元素（花、耳饰）本来就只占几格，
        #    门槛设成"总格数的 1/400"会把它们整个丢掉。
        if m.sum() < 3:
            continue
        v = lum[m]
        lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        d = np.clip((lum - lo) / max(1e-6, hi - lo), 0, 1).astype(np.float32)
        # 若给了采样调色板就用它（256 色模式），否则用手写的
        pal = sampled_pals.get(name) if sampled_pals else None
        if not pal:
            pal = PALETTES[name]
        # ⚠️ 取色前必须平滑：直接用参考图亮度会把**线稿和抗锯齿**也当成明暗，
        #    细看全是噪点。3x3 两遍足够抹掉线稿宽度（1~2 格）而保留衣物褶皱。
        d = _box_blur(d, 3, 2)
        # 不给 octaves —— 细节就在参考图里，加随机噪声会让平涂区脏掉
        sc.add(name, m, pal, detail=d, base_index=None)
        used.append((name, int(m.sum())))
    return sc, used


if __name__ == '__main__':
    DESK = os.environ.get('REF_DIR')
    if not DESK:
        raise SystemExit('请用 REF_DIR=<图片所在目录> 指定参考图目录')
    OUT = os.environ.get('BEAD_OUT') or os.path.join(os.getcwd(), 'out')
    os.makedirs(OUT, exist_ok=True)
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DESK, '115.png')
    size = tuple(int(v) for v in (os.environ.get('BEAD_SIZE') or '56x88').split('x'))
    lv = int(os.environ.get('BEAD_LEVELS') or 5)
    print('参考图：%s' % src)
    print('目标格数 %dx%d   %d 层' % (size[0], size[1], lv))
    keep_bg = (os.environ.get('BEAD_KEEP_BG') or '0') == '1'
    sc, used = build_scene(src, size=size, levels=lv, keep_bg=keep_bg)
    print()
    print('元素分区：')
    for n, c in used:
        print('   %-8s %6d 格  %5.1f%%   %d 色'
              % (n, c, 100 * c / (size[0] * size[1]), len(PALETTES[n])))
    lvs = sc.grow()
    print()
    print('%-5s %-11s %-8s %-9s %s' % ('层', '格阵', '格尺寸', '豆子数', '用色数'))
    for L, (im, cells, st) in enumerate(lvs):
        print('%-5d %-11s %-8d %-9d %d'
              % (L, '%dx%d' % st['grid'], st['cell'], st['total_beads'], st['n_colors']))
    tag = os.path.splitext(os.path.basename(src))[0]
    for L, (im, cells, st) in enumerate(lvs):
        im.save(os.path.join(OUT, '%s_bead_L%d.png' % (tag, L)))
    # 对照图
    from PIL import ImageDraw
    TH = 480
    row = []
    for im, _, _ in lvs:
        k = TH / im.height
        row.append(im.resize((max(1, int(im.width * k)), TH), Image.NEAREST))
    pad = 14
    cw = sum(t.width for t in row) + pad * (len(row) + 1)
    canvas = Image.new('RGB', (cw, TH + 44), (26, 26, 30))
    d = ImageDraw.Draw(canvas)
    x = pad
    for L, t in enumerate(row):
        canvas.paste(t, (x, 34))
        d.text((x + 4, 11), 'L%d %dx%d %d色 %d豆'
               % (L, lvs[L][2]['grid'][0], lvs[L][2]['grid'][1],
                  lvs[L][2]['n_colors'], lvs[L][2]['total_beads']), fill=(224, 228, 236))
        x += t.width + pad
    p = os.path.join(OUT, '%s_bead_all.png' % tag)
    canvas.save(p)
    print()
    print('-> %s' % p)
    top = lvs[-1][2]
    print('-> 最细层 %d 颗，%d 色' % (top['total_beads'], top['n_colors']))
    with open(os.path.join(OUT, '%s_shopping.txt' % tag), 'w', encoding='utf-8') as f:
        for L, (_, _, st) in enumerate(lvs):
            f.write('# L%d %dx%d  %d 颗  %d 色\n'
                    % (L, st['grid'][0], st['grid'][1], st['total_beads'], st['n_colors']))
            for c in st['colors']:
                f.write('%s\t%d\n' % (c['hex'], c['n']))
            f.write('\n')
