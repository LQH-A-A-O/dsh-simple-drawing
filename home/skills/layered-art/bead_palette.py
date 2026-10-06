#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""256 色调色板：不再手写每区 5~11 色，改从参考图**按元素采样**。

用户要求："从八位 rgb 换成 256 位，细节色更多了"

设计：
  · 每个元素在该遮罩内的真实像素按**亮度排序取分位数** -> 得到一组有序颜色
  · 色数按元素面积分配（大区域分更多色），总数约 256
  · 这样"天空从深到浅有 30 个层次"是**从参考图量出来的**，不是我编的

与手写调色板的区别：手写的只有 5~11 色，色阶之间是跳的；
采样出来的有几十色，色阶连续，细节自然多。
"""
import numpy as np
from PIL import Image

__all__ = ['build_element_palettes', 'PALETTE_MODES']


def _quantile_palette(px, n):
    """把一组像素（按亮度）分成 n 档，每档取中位色。返回暗->亮的有序调色板。"""
    if len(px) == 0:
        return []
    lum = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
    order = np.argsort(lum)
    px = px[order]
    n = max(1, min(n, len(px)))
    # 用分位边界切段，段内取中位（比均匀采样稳，避免抗锯齿孤点主导）
    edges = np.linspace(0, len(px), n + 1).astype(int)
    out = []
    for i in range(n):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        seg = px[a:b]
        if len(seg) == 0:
            continue
        med = np.median(seg, axis=0).astype(int)
        out.append((int(med[0]), int(med[1]), int(med[2])))
    # 去重但保序
    ded = []
    for c in out:
        if not ded or ded[-1] != c:
            ded.append(c)
    return ded


def build_element_palettes(biga, cls_big, elements, total_colors=256,
                           min_per=3, gamma=0.75):
    """按元素面积把 total_colors 分配下去，每个元素从参考图采出自己的调色板。

    biga / cls_big 要在**同一分辨率**上（分区用的那张细图）。
    gamma < 1 让中等区域也能分到较多色（纯按面积会让小区域只剩 1 色）。
    """
    H, W = cls_big.shape
    flat_a = biga.reshape(-1, 3)
    flat_c = cls_big.reshape(-1)
    areas = {}
    for i, (name, _) in enumerate(elements):
        areas[i] = int((flat_c == i).sum())
    tot = sum(areas.values()) or 1
    # 权重 = 面积^gamma，且面积占比低于万分之五的直接给 min_per
    w = {}
    for i, a in areas.items():
        if a <= 0:
            w[i] = 0.0
        elif a < tot * 5e-4:
            w[i] = 0.0
        else:
            w[i] = (a / tot) ** gamma
    sw = sum(w.values()) or 1.0
    pals = {}
    for i, (name, _) in enumerate(elements):
        if areas[i] <= 0:
            continue
        if w[i] <= 0:
            n = min_per
        else:
            n = max(min_per, int(round(w[i] / sw * total_colors)))
        px = flat_a[flat_c == i]
        # 采样上限：太多色没意义（一个区域少于 4 个不同色的像素时）
        uniq = len(np.unique((px // 4 * 4), axis=0))
        n = max(1, min(n, max(2, uniq)))
        pal = _quantile_palette(px, n)
        if pal:
            pals[name] = pal
    return pals


PALETTE_MODES = ('hand', 'sampled')


def report(pals):
    lines = []
    tot = 0
    for name, p in sorted(pals.items(), key=lambda kv: -len(kv[1])):
        tot += len(p)
        hexes = ' '.join('#%02X%02X%02X' % c for c in p[:3])
        lines.append('   %-8s %3d 色   %s%s'
                     % (name, len(p), hexes, ' ...' if len(p) > 3 else ''))
    lines.append('   合计 %d 色' % tot)
    return '\n'.join(lines)


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bead_from_ref as BR

    DESK = os.environ.get('REF_DIR')
    if not DESK:
        raise SystemExit('请用 REF_DIR=<图片所在目录> 指定参考图目录')
    for fn in ('115.png', '116.png'):
        im = Image.open(os.path.join(DESK, fn)).convert('RGB')
        big = im.resize((im.width // 3, im.height // 3), Image.LANCZOS)
        biga = np.asarray(big).astype(np.int32)
        cls = BR.classify(biga)
        print('=== %s ===' % fn)
        for tot in (64, 256):
            pals = build_element_palettes(biga, cls, BR.ELEMENTS, total_colors=tot)
            print('  总色数目标 %d：' % tot)
            print(report(pals))
        print()
