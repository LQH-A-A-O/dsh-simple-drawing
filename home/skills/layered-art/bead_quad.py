#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自适应四叉树拼豆 —— 按"和原图的误差"决定哪里继续切。

与固定格数路线的区别
--------------------
固定格数（bead_grow）：
    画布定死 W×H，第 L 层的格子尺寸 = 2^(N-1-L)，所有区域**均匀**细分。
    问题：平坦区（背景、白风衣）和高细节区（脸、花）用一样大的格子 ——
          要么平坦区浪费豆子，要么细节区不够精细。

自适应细分（本模块）：
    从"整幅一块"开始。对每块问一句"**一个颜色能不能代表这块**"：
        能  -> 保留（大色块）
        不能 -> 切成四份，每份再问一遍
    切到误差够小、或达到深度上限、或切到单格为止。

    这就是用户说的"初次分割 -> 填充 -> 在原图基础上再细分 -> 反复"，
    也就是**四叉树图像逼近**。好处：
      · 豆子花在刀刃上（平坦区省，细节区细）
      · 画布不再受"必须被 2 的幂整除"约束（见 _split 里对奇数的处理）
      · 精细度只受参考图分辨率和误差阈值限制 —— 可以一直切下去

每个区域有自己的调色板（沿用"染色机制"），所以细分的颜色只在本区调色板里选，
不会串色。
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

__all__ = ['QuadBeads']


def _split(x0, y0, w, h):
    """把一块切成四份。**支持奇数尺寸**（这是"画布任意大"的关键）。

    w=5 -> 2 和 3；w=1 -> 不再切。
    """
    if w <= 1 and h <= 1:
        return []
    aw = w // 2 if w > 1 else 1
    bw = w - aw
    ah = h // 2 if h > 1 else 1
    bh = h - ah
    out = []
    if aw > 0 and ah > 0:
        out.append((x0, y0, aw, ah))
    if bw > 0 and ah > 0:
        out.append((x0 + aw, y0, bw, ah))
    if aw > 0 and bh > 0:
        out.append((x0, y0 + ah, aw, bh))
    if bw > 0 and bh > 0:
        out.append((x0 + aw, y0 + ah, bw, bh))
    return out


class QuadBeads:
    """自适应四叉树拼豆。

    size     画布格数 (W, H) —— 任意正整数，不必是 2 的幂
    ref      参考图（PIL）。会缩放到 size；细分依据是它的像素
    regions  [(name, mask(HxW bool), palette[(r,g,b)...])]，顺序 = 优先级
    tol      误差阈值（0~255 的每通道平均绝对差）。越小越细
    max_depth 最大切分深度
    min_size 最小格边长（1 = 切到单格）
    """

    def __init__(self, size, ref, regions, tol=10.0, max_depth=12,
                 min_size=1, bg=(32, 34, 42)):
        self.W, self.H = int(size[0]), int(size[1])
        assert self.W > 0 and self.H > 0
        im = ref.convert('RGB').resize((self.W, self.H), Image.LANCZOS)
        self.ref = np.asarray(im).astype(np.float32)
        self.tol = float(tol)
        self.max_depth = int(max_depth)
        self.min_size = max(1, int(min_size))
        self.bg = tuple(int(v) for v in bg[:3])
        self.regions = []
        for name, mask, pal in regions:
            m = np.asarray(mask, bool)
            assert m.shape == (self.H, self.W), \
                '区域 %s 遮罩形状 %s 应为 (%d,%d)' % (name, m.shape, self.H, self.W)
            p = np.asarray([[int(v) for v in c[:3]] for c in pal], np.float32)
            assert len(p), '区域 %s 调色板为空' % name
            self.regions.append((name, m, p, np.arange(len(p))))
        self.cells = []

    # ---------------- 内部
    def _region_of(self, x0, y0, w, h):
        """这一块的归属 = 覆盖格数最多的区域（并列取优先级高的）。"""
        best, bi, bn = -1, 0, ''
        for i, (name, m, _p, _ix) in enumerate(self.regions):
            c = int(m[y0:y0 + h, x0:x0 + w].sum())
            if c > best:
                best, bi, bn = c, i, name
        return bi, bn, best

    def _fit(self, x0, y0, w, h, pal):
        """这一块用调色板里哪个色最好，误差多大。"""
        px = self.ref[y0:y0 + h, x0:x0 + w].reshape(-1, 3)
        if px.size == 0:
            return pal[0], 0.0
        d = np.abs(px[:, None, :] - pal[None, :, :]).sum(2)   # (N, K)
        j = int(d.mean(0).argmin())
        return pal[j], float(d[:, j].mean() / 3.0)            # 每通道平均绝对差

    def refine(self, verbose=False):
        """跑细分。返回 cell 列表 [(x, y, w, h, (r,g,b), region, depth)]。"""
        cells = []
        stack = [(0, 0, self.W, self.H, 0)]
        n_split = 0
        while stack:
            x0, y0, w, h, depth = stack.pop()
            ri, rname, _ = self._region_of(x0, y0, w, h)
            pal = self.regions[ri][2]
            col, err = self._fit(x0, y0, w, h, pal)
            can_split = (max(w, h) > self.min_size) and (depth < self.max_depth)
            if err > self.tol and can_split:
                sub = _split(x0, y0, w, h)
                if sub:
                    n_split += 1
                    for (sx, sy, sw, sh) in sub:
                        stack.append((sx, sy, sw, sh, depth + 1))
                    continue
            cells.append((x0, y0, w, h,
                          (int(col[0]), int(col[1]), int(col[2])), rname, depth))
        self.cells = cells
        if verbose:
            print('   切分 %d 次 -> %d 块' % (n_split, len(cells)))
        return cells

    # ---------------- 统计
    def stats(self):
        if not self.cells:
            return {}
        beads = sum(w * h for _x, _y, w, h, _c, _r, _d in self.cells)
        cols = {}
        for _x, _y, w, h, c, _r, _d in self.cells:
            cols[c] = cols.get(c, 0) + w * h
        dep = {}
        for *_z, d in [(c[6],) for c in self.cells]:
            pass
        dep = {}
        for cell in self.cells:
            dep[cell[6]] = dep.get(cell[6], 0) + 1
        reg = {}
        for cell in self.cells:
            reg[cell[5]] = reg.get(cell[5], 0) + cell[2] * cell[3]
        return {
            'cells': len(self.cells),
            'beads': beads,
            'canvas': (self.W, self.H),
            'fill_ratio': beads / float(self.W * self.H),
            'n_colors': len(cols),
            'colors': sorted(cols.items(), key=lambda kv: -kv[1]),
            'depth_hist': dict(sorted(dep.items())),
            'region_beads': dict(sorted(reg.items(), key=lambda kv: -kv[1])),
        }

    # ---------------- 渲染
    def render(self, scale=1, bead_style=True):
        """cell 列表 -> 图。每格边长 w 格 = w*scale 像素。"""
        W, H = self.W * scale, self.H * scale
        if not bead_style or scale < 4:
            im = Image.new('RGB', (W, H), self.bg)
            d = ImageDraw.Draw(im)
            for x0, y0, w, h, c, _r, _d in self.cells:
                d.rectangle([x0 * scale, y0 * scale,
                             (x0 + w) * scale - 1, (y0 + h) * scale - 1], fill=c)
            return im
        ss = 3
        base = Image.new('RGB', (W * ss, H * ss), self.bg)
        d = ImageDraw.Draw(base)
        u = scale * ss                      # 一颗豆的像素（含超采样）
        gap = max(1, int(u * 0.055))
        rad = max(1, int(u * 0.22))
        for x0, y0, w, h, c, _r, _d in self.cells:
            r, g, b = c
            hl = tuple(min(255, int(v + (255 - v) * 0.30)) for v in (r, g, b))
            # ⚠️ 必须**逐颗**画，块只是"这部分都是同色"的压缩表示。
            #    原来对整块画一个圆角矩形 -> 一个 8x8 的块变成一颗大方块，
            #    看起来像贴瓷砖而不像拼豆。
            hw = max(1, int(u * 0.20))
            for yy in range(y0, y0 + h):
                py = yy * u
                for xx in range(x0, x0 + w):
                    px = xx * u
                    d.rounded_rectangle([px + gap, py + gap,
                                         px + u - gap - 1, py + u - gap - 1],
                                        radius=rad, fill=(r, g, b))
                    if u >= 8:
                        d.ellipse([px + gap + hw * 0.35, py + gap + hw * 0.35,
                                   px + gap + hw * 1.35, py + gap + hw * 1.35],
                                  fill=hl)
        return base.resize((W, H), Image.LANCZOS)


if __name__ == '__main__':
    # 自检：造一个已知的图，看细分是不是"平的地方粗、细的地方细"
    print('=' * 74)
    print('自适应四叉树拼豆 · 自检')
    print('=' * 74)
    W = H = 128
    # 左半纯色（应当只切很少），右半高频条纹（应当切到很深）
    a = np.zeros((H, W, 3), np.uint8)
    a[:, :W // 2] = (200, 60, 60)
    for i in range(W // 2, W):
        a[:, i] = (250, 250, 250) if (i // 2) % 2 == 0 else (20, 20, 20)
    ref = Image.fromarray(a, 'RGB')
    m_all = np.ones((H, W), bool)
    pal = [(200, 60, 60), (250, 250, 250), (20, 20, 20)]
    qb = QuadBeads((W, H), ref, [('all', m_all, pal)], tol=8.0, max_depth=10)
    qb.refine(verbose=True)
    st = qb.stats()
    print()
    print('  切分块数 %d   覆盖豆数 %d / %d（%.0f%%）'
          % (st['cells'], st['beads'], W * H, 100 * st['fill_ratio']))
    print('  深度分布 %s' % st['depth_hist'])
    # 左半应该是大块（浅），右半应该切到深
    left = [c for c in qb.cells if c[0] < W // 2]
    right = [c for c in qb.cells if c[0] >= W // 2 - 8]
    lavg = np.mean([c[6] for c in left]) if left else 0
    ravg = np.mean([c[6] for c in right]) if right else 0
    print('  左半（纯色）平均深度 %.1f   右半（高频）平均深度 %.1f' % (lavg, ravg))
    ok = ravg > lavg
    print('  %s 高频区确实切得更深' % ('✅' if ok else '❌'))
    # 画布任意大小（不必是 2 的幂）
    for sz in ((100, 77), (333, 199), (7, 5)):
        q2 = QuadBeads(sz, ref, [('all', np.ones((sz[1], sz[0]), bool), pal)],
                       tol=8.0, max_depth=10)
        q2.refine()
        s2 = q2.stats()
        cover = s2['beads']
        print('  %-10s 任意尺寸可跑，覆盖 %d / %d %s'
              % ('%dx%d' % sz, cover, sz[0] * sz[1],
                 '✅' if cover == sz[0] * sz[1] else '❌ 有空洞'))
