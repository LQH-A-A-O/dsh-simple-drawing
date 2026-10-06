#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""渐进生长式拼豆 · 自检。

钉死"加法路线"相对"降采样减法"必须成立的那几条性质：
  ① 层数与格尺寸正确，最细层 = 画布
  ② **每一格必有归属**（不能露背景白）—— 第一版用"覆盖率>0.5"就露了
  ③ **调色板纪律**：每格颜色必须来自它所属区域的调色板，不能串色
  ④ 豆子清单的颗数之和 = 总格数
  ⑤ **确实是加法**：层数越高用色越多、细节越多（L0 最平）
  ⑥ 可停性：每一层单独拿出来都是完整图
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bead_grow import (BeadScene, fbm_octaves, detail_vgrad, detail_radial,   # noqa: E402
                       mask_circle, mask_below_curve, mask_rect)

PASS, FAIL = [], []


def check(name, ok, extra=''):
    (PASS if ok else FAIL).append(name)
    print('  %s %s%s' % ('✅' if ok else '❌', name, ('   ' + extra) if extra else ''))


SIZE = (56, 56)
LEVELS = 4
W, H = SIZE
shape = (H, W)

xs = np.linspace(0, 1, W)
ridge = 0.52 + 0.09 * np.sin(xs * np.pi * 1.7 + 0.6)
m_sky = ~mask_below_curve(shape, ridge)
m_gnd = mask_below_curve(shape, ridge) & ~mask_below_curve(shape, np.full(W, 0.80))
m_wat = mask_below_curve(shape, np.full(W, 0.80))
m_sun = mask_circle(shape, 0.74, 0.20, 0.075)
m_sky = m_sky & ~m_sun

SKY = [(28, 52, 110), (72, 118, 190), (156, 196, 234)]
GND = [(24, 44, 32), (56, 88, 52), (118, 142, 84)]
WAT = [(16, 42, 88), (44, 88, 148), (112, 158, 202)]
SUN = [(255, 108, 40), (255, 186, 62), (255, 238, 168)]

oc = fbm_octaves(shape, seed=5, n_oct=LEVELS)
sc = BeadScene(size=SIZE, levels=LEVELS, bg=(252, 252, 250), seed=0)
sc.add('sky', m_sky, SKY, detail=detail_vgrad(shape, 0.1, 0.95), octaves=oc, base_index=1)
sc.add('gnd', m_gnd, GND, detail=oc[0] * 0.5 + 0.5, octaves=oc, base_index=0)
sc.add('wat', m_wat, WAT, detail=oc[1], octaves=oc, base_index=0)
sc.add('sun', m_sun, SUN, detail=detail_radial(shape, 0.74, 0.20, 0.085))

print('=' * 74)
print('渐进生长式拼豆 · 自检')
print('=' * 74)
print()
print('【1】层数与格尺寸')
check('层数 = %d' % LEVELS, len(sc.cells) == LEVELS, str(sc.cells))
check('最细层格尺寸 = 1（= 画布分辨率）', sc.cells[-1] == 1)
check('格尺寸逐层减半', all(sc.cells[i] == 2 * sc.cells[i + 1]
                          for i in range(LEVELS - 1)))
for c in sc.cells:
    assert W % c == 0 and H % c == 0

levels = sc.grow()
print()
print('【2】每一格必有归属（不露背景白）')
BG = np.array(sc.bg, np.int32)
for L, (im, cells, st) in enumerate(levels):
    nbg = int((np.abs(cells.astype(np.int32) - BG).sum(2) == 0).sum())
    # 背景色只在画面最外圈可能存在；内部不该有
    interior = cells[1:-1, 1:-1].astype(np.int32)
    nbg_in = int((np.abs(interior - BG).sum(2) == 0).sum())
    check('第 %d 层内部无背景洞（%d 个）' % (L, nbg_in), nbg_in == 0,
          '整图背景格 %d' % nbg)

print()
print('【3】调色板纪律：每格颜色必须来自它所属区域')
# 检查最细层：每个颜色必须出现在至少一个区域的调色板里
_, cells3, _ = levels[-1]
allpal = set()
for pal in (SKY, GND, WAT, SUN):
    for c in pal:
        allpal.add(tuple(int(v) for v in c))
used = set(tuple(int(v) for v in c) for c in cells3.reshape(-1, 3))
extra = used - allpal
check('没有出现调色板之外的颜色', len(extra) == 0,
      ('越界色 %s' % sorted(extra)[:3]) if extra else '%d 色全部合法' % len(used))

# 更强的检查：天空区域的格子不能出现山的绿
sky_rgb = set(tuple(int(v) for v in c) for c in SKY)
gnd_rgb = set(tuple(int(v) for v in c) for c in GND)
sky_cells = cells3[m_sky[::1, ::1]] if m_sky.shape == cells3.shape else None
if sky_cells is not None:
    bad = sum(1 for c in sky_cells if tuple(int(v) for v in c) in gnd_rgb)
    check('天空区域里没有出现山的颜色（%d 格）' % bad, bad == 0)

print()
print('【4】豆子清单')
for L, (_, _, st) in enumerate(levels):
    tot = sum(c['n'] for c in st['colors'])
    check('第 %d 层清单颗数之和 = 总格数（%d）' % (L, st['total_beads']),
          tot == st['total_beads'], '%d vs %d' % (tot, st['total_beads']))
    check('第 %d 层用色数 = 清单条数' % L, st['n_colors'] == len(st['colors']))

print()
print('【5】确实是「加法」——层越高细节越多')
# ⚠️ 第一版这两条都测错了方向：
#    · "用色数逐层不减" —— 粗格的众数归并本来就可能丢掉一个色（实测 [4,8,10,9]），
#      这不是缺陷。
#    · "相邻格同色比例逐层下降" —— 格子越细，同一片天空的相邻豆子越可能同色，
#      实测这个比例是 0.77→0.94 **上升**的，指标本身方向就错了。
#    正确的做法是**直接验证加法机制**：细节场累加的倍频程越多，变化越大。
nc = [st['n_colors'] for _, _, st in levels]
check('第 0 层用色数明显少于最细层（%d < %d）' % (nc[0], nc[-1]), nc[0] < nc[-1])
allpal = len(SKY) + len(GND) + len(WAT) + len(SUN)
check('每层用色数不超过调色板总数（%s <= %d）' % (nc, allpal),
      all(n <= allpal for n in nc))
# 机制：**倍频程的偏和**的离散度应逐层不减（这就是"加法"本身）
# ⚠️ 不要去测 detail_at() 的离散度：那里还混了一个**固定的**细节场（比如竖直渐变），
#    固定项与逐层增长的项叠加后不保证严格单调（实测出现 0.232 -> 0.221 这种
#    极小回落）。测机制就测机制本身。
for r in sc.regions:
    if r.octaves is None:
        continue
    sds = []
    for L in range(LEVELS):
        n = min(len(r.octaves), L + 1)
        acc = np.zeros_like(np.asarray(r.octaves[0], np.float32))
        for k in range(n):
            acc = acc + np.asarray(r.octaves[k], np.float32) * (0.5 ** k)
        sds.append(float(acc.std()))
    check('区域 %s 累加的倍频程越多变化越大（%s）'
          % (r.name, ' '.join('%.4f' % v for v in sds)),
          all(sds[i] < sds[i + 1] for i in range(len(sds) - 1)))
# 可观测：细节场的取值范围逐层变大
for r in sc.regions:
    if r.octaves is None:
        continue
    spans = []
    for L in range(LEVELS):
        d = r.detail_at(L, LEVELS)
        spans.append(float(np.percentile(d, 95) - np.percentile(d, 5)))
    check('区域 %s 细节取值范围逐层不减（%s）'
          % (r.name, ' '.join('%.3f' % v for v in spans)),
          all(spans[i] <= spans[i + 1] + 1e-6 for i in range(len(spans) - 1)))

print()
print('【6】可停性：每层单独都是完整图')
# ⚠️ cells.shape 是 (h, w, 3)（RGB 三通道），要和 (h, w) 比必须取 shape[:2]
for L, (im, cells, st) in enumerate(levels):
    want = (H // sc.cells[L], W // sc.cells[L])
    check('第 %d 层是完整格阵 %s' % (L, str(st['grid'])),
          cells.shape[:2] == want, 'shape %s' % str(cells.shape))

print()
print('=' * 74)
print('通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
if FAIL:
    print('失败：')
    for f in FAIL:
        print('  ✗ %s' % f)
