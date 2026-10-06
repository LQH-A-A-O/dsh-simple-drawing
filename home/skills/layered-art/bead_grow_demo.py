#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""渐进生长式拼豆 · 示范：一幅山水画。

正是用户描述的那个例子：
    第 0 层：天空纯蓝、太阳橙红、山深绿、水蓝   —— 几块大色块
    往上：天空由深到浅、太阳有光晕、山露出岩石、水面有波纹

跑法：
    python bead_grow_demo.py            # 出 4 层图 + 每层的豆子清单
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bead_grow import (BeadScene, fbm_octaves, detail_vgrad, detail_radial,   # noqa: E402
                       detail_hband, mask_circle, mask_below_curve, _norm01)

OUT = os.environ.get('BEAD_OUT') or os.path.join(os.getcwd(), 'out')
SIZE = (56, 56)          # 最终 56x56 格（约等于两块 29x29 拼豆板）
LEVELS = 4               # 层数：7x7 -> 14x14 -> 28x28 -> 56x56
os.makedirs(OUT, exist_ok=True)

W, H = SIZE
shape = (H, W)

print('=' * 74)
print('渐进生长式拼豆 · 山水画')
print('画布 %dx%d 格   %d 层' % (W, H, LEVELS))
print('=' * 74)

# ================================================================ ① 提示词机制
# 对画面的概括。这里用几何图元写死；将来接"提示词 -> 区域"的模型时，
# 换掉的只是这一段，下面的染色与生长逻辑完全不用动。
print()
print('【① 提示词机制】画面的语义分区')

# 山的轮廓：一条起伏的曲线
xs = np.linspace(0, 1, W)
ridge = 0.50 + 0.10 * np.sin(xs * np.pi * 1.7 + 0.6) + \
    0.05 * np.sin(xs * np.pi * 4.3 + 1.9)
far_ridge = 0.42 + 0.07 * np.sin(xs * np.pi * 2.3 + 2.4)

m_sky = ~mask_below_curve(shape, far_ridge)                     # 天空在远山之上
m_far = mask_below_curve(shape, far_ridge) & ~mask_below_curve(shape, ridge)
m_near = mask_below_curve(shape, ridge) & ~mask_below_curve(shape, np.full(W, 0.78))
m_water = mask_below_curve(shape, np.full(W, 0.78))
m_sun = mask_circle(shape, 0.74, 0.20, 0.075)
# 太阳盖在天空上
m_sky = m_sky & ~m_sun

print('   天空 %d 格   远山 %d 格   近山 %d 格   水面 %d 格   太阳 %d 格'
      % (m_sky.sum(), m_far.sum(), m_near.sum(), m_water.sum(), m_sun.sum()))

# ================================================================ ② 染色机制
# 每个区域一个**固定调色板**。细节只决定在这一区里取哪个色。
print()
print('【② 染色机制】每个区域的固定调色板')

SKY = [(28, 52, 110), (46, 82, 156), (72, 118, 190), (108, 156, 214),
       (156, 196, 234), (204, 226, 246)]
FAR = [(38, 52, 62), (54, 72, 80), (74, 96, 100), (104, 126, 126)]
NEAR = [(24, 44, 32), (38, 66, 42), (56, 88, 52), (84, 116, 64),
        (118, 142, 84), (96, 88, 72), (140, 130, 112)]
WATER = [(16, 42, 88), (28, 62, 118), (44, 88, 148), (72, 120, 176),
         (112, 158, 202)]
SUN = [(255, 108, 40), (255, 148, 48), (255, 186, 62), (255, 216, 104),
       (255, 238, 168)]
for nm, pal in (('天空', SKY), ('远山', FAR), ('近山', NEAR),
                ('水面', WATER), ('太阳', SUN)):
    print('   %-4s %2d 色  %s' % (nm, len(pal),
                                 ' '.join('#%02X%02X%02X' % c for c in pal[:4])
                                 + (' ...' if len(pal) > 4 else '')))

# 细节场：拆成倍频程，分层累加
oct_sky = fbm_octaves(shape, seed=11, n_oct=LEVELS)
oct_near = fbm_octaves(shape, seed=27, n_oct=LEVELS)
oct_far = fbm_octaves(shape, seed=33, n_oct=LEVELS)
oct_water = fbm_octaves(shape, seed=41, n_oct=LEVELS)

# 天空：上深下浅（竖直渐变）混一点云
d_sky = np.clip(0.72 * detail_vgrad(shape, 0.12, 0.95) +
                0.28 * _norm01(oct_sky[0]), 0, 1)
# 远山：整体偏暗，只有轻微起伏
d_far = np.clip(0.30 + 0.35 * _norm01(oct_far[0]), 0, 1)
# 近山：完整的噪声细节（岩石与植被交错）
d_near = np.clip(0.15 + 0.85 * _norm01(oct_near[0]), 0, 1)
# 水面：近处亮远处暗 + 波纹
d_water = np.clip(0.25 * detail_vgrad(shape, 0.0, 1.0) +
                  0.45 * detail_hband(shape, freq=11.0) +
                  0.30 * _norm01(oct_water[0]), 0, 1)
# 太阳：中心亮边缘红
d_sun = detail_radial(shape, 0.74, 0.20, 0.085)

# ================================================================ ③ 生长
print()
print('【③ 生长】从白画布逐层加色')
sc = BeadScene(size=SIZE, levels=LEVELS, bg=(252, 252, 250), seed=0)
sc.add('sky', m_sky, SKY, detail=d_sky, octaves=oct_sky, base_index=2)
sc.add('far', m_far, FAR, detail=d_far, base_index=1)
sc.add('near', m_near, NEAR, detail=d_near, octaves=oct_near, base_index=1)
sc.add('water', m_water, WATER, detail=d_water, octaves=oct_water, base_index=1)
sc.add('sun', m_sun, SUN, detail=d_sun, base_index=None)

levels = sc.grow()
print()
print('%-6s %-12s %-9s %-9s %s' % ('层', '格阵', '格尺寸', '豆子数', '用色数'))
for L, (im, cells, st) in enumerate(levels):
    print('%-6d %-12s %-9d %-9d %d'
          % (L, '%dx%d' % st['grid'], st['cell'], st['total_beads'], st['n_colors']))

# ================================================================ ④ 输出
# 每层单独存 + 一张拼在一起的对照图
tiles = []
for L, (im, cells, st) in enumerate(levels):
    p = os.path.join(OUT, 'landscape_L%d.png' % L)
    im.save(p)
    tiles.append(im)

# 每层放大到同一高度做对比
TH = 420
row = []
for im in tiles:
    k = TH / im.height
    row.append(im.resize((max(1, int(im.width * k)), TH), Image.NEAREST))
pad = 16
cw = sum(t.width for t in row) + pad * (len(row) + 1)
canvas = Image.new('RGB', (cw, TH + 46), (24, 24, 28))
d = ImageDraw.Draw(canvas)
x = pad
for L, t in enumerate(row):
    canvas.paste(t, (x, 36))
    d.text((x + 4, 12), 'L%d  %dx%d  %d色  %d豆'
           % (L, levels[L][2]['grid'][0], levels[L][2]['grid'][1],
              levels[L][2]['n_colors'], levels[L][2]['total_beads']),
           fill=(220, 226, 236))
    x += t.width + pad
canvas.save(os.path.join(OUT, 'landscape_all_levels.png'))
print()
print('-> %s' % os.path.join(OUT, 'landscape_all_levels.png'))

# 刚需：最细那层的豆子清单
top = levels[-1][2]
print()
print('【④ 最细层（%dx%d）豆子清单 —— 直接照着买】'
      % (top['grid'][0], top['grid'][1]))
print('   共 %d 颗，%d 种颜色' % (top['total_beads'], top['n_colors']))
for c in top['colors'][:14]:
    print('   %-9s %5d 颗  %s' % (c['hex'], c['n'], '#' * int(c['n'] / top['colors'][0]['n'] * 40)))
if len(top['colors']) > 14:
    print('   ... 另有 %d 种' % (len(top['colors']) - 14))
with open(os.path.join(OUT, 'shopping_list.txt'), 'w', encoding='utf-8') as f:
    for L, (_, _, st) in enumerate(levels):
        f.write('# 第 %d 层  %dx%d  %d 颗  %d 色\n'
                % (L, st['grid'][0], st['grid'][1], st['total_beads'], st['n_colors']))
        for c in st['colors']:
            f.write('%s\t%d\n' % (c['hex'], c['n']))
        f.write('\n')
print()
print('-> %s' % os.path.join(OUT, 'shopping_list.txt'))
