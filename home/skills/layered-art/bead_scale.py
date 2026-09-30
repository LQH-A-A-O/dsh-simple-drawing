# -*- coding: utf-8 -*-
"""7 头身 MC 体型：**不同头高能做多细的脸**？

背景
----
MC 玩家模型是 **4 头身**（腿12+身12+头8=32px，头占 8px -> 32/8 = 4.0），
不是 7 头身。所以"日漫 7 头身 + MC 体型"必须把两件事拆开：

    保留  MC 的**构造**：每个部位都是轴对齐方盒，零曲线；
          宽度方案：身宽 = 1 头宽，手臂各 0.5 头宽，两腿合起来 1 头宽。
    换掉  MC 的**纵向比例**：躯干和腿拉长，凑到 7 头。

本脚本把同一张"MC 方盒 + 7 头身"的角色在 **头 = 8 / 12 / 16 / 20 格** 下各渲一遍，
直接把"能画多细"摆出来，而不是靠猜。
另外附带 MC 真实比例的 4 头身做对照。
"""
import os


def _out_dir(name):
    """输出目录：环境变量 LAYERED_ART_OUT 优先，否则落在系统临时目录。

    ⚠️ 不要在这里写死自己机器的路径 —— 打包分发时会被拒绝，
       别人机器上也不存在。
    """
    import os
    import tempfile
    base = os.environ.get('LAYERED_ART_OUT') or os.path.join(
        tempfile.gettempdir(), 'layered-art')
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    return d


import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 不写绝对路径，打包才过得了净化检查
from bead import Grid                     # noqa: E402

OUT = _out_dir('rig')
os.makedirs(OUT, exist_ok=True)

CELL = 2
PAL = {
    'hair': (32, 30, 42, 255), 'hair2': (66, 62, 86, 255),
    'skin': (242, 222, 208, 255), 'skin2': (214, 186, 170, 255),
    'blouse': (238, 243, 252, 255), 'navy': (38, 48, 84, 255),
    'navy2': (28, 36, 64, 255), 'red': (190, 60, 74, 255),
    'shoe': (34, 34, 42, 255), 'eye': (30, 32, 48, 255),
    'hilite': (248, 252, 255, 255), 'mouth': (198, 116, 116, 255),
    'blush': (244, 190, 188, 255), 'bg': (248, 250, 253, 255),
    'line': (24, 26, 36, 255),
}

# 7 头身（头高 = 1 单位）各关节高度。来自 CANON7。
LEV7 = {'头顶': 7.0, '下巴': 6.0, '肩': 5.5, '腰': 4.0, '胯': 3.5,
        '膝': 2.0, '脚踝': 0.30}
# MC 真实比例（4 头身）：头 8 / 身 12 / 腿 12，合计 32
LEV4 = {'头顶': 4.0, '下巴': 3.0, '肩': 2.75, '腰': 2.0, '胯': 1.5,
        '膝': 0.75, '脚踝': 0.15}


def build(H, levels, head_count, palette_face=True):
    """H = 头占多少格。返回 (Grid, 行表)。"""
    fw = 2 * (H + 4)                       # 画布宽（格），保证偶数
    fh = int(round(head_count * H)) + 2 * 4
    if fh % 2:
        fh += 1
    G = Grid((fw * CELL, fh * CELL), cell=CELL)
    assert G.cw == fw and G.ch == fh
    A = G.axis
    CX = A - 0.5
    sole = fh - 4
    R = {k: int(round(sole - v * H)) for k, v in levels.items()}
    G.a[:, :] = PAL['bg']

    half_body = H / 2.0                    # 身宽 = 1 头宽 -> 半宽 0.5H
    half_arm = H / 4.0                     # 手臂各 0.5 头宽
    half_leg = H / 4.0                     # 两腿合起来 1 头宽

    def rect_axis(hw, y0, y1, c):
        """以中轴为中心、左右各 hw 格的矩形（**偶数格宽，天然对称**）。"""
        G.rect(int(round(A - hw)), y0, int(round(A + hw)) - 1, y1, c)

    # ---- 腿（两根方柱）
    for s in (-1, 1):
        x0 = int(round(CX + s * half_leg - half_leg))
        x1 = int(round(CX + s * half_leg + half_leg)) - 1
        G.rect(x0, R['胯'], x1, R['脚踝'], PAL['skin'])
        G.rect(x0, R['脚踝'], x1, sole - 1, PAL['shoe'])
    # ---- 躯干（方盒；MC 里身宽正好 = 头宽）
    rect_axis(half_body, R['肩'], R['胯'] + 1, PAL['blouse'])
    # ---- 手臂（方柱，垂在身侧）
    for s in (-1, 1):
        xa = A + s * (half_body + half_arm)
        x0 = int(round(xa - half_arm))
        x1 = int(round(xa + half_arm)) - 1
        G.rect(x0, R['肩'] + 1, x1, R['腰'] + 2, PAL['blouse'])
        G.rect(x0, R['腰'] + 3, x1, R['胯'] + 1, PAL['skin'])
    # ---- 脖子 + 水手领 + 领结
    rect_axis(max(1, H / 8.0), R['下巴'], R['肩'], PAL['skin2'])
    rect_axis(half_body, R['肩'], R['肩'] + max(1, H // 6), PAL['navy'])
    rect_axis(max(1, H / 8.0), R['肩'] + max(1, H // 6) + 1,
              R['肩'] + max(1, H // 6) + max(1, H // 8), PAL['red'])
    # ---- 裙子（方盒，下摆到膝上）
    hem = R['胯'] + int(round((R['膝'] - R['胯']) * 0.55))
    rect_axis(half_body + max(1, H // 8), R['胯'], hem, PAL['navy2'])
    for i in range(1, 4):
        x = int(round(A - (half_body + H // 8) + (2 * (half_body + H // 8)) * i / 4.0))
        G.rect(x, R['胯'], x, hem, PAL['navy'])

    # ---- 头（方盒）—— MC 的头就是方的，这里保留
    hy0, hy1 = R['头顶'], R['下巴']        # 注意：头顶是 y 小的那端
    G.rect(int(round(A - H / 2.0)), hy0, int(round(A + H / 2.0)) - 1, hy1, PAL['skin'])
    # 头发：顶 + 两侧（保留方盒感，不做圆弧）
    G.rect(int(round(A - H / 2.0)), hy0, int(round(A + H / 2.0)) - 1,
           hy0 + max(1, H // 4), PAL['hair'])
    for s in (-1, 1):
        x = int(round(A + s * (H / 2.0 - max(1, H // 8)))) - (1 if s < 0 else 0)
        G.rect(x, hy0, x, hy1 - max(1, H // 4), PAL['hair'])

    if palette_face:
        draw_face(G, H, A, R, hy0, hy1)
    return G, R, A


def draw_face(G, H, A, R, hy0, hy1):
    """按 H 决定能画到什么程度的脸。**只画左眼，最后整体镜像**。"""
    face_top = hy0 + max(1, H // 4)
    face_bot = hy1
    mid = (face_top + face_bot) // 2
    # 眼睛尺寸随 H 增长
    ew = 2 if H < 16 else 3
    eh = 2 if H < 12 else (2 if H < 20 else 3)
    gap = 1 if H < 16 else 2                     # 两眼中缝
    ex1 = A - (gap + 1) // 2 - 1                 # 左眼右边界（col）
    ex0 = ex1 - ew + 1
    ey = mid - eh // 2
    G.rect(ex0, ey, ex1, ey + eh - 1, PAL['eye'])
    if H >= 12:                                   # 高光
        G.rect(ex1, ey, ex1, ey, PAL['hilite'])
    if H >= 16:                                   # 瞳孔
        G.rect(ex0, ey + eh - 1, ex0, ey + eh - 1, PAL['hilite'])

    # 嘴（跨中轴，天然对称）
    mw = 1 if H < 12 else 2
    my = face_bot - max(1, H // 6)
    G.rect(A - mw, my, A + mw - 1, my, PAL['mouth'])
    # 腮红
    if H >= 12:
        by = my - 1
        G.rect(A - gap - ew - 1, by, A - gap - ew - 1, by + 1, PAL['blush'])
    # 眉毛
    if H >= 16:
        G.rect(ex0, ey - 2, ex1, ey - 2, PAL['hair2'])
    # 头发高光（斜向一道，用方块条纹保持 MC 感）
    if H >= 16:
        for i in range(3):
            G.rect(int(round(A - H / 2.0)) + 1 + i, hy0 + 1 + i,
                   int(round(A - H / 2.0)) + 2 + i, hy0 + 1 + i, PAL['hair2'])


def finish(G, asym=True):
    """对称件画完 -> 整体镜像 -> 再画非对称件。"""
    G.mirror_left_to_right()
    if asym:
        # 非对称件放镜像之后：左侧发夹
        A = G.axis
        G.rect(A - int(A * 0.55), 0, A - int(A * 0.55), 0, PAL['bg'])   # 占位，防越界
    return G


def sym_ok(G, A):
    L = G.a[:, :A]
    Rr = G.a[:, A:][:, ::-1]
    return bool(np.array_equal(L, Rr))


def tile(H, levels, head_count, label):
    G, R, A = build(H, levels, head_count)
    G.mirror_left_to_right()
    ok = sym_ok(G, A)
    img = G.to_image()
    # 脸的特写
    hx0 = int(round(A - H / 2.0)) - 1
    hx1 = int(round(A + H / 2.0))
    face = G.zoom((hx0, R['头顶'] - 1, hx1, R['下巴'] + 1), k=max(4, 120 // H))
    return img, face, ok, G


cases = [
    (8, LEV7, 7.0, '7 头身 · 头 8 格 (16px)'),
    (12, LEV7, 7.0, '7 头身 · 头 12 格 (24px)'),
    (16, LEV7, 7.0, '7 头身 · 头 16 格 (32px)'),
    (20, LEV7, 7.0, '7 头身 · 头 20 格 (40px)'),
    (8, LEV4, 4.0, '对照：MC 真实 4 头身 · 头 8 格'),
]

rows = []
print('%-30s %-12s %-10s %-8s %s' % ('方案', '画布(px)', '格阵(格)', '头(px)', '左右对称'))
for H, lv, n, label in cases:
    img, face, ok, G = tile(H, lv, n, label)
    rows.append((label, img, face, G))
    print('%-30s %-12s %-10s %-8d %s'
          % (label, '%dx%d' % G.px, '%dx%d' % (G.cw, G.ch), H * CELL,
             '是 ✓' if ok else '否 ✗'))

# ---------------- 排版：上半全身，下半脸特写
pad = 12
tw = max(im.width for _, im, _, _ in rows)
th = max(im.height for _, im, _, _ in rows)
k = max(1, 260 // th)
w = tw * k
h = th * k
face_h = max(f.height for _, _, f, _ in rows)
sheet_w = sum(max(w, f.width) + pad for _, _, f, _ in rows) + pad
sheet = Image.new('RGB', (sheet_w, h + face_h + 60), (22, 24, 28))
d = ImageDraw.Draw(sheet)
x = pad
for label, img, face, G in rows:
    up = img.resize((img.width * k, img.height * k), Image.NEAREST)
    sheet.paste(up, (x, 34))
    sheet.paste(face.convert('RGB'), (x, h + 52))
    d.text((x + 2, 12), label, fill=(200, 210, 225))
    d.text((x + 2, h + 36), '脸特写 + 格线', fill=(150, 165, 185))
    x += max(w, face.width) + pad
sheet.save(os.path.join(OUT, '头身比_能做到什么程度.png'))
print('\n对照图 ->', os.path.join(OUT, '头身比_能做到什么程度.png'), sheet.size)
