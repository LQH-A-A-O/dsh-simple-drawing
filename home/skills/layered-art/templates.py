# -*- coding: utf-8 -*-
"""两套独立模板：**日漫 7 头身** 和 **MC 4 头身**

用户要的是"分别来，不要混在一起"。所以这里两个构建器各按各的正确画法：

    anime_7head   头 16 格/32px，7 头身，**锥形躯干 + 胶囊四肢**（日漫身段）
    mc_4head      头  8 格/16px，**4 头身**，**轴对齐方盒**（MC 真实比例）
                  MC 玩家模型：头 8×8、身 8×12、臂 4×12、腿 4×12 -> 高 32 宽 16

两者共用同一套格阵与"画左半 + 整体镜像"流程，所以：
  · 都满足 cell 整除、偶数列、中轴在格线上
  · 都能逐个部件精修 + 镜像
  · 都能参数化改头身比/肩宽（各自的取值域不同）

⚠️ 4 头身的 MC 头只有 8 格，脸**只能**画两个色块当眼睛 + 一条嘴。
   这不是实现不好，是 MC 尺寸的物理上限 —— 真 MC 皮肤也是这样。
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
from bead import Grid, snap_even            # noqa: E402

OUT = _out_dir('rig')
os.makedirs(OUT, exist_ok=True)

CELL = 2
P = {
    'hair': (34, 32, 44, 255), 'hair2': (70, 66, 92, 255),
    'hair3': (120, 116, 140, 255),
    'skin': (242, 222, 208, 255), 'skin2': (212, 182, 166, 255),
    'blouse': (238, 243, 252, 255), 'blouse2': (200, 212, 232, 255),
    'navy': (38, 48, 84, 255), 'navy2': (26, 34, 62, 255),
    'red': (190, 60, 74, 255), 'shoe': (34, 34, 42, 255),
    'eye': (30, 32, 48, 255), 'hilite': (248, 252, 255, 255),
    'mouth': (198, 116, 116, 255), 'blush': (246, 192, 190, 255),
    'bg': (248, 250, 253, 255),
}


def newgrid(head_cells, head_count, side_heads):
    """画布宽 = 2*(头格 + 余量)，高 = 头身比*头格 + 上下留白。全部保证偶数格。"""
    fw = 2 * (head_cells + 4)
    fh = int(round(head_count * head_cells)) + 8
    if fh % 2:
        fh += 1
    G = Grid((fw * CELL, fh * CELL), cell=CELL)
    A = G.axis
    G.a[:, :] = P['bg']
    return G, A, fw, fh


# ================================================================ 7 头身（日漫）
def build_anime(head_cells=16, head_count=7.0, face_detail=True):
    H = head_cells
    G, A, fw, fh = newgrid(H, head_count, 1.5)
    CX = A - 0.5
    sole = fh - 4
    L = {'头顶': 7.0, '下巴': 6.0, '肩': 5.5, '腰': 4.0, '胯': 3.5,
         '膝': 2.0, '脚踝': 0.30}
    R = {k: int(round(sole - v * H)) for k, v in L.items()}
    half_body, half_arm = H * 0.42, H * 0.11          # 日漫身段：肩窄
    half_leg = H * 0.19

    # 腿（胶囊，有粗细变化）
    for s in (-1, 1):
        cx = CX + s * H * 0.20
        G.capsule(cx, R['胯'] + 1, cx + s * 0.5, R['膝'], H * 0.185, P['skin'])
        G.capsule(cx + s * 0.5, R['膝'], cx + s * 0.7, R['脚踝'], H * 0.135, P['skin'])
        G.rect(int(round(cx - H * 0.14)), R['脚踝'], int(round(cx + H * 0.13)),
               sole - 1, P['shoe'])
    # 躯干（锥形：肩宽 -> 腰细 -> 胯略宽）—— 日漫身段的关键
    for y in range(R['肩'], R['胯'] + 2):
        t = (y - R['肩']) / max(1, R['胯'] + 1 - R['肩'])
        w = (half_body * (1 - t) + H * 0.30 * t)
        if t > 0.75:                                   # 胯部再放开一点
            w += H * 0.05 * ((t - 0.75) / 0.25)
        G.rect(int(round(A - w)), y, int(round(A + w)) - 1, y, P['blouse'])
    # 手臂（上臂白袖，前臂皮肤）
    for s in (-1, 1):
        x0 = A + s * (half_body + half_arm)
        G.capsule(x0, R['肩'] + 2, x0 + s * H * 0.06, R['腰'] + 1, H * 0.115, P['blouse'])
        G.capsule(x0 + s * H * 0.06, R['腰'] + 1, x0 + s * H * 0.10, R['胯'],
                  H * 0.095, P['skin'])
    # 裙子（A 字，向下一路放宽）
    hem = R['胯'] + int(round((R['膝'] - R['胯']) * 0.45))
    for y in range(R['胯'], hem + 1):
        t = (y - R['胯']) / max(1, hem - R['胯'])
        w = H * 0.44 + H * 0.16 * t
        G.rect(int(round(A - w)), y, int(round(A + w)) - 1, y, P['navy2'])
    for i in range(1, 5):                              # 褶
        x = int(round(A - (H * 0.60) + (H * 1.20) * i / 5.0))
        G.rect(x, R['胯'] + 1, x, hem, P['navy'])
    # 脖子 + 水手领 + 领结
    G.rect(int(round(A - H * 0.11)), R['下巴'], int(round(A + H * 0.11)) - 1,
           R['肩'] + 1, P['skin2'])
    G.rect(int(round(A - half_body)), R['肩'], int(round(A + half_body)) - 1,
           R['肩'] + 2, P['navy'])
    G.rect(A - 1, R['肩'] + 3, A, R['肩'] + 4, P['red'])

    # ---- 头（**圆**，不是方块 —— 7 头身走日漫）
    ch = (R['头顶'] + R['下巴']) / 2.0
    hr = H / 2.0
    # ⚠️ 头发不能画成"和头一样大的整圆" —— 那样脸缩在中间，
    #    外面剩一圈黑环，看着像头盔。头发要**上移 + 上厚下薄**。
    G.ellipse(CX, ch - H * 0.10, hr, hr * 0.98, P['hair'])          # 头发整体上移
    G.rect(int(round(CX - hr)), R['下巴'] - 1, int(round(CX + hr)),
           R['下巴'] - 1, P['bg'])                                   # 下巴以下不留头发
    # 后发（垂到肩）
    for s in (-1, 1):
        x = int(round(CX + s * (hr - 1)))
        G.rect(x, R['头顶'] + 3, x, R['肩'] - 1, P['hair'])
    # 脸（**下移 + 扁一点**，让额头留出头发）
    G.ellipse(CX, ch + H * 0.04, hr - H * 0.24, hr - H * 0.28, P['skin'])
    # 发丝高光：**必须在镜像之前画**。留在镜像之后会只出现在左半边，
    # 整个角色就不对称了（断言会直接报出来）。
    if face_detail and H >= 16:
        for i in range(3):
            G.rect(int(round(CX - hr)) + 1 + i, R['头顶'] + 1 + i,
                   int(round(CX - hr)) + 2 + i, R['头顶'] + 1 + i, P['hair3'])

    # ⚠️ 描边的正确顺序：**先描边，再画脸部细节**。
    #    盲描（按颜色差找边界）会把眼、嘴、腮红也描黑 —— 整张脸糊成一团（实测踩过）。
    #    正解是让细节后画，落在描边之上。
    G.mirror_left_to_right()
    G.add_outline(k=0.72, bg=P['bg'])

    if face_detail:
        fy0, fy1 = R['头顶'], R['下巴']
        ew = 3 if H >= 16 else 2
        eh = 3 if H >= 18 else 2
        gap = 2 if H >= 16 else 1
        ex1 = A - (gap + 1) // 2 - 1
        ex0 = ex1 - ew + 1
        ey = int(round(ch)) + (1 if H >= 16 else 0)
        G.rect(ex0, ey, ex1, ey + eh - 1, P['eye'])
        G.rect(ex1, ey, ex1, ey, P['hilite'])                  # 高光
        G.rect(ex0, ey + eh - 1, ex0, ey + eh - 1, P['hilite'])
        G.mirror_cols(ex0, ex1)                                 # 镜像出右眼
        if H >= 16:
            G.rect(ex0, ey - 2, ex1, ey - 2, P['hair2'])       # 眉
            G.mirror_cols(ex0, ex1)
        my = fy1 - max(1, H // 7)
        G.rect(A - 1, my, A, my, P['mouth'])
        bx = A - gap - ew - 2
        G.rect(bx, my, bx, my + 1, P['blush'])
        G.mirror_cols(bx, bx)
    return G, R, A


# ================================================================ MC 4 头身（方盒）
def build_mc(head_cells=8, face_detail=True):
    """**严格按 MC 玩家模型的尺寸**：头 8×8、身 8×12、臂 4×12、腿 4×12。"""
    H = head_cells
    G, A, fw, fh = newgrid(H, 4.0, 2.0)
    CX = A - 0.5
    # 注意这里**不用 CANON7** —— MC 的比例就是 12/8/8，不是解剖比例
    leg_h, bod_h = 12 * H // 8, 12 * H // 8
    sole = fh - 4
    R = {'脚底': sole, '脚踝': sole - leg_h, '胯': sole - leg_h,
         '肩': sole - leg_h - bod_h, '下巴': sole - leg_h - bod_h,
         '头顶': sole - leg_h - bod_h - H}
    hw_h, hw_b = H // 2, (8 * H // 8) // 2                 # 头半宽 / 身半宽（都是 8 格宽）
    aw = 4 * H // 8                                        # 臂宽（4 格）
    # 腿：两条 4 宽方柱
    # ⚠️ 平面渲染没有立体阴影，两条同色腿会**糊成一块**。
    #    必须自己画一条 1 格的分隔缝（立体渲染里这缝是自然出现的）。
    for s in (-1, 1):
        x0 = int(round(A + s * aw - aw)) if s > 0 else int(round(A - aw))
        x1 = x0 + aw - 1
        G.rect(x0, R['胯'], x1, R['脚踝'] - 1, P['navy2'])
        G.rect(x0, R['脚踝'] - 3, x1, R['脚底'] - 1, P['shoe'])      # 鞋用可区分的灰
    G.rect(A - 1, R['胯'], A, R['脚踝'] - 4, P['navy'])              # 腿缝
    # 躯干：8 宽方柱
    G.rect(A - hw_b, R['肩'], A + hw_b - 1, R['胯'] - 1, P['blouse'])
    G.rect(A - hw_b, R['肩'], A + hw_b - 1, R['肩'] + H // 4, P['navy'])   # 领
    G.rect(A - 1, R['肩'] + H // 4 + 1, A, R['肩'] + H // 4 + 2, P['red'])  # 领结
    # 手臂：两根 4 宽方柱，贴在外侧
    for s in (-1, 1):
        x0 = (A + s * (hw_b + aw)) - aw if s > 0 else (A - hw_b - aw)
        x1 = x0 + aw - 1
        G.rect(x0, R['肩'], x1, R['肩'] + bod_h // 2, P['blouse'])
        G.rect(x0, R['肩'] + bod_h // 2 + 1, x1, R['胯'] - 1, P['skin'])
    # 头：8×8 方块（MC 的头就是方的）
    G.rect(A - hw_h, R['头顶'], A + hw_h - 1, R['下巴'] - 1, P['skin'])
    G.rect(A - hw_h, R['头顶'], A + hw_h - 1, R['头顶'] + H // 4, P['hair'])
    for s in (-1, 1):                                       # 两侧头发
        x = A + s * hw_h - (1 if s < 0 else 0)
        G.rect(x, R['头顶'], x, R['下巴'] - H // 4, P['hair'])

    # 同样：先描边，再画脸
    G.mirror_left_to_right()
    G.add_outline(k=0.72, bg=P['bg'])

    if face_detail:
        # 4 头身的头只有 8 格 —— 眼睛最多 2 格，再没有余量
        ey = R['头顶'] + H // 2
        ex1 = A - 2
        ex0 = ex1 - 1
        G.rect(ex0, ey, ex1, ey + 1, P['eye'])
        G.rect(ex1, ey, ex1, ey, P['hilite'])
        G.mirror_cols(ex0, ex1)
        my = R['下巴'] - 2
        G.rect(A - 1, my, A, my, P['mouth'])
    return G, R, A


def sym_ok(G):
    A = G.axis
    return bool(np.array_equal(G.a[:, :A], G.a[:, A:][:, ::-1]))


if __name__ == '__main__':
    import math
    out = []
    for name, fn, hc, label in (
            ('7 头身（日漫）', build_anime, 16, 'anime'),
            ('MC 4 头身（方盒）', build_mc, 8, 'mc')):
        G, R, A = fn()
        ok = sym_ok(G)
        img = G.to_image()
        # 脸特写
        fx0, fx1 = A - (16 if hc == 16 else 8), A + (15 if hc == 16 else 7)
        fy0 = max(0, R['头顶'] - 1)
        fy1 = min(G.ch - 1, R['下巴'] + 1)
        face = G.zoom((fx0, fy0, fx1, fy1), k=max(6, 200 // (fx1 - fx0 + 1)))
        out.append((name, img, face, G, ok))
        print('%-18s 画布 %-9s 格阵 %-9s 头 %2d 格 / %2dpx  左右对称 %s'
              % (name, '%dx%d' % G.px, '%dx%d' % (G.cw, G.ch), hc, hc * CELL,
                 '是 ✓' if ok else '否 ✗'))

    # 排版：各自按显示高度 420 等比放大，脸特写放下面
    pad, DH = 16, 420
    cols = []
    for name, img, face, G, ok in out:
        k = max(1, DH // img.height)
        side = img.resize((img.width * k, img.height * k), Image.NEAREST)
        cols.append((name, side, face, G))
    W = sum(max(c[1].width, c[2].width) + pad for c in cols) + pad
    fh_ = max(c[2].height for c in cols)
    sheet = Image.new('RGB', (W, DH + fh_ + 74), (22, 24, 28))
    d = ImageDraw.Draw(sheet)
    x = pad
    for name, side, face, G in cols:
        sheet.paste(side, (x, 36))
        sheet.paste(face.convert('RGB'), (x, DH + 62))
        d.text((x + 2, 12), name, fill=(210, 220, 235))
        d.text((x + 2, DH + 42), '脸特写 · 头 %d 格' % (G.ch and 0 or 0) or
               '脸特写 + 格线', fill=(150, 165, 185))
        d.text((x + 2, DH + 20), '%dx%d px' % G.px, fill=(140, 155, 175))
        x += max(side.width, face.width) + pad
    sheet.save(os.path.join(OUT, '两套比例_分别.png'))
    print('\n->', os.path.join(OUT, '两套比例_分别.png'), sheet.size)
