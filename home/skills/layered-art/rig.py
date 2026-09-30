# -*- coding: utf-8 -*-
"""赛博拼豆：二维骨架模板（形变 + 姿势 + 服装吸附）

单位约定（这是本模块的地基，改它等于改一切）
--------------------------------------------
**头高 = 1.0**。不是像素、不是身高比例。

    y：脚底 = 0，头顶 = head_count，下巴 = head_count - 1
    x：以中轴为 0，右为正
    横向量（肩宽/腰宽/肢体粗细）同样以**头高**为单位

为什么不用"身高比例"：一改头身比，身高比例就得全部重算；
用头高做单位时，头永远占 1 格，只有下巴以下重新分配 —— 这才是
"6 头身 / 7 头身"的真实语义。

三处对原始设计稿的修正（都在下面代码里标了 ⚠️）
------------------------------------------------
1. **锚点不用绝对像素**，用"沿某根骨头走百分之几"。否则改头身比时
   裙摆会悬空，而且不报错。
2. **每根骨头带 depth**，否则二维骨架表达不了"胳膊在身体前面"，
   手插腰/抱臂全做不了。
3. **形变不是整体缩放**，下巴以下按 (n-1)/(base-1) 重分布。
"""
import math


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


import os

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# 模板：全部以"头高"为单位。数值由原始设计稿的像素坐标归一化得来
# （原稿 128×192、头高 22px、脚底 y≈168），并**规整到自洽的 7 头身**：
# 原稿头顶 7.23 头高、下巴不在 6.0 上，不自洽，这里统一成顶=7、下巴=6。
# ---------------------------------------------------------------------------
BASE_HEAD_COUNT = 7.0

TEMPLATE = {
    'name': 'female_standing',
    'head_count': BASE_HEAD_COUNT,
    'joints': {
        'head_c':   (0.000, 6.500),
        'head_r':   0.500,
        'neck':     (0.000, 5.944),
        'chest':    (0.000, 4.976),
        'waist':    (0.000, 3.963),
        'shoulder': {'L': (-0.793, 5.592), 'R': (0.793, 5.592)},
        'elbow':    {'L': (-1.057, 4.491), 'R': (1.057, 4.491)},
        'wrist':    {'L': (-1.145, 3.435), 'R': (1.145, 3.435)},
        'hip':      {'L': (-0.528, 3.214), 'R': (0.528, 3.214)},
        'knee':     {'L': (-0.617, 1.761), 'R': (0.617, 1.761)},
        'ankle':    {'L': (-0.661, 0.264), 'R': (0.661, 0.264)},
    },
    # 各段粗细（半径，头高为单位）
    'radius': {'neck': 0.170, 'upper_arm': 0.160, 'fore_arm': 0.135,
               'thigh': 0.265, 'shin': 0.205, 'foot': 0.150},
    # 躯干轮廓（半宽，头高为单位），按高度从下往上
    'torso': [(3.214, 0.560), (3.963, 0.520), (4.976, 0.660), (5.592, 0.760)],
    # ⚠️ 修正 2：每根骨头的深度。数值越大越靠前（后画）。
    #    没有这个就做不了遮挡，手插腰会变成"手在身体里"。
    'depth': {
        'leg_L': 10, 'leg_R': 10,
        'torso': 20,
        'arm_R': 40,      # 右臂在躯干之前
        'arm_L': 5,       # 左臂在躯干之后 -> 能做出"手插腰/抱臂"
        'head': 30,
        'hair_back': 1,
    },
}

# ⚠️ 修正 1：服装锚点 = (骨头甲, 骨头乙, 沿路比例)，**不是绝对像素**。
#    甲→乙 是方向；比例 0 = 甲，1 = 乙。改头身比 / 摆姿势后自动跟着走。
ANCHORS = {
    'collar':      ('neck', 'chest', 0.18),
    'shoulder_L':  ('neck', 'shoulder_L', 1.0),
    'shoulder_R':  ('neck', 'shoulder_R', 1.0),
    'chest':       ('neck', 'chest', 0.85),
    'waist':       ('chest', 'waist', 1.0),
    'hip_L':       ('waist', 'hip_L', 1.0),
    'hip_R':       ('waist', 'hip_R', 1.0),
    'hem_L':       ('hip_L', 'knee_L', 0.62),      # 百褶裙下摆
    'hem_R':       ('hip_R', 'knee_R', 0.62),
    'cuff_L':      ('elbow_L', 'wrist_L', 0.78),   # 袖口
    'cuff_R':      ('elbow_R', 'wrist_R', 0.78),
}


class Rig:
    """一套可以形变、摆姿势、吸附服装的二维骨架。"""

    def __init__(self, tpl=None):
        import copy
        self.t = copy.deepcopy(tpl or TEMPLATE)

    # ---------------- 取值
    def J(self, name, side=None):
        v = self.t['joints'][name]
        if isinstance(v, dict):
            if side is None:
                raise KeyError('%s 需要 side' % name)
            return v[side]
        return v

    def anchor(self, name):
        a = ANCHORS[name]
        p0 = self._resolve_bone_point(a[0])
        p1 = self._resolve_bone_point(a[1])
        k = a[2]
        return (p0[0] + (p1[0] - p0[0]) * k, p0[1] + (p1[1] - p0[1]) * k)

    def _resolve_bone_point(self, spec):
        """骨头名 -> 点。'shoulder_L' 这种带 _L/_R 后缀的要拆开。"""
        if spec in self.t['joints']:
            return self.J(spec)
        for base in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'):
            if spec.startswith(base + '_'):
                return self.J(base, spec[len(base) + 1:])
        raise KeyError(spec)

    # ---------------- 形变
    def morph_head_count(self, n):
        """改头身比。

        ⚠️ 修正 3：**不是整体缩放**。头顶 = n，下巴 = n-1，脚底 = 0；
           下巴以下每个关节的"离下巴距离"按 (n-1)/(base-1) 缩放。
           这样脚底永远在 0、头永远是 1 单位。
        """
        base_n = self.t['head_count']
        chin0 = base_n - 1.0
        k = (n - 1.0) / (base_n - 1.0)
        chin = n - 1.0

        def fy(y):
            return chin - (chin0 - y) * k

        j = self.t['joints']
        for key, val in list(j.items()):
            if key == 'head_r':
                continue
            if isinstance(val, dict):
                for s in val:
                    x, y = val[s]
                    val[s] = (x * k, fy(y))       # 横向也按同一比例，保持等比
            else:
                x, y = val
                j[key] = (x * k, fy(y))
        j['head_c'] = (0.0, n - 0.5)
        j['head_r'] = 0.5
        self.t['head_count'] = n
        self.t['torso'] = [(fy(y), w * k) for (y, w) in self.t['torso']]
        return self

    def morph_shoulder(self, ratio):
        """改肩宽。ratio = 肩宽 / 头高（建议 1.4~1.9）。"""
        cur = abs(self.J('shoulder', 'R')[0] - self.J('shoulder', 'L')[0])
        if cur <= 1e-6:
            return self
        k = ratio / cur
        for key in ('shoulder', 'elbow', 'wrist'):
            for s in ('L', 'R'):
                x, y = self.J(key, s)
                self.t['joints'][key][s] = (x * k, y)
        self.t['torso'] = [(y, w * k) for (y, w) in self.t['torso']]
        return self

    # ---------------- 姿势（正向运动学）
    def pose(self, angles):
        """angles: {'arm_L': 度, 'fore_L': 度, 'thigh_R': 度, ...}

        绕关节旋转下游骨段。**必须在世界坐标里逐段累加** ——
        每段继承父段的旋转，否则前臂不会跟着上臂转。
        """
        J = self.t['joints']

        def rot(p, o, deg):
            a = math.radians(deg)
            dx, dy = p[0] - o[0], p[1] - o[1]
            return (o[0] + dx * math.cos(a) - dy * math.sin(a),
                    o[1] + dx * math.sin(a) + dy * math.cos(a))

        for s in ('L', 'R'):
            # 手臂：肩 -> 肘 -> 腕
            a1 = angles.get('arm_' + s, 0.0)
            a2 = angles.get('fore_' + s, 0.0)
            sh = J['shoulder'][s]
            J['elbow'][s] = rot(J['elbow'][s], sh, a1)
            J['wrist'][s] = rot(J['wrist'][s], J['elbow'][s], a1 + a2)
            # 腿：胯 -> 膝 -> 踝
            b1 = angles.get('thigh_' + s, 0.0)
            b2 = angles.get('shin_' + s, 0.0)
            hp = J['hip'][s]
            J['knee'][s] = rot(J['knee'][s], hp, b1)
            J['ankle'][s] = rot(J['ankle'][s], J['knee'][s], b1 + b2)
        # 躯干倾斜
        tilt = angles.get('tilt', 0.0)
        if tilt:
            base = J['hip']['L']
            for key in ('neck', 'chest', 'waist', 'head_c'):
                J[key] = rot(J[key], base, tilt)
            for name in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'):
                for s in ('L', 'R'):
                    J[name][s] = rot(J[name][s], base, tilt)
        return self


# ---------------------------------------------------------------------------
# 光栅化：直接在 128×192 的格子上画硬边（拼豆就是硬边）
# ---------------------------------------------------------------------------
class Canvas:
    def __init__(self, rig, size=(128, 192), unit_px=None, margin=8.0):
        """unit_px: 1 头高 = 多少像素。None 时按画布自动填满。"""
        self.rig = rig
        self.size = size
        hc = rig.t['head_count']
        if unit_px is None:
            unit_px = (size[1] - 2 * margin) / hc
        self.u = unit_px
        self.cx = size[0] / 2.0
        # 脚底在 y=0 -> 屏幕 y = size[1] - margin
        self.base_y = size[1] - margin

    def p(self, pt):
        """头单位坐标 -> 屏幕像素。"""
        return (self.cx + pt[0] * self.u, self.base_y - pt[1] * self.u)

    def r(self, v):
        return v * self.u


def capsule(d, p0, p1, r, fill):
    """两端圆头的粗线。拼豆风格要硬边，所以 line 不开抗锯齿。"""
    d.line([p0, p1], fill=fill, width=max(1, int(round(r * 2))))
    for p in (p0, p1):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)


PAL = {
    'skin': (238, 216, 202, 255),
    'hair': (30, 30, 40, 255),
    'hair2': (52, 52, 70, 255),
    'blouse': (240, 244, 252, 255),
    'navy': (36, 46, 80, 255),
    'navy2': (26, 34, 62, 255),
    'red': (186, 58, 72, 255),
    'shoe': (32, 32, 40, 255),
    'bg': (247, 250, 255, 255),
}


def draw_figure(rig, size=(128, 192), garment=True, unit_px=None):
    """把骨架画成一张拼豆图。返回 RGBA。"""
    img = Image.new('RGBA', size, PAL['bg'])
    d = ImageDraw.Draw(img)
    C = Canvas(rig, size, unit_px)
    R = rig.t['radius']
    dep = rig.t['depth']

    # ---- 按 depth 排序，从后往前画（⚠️ 修正 2 的用处）
    layers = sorted(dep.items(), key=lambda kv: kv[1])

    def draw_hair_back():
        hc = C.p(rig.J('head_c'))
        hr = C.r(rig.t['joints']['head_r'])
        d.ellipse([hc[0] - hr, hc[1] - hr, hc[0] + hr, hc[1] + hr * 1.15], fill=PAL['hair'])

    def draw_leg(s):
        hip, knee, ank = (C.p(rig.J('hip', s)), C.p(rig.J('knee', s)),
                          C.p(rig.J('ankle', s)))
        capsule(d, hip, knee, C.r(R['thigh']), PAL['skin'])
        capsule(d, knee, ank, C.r(R['shin']), PAL['skin'])
        foot = (ank[0], C.base_y - 2)
        capsule(d, ank, foot, C.r(R['foot']), PAL['shoe'])

    def draw_torso():
        pts = [C.p((0, y)) for (y, w) in rig.t['torso']]
        poly = []
        for (y, w), p in zip(rig.t['torso'], pts):
            poly.append((C.cx - C.r(w), p[1]))
        for (y, w), p in reversed(list(zip(rig.t['torso'], pts))):
            poly.append((C.cx + C.r(w), p[1]))
        d.polygon(poly, fill=PAL['skin'])

    def draw_arm(s):
        sh, el, wr = (C.p(rig.J('shoulder', s)), C.p(rig.J('elbow', s)),
                      C.p(rig.J('wrist', s)))
        capsule(d, sh, el, C.r(R['upper_arm']), PAL['skin'])
        capsule(d, el, wr, C.r(R['fore_arm']), PAL['skin'])

    def draw_head():
        hc = C.p(rig.J('head_c'))
        hr = C.r(rig.t['joints']['head_r'])
        # 头发壳（比头大一圈）
        d.ellipse([hc[0] - hr * 1.06, hc[1] - hr * 1.06,
                   hc[0] + hr * 1.06, hc[1] + hr * 1.06], fill=PAL['hair'])
        d.polygon([(hc[0] - hr * 1.06, hc[1] - hr * 0.1),
                   (hc[0] + hr * 1.06, hc[1] - hr * 0.1),
                   (hc[0] + hr * 1.06, hc[1] + hr * 1.1),
                   (hc[0] - hr * 1.06, hc[1] + hr * 1.1)], fill=PAL['hair'])
        # 脸
        d.ellipse([hc[0] - hr * 0.92, hc[1] - hr * 0.92,
                   hc[0] + hr * 0.92, hc[1] + hr * 0.98], fill=PAL['skin'])

    def draw_garment():
        # 上衣：领口 -> 腰，用骨架点驱动
        col = C.p(rig.anchor('collar'))
        wa = C.p(rig.anchor('waist'))
        sh_l, sh_r = C.p(rig.anchor('shoulder_L')), C.p(rig.anchor('shoulder_R'))
        d.polygon([(sh_l[0], sh_l[1]), (sh_r[0], sh_r[1]),
                   (wa[0] + C.r(0.62), wa[1]), (wa[0] - C.r(0.62), wa[1])],
                  fill=PAL['blouse'])
        # 水手领
        d.polygon([(sh_l[0], sh_l[1]), (sh_r[0], sh_r[1]),
                   (col[0] + C.r(0.30), col[1] - C.r(0.30)),
                   (col[0], col[1] + C.r(0.46)),
                   (col[0] - C.r(0.30), col[1] - C.r(0.30))], fill=PAL['navy'])
        # 领结
        rib = C.r(0.16)
        d.ellipse([col[0] - rib, col[1] + C.r(0.20) - rib,
                   col[0] + rib, col[1] + C.r(0.20) + rib], fill=PAL['red'])
        # 袖子：沿"肩 -> 肘"方向生成 -> 姿势一变自动跟着（这是整套设计的关键证明）
        for s in ('L', 'R'):
            sh, el, cu = (C.p(rig.J('shoulder', s)), C.p(rig.J('elbow', s)),
                          C.p(rig.anchor('cuff_' + s)))
            d_ = (el[0] - sh[0], el[1] - sh[1])
            n = math.hypot(*d_) or 1.0
            end = (sh[0] + d_[0] / n * math.hypot(cu[0] - sh[0], cu[1] - sh[1]) * 0.55,
                   sh[1] + d_[1] / n * math.hypot(cu[0] - sh[0], cu[1] - sh[1]) * 0.55)
            capsule(d, sh, end, C.r(R['upper_arm'] * 1.22), PAL['blouse'])
        # 百褶裙：胯 -> 下摆。下摆锚点沿"胯->膝"走 62%，改头身比也不会错位
        hl, hr = C.p(rig.anchor('hem_L')), C.p(rig.anchor('hem_R'))
        hip_l, hip_r = C.p(rig.J('hip', 'L')), C.p(rig.J('hip', 'R'))
        d.polygon([(hip_l[0] - C.r(0.10), hip_l[1] + C.r(0.12)),
                   (hip_r[0] + C.r(0.10), hip_r[1] + C.r(0.12)),
                   (hr[0] + C.r(0.30), hr[1]), (hl[0] - C.r(0.30), hl[1])],
                  fill=PAL['navy2'])
        for i in range(1, 5):
            t = i / 5.0
            x0 = (hip_l[0] - C.r(0.10)) + ((hip_r[0] + C.r(0.10)) - (hip_l[0] - C.r(0.10))) * t
            x1 = (hl[0] - C.r(0.30)) + ((hr[0] + C.r(0.30)) - (hl[0] - C.r(0.30))) * t
            d.line([(x0, hip_l[1] + C.r(0.12)), (x1, hl[1])], fill=PAL['navy'], width=1)

    for name, _ in layers:
        if name == 'hair_back':
            draw_hair_back()
        elif name.startswith('leg'):
            draw_leg(name[-1])
        elif name == 'torso':
            draw_torso()
        elif name.startswith('arm'):
            draw_arm(name[-1])
        elif name == 'head':
            if garment:
                draw_garment()
            draw_head()
    return img


def up(img, k=4):
    return img.resize((img.width * k, img.height * k), Image.NEAREST)


def bead(img, cell=3):
    """吸附到**粗格子** —— 这才是"拼豆"的关键，不是"低分辨率"。

    直接在像素级画出来的是「低分辨率矢量图」；拼豆要的是
    **每个格子一个纯色**，所以必须先把图降采样到 cell 粒度的格阵，
    再 NEAREST 放大回来（硬边、格子边界可见）。

    ⚠️ 每格取**众数**，不是平均 —— 平均会在边界糊出中间色，
       格子就看不出来了，"拼豆"感全丢（这条在 pixel.py 里已经踩过一次）。
    """
    import numpy as np
    a = np.asarray(img.convert('RGBA'))
    h, w = a.shape[:2]
    H, W = h // cell, w // cell
    if H == 0 or W == 0:
        return img
    a = a[:H * cell, :W * cell].reshape(H, cell, W, cell, 4)
    a = a.transpose(0, 2, 1, 3, 4).reshape(H, W, cell * cell, 4)
    out = np.zeros((H, W, 4), np.uint8)
    for i in range(H):
        for j in range(W):
            px = a[i, j].astype(np.int64)
            key = (px[:, 0] << 24) | (px[:, 1] << 16) | (px[:, 2] << 8) | px[:, 3]
            vals, cnt = np.unique(key, return_counts=True)
            k = int(vals[cnt.argmax()])
            out[i, j] = ((k >> 24) & 255, (k >> 16) & 255, (k >> 8) & 255, k & 255)
    small = Image.fromarray(out, 'RGBA')
    return small.resize((W * cell, H * cell), Image.NEAREST)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    OUT = _out_dir('rig')
    os.makedirs(OUT, exist_ok=True)

    def sheet(items, path, k=4, cols=None):
        cols = cols or len(items)
        rows = (len(items) + cols - 1) // cols
        w, h = items[0][1].width * k, items[0][1].height * k
        pad = 10
        out = Image.new('RGB', (cols * (w + pad) + pad, rows * (h + pad + 18) + pad),
                        (24, 26, 30))
        dd = ImageDraw.Draw(out)
        for i, (name, im) in enumerate(items):
            r, c = divmod(i, cols)
            x = pad + c * (w + pad)
            y = pad + r * (h + pad + 18)
            out.paste(up(im, k).convert('RGB'), (x, y))
            dd.text((x + 2, y + h + 3), name, fill=(200, 210, 225))
        out.save(path)
        return out

    # 1) 基准
    a = Rig()
    # 2) 摆姿势：右臂上举、左臂后摆（靠 depth 做遮挡）、右腿前迈
    b = Rig().pose({'arm_R': -48, 'fore_R': -35, 'arm_L': 22,
                    'thigh_R': 26, 'shin_R': -18, 'tilt': -3})
    # 3) 改头身比到 6 头身（同一套姿势）
    c = Rig().morph_head_count(6.0).pose({'arm_R': -48, 'fore_R': -35, 'arm_L': 22,
                                          'thigh_R': 26, 'shin_R': -18, 'tilt': -3})
    # 4) 7 头身 + 宽肩 1.9
    e = Rig().morph_shoulder(1.9)

    items = [('基准 7 头身', draw_figure(a)),
             ('摆姿势（含遮挡）', draw_figure(b)),
             ('6 头身（同姿势）', draw_figure(c)),
             ('宽肩 1.9', draw_figure(e))]
    s = sheet(items, os.path.join(OUT, '骨架对比.png'), k=3)
    print('对比图 ->', s.size)

    # 5) 单张放大，看拼豆的格子
    up(draw_figure(a), 4).save(os.path.join(OUT, '拼豆_放大.png'))
    # 6) 锚点/骨架叠图（用来看关节与锚点位置对不对）
    dbg = draw_figure(a).convert('RGB')
    dd = ImageDraw.Draw(dbg)
    C = Canvas(a)
    for k_, v in a.t['joints'].items():
        if k_ == 'head_r':                 # ⚠️ 半径是标量，不是坐标点
            continue
        pts = v.values() if isinstance(v, dict) else [v]
        for pt in pts:
            x, y = C.p(pt)
            dd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 60, 60))
    for name in ANCHORS:
        x, y = C.p(a.anchor(name))
        dd.rectangle([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(30, 200, 90))
    up(dbg, 4).save(os.path.join(OUT, '骨架锚点.png'))
    print('产物 ->', OUT)
