# -*- coding: utf-8 -*-
"""立绘动画：**骨架驱动** + 表情参数 + 帧导出（GIF / mp4）

在此之前的问题
--------------
`rig.py` 早就有 `pose()`（正向运动学：抬臂 / 屈肘 / 迈腿 / 躯干倾斜），
但 `composite.py` 画立绘用的是**写死的头高坐标** —— 所以动骨架不会重画。
本模块把渲染接上骨架：**每个部位的坐标都从 `rig.J()` / `rig.anchor()` 现取**，
于是 `pose()` 一改，整套图跟着动。

三层结构
--------
    Clip    时间轴：关键帧 + 缓动 -> 每帧的 (姿势角, 表情)
    Rig     骨架：pose(角度) -> 关节坐标（含遮挡深度）
    draw_char  渲染：从骨架现取坐标画出来

表情参数（`Face`）
------------------
    eye_open  0=闭 1=睁（左右可分开，做单眼眨）
    mouth    -1=撇嘴 0=平 1=张嘴
    brow     -1=皱眉 0=平 1=挑眉

⚠️ 硬像素边、不抗锯齿（和路线 E 一致）。放大只能用 NEAREST。
"""
import math
import os
import shutil
import subprocess

from PIL import Image, ImageDraw

__all__ = ['Face', 'Clip', 'Cam', 'draw_char', 'export_gif', 'export_mp4',
           'find_ffmpeg', 'PALETTE']


def _clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def _smooth(t):
    """平滑缓动（3t²-2t³）。比线性自然，又不用引入曲线库。"""
    t = _clamp(t)
    return t * t * (3 - 2 * t)


def _lerp(a, b, k):
    return a + (b - a) * k


# ================================================================ 表情
class Face:
    """一帧的表情。所有量都是 0..1 或 -1..1，**可以插值**。"""

    __slots__ = ('eye_open_l', 'eye_open_r', 'mouth', 'brow', 'blush')

    def __init__(self, eye_open=1.0, mouth=0.0, brow=0.0, blush=1.0,
                 eye_open_l=None, eye_open_r=None):
        self.eye_open_l = eye_open if eye_open_l is None else eye_open_l
        self.eye_open_r = eye_open if eye_open_r is None else eye_open_r
        self.mouth = mouth
        self.brow = brow
        self.blush = blush

    def lerp(self, other, k):
        return Face(eye_open_l=_lerp(self.eye_open_l, other.eye_open_l, k),
                    eye_open_r=_lerp(self.eye_open_r, other.eye_open_r, k),
                    mouth=_lerp(self.mouth, other.mouth, k),
                    brow=_lerp(self.brow, other.brow, k),
                    blush=_lerp(self.blush, other.blush, k))

    def flip(self):
        """左右互换（做 wink 时用）。"""
        return Face(eye_open_l=self.eye_open_r, eye_open_r=self.eye_open_l,
                    mouth=self.mouth, brow=self.brow, blush=self.blush)

    def __repr__(self):
        return 'Face(eye %.2f/%.2f mouth %+.2f brow %+.2f)' % (
            self.eye_open_l, self.eye_open_r, self.mouth, self.brow)


# ================================================================ 时间轴
class Clip:
    """关键帧时间轴。

    用法::

        c = Clip(fps=12)
        c.key(0.0, pose={}, face=Face())                       # 站姿
        c.key(0.5, pose={'arm_R': 55}, face=Face())            # 抬手
        c.key(0.8, pose={}, face=Face(eye_open=0.0))           # 眨眼
        for pose, face in c.frames(1.2):
            ...

    角度以外的一切都是标量，所以插值就是逐字段 lerp。`pose` 里没写的键
    按 0 处理 —— 这样关键帧可以只写"变化的那几个"。
    """

    def __init__(self, fps=12):
        self.fps = fps
        self.keys = []          # [(t, pose_dict, Face, ease)]

    def key(self, t, pose=None, face=None, ease='smooth'):
        self.keys.append((float(t), dict(pose or {}),
                          face or Face(), ease))
        self.keys.sort(key=lambda k: k[0])
        return self

    def sample(self, t):
        ks = self.keys
        if not ks:
            return {}, Face()
        if t <= ks[0][0]:
            return dict(ks[0][1]), ks[0][2]
        if t >= ks[-1][0]:
            return dict(ks[-1][1]), ks[-1][2]
        for i in range(len(ks) - 1):
            t0, p0, f0, e0 = ks[i]
            t1, p1, f1, _ = ks[i + 1]
            if t0 <= t <= t1:
                k = (t - t0) / max(1e-9, t1 - t0)
                k = _smooth(k) if e0 == 'smooth' else k
                keys = set(p0) | set(p1)
                pose = {kk: _lerp(p0.get(kk, 0.0), p1.get(kk, 0.0), k)
                        for kk in keys}
                return pose, f0.lerp(f1, k)
        return dict(ks[-1][1]), ks[-1][2]

    def frames(self, dur):
        n = max(1, int(round(dur * self.fps)))
        return [self.sample(i / float(self.fps)) for i in range(n)]


# ================================================================ 相机
class Cam:
    """头高坐标 -> 像素。骨架里 1 单位 = 1 头高，脚底 y=0。"""

    def __init__(self, size=(420, 900), head_units=7.0,
                 span=0.90, sole=0.95):
        self.w, self.h = size
        self.hu = size[1] * span / head_units
        self.cx = size[0] / 2.0
        self.base = size[1] * sole
        self.bob = 0.0                      # 呼吸用的整体上下偏移（像素）

    def x(self, v):
        return self.cx + v * self.hu

    def y(self, v):
        return self.base + self.bob - v * self.hu

    def r(self, v):
        return v * self.hu


# ================================================================ 调色板
# 取自 363.png 那个角色（青绿发 + 粉红发带 + 淡紫衣 + 深蓝四肢）
PALETTE = {
    'hair':     (132, 219, 213), 'hair_d': (95, 158, 170), 'hair_l': (168, 236, 230),
    'hair_s':   (138, 158, 189),
    'skin':     (251, 223, 200), 'skin_d': (226, 196, 178), 'blush': (234, 191, 192),
    'ribbon':   (210, 101, 124), 'ribbon_d': (164, 79, 97),
    'cloth':    (222, 211, 228), 'cloth_d': (191, 181, 205),
    'dark':     (54, 54, 85), 'dark_d': (42, 42, 68), 'dark_k': (24, 24, 30),
    'trim':     (132, 219, 213),
    'eye':      (124, 176, 200), 'iris2': (74, 120, 160), 'pupil': (26, 28, 42),
    'lash':     (30, 28, 44), 'hilite': (252, 254, 255),
    'bg':       (249, 250, 253),
}


def _rr(d, x0, y0, x1, y1, c):
    """坐标安全的矩形（对称件传进来的 x 常常是 x1 < x0）。"""
    d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=c)


def _poly(d, pts, c):
    d.polygon(pts, fill=c)


def _ell(d, cx, cy, rx, ry, c):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=c)


# ================================================================ 渲染
def draw_char(rig, face=None, pal=None, cam=None, size=(420, 900),
              head_units=7.0):
    """**从骨架现取坐标**画一张立绘。传进来的 rig 应该已经 pose 过。

    返回 PIL.Image（RGBA）。
    """
    face = face or Face()
    pal = dict(PALETTE if pal is None else pal)
    cam = cam or Cam(size, head_units)
    img = Image.new('RGBA', size, pal['bg'])
    d = ImageDraw.Draw(img)

    J = rig.J
    A = rig.anchor
    HU = head_units

    # ---- 部位绘制函数（每个都只读骨架）
    def hair_back():
        _ell(d, cam.x(J('head_c')[0]), cam.y(J('head_c')[1]),
             cam.r(0.60), cam.r(0.62), pal['hair_d'])
        _poly(d, [(cam.x(-0.58), cam.y(6.86)), (cam.x(0.58), cam.y(6.86)),
                  (cam.x(0.70), cam.y(3.30)), (cam.x(0.56), cam.y(2.45)),
                  (cam.x(0.28), cam.y(2.70)), (cam.x(-0.28), cam.y(2.70)),
                  (cam.x(-0.56), cam.y(2.45)), (cam.x(-0.70), cam.y(3.30))],
              pal['hair_d'])

    def leg(s):
        hp, kn, an = J('hip', s), J('knee', s), J('ankle', s)
        _poly(d, [(cam.x(hp[0] - 0.20), cam.y(hp[1])), (cam.x(hp[0] + 0.20), cam.y(hp[1])),
                  (cam.x(kn[0] + 0.16), cam.y(kn[1])), (cam.x(kn[0] - 0.16), cam.y(kn[1]))],
              pal['dark'])
        _poly(d, [(cam.x(kn[0] - 0.16), cam.y(kn[1])), (cam.x(kn[0] + 0.16), cam.y(kn[1])),
                  (cam.x(an[0] + 0.13), cam.y(an[1])), (cam.x(an[0] - 0.13), cam.y(an[1]))],
              pal['dark'])
        # 袜口
        _rr(d, cam.x(an[0] - 0.14), cam.y(an[1] + 0.26),
            cam.x(an[0] + 0.14), cam.y(an[1] + 0.10), pal['trim'])
        # 鞋
        _poly(d, [(cam.x(an[0] - 0.14), cam.y(an[1] + 0.04)),
                  (cam.x(an[0] + 0.14), cam.y(an[1] + 0.04)),
                  (cam.x(an[0] + 0.16), cam.y(0.0)), (cam.x(an[0] - 0.16), cam.y(0.0))],
              pal['dark_k'])

    def torso():
        """躯干 / 裙 —— **全部从关节现取**。

        ⚠️ 之前这里只从关节取了 y，x 和裙摆还写死（±0.60 / 4.60 / 2.55 …）。
           后果：`tilt` 只转动了关节和头，**身体不跟着倾**，头和身子会脱开。
           正解是横竖都从关节算，宽度用"肩关节间距"推。
        """
        sl, sr = J('shoulder', 'L'), J('shoulder', 'R')
        wa = J('waist')
        hl, hr = J('hip', 'L'), J('hip', 'R')
        sw = abs(sr[0] - sl[0]) / 2.0          # 肩关节间距的一半
        tw, ww, hw = sw * 0.76, sw * 0.58, sw * 0.72
        s_cx = (sl[0] + sr[0]) / 2.0
        h_cx = (hl[0] + hr[0]) / 2.0
        # 上衣 + 裙摆：肩 -> 腰 -> 胯 -> 下摆
        hem_l, hem_r = A('hem_L'), A('hem_R')
        _poly(d, [(cam.x(s_cx - tw), cam.y(sl[1] + 0.05)),
                  (cam.x(s_cx + tw), cam.y(sr[1] + 0.05)),
                  (cam.x(wa[0] + ww), cam.y(wa[1])),
                  (cam.x(h_cx + hw), cam.y(hl[1])),
                  (cam.x(hem_r[0] + 0.16), cam.y(hem_r[1])),
                  (cam.x(hem_l[0] - 0.16), cam.y(hem_l[1])),
                  (cam.x(h_cx - hw), cam.y(hl[1])),
                  (cam.x(wa[0] - ww), cam.y(wa[1]))], pal['cloth'])
        # 胸前阴影带
        _poly(d, [(cam.x(wa[0] - ww * 0.86), cam.y(sl[1] - 0.95)),
                  (cam.x(wa[0] + ww * 0.86), cam.y(sr[1] - 0.95)),
                  (cam.x(wa[0] + ww * 0.72), cam.y(wa[1] + 0.15)),
                  (cam.x(wa[0] - ww * 0.72), cam.y(wa[1] + 0.15))], pal['cloth_d'])
        # 腰带
        _rr(d, cam.x(wa[0] - ww * 1.05), cam.y(wa[1] + 0.10),
            cam.x(wa[0] + ww * 1.05), cam.y(wa[1] - 0.04), pal['trim'])
        # 下摆滚边
        _poly(d, [(cam.x(hem_l[0] - 0.16), cam.y(hem_l[1] + 0.10)),
                  (cam.x(hem_r[0] + 0.16), cam.y(hem_r[1] + 0.10)),
                  (cam.x(hem_r[0] + 0.14), cam.y(hem_r[1] - 0.06)),
                  (cam.x(hem_l[0] - 0.14), cam.y(hem_l[1] - 0.06))], pal['trim'])
        # 裙褶（跟着下摆走）
        for i in (-3, -2, -1, 1, 2, 3):
            t = i / 7.0
            tl = 0.5 + t
            d.line([(cam.x(h_cx - hw + 2 * hw * tl), cam.y(hl[1])),
                    (cam.x(hem_l[0] - 0.16 + (hem_r[0] - hem_l[0] + 0.32) * tl),
                     cam.y(hem_l[1]))],
                   fill=pal['cloth_d'], width=max(2, int(cam.r(0.02))))
        # 水手领（跟脖子和肩走 -> 点头/倾斜时也对）
        nk = J('neck')
        _poly(d, [(cam.x(sl[0]), cam.y(sl[1] + 0.02)),
                  (cam.x(sr[0]), cam.y(sr[1] + 0.02)),
                  (cam.x(nk[0] + 0.30), cam.y(nk[1] - 0.16)),
                  (cam.x(nk[0] - 0.30), cam.y(nk[1] - 0.16))], pal['cloth_d'])
        _poly(d, [(cam.x(nk[0] - 0.28), cam.y(nk[1] - 0.14)),
                  (cam.x(nk[0] + 0.28), cam.y(nk[1] - 0.14)),
                  (cam.x(nk[0] + 0.15), cam.y(nk[1] - 0.40)),
                  (cam.x(nk[0] - 0.15), cam.y(nk[1] - 0.40))], pal['trim'])

    def arm(s):
        sh, el, wr = J('shoulder', s), J('elbow', s), J('wrist', s)
        cu = A('cuff_' + s)

        def thick(p, q, w):
            dx, dy = q[0] - p[0], q[1] - p[1]
            n = math.hypot(dx, dy) or 1.0
            px, py = -dy / n * w, dx / n * w
            _poly(d, [(cam.x(p[0] + px), cam.y(p[1] + py)),
                      (cam.x(q[0] + px), cam.y(q[1] + py)),
                      (cam.x(q[0] - px), cam.y(q[1] - py)),
                      (cam.x(p[0] - px), cam.y(p[1] - py))], pal['dark'])
        thick(sh, el, 0.135)
        thick(el, cu, 0.115)
        thick(cu, wr, 0.100)
        # ⚠️ 肩头补一个圆盘：手臂一抬起来，锥形的上缘就和躯干脱开一条缝。
        #    补一个略大的圆盖住接缝，动起来才连得上。
        _ell(d, cam.x(sh[0]), cam.y(sh[1]), cam.r(0.145), cam.r(0.145), pal['dark'])
        # 袖口青边
        thick((cu[0] - (wr[0] - cu[0]) * 0.22, cu[1] - (wr[1] - cu[1]) * 0.22), cu, 0.12)
        _poly(d, [(cam.x(cu[0] - 0.12), cam.y(cu[1] + 0.10)),
                  (cam.x(cu[0] + 0.12), cam.y(cu[1] + 0.10)),
                  (cam.x(cu[0] + 0.12), cam.y(cu[1] - 0.06)),
                  (cam.x(cu[0] - 0.12), cam.y(cu[1] - 0.06))], pal['trim'])
        # 手（沿"肘 -> 腕"再往外延一点，手腕动它跟着动）
        hx, hy = wr[0] + (wr[0] - cu[0]) * 0.35, wr[1] + (wr[1] - cu[1]) * 0.35
        _ell(d, cam.x(hx), cam.y(hy), cam.r(0.13), cam.r(0.15), pal['skin'])

    def head():
        hc = J('head_c')
        hr = rig.t['joints']['head_r'] if not isinstance(
            rig.t['joints']['head_r'], dict) else 0.5
        cx, cy = cam.x(hc[0]), cam.y(hc[1])
        rh = cam.r(hr)
        _ell(d, cx, cy, rh * 1.08, rh * 1.18, pal['hair'])          # 颅骨/头发
        # 脸
        _ell(d, cx, cy - rh * 0.10, rh * 0.84, rh * 0.94, pal['skin'])
        _poly(d, [(cx - rh * 0.82, cy + rh * 0.10), (cx - rh * 0.74, cy + rh * 0.54),
                  (cx - rh * 0.46, cy + rh * 0.90), (cx, cy + rh * 1.06),
                  (cx + rh * 0.46, cy + rh * 0.90), (cx + rh * 0.74, cy + rh * 0.54),
                  (cx + rh * 0.82, cy + rh * 0.10)], pal['skin'])
        # 刘海：顶边沿颅骨弧线，下缘浅波
        pts = []
        for i in range(15):
            t = -0.96 + 1.92 * i / 14.0
            pts.append((cx + rh * 1.12 * t,
                        cy - rh * 1.20 * math.sqrt(max(0.0, 1 - t * t))))
        for t, dep in ((0.92, 0.30), (0.62, 0.38), (0.32, 0.28), (0.02, 0.36),
                       (-0.30, 0.26), (-0.62, 0.36), (-0.92, 0.30)):
            pts.append((cx + rh * 1.06 * t, cy - rh * (1.20 - dep * 1.4)))
        _poly(d, pts, pal['hair'])
        # 鬓发
        for s in (-1, 1):
            _poly(d, [(cx + s * rh * 0.94, cy - rh * 0.70),
                      (cx + s * rh * 1.16, cy - rh * 0.78),
                      (cx + s * rh * 1.22, cy + rh * 1.60),
                      (cx + s * rh * 0.94, cy + rh * 1.54)], pal['hair'])
        # 头发高光
        _poly(d, [(cx - rh * 0.70, cy - rh * 0.94), (cx + rh * 0.26, cy - rh * 1.02),
                  (cx + rh * 0.20, cy - rh * 0.82), (cx - rh * 0.76, cy - rh * 0.74)],
              pal['hair_l'])
        _poly(d, [(cx - rh * 0.64, cy - rh * 0.62), (cx + rh * 0.04, cy - rh * 0.70),
                  (cx - rh * 0.02, cy - rh * 0.56), (cx - rh * 0.70, cy - rh * 0.48)],
              pal['hair_l'])

        # ---- 五官
        ey = cy + rh * 0.16                      # 眼线
        ew, eh = rh * 0.30, rh * 0.155           # **扁**：宽约 1.9 倍高
        for s, op in ((-1, face.eye_open_l), (1, face.eye_open_r)):
            ex = cx + s * rh * 0.44
            _eye(d, ex, ey, ew, eh, _clamp(op), pal)
        # 眉（跟 brow 抬/皱）
        by = ey - rh * 0.34 - rh * 0.10 * face.brow
        for s in (-1, 1):
            tilt = (-rh * 0.05 * face.brow) * s
            d.line([(cx + s * rh * 0.20, by - tilt), (cx + s * rh * 0.70, by + tilt)],
                   fill=pal['hair_s'], width=max(2, int(rh * 0.06)))
        # 鼻
        d.line([(cx + rh * 0.02, ey + rh * 0.30), (cx + rh * 0.10, ey + rh * 0.34)],
               fill=pal['skin_d'], width=max(2, int(rh * 0.04)))
        # 嘴（mouth: -1 撇嘴 / 0 平 / 1 张嘴）
        my = ey + rh * 0.62
        mw = rh * 0.16
        if face.mouth > 0.25:
            _ell(d, cx, my + rh * 0.04 * face.mouth, mw * (0.7 + face.mouth * 0.6),
                 rh * 0.06 + rh * 0.10 * face.mouth, pal['ribbon_d'])
        else:
            dy = -rh * 0.06 * face.mouth
            d.line([(cx - mw, my + dy), (cx, my + rh * 0.03), (cx + mw, my + dy)],
                   fill=pal['ribbon_d'], width=max(2, int(rh * 0.045)))
        # 腮红（跟 blush 淡入淡出）
        if face.blush > 0.02:
            for s in (-1, 1):
                _ell(d, cx + s * rh * 0.76, ey + rh * 0.42,
                     rh * 0.20, rh * 0.075 * face.blush, pal['blush'])

    def neck():
        nk = J('neck')
        _poly(d, [(cam.x(nk[0] - 0.13), cam.y(nk[1] + 0.22)),
                  (cam.x(nk[0] + 0.13), cam.y(nk[1] + 0.22)),
                  (cam.x(nk[0] + 0.16), cam.y(nk[1] - 0.16)),
                  (cam.x(nk[0] - 0.16), cam.y(nk[1] - 0.16))], pal['skin_d'])

    def ribbon():
        nk = J('neck')
        bx = cam.x(nk[0] - 0.42)
        by = cam.y(nk[1] + 0.66)
        bw, bh = cam.r(0.16), cam.r(0.12)
        _poly(d, [(bx, by), (bx - bw, by - bh), (bx - bw, by + bh)], pal['ribbon'])
        _poly(d, [(bx, by), (bx + bw * 0.85, by - bh * 0.85),
                  (bx + bw * 0.85, by + bh * 0.85)], pal['ribbon_d'])
        _rr(d, bx - cam.r(0.03), by - cam.r(0.05),
            bx + cam.r(0.03), by + cam.r(0.05), pal['ribbon_d'])

    # ---- 层序：**按骨架给的 depth 决定前后**，这就是 rig.t['depth'] 的用处。
    #      典型用途：左臂 depth=5（在躯干之后）、右臂 depth=40（在躯干之前），
    #      于是"手插腰/抱胸"能成立；只改一个数就能翻转遮挡。
    dep = rig.t['depth']
    torso_d = dep.get('torso', 20)
    hair_back()
    for s in ('L', 'R'):
        leg(s)
    for s in ('L', 'R'):
        if dep.get('arm_' + s, 30) < torso_d:
            arm(s)
    torso()
    for s in ('L', 'R'):
        if dep.get('arm_' + s, 30) >= torso_d:
            arm(s)
    neck()
    head()
    ribbon()
    return img


def _eye(d, cx, cy, ew, eh, op, pal):
    """一只眼睛。op = 睁开程度 0..1（上睑自上而下盖）。"""
    if op < 0.08:
        # 闭眼：一条下弯的线
        d.line([(cx - ew * 0.92, cy), (cx, cy + eh * 0.52), (cx + ew * 0.92, cy)],
               fill=pal['lash'], width=max(2, int(eh * 0.44)))
        return
    top = cy - eh * op
    bot = cy + eh * 0.92
    # 眼白
    _ell(d, cx, (top + bot) / 2, ew, max(1.0, (bot - top) / 2), pal['hilite'])
    # 虹膜 / 瞳孔（跟着可见高度收）
    rr = max(1.0, (bot - top) / 2)
    _ell(d, cx, (top + bot) / 2 + rr * 0.05, ew * 0.52, rr * 0.80, pal['eye'])
    _ell(d, cx, (top + bot) / 2 + rr * 0.05, ew * 0.28, rr * 0.62, pal['pupil'])
    if op > 0.5:
        _ell(d, cx - ew * 0.26, (top + bot) / 2 - rr * 0.30,
             ew * 0.15, rr * 0.24, pal['hilite'])
    # 上眼睑（粗线，外眼角上挑）
    d.line([(cx - ew * 0.94, top + eh * 0.10), (cx + ew * 1.06, top - eh * 0.18)],
           fill=pal['lash'], width=max(3, int(eh * 0.34)))
    # 外眼角睫毛
    d.polygon([(cx + ew * 1.06, top - eh * 0.18),
               (cx + ew * 1.42, top - eh * 0.46),
               (cx + ew * 1.30, top + eh * 0.06)], fill=pal['lash'])
    # 下眼睑
    d.line([(cx - ew * 0.84, bot - eh * 0.06), (cx + ew * 0.94, bot - eh * 0.10)],
           fill=pal['lash'], width=max(1, int(eh * 0.10)))


# ================================================================ 导出
def find_ffmpeg():
    """找 ffmpeg：环境变量 FFMPEG -> PATH -> 几个常见位置。找不到返回 None。"""
    p = os.environ.get('FFMPEG')
    if p and os.path.exists(p):
        return p
    w = shutil.which('ffmpeg')
    if w:
        return w
    for c in (r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'):
        if os.path.exists(c):
            return c
    return None


def export_gif(frames, path, fps=12, loop=0):
    """导出 GIF。帧数多时体积会很大 —— 拼豆/像素风的图压缩率好，问题不大。"""
    if not frames:
        raise ValueError('没有帧')
    d = max(20, int(1000.0 / fps))
    ims = [f.convert('P', palette=Image.ADAPTIVE, colors=255) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:],
                duration=d, loop=loop, disposal=2, optimize=False)
    return path


def export_mp4(frames, path, fps=12, ffmpeg=None):
    """导出 mp4：先落 PNG 序列再喂 ffmpeg。

    ⚠️ 帧尺寸必须偶数（H.264 的 yuv420p 要求）—— 奇数宽高会让 ffmpeg 直接报错。
    """
    ff = ffmpeg or find_ffmpeg()
    if not ff:
        return None
    import tempfile
    tmp = tempfile.mkdtemp(prefix='layeredart-anim-')
    try:
        for i, f in enumerate(frames):
            im = f.convert('RGB')
            if im.width % 2 or im.height % 2:
                im = im.crop((0, 0, im.width - im.width % 2, im.height - im.height % 2))
            im.save(os.path.join(tmp, 'f%05d.png' % i))
        subprocess.run([ff, '-y', '-framerate', str(fps), '-i',
                        os.path.join(tmp, 'f%05d.png'),
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
                        '-pix_fmt', 'yuv420p', path],
                       capture_output=True, check=True)
        return path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
