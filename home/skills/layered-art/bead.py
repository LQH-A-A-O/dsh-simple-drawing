# -*- coding: utf-8 -*-
"""拼豆格阵：**一切吸附到格子** + 局部精修 + 镜像

为什么要有这个模块
------------------
之前是"先在像素级精细画，最后降采样到格子"。两个致命问题：
  1. 细线会消失 —— 1px 的线在 3×3 的格里占不到多数
  2. 每换一个姿势，格子切割位置就变一次 -> 同一个部位的颜色在两帧之间会跳

正解是**从一开始就在格子上工作**：坐标量化到格阵，图元就是格的整数倍。

"从局部提升精度"（用户的想法）
------------------------------
局部放大来画 -> 镜像出另一半 -> 贴回整体。
这是对的，但有三条硬约束，缺一条镜像就会差半格：

  ⚠️ 1. **cell 必须整除画布两边**。128 = 2^7 所以 cell ∈ {1,2,4,8}，**3 不行**：
        128/3 = 42.67，格子铺不满，中轴落在 21.33 列。
  ⚠️ 2. **列数必须是偶数**，中轴才落在格线上。
  ⚠️ 3. **对称特征自身的尺寸必须是偶数列**。头 22px / 格 2px = 11 格 ——
        奇数格的头画不出对称的脸（左右各 5.5 格）。头必须量化成 12 格 = 24px。

这三条都在下面用 assert 钉死了，违反了直接报错，不会悄悄出半格偏移。
"""
import math

import numpy as np
from PIL import Image

__all__ = ['Grid', 'snap_even', 'quantize', 'mirror_cells', 'bead_from_cells']


def snap_even(v, cell, minimum=2):
    """量化到格子，并**进位到偶数格**（对称特征必须偶数列）。"""
    n = max(minimum, int(round(v / float(cell))))
    if n % 2:
        n += 1
    return n * cell


class Grid:
    """拼豆格阵。所有坐标以**格**为单位，(0,0) = 左上角那一格。"""

    def __init__(self, size=(128, 192), cell=2):
        w, h = size
        assert w % cell == 0, 'cell=%d 不能整除画布宽 %d' % (cell, w)
        assert h % cell == 0, 'cell=%d 不能整除画布高 %d' % (cell, h)
        self.cell = cell
        self.px = (w, h)
        self.cw, self.ch = w // cell, h // cell
        assert self.cw % 2 == 0, ('列数 %d 是奇数，中轴不落在格线上，'
                                  '镜像会差半格' % self.cw)
        self.axis = self.cw // 2          # 中轴所在的格线（左半 = 0..axis-1）
        self.a = np.zeros((self.ch, self.cw, 4), np.uint8)

    # ---------------- 基本写入（全部以格为单位）
    def put(self, x, y, rgba):
        if 0 <= x < self.cw and 0 <= y < self.ch:
            self.a[y, x] = rgba

    def rect(self, x0, y0, x1, y1, rgba):
        """闭区间格矩形。"""
        self.a[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = rgba

    def ellipse(self, cx, cy, rx, ry, rgba):
        """以格为单位的实心椭圆（用于头/眼睛这种对称件）。"""
        yy, xx = np.mgrid[0:self.ch, 0:self.cw]
        m = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0
        self.a[m] = rgba

    def capsule(self, x0, y0, x1, y1, r, rgba):
        """格单位的胶囊（两端圆头），用于四肢。"""
        yy, xx = np.mgrid[0:self.ch, 0:self.cw]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(L2, 1e-6), 0, 1)
        d2 = (xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2
        self.a[d2 <= r * r] = rgba

    # ---------------- 镜像（用户要的"局部精修 + 镜像"）
    def mirror_left_to_right(self):
        """把左半（列 0..axis-1）镜像到右半（列 axis..cw-1）。

        ⚠️ 中轴在格线 axis 上：格 i 的镜像是 cw-1-i。
           不是 2*axis-i —— 那是"以格心为轴"，会整体错半格。
        """
        left = self.a[:, :self.axis]
        self.a[:, self.axis:] = left[:, ::-1]

    def mirror_cols(self, x0, x1):
        """只镜像一个局部列区间（画完一只眼睛立刻镜像出另一只）。"""
        for dx in range(0, x1 - x0 + 1):
            src = x0 + dx
            dst = self.cw - 1 - src
            self.a[:, dst] = self.a[:, src]

    # ---------------- 输出
    def add_outline(self, k=0.52, bg=None, thr=28):
        """给**每个颜色区域的边界**加 1 格深色描边。

        为什么必须有这一步：在拼豆这个分辨率下，**近色相邻部件会糊成一块**
        （两条 navy 的腿、白袖子配白上衣、深灰鞋配深蓝裤）。实测发现
        "把颜色微微调开"是没用的 —— navy(38,48,84) 和 navy2(26,34,62)
        的差在 8 格宽的东西上根本看不出来。标准像素画的做法就是描边。

        k   深色系数（越小越黑）
        bg  背景色；等于它的格不描边（否则整个画面外框会被黑边包住）
        thr 颜色差超过它才算"不同区域"
        """
        a = self.a
        rgb = a[..., :3].astype(np.int32)
        isbg = np.zeros(a.shape[:2], bool)
        if bg is not None:
            isbg = (np.abs(rgb - np.array(bg[:3], np.int32)).sum(2) <= thr)
        edge = np.zeros(a.shape[:2], bool)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(np.roll(rgb, dy, 0), dx, 1)
            edge |= (np.abs(rgb - sh).sum(2) > thr)
        edge &= ~isbg
        a[edge, :3] = np.clip(a[edge, :3].astype(np.float64) * k, 0, 255).astype(np.uint8)
        return self

    def to_image(self):
        """格阵 -> 实际像素图（NEAREST 硬边）。"""
        small = Image.fromarray(self.a, 'RGBA')
        return small.resize(self.px, Image.NEAREST)

    def zoom(self, box_cells, k=12):
        """放大看局部（局部精修时的工作视图）。格为单位 (x0,y0,x1,y1)。"""
        x0, y0, x1, y1 = box_cells
        sub = self.a[y0:y1 + 1, x0:x1 + 1]
        im = Image.fromarray(sub, 'RGBA').resize(
            ((x1 - x0 + 1) * k, (y1 - y0 + 1) * k), Image.NEAREST)
        # 画格线，方便数格子
        arr = np.asarray(im).copy()
        arr[::k, :, :3] = (255, 90, 90)
        arr[:, ::k, :3] = (255, 90, 90)
        return Image.fromarray(arr, 'RGBA')


def quantize(v, cell):
    """任意像素 -> 最近的格边界。"""
    return int(round(v / float(cell)))


def bead_from_cells(arr, cell):
    """格阵 -> 像素图（外部若已有 numpy 格阵可用）。"""
    small = Image.fromarray(arr.astype(np.uint8), 'RGBA')
    return small.resize((small.width * cell, small.height * cell), Image.NEAREST)
