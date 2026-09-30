# -*- coding: utf-8 -*-
"""部件级画布 + 装配：**一个部件一块画布，最后合起来**

用户的构想
----------
"一个部件一块画布，最后在一大块画布上合起来变成完整的图片，
 这样局部的比例也不会变得很奇怪。"

对的，这是游戏角色装配的标准做法。但要成立必须满足三条，
缺一条就出问题（都在下面用 assert 钉死）：

  ① **整数倍**：部件画布格数 = 成品格数 × 整数倍。
     非整数倍降采样会错位。
  ② **整数格偏移装配**：放的位置必须是整数格，否则半格错位，
     眼睛的格子和脸的格子对不上（和 mirror 那个坑同源）。
  ③ **按格取众数降采样**，不能插值 —— 插值会把拼豆糊成灰。

⚠️ 分辨率守恒：画布画得再大，**成品精度仍由"成品占几格"决定**。
   画大只买到两样东西：手好画、比例准。买不到更多细节。
   所以部件内部任何特征的宽度必须 ≥ 成品 1 格，否则降采样时直接消失。
"""
import numpy as np
from PIL import Image

__all__ = ['Part', 'Assembler']


def _mode_down(a, sub):
    """按 sub×sub 取众数降采样。

    ⚠️ 不能插值、不能平均 —— 平均会糊出中间色，拼豆的格子边界就没了。
       这条在 pixel.py / bead.py 里都踩过。
    """
    h, w = a.shape[:2]
    H, W = h // sub, w // sub
    a = a[:H * sub, :W * sub].reshape(H, sub, W, sub, 4).transpose(0, 2, 1, 3, 4)
    a = a.reshape(H, W, sub * sub, 4)
    out = np.zeros((H, W, 4), np.uint8)
    for i in range(H):
        for j in range(W):
            px = a[i, j].astype(np.int64)
            key = (px[:, 0] << 24) | (px[:, 1] << 16) | (px[:, 2] << 8) | px[:, 3]
            vals, cnt = np.unique(key, return_counts=True)
            k = int(vals[cnt.argmax()])
            out[i, j] = ((k >> 24) & 255, (k >> 16) & 255, (k >> 8) & 255, k & 255)
    return out


class Part:
    """一个部件自己的画布。

    内部以 **sub-格** 为单位画（这样手有空间、比例好控），
    导出时按整数倍 sub 降到成品格。
    """

    def __init__(self, cells, sub=6, bg=(0, 0, 0, 0)):
        assert isinstance(cells, tuple) and len(cells) == 2
        self.cw, self.ch = cells          # 成品格数
        self.sub = sub                    # 整数倍
        self.a = np.zeros((self.ch * sub, self.cw * sub, 4), np.uint8)
        if bg[3]:
            self.a[:, :] = bg

    # ---- 内部绘图：坐标单位是 sub-格
    def rect(self, x0, y0, x1, y1, c):
        self.a[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = c

    def ellipse(self, cx, cy, rx, ry, c):
        yy, xx = np.mgrid[0:self.a.shape[0], 0:self.a.shape[1]]
        m = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0
        self.a[m] = c

    def capsule(self, x0, y0, x1, y1, r, c):
        yy, xx = np.mgrid[0:self.a.shape[0], 0:self.a.shape[1]]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / max(L2, 1e-6), 0, 1)
        self.a[(xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2 <= r * r] = c

    def mirror_left_to_right(self):
        """以**中轴格线**镜像：格 i 的镜像是 cw-1-i。"""
        assert self.cw % 2 == 0, '列数必须偶数，中轴才在格线上'
        A = self.cw // 2
        self.a = self.a.copy()
        L = self.a[:, :A * self.sub]
        self.a[:, A * self.sub:] = L[:, ::-1]
        return self

    def mirror_cols(self, x0, x1):
        """镜像局部列区间（sub-格坐标）。"""
        W = self.a.shape[1]
        for d in range(0, x1 - x0 + 1):
            self.a[:, W - 1 - (x0 + d)] = self.a[:, x0 + d]
        return self

    # ---- 装配（把另一个部件的成品格数组放进来）
    def place(self, arr, x_cell, y_cell, label=''):
        """把 bake() 过的部件放进本部件，偏移是**整数格**。

        内部按 sub 倍 NEAREST 放大再贴 —— 这样两边的格子严格对齐，
        不会出现"眼睛的格和脸的格错开半格"。
        """
        assert float(x_cell).is_integer() and float(y_cell).is_integer(), \
            '%s 装配偏移必须是整数格，给的是 (%s, %s)' % (label, x_cell, y_cell)
        up = np.repeat(np.repeat(arr, self.sub, 0), self.sub, 1)
        x, y = int(x_cell) * self.sub, int(y_cell) * self.sub
        H, W = self.a.shape[:2]
        h, w = up.shape[:2]
        x1, y1 = min(W, x + w), min(H, y + h)
        sx, sy = max(0, -x), max(0, -y)
        sub = up[sy:sy + (y1 - max(0, y)), sx:sx + (x1 - max(0, x))]
        dst = self.a[max(0, y):y1, max(0, x):x1]
        m = sub[..., 3] > 0
        dst[m] = sub[m]
        return self

    # ---- 导出
    def bake(self):
        """降到成品格。返回 (ch, cw, 4) 数组。"""
        return _mode_down(self.a, self.sub)

    def preview(self, k=10):
        """作者视图：显示 sub-格画布 + 粗格线（成品格边界）。"""
        im = Image.fromarray(self.a, 'RGBA')
        arr = np.asarray(im).copy()
        s = self.sub
        arr[::s, :, :3] = (255, 90, 90)
        arr[:, ::s, :3] = (255, 90, 90)
        return Image.fromarray(arr, 'RGBA').resize(
            (im.width * k, im.height * k), Image.NEAREST)


class Assembler:
    """总画布。把各部件按**整数格偏移**装配上去。"""

    def __init__(self, cells, bg=(248, 250, 253, 255)):
        self.cw, self.ch = cells
        self.a = np.zeros((self.ch, self.cw, 4), np.uint8)
        self.a[:, :] = bg

    def place(self, arr, x_cell, y_cell, label=''):
        """arr 是 bake() 出来的成品格数组。

        ⚠️ 偏移必须是**整数格** —— 半格偏移会让部件的格子和总画布的
           格子错开，看起来就是"眼睛歪了半格"（和 mirror 是同一类坑）。
        """
        assert float(x_cell).is_integer() and float(y_cell).is_integer(), \
            '%s 的装配偏移必须是整数格，给的是 (%s, %s)' % (label, x_cell, y_cell)
        x, y = int(x_cell), int(y_cell)
        h, w = arr.shape[:2]
        x1, y1 = min(self.cw, x + w), min(self.ch, y + h)
        if x >= self.cw or y >= self.ch or x1 <= 0 or y1 <= 0:
            raise ValueError('%s 装配位置 (%d,%d) 完全在画布外' % (label, x, y))
        sx, sy = max(0, -x), max(0, -y)
        sub = arr[sy:sy + (y1 - max(0, y)), sx:sx + (x1 - max(0, x))]
        dst = self.a[max(0, y):y1, max(0, x):x1]
        m = sub[..., 3] > 0
        dst[m] = sub[m]
        return self

    def add_outline(self, k=0.72, bg=(248, 250, 253, 255), thr=28, parts_mask=None):
        """描边。**只描部件边界**（parts_mask 给出每个格属于哪个部件）。

        ⚠️ 按颜色差全局描边在低分辨率下会失败 —— 几乎每格都紧邻异色格，
           等于整体压暗，脸会糊掉（这个否定结果实测了三轮）。
        """
        a = self.a
        rgb = a[..., :3].astype(np.int32)
        if parts_mask is None:
            edge = np.zeros(a.shape[:2], bool)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sh = np.roll(np.roll(rgb, dy, 0), dx, 1)
                edge |= (np.abs(rgb - sh).sum(2) > thr)
        else:
            edge = np.zeros(a.shape[:2], bool)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sh = np.roll(np.roll(parts_mask, dy, 0), dx, 1)
                edge |= (sh != parts_mask)
        isbg = (np.abs(rgb - np.array(bg[:3], np.int32)).sum(2) <= thr)
        edge &= ~isbg
        a[edge, :3] = np.clip(a[edge, :3].astype(np.float64) * k, 0, 255).astype(np.uint8)
        return self

    def to_image(self, cell=2):
        im = Image.fromarray(self.a, 'RGBA')
        return im.resize((self.cw * cell, self.ch * cell), Image.NEAREST)
