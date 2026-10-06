# -*- coding: utf-8 -*-
"""渐进生长式拼豆 —— 从白画布开始"加"出图来。

与原来的路线相反
----------------
原路线（减法）：
    高分辨率母版 -> 降采样归并 -> 拼豆
    问题：细节不受控地丢失；每换一次格距结果就变；没法控制用了哪些颜色。

本模块（加法）：
    白画布 -> 语义分区 -> 每区上基色 -> 逐层细分加变化 -> 成品
    每一层都是一张**完整可用**的拼豆图，可以停在任何一层。

两个机制（用户提出）
--------------------
① **提示词机制**：先有对画面的概括（天空/太阳/山/水），据此划定区域。
   本模块不依赖任何模型 —— 区域用几何图元描述（矩形/圆/多边形/半平面）。

② **染色机制**：每个区域有自己的**固定调色板**。细节只决定"在这一区的
   调色板里取哪个色"，不会串到别的区去。这正好对应拼豆"按颜色买豆子"
   的现实：天空只用蓝色系、山只用绿/灰/黑。

细节从哪来
----------
细节场（0..1 浮点）决定颜色在调色板里的位置。三个来源：
  · 程序化（渐变/噪声/条纹）—— 不需要任何素材
  · 参考图（灰度 -> 0..1）—— 想照着某张图画时用
  · 混合

分层的关键：细节场拆成**倍频程（octave）**。第 L 层只累加前 L+1 个倍频程，
所以第 0 层几乎是平的（只有最粗的结构），越往上变化越细。
这就是"加法"的数学形式 —— 和扩散模型"一次次去噪"正好相反。
"""
import numpy as np
from PIL import Image

__all__ = ['BeadScene', 'fbm_octaves', 'Region', 'mask_ellipse']

# ================================================================ 程序化细节场

def fbm_octaves(shape, seed=0, n_oct=6, lacunarity=2.0):
    """返回一组**零均值**的倍频程噪声，从最粗（波长≈整幅）到最细。

    ⚠️ **必须零均值**，这是"加法"能成立的前提。
       第一版返回的是 0..1 的非负场，各层均值还不一样（实测 0.602/0.469/0.529/0.469），
       叠加之后的方差不可控 —— 实测倍频程之间出现**负相关**（oct0 与 oct1 相关 -0.339），
       于是"多累加一层"反而让总方差**下降**（0.2070 -> 0.1991），加法语义直接没了。
       零均值之后 Var(Σ w·oct) = Σ w²·Var(oct)，严格随层数增长。

    生成方式：在低频网格上取随机值再双线性放大到目标尺寸，最后减去自身均值。
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    raw = []
    for k in range(n_oct):
        gh = max(2, int(round(2 * (lacunarity ** k))))
        gw = max(2, int(round(2 * (lacunarity ** k))))
        g = rng.random((gh, gw)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8), 'L').resize((w, h), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32) / 255.0
        s = float(a.std())
        if s > 1e-6:
            a = (a - float(a.mean())) / s      # 零均值 + 单位方差
        raw.append(a.astype(np.float32))

    # ---- **正交化**：让各倍频程互不相关
    # ⚠️ 只做零均值还不够。实测粗网格放大后本质是个双线性曲面（系统性倾斜），
    #    和更细的场出现 **-0.369** 的负相关 —— 于是"多累加一层"反而让总方差下降
    #    （1.0000 -> 0.9385 -> 0.8866），加法语义丢失。
    #    正交化之后各层内积为 0，Var(Σ w·oct) = Σ w²·Var(oct)，严格随层数增长。
    basis, outs = [], []
    for a in raw:
        v = a.ravel().astype(np.float64).copy()
        for u in basis:
            v -= (v @ u) * u
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            continue
        v = v / n
        basis.append(v)
        r = v.reshape(shape)
        rs = float(r.std())
        if rs > 1e-9:
            r = (r - float(r.mean())) / rs      # 再归一回单位方差，便于调权重
        outs.append(r.astype(np.float32))
    return outs


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a, np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def detail_vgrad(shape, top=1.0, bottom=0.0):
    """竖直渐变（天空上深下浅、水面远处深近处浅）。"""
    h, w = shape
    t = np.linspace(top, bottom, h, dtype=np.float32)[:, None]
    return np.repeat(t, w, axis=1)


def detail_radial(shape, cx, cy, r, invert=False):
    """径向（太阳的光晕、树冠的明暗）。cx/cy/r 用 0..1 的相对坐标。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / w - cx) ** 2 + (yy / h - cy) ** 2)) / max(1e-6, r)
    v = np.clip(1.0 - d, 0.0, 1.0)
    return (1.0 - v) if invert else v


def detail_hband(shape, freq=8.0, amp=0.35):
    """水平条纹（水波、横向的岩层）。"""
    h, w = shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    return _norm01((0.5 + 0.5 * np.sin(yy / max(1.0, h / freq) * np.pi * 2)) * amp + 0.5)


# ================================================================ 区域

class Region:
    """一个语义区域：遮罩 + 固定调色板 + 细节场。

    palette: 有序颜色列表（[(r,g,b), ...]）。顺序有意义 ——
             细节值 0 取第一个，1 取最后一个。所以按"暗->亮"或
             "远->近"排好，出来的画面才符合直觉。
    detail:  0..1 的浮点数组；给 None 表示这一区是纯色（不细分）。
    octaves: 细节的倍频程列表（由 fbm_octaves 得到）。给了它就会按层
             逐级累加，这是"加法"渐变的关键。
    """

    def __init__(self, name, mask, palette, detail=None, octaves=None,
                 base_index=None, idx_map=None):
        self.name = name
        # idx_map：逐格的调色板下标（可选）。给了就直接用，不靠 detail 推。
        # 用途：脸部这类"从参考图贴过来"的区域，参考图直接给出每格该是什么色。
        self.idx_map = None if idx_map is None else np.asarray(idx_map, np.int32)
        self.mask = np.asarray(mask, bool)
        self.palette = [tuple(int(v) for v in c[:3]) for c in palette]
        assert self.palette, '区域 %s 的调色板是空的' % name
        self.detail = None if detail is None else np.asarray(detail, np.float32)
        self.octaves = octaves
        self.base_index = base_index
        # ⚠️ 归一化尺度必须**只用全部倍频程的和算一次**，各层共用同一把尺子。
        #    第一版是每层各自 _norm01(累加结果)，结果"多累加一层"带来的幅度增长
        #    被归一化抹平 —— 实测某区域的细节离散度反而**逐层下降**
        #    （0.200 -> 0.141 -> 0.133 -> 0.128），加法语义丢失。
        #    第二版又把 _lo/_hi 算成"倍频程+细节场"的范围，却拿它去归一化
        #    **只有倍频程**的累加值 —— 尺子和被量的东西不一致，仍然不单调。
        #    现在：_lo/_hi 只由**倍频程之和**决定，细节场在归一化之后再混入。
        if octaves is not None and len(octaves):
            full = np.zeros_like(np.asarray(octaves[0], np.float32))
            for k in range(len(octaves)):
                full = full + np.asarray(octaves[k], np.float32) * (0.5 ** k)
            self._lo = float(full.min())
            self._hi = float(full.max())

    def detail_at(self, level, n_levels):
        """第 level 层的细节场（0..1）。

        ⚠️ 这里是"加法"的核心：第 0 层只取最粗的一个倍频程，
           每升一层多累加一个，所以细节是**逐层加上去**的，不是减出来的。
           归一化用构造时定好的**同一把尺子**（见 __init__ 的说明）。
        """
        if self.octaves is not None:
            n = min(len(self.octaves), level + 1)
            acc = np.zeros_like(self.octaves[0], np.float32)
            for k in range(n):
                acc = acc + np.asarray(self.octaves[k], np.float32) * (0.5 ** k)
            v = (acc - self._lo) / max(1e-6, self._hi - self._lo)
            v = np.clip(v, 0, 1).astype(np.float32)
            if self.detail is not None:
                v = np.clip(0.5 * v + 0.5 * _norm01(self.detail), 0, 1)
            return v
        if self.detail is not None:
            # 没有倍频程时，用"向基准收敛"的方式模拟分层：
            # 低层往基准色收，高层才显出完整细节
            base = 0.5 if self.base_index is None else self.base_index / \
                max(1, len(self.palette) - 1)
            t = (level + 1) / float(n_levels)
            return np.clip(base + (self.detail - base) * t, 0, 1).astype(np.float32)
        # 纯色区：常量
        b = 0.5 if self.base_index is None else self.base_index / \
            max(1, len(self.palette) - 1)
        return np.full(self.mask.shape, b, np.float32)

    def color_index(self, level, n_levels):
        """该层每格的调色板下标。

        有 idx_map 就直接用它（逐格指定，最准）；
        否则由细节值推（细节值 -> 下标）。
        """
        if self.idx_map is not None:
            K = len(self.palette)
            return np.clip(self.idx_map, 0, K - 1).astype(np.int32)
        d = self.detail_at(level, n_levels)
        K = len(self.palette)
        idx = np.clip((d * K).astype(np.int32), 0, K - 1)
        if self.base_index is not None and level == 0:
            idx = np.full_like(idx, int(np.clip(self.base_index, 0, K - 1)))
        return idx


# ================================================================ 场景

class BeadScene:
    """从白画布逐层长出拼豆图。

    size:   最终格数 (W, H)
    levels: 总层数。第 0 层最粗，第 levels-1 层最细（= size）。
    """

    def __init__(self, size=(56, 56), levels=4, bg=(255, 255, 255), seed=0):
        self.W, self.H = int(size[0]), int(size[1])
        assert self.W > 0 and self.H > 0
        self.levels = int(levels)
        assert self.levels >= 1
        self.bg = tuple(int(v) for v in bg[:3])
        self.seed = seed
        self.regions = []
        # 第 L 层的格尺寸（必须能整除 size）
        self.cells = [max(1, 2 ** (self.levels - 1 - L)) for L in range(self.levels)]
        for L, c in enumerate(self.cells):
            assert self.W % c == 0 and self.H % c == 0, \
                '第 %d 层的格尺寸 %d 不能整除画布 %dx%d' % (L, c, self.W, self.H)

    # ---------------- 建区域
    def add(self, name, mask, palette, detail=None, octaves=None, base_index=None,
            idx_map=None):
        m = np.asarray(mask, bool)
        assert m.shape == (self.H, self.W), \
            '区域 %s 的遮罩形状 %s 与画布 (%d,%d) 不符' % (name, m.shape, self.H, self.W)
        if idx_map is not None:
            assert np.asarray(idx_map).shape == (self.H, self.W), \
                '区域 %s 的 idx_map 形状不符' % name
        self.regions.append(Region(name, m, palette, detail, octaves, base_index,
                                   idx_map))
        return self

    # ---------------- 渲染一层
    def render_level(self, level, scale=None, bead_style=False):
        """渲染第 level 层，返回 (PIL 图, 每格颜色数组 (H,W,3), 统计)。

        ⚠️ 归属用**覆盖率取最大**，不是"覆盖率 > 0.5 才算"。
           第一版用阈值，结果区域交界处的细格两边都不满足 -> 没有归属 ->
           露出背景白（实测山上出现白洞、水上方有一条白带）。
           取最大则保证每个格**必有**归属。
        """
        assert 0 <= level < self.levels, '层号 %d 越界（共 %d 层）' % (level, self.levels)
        c = self.cells[level]
        ch, cw = self.H // c, self.W // c
        out = np.zeros((ch, cw, 3), np.uint8)
        out[:] = self.bg

        # ① 每个区域在每个粗格里的覆盖率
        cov = np.zeros((len(self.regions), ch, cw), np.float32)
        for ri, r in enumerate(self.regions):
            cov[ri] = r.mask[:ch * c, :cw * c].reshape(ch, c, cw, c).mean((1, 3))
        # ② 归属 = 覆盖率最大的那个区域（并列时取先加入的）
        owner = cov.argmax(0)
        # 覆盖率全为 0 的格子（理论上不该有）保持背景
        empty = cov.max(0) <= 0

        # ③ 每个区域在该层的颜色 -> 归并到粗格
        for ri, r in enumerate(self.regions):
            idx = r.color_index(level, self.levels)
            K = len(r.palette)
            pick = np.zeros((self.H, self.W), np.int32)
            for k in range(K):
                pick[r.mask & (idx == k)] = k
            blk = pick[:ch * c, :cw * c].reshape(ch, c, cw, c)
            blk = blk.transpose(0, 2, 1, 3).reshape(ch, cw, c * c)
            counts = np.zeros((ch, cw, K), np.int32)
            for k in range(K):
                counts[:, :, k] = (blk == k).sum(2)
            mode = counts.argmax(2)

            sel = (owner == ri) & ~empty
            for k in range(K):
                s = sel & (mode == k)
                if s.any():
                    out[s] = r.palette[k]

        stats = self._stats(out, c)
        return self._to_image(out, c, scale=scale, bead_style=bead_style), out, stats

    def grow(self, scale=None, bead_style=False):
        """渲染所有层。返回 [(图, 每格数组, 统计), ...]，从最粗到最细。"""
        return [self.render_level(L, scale=scale, bead_style=bead_style)
                for L in range(self.levels)]

    # ---------------- 输出
    @staticmethod
    def scale_for(cells, c, target_h=None, default=None):
        """按目标出图高度算每颗豆的渲染像素。

        target_h 给了就按它算（比如 1080）；否则用 default（没给则 12）。
        必须取整，且至少 1。
        """
        if target_h:
            return max(1, int(round(target_h / cells.shape[0])))
        return max(1, int(default if default else 12))

    def _to_image(self, cells, c, scale=None, bead_style=False):
        """格阵 -> 出图。

        scale       每颗豆渲染成多少像素（None = 按原来的自适应值）
        bead_style  True 时把每颗豆画成圆角 + 高光 + 细缝，像真的拼豆
        """
        if scale is None:
            k = max(4, 16 // max(1, c // 2)) if c > 1 else 12
        else:
            k = max(1, int(scale))
        if not bead_style or k < 4:
            im = Image.fromarray(cells, 'RGB')
            return im.resize((cells.shape[1] * k, cells.shape[0] * k), Image.NEAREST)
        return self._bead_render(cells, k)

    @staticmethod
    def _bead_render(cells, k):
        """把每颗豆画成真的拼豆：方底 + 圆角 + 左上高光 + 右下暗边 + 细缝。

        在 k >= 8 时效果明显；k 小时自动退化成方块（见 _to_image 的判断）。
        """
        ch, cw = cells.shape[:2]
        W, H = cw * k, ch * k
        # 用超采样画圆角，边缘干净
        ss = 4
        base = Image.new('RGB', (W * ss, H * ss), (0, 0, 0))
        from PIL import ImageDraw
        d = ImageDraw.Draw(base)
        gap = max(1, k // 12) * ss                     # 豆与豆之间的缝
        rad = max(1, k // 5) * ss                      # 圆角半径
        for y in range(ch):
            for x in range(cw):
                r, g, b = (int(v) for v in cells[y, x])
                x0, y0 = x * k * ss + gap, y * k * ss + gap
                x1, y1 = (x + 1) * k * ss - gap, (y + 1) * k * ss - gap
                if x1 <= x0 or y1 <= y0:
                    continue
                d.rounded_rectangle([x0, y0, x1 - 1, y1 - 1], radius=rad,
                                    fill=(r, g, b))
                # 左上高光
                hl = tuple(min(255, int(v + (255 - v) * 0.34)) for v in (r, g, b))
                hh = max(1, int((y1 - y0) * 0.30))
                hw = max(1, int((x1 - x0) * 0.30))
                d.ellipse([x0 + (x1 - x0) * 0.12, y0 + (y1 - y0) * 0.12,
                           x0 + (x1 - x0) * 0.12 + hw, y0 + (y1 - y0) * 0.12 + hh],
                          fill=hl)
                # 右下暗边
                sh = tuple(max(0, int(v * 0.72)) for v in (r, g, b))
                d.arc([x0, y0, x1 - 1, y1 - 1], start=20, end=160,
                      fill=sh, width=max(1, k // 10) * ss)
        return base.resize((W, H), Image.LANCZOS)

    def _stats(self, cells, c):
        """每层的豆子清单 —— 这是"加法"路线相比降采样最实用的产出。"""
        flat = cells.reshape(-1, 3)
        uniq, cnt = np.unique(flat, axis=0, return_counts=True)
        order = np.argsort(-cnt)
        lst = []
        for i in order:
            r, g, b = (int(v) for v in uniq[i])
            lst.append({'rgb': (r, g, b), 'hex': '#%02X%02X%02X' % (r, g, b),
                        'n': int(cnt[i])})
        return {'cell': c, 'grid': (cells.shape[1], cells.shape[0]),
                'total_beads': int(cells.shape[0] * cells.shape[1]),
                'n_colors': len(lst), 'colors': lst}


# ================================================================ 快捷遮罩

def mask_rect(shape, x0, y0, x1, y1):
    """相对坐标 (0..1) 的矩形。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx / w >= x0) & (xx / w < x1) & (yy / h >= y0) & (yy / h < y1))


def mask_circle(shape, cx, cy, r):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx / w - cx) ** 2 + (yy / h - cy) ** 2) <= r * r


def mask_ellipse(shape, cx, cy, rx, ry):
    """**按格数**定半径的椭圆（cx/cy 用归一化位置，rx/ry 用格数）。

    ⚠️ 别用 mask_circle：它用归一化坐标，在非正方画布上圆会变成椭圆。
       实测 112x192 上 r=0.14 得到 横15.7格 × 竖26.9格，脸被拉成竖长条。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - cx * w) / max(1e-6, rx)) ** 2 +
            ((yy - cy * h) / max(1e-6, ry)) ** 2) <= 1.0


def mask_poly(shape, pts):
    """相对坐标多边形（射线法）。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = xx / w, yy / h
    inside = np.zeros((h, w), bool)
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cond = ((y0 > Y) != (y1 > Y))
        with np.errstate(divide='ignore', invalid='ignore'):
            xin = (x1 - x0) * (Y - y0) / np.where(y1 - y0 == 0, 1e-9, y1 - y0) + x0
        inside ^= cond & (X < xin)
    return inside


def mask_below_curve(shape, curve_y):
    """曲线下方的区域。curve_y 是长度 = 宽度的 0..1 数组（山的轮廓用）。"""
    h, w = shape
    cy = np.asarray(curve_y, np.float32)
    if cy.shape[0] != w:
        cy = np.interp(np.linspace(0, 1, w), np.linspace(0, 1, cy.shape[0]), cy)
    yy = np.arange(h, dtype=np.float32)[:, None] / h
    return yy >= cy[None, :]
