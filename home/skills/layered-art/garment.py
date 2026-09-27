# -*- coding: utf-8 -*-
"""从**人体十块**上生成贴身的衣壳 —— "先画人体，再画适配的衣服"。

MakeHuman 自带的 clothes 里**没有 JK 制服**（实测 19 个全是西装/运动服/连体裤，
`male_*` 还是男装），所以衣服自己画。

做法：取人体某个块的三角面，沿**顶点法线**外移一个距离 -> 得到一层贴身的衣壳。
外移量可以随高度变化（裙摆外扩）或随周向角度变化（百褶裙的褶）。
衣壳的边缘是开放边界（不封口），平面渲染色块看不出问题。
"""
import numpy as np

__all__ = ['vertex_normals', 'offset_shell', 'blouse', 'skirt', 'plate']


def vertex_normals(V, F, sel=None):
    """选中面的顶点法线（相邻面法线平均）。"""
    idx = np.nonzero(sel)[0] if sel is not None else np.arange(len(F))
    nrm = np.zeros_like(V)
    a, b, c = V[F[idx, 0]], V[F[idx, 1]], V[F[idx, 2]]
    fn = np.cross(b - a, c - a)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    np.add.at(nrm, F[idx, 0], fn)
    np.add.at(nrm, F[idx, 1], fn)
    np.add.at(nrm, F[idx, 2], fn)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return nrm, idx


def offset_shell(V, F, sel, dist=0.02, side_axis=0, up_axis=1):
    """取 sel 选中的面，沿顶点法线外移，返回 (新顶点表, 新面表)。

    dist : 常数，或 callable(顶点数组) -> 每顶点一个外移量。
           位置相关的量故意都从顶点坐标算，不依赖面索引。
    """
    nrm, idx = vertex_normals(V, F, sel)
    used = np.unique(F[idx])
    Vn = V.copy()
    if callable(dist):
        d = np.asarray(dist(V[used]), float)
    else:
        d = np.full(len(used), float(dist))
    Vn[used] = V[used] + nrm[used] * d[:, None]
    return Vn, F[idx]


def blouse(V, F, part, up_axis=1, ua=None, dist=0.022, sleeve_t=0.70):
    """JK 衬衫（水手服上衣）：胸 + 腰 + 胳膊上半段（短袖）。"""
    ua = up_axis if ua is None else ua
    y = V[:, ua]
    h = float(y.max() - y.min())
    t = (y - y.min()) / max(h, 1e-9)
    tc = V[F].mean(axis=1)[:, ua]
    tc = (tc - y.min()) / max(h, 1e-9)

    sel = (part == '胸') | (part == '腰')
    arm = (part == '胳膊') & (tc > sleeve_t)
    sel = sel | arm
    return offset_shell(V, F, sel, dist=dist, up_axis=ua), sel


def skirt(V, F, part, up_axis=1, side_axis=0, depth_axis=2,
          hem_t=0.38, base=0.030, flare=0.075, pleat=0.16, pleat_n=14):
    """JK 百褶裙：从胯块向下，越往下外扩越多；周向按 cos 调制出褶。

    hem_t  : 裙摆高度（七头身里膝在 0.286，JK 裙在膝上 -> 0.38 左右）
    flare  : 从胯（t=0.50）到裙摆的外扩量
    pleat  : 褶的幅度（相对 base+flare 的比例）
    """
    ua = up_axis
    y = V[:, ua]
    h = float(y.max() - y.min())
    t = (y - y.min()) / max(h, 1e-9)

    # 取 胯 + 大腿的**上半段**（大腿下半段要露出来）
    tc = V[F].mean(axis=1)[:, ua]
    tc = (tc - y.min()) / max(h, 1e-9)
    sel = (part == '胯') | ((part == '大腿') & (tc > hem_t))
    sidx = np.nonzero(sel)[0]
    used = np.unique(F[sidx])

    xc = float((V[:, side_axis].max() + V[:, side_axis].min()) / 2)
    zc = float((V[:, depth_axis].max() + V[:, depth_axis].min()) / 2)

    def dist(pts):
        tt = (pts[:, ua] - y.min()) / max(h, 1e-9)
        # 从胯往下线性外扩
        f = np.clip((0.50 - tt) / max(0.50 - hem_t, 1e-6), 0.0, 1.0)
        d = base + flare * f
        # 周向褶：绕竖直轴的角度
        ang = np.arctan2(pts[:, side_axis] - xc, pts[:, depth_axis] - zc)
        d = d * (1.0 + pleat * np.cos(pleat_n * ang))
        return d

    return offset_shell(V, F, sel, dist=dist, up_axis=ua), sel


def sailor_collar(V, F, part, up_axis=1, side_axis=0, depth_axis=2,
                  neck_t=0.80, apex_t=0.70, out=0.030):
    """水手领：**前 V 字两条带 + 后领方片**。

    ⚠️ 上一版只做了一块"后领方片"，结果它是背面的板，从正面看就是胸前两块
       莫名其妙的方块。真水手服的结构是：
         后领 = 横跨肩背的大方片；前领 = 从两肩斜下到胸中央的 V 字两条。
       这两样必须分开建面，位置也不一样。
    返回 (verts, faces)。
    """
    ua = up_axis
    y = V[:, ua]
    h = float(y.max() - y.min())
    tc = V[F].mean(axis=1)[:, ua]
    tc = (tc - y.min()) / max(h, 1e-9)

    sel = (part == '胸') & (tc >= apex_t - 0.02) & (tc <= neck_t + 0.04)
    if not sel.any():
        return None, None
    vidx = np.unique(F[sel])
    p = V[vidx]
    xs = p[:, side_axis]
    half = float(np.percentile(np.abs(xs - (xs.max() + xs.min()) / 2), 95))
    mid = float((xs.max() + xs.min()) / 2)
    zf = float(np.percentile(p[:, depth_axis], 100)) + out      # 前面
    zb = float(np.percentile(p[:, depth_axis], 0)) - out        # 背面
    y_top = y.min() + neck_t * h
    y_apex = y.min() + apex_t * h
    y_bot = y.min() + (neck_t + 0.035) * h

    verts, faces = [], []
    # 前 V：左右各一条梯形带，从肩外侧斜到胸中央
    for sgn in (-1, 1):
        i0 = len(verts)
        x_out = mid + sgn * half * 0.98
        x_in_top = mid + sgn * half * 0.30
        x_in_bot = mid + sgn * half * 0.06
        verts += [[x_out, y_top, zf * 0.995],
                  [x_in_top, y_top, zf],
                  [x_in_bot, y_apex, zf],
                  [x_out, y_apex + 0.012 * h, zf * 0.995]]
        faces += [[i0, i0 + 1, i0 + 2], [i0, i0 + 2, i0 + 3]]
    # 后领：横跨肩背的一片。⚠️ 不能做太宽、也不能提到脖子高度 ——
    #   上一版宽 1.02 倍胸半宽 + 顶到 t=0.80，结果从正面看是颈侧戳出的两根横杠
    #   （像肩章）。水手服的后领片是**贴着上背**的，正面只该在肩头露一线。
    i0 = len(verts)
    bw = half * 0.74
    yb_lo = y.min() + (apex_t + 0.010) * h
    yb_hi = y.min() + (neck_t - 0.010) * h
    verts += [[mid - bw, yb_lo, zb],
              [mid + bw, yb_lo, zb],
              [mid + bw, yb_hi, zb],
              [mid - bw, yb_hi, zb]]
    faces += [[i0, i0 + 1, i0 + 2], [i0, i0 + 2, i0 + 3]]
    return np.array(verts, float), np.array(faces, np.int32)
