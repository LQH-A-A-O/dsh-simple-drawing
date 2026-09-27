# -*- coding: utf-8 -*-
"""方块人体 —— 像《我的世界》那样用**轴对齐方块**拼角色，
"粗细"不靠几何锥度，靠**像素梯度**（每个侧面切成竖条，逐条一个亮度）。

和 mesh 那条路的区别
--------------------
· 网格路线：面可以任意朝向 -> 圆润，但边界只能沿三角面走 -> 一定锯齿。
· 方块路线：只有轴对齐的面 -> 轮廓天然是直角/阶梯，正是 MC 的味道；
  "圆柱感"和"变细"全靠侧面上的竖条亮度梯度造出来。

用法
----
    blocks = figure_boxes()            # 十块，按七头身 canon 摆好
    V, F, rgb = merge(blocks)          # 合成一份顶点/面/颜色
    img = mesh.render_regions(V, F, SIZE, rgb, shade=False, ...)

⚠️ shade 必须关掉：亮度**已经写进 rgb 了**（每条竖条一个色）。
   再叠一次光照会得到两套明暗打架的脏结果。
"""
import numpy as np

__all__ = ['CANON_BOX', 'box', 'figure_boxes', 'merge', 'RAMP']

# 七头身 canon（同 build_char.CANON7）
CY = {'脚底': 0.000, '脚踝': 0.043, '膝': 0.286, '胯': 0.500, '腰': 0.571,
      '胸': 0.643, '肩': 0.786, '下巴': 0.857, '头顶': 1.000}

# 侧面竖条的亮度梯度：暗-亮-暗，看着就是根圆柱。
# 条数取奇数，中间一条最亮 —— 这就是"靠像素梯度做出粗细"的核心。
RAMP = (0.74, 0.88, 1.00, 0.90, 0.78)

# 十块的参数：(下界t, 上界t, 半宽, 半深, 中心x)
# 半宽/半深都是**相对身高**。取值来自之前实测的宽度（胸 0.180、腰 0.162、
# 胯 0.190、头 0.099、脖 0.062、单腿小腿 0.065 ...）。
CANON_BOX = [
    ('头',   CY['下巴'], CY['头顶'], 0.050, 0.058, 0.000, 0.000, 'skin'),
    ('脖子', CY['肩'],   CY['下巴'] + 0.006, 0.031, 0.034, 0.000, 0.000, 'skin'),
    ('胸',   CY['胸'],   CY['肩'] + 0.004, 0.090, 0.056, 0.000, 0.000, 'cloth'),
    ('腰',   CY['腰'],   CY['胸'], 0.081, 0.050, 0.000, 0.000, 'cloth'),
    ('胯',   CY['胯'],   CY['腰'], 0.095, 0.058, 0.000, 0.000, 'cloth2'),
    ('大腿', CY['膝'],   CY['胯'], 0.038, 0.044, 0.050, 0.000, 'skin'),
    ('小腿', CY['脚踝'], CY['膝'], 0.033, 0.038, 0.048, 0.000, 'skin'),
    ('脚',   CY['脚底'], CY['脚踝'], 0.034, 0.056, 0.048, 0.000, 'foot'),
    ('胳膊', 0.560,      CY['肩'] + 0.004, 0.024, 0.027, 0.112, 0.000, 'cloth'),
    ('手',   0.440,      0.560, 0.022, 0.025, 0.112, 0.000, 'skin'),
    # ---- 头发：顶盖 + 两侧发束（都是方块，MC 味）----
    ('发顶', 0.930, 1.008, 0.054, 0.062, 0.000, 0.000, 'hair'),
    ('发侧', 0.800, 0.935, 0.011, 0.050, 0.052, 0.000, 'hair'),
    ('发后', 0.860, 0.935, 0.052, 0.012, 0.000, -0.056, 'hair'),
    # ---- 五官：贴在头面 z=+0.058 上一层的薄片 ----
    ('眼',   0.893, 0.911, 0.012, 0.004, 0.022, 0.057, 'eye'),
    ('嘴',   0.872, 0.876, 0.013, 0.003, 0.000, 0.057, 'eye'),
    # ---- 水手领：胸前的 V 字简化成两块方片 ----
    ('领',   0.760, 0.792, 0.052, 0.006, 0.038, 0.055, 'cloth2'),
]


def _tone(rgb, k):
    return tuple(int(min(255, max(0, c * k))) for c in rgb)


def box(t0, t1, hw, hd, cx=0.0, cz=0.0, ramp=RAMP, y0=0.0, h=1.0,
        base=(255, 255, 255), mirror=True,
        axis_up=1, axis_side=0, axis_depth=2):
    """一个轴对齐方块。返回 (verts, faces, rgb)。

    四个侧面各切成 len(ramp) 条竖条，每条一个亮度 —— 方块因此读起来是圆柱。
    前后两个面不切（正面/背面本来就对着相机）。
    mirror=True 时同时在 -cx 处镜像一个（左右对称的肢体）。
    """
    n = len(ramp)
    out_v, out_f, out_c = [], [], []
    for sgn in ((1, -1) if (mirror and cx != 0) else (1,)):
        x = cx * sgn
        ya, yb = y0 + t0 * h, y0 + t1 * h
        # 每条竖条在宽度上的边界（从 -hw 到 +hw 均分）
        xs = np.linspace(-hw, hw, n + 1)
        vo = len(out_v)

        def add(p):
            out_v.append([0.0, 0.0, 0.0])
            out_v[-1][axis_side] = x + p[0]
            out_v[-1][axis_up] = p[1]
            out_v[-1][axis_depth] = cz + p[2]
            return len(out_v) - 1

        # 前面（z=+hd）：不切条，一个整面用最亮的中间调
        f0 = add((-hw, yb, hd)); f1 = add((hw, yb, hd))
        f2 = add((hw, ya, hd)); f3 = add((-hw, ya, hd))
        out_f += [[f0, f1, f2], [f0, f2, f3]]
        out_c += [_tone(base, ramp[n // 2])] * 2
        # 后面（z=-hd）
        b0 = add((-hw, yb, -hd)); b1 = add((hw, yb, -hd))
        b2 = add((hw, ya, -hd)); b3 = add((-hw, ya, -hd))
        out_f += [[b0, b2, b1], [b0, b3, b2]]
        out_c += [_tone(base, ramp[0])] * 2
        # 左右两个侧面：切成 n 条竖条，逐条一个亮度
        for zs, order in ((hd, 1), (-hd, -1)):
            for k in range(n):
                xa, xb = xs[k], xs[k + 1]
                # 侧面的条：在深度方向只有一条，宽度方向是 xa->xb
                if order > 0:
                    p = [(xa, yb, zs), (xb, yb, zs), (xb, ya, zs), (xa, ya, zs)]
                else:
                    p = [(xb, yb, zs), (xa, yb, zs), (xa, ya, zs), (xb, ya, zs)]
                i0 = add(p[0]); i1 = add(p[1]); i2 = add(p[2]); i3 = add(p[3])
                out_f += [[i0, i1, i2], [i0, i2, i3]]
                out_c += [_tone(base, ramp[k])] * 2
        # 上/下两个面（只看得到顶面，底面会被挡住）
        t0v = add((-hw, yb, -hd)); t1v = add((hw, yb, -hd))
        t2v = add((hw, yb, hd)); t3v = add((-hw, yb, hd))
        out_f += [[t0v, t1v, t2v], [t0v, t2v, t3v]]
        out_c += [_tone(base, ramp[n // 2])] * 2
        bo0 = add((-hw, ya, hd)); bo1 = add((hw, ya, hd))
        bo2 = add((hw, ya, -hd)); bo3 = add((-hw, ya, -hd))
        out_f += [[bo0, bo1, bo2], [bo0, bo2, bo3]]
        out_c += [_tone(base, ramp[0])] * 2
    return (np.array(out_v, float), np.array(out_f, np.int32),
            np.array(out_c, float))


def figure_boxes(pal, h=15.279, ramp=RAMP, uniform_limbs=False):
    """按 canon 拼出十块。返回 (V, F, rgb) 可直接喂 render_regions。

    uniform_limbs=True 时把大腿/小腿（以及上臂/手）的方块**设成同一宽度**，
    粗细之分完全交给像素梯度 —— 就是"人物是矩形的，靠梯度做粗细"的字面版。
    """
    VS, FS, CS = [], [], []
    off = 0
    for name, t0, t1, hw, hd, cx, cz, col in CANON_BOX:
        if uniform_limbs:
            if name in ('大腿', '小腿'):
                hw, hd = 0.036, 0.042
            if name in ('胳膊', '手'):
                hw, hd = 0.023, 0.026
        # ⚠️ canon 里的半宽/半深是**相对身高**的比例，必须乘身高换成绝对长度。
        #    漏了这一步人物会被压成 1:56 的细条（实测小图变成 16x767）。
        V, F, C = box(t0, t1, hw * h, hd * h, cx=cx * h, cz=cz * h, ramp=ramp,
                      h=h, base=pal.get(col, (200, 60, 60)),
                      mirror=(name not in ('发后', '眼', '嘴', '领')))
        VS.append(V); FS.append(F + off); CS.append(C)
        off += len(V)
    return np.vstack(VS), np.vstack(FS), np.vstack(CS)


def merge(parts):
    """[(V,F,rgb), ...] -> 合并成一份。"""
    VS, FS, CS, off = [], [], [], 0
    for V, F, C in parts:
        VS.append(V); FS.append(F + off); CS.append(C)
        off += len(V)
    return np.vstack(VS), np.vstack(FS), np.vstack(CS)
