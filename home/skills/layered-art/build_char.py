"""完整链路 v3：MakeHuman 变形 → **七头身十块分区** → 上色出图。

v1 失败的原因：基础网格没有按身体部位分组（72.4% 的面同属一个 "body" 组，
`fgstr`/`fgidx` 还是偏移表不是组号），所以拿不到"头/躯干/四肢"的语义分块。

v2 改成**按面重心的三维位置分区** —— 位置是确定的，不依赖网格给不给语义：

    高度 t（0=脚底, 1=头顶） + 横向偏移 w（|x-中轴|/身高）
        t > 0.885             → 头发
        0.795 < t ≤ 0.885     → 头
        0.755 < t ≤ 0.795 且 w<0.06 → 颈
        w > 0.20              → 手
        w > 0.105 且 t > 0.45 → 手臂
        t < 0.06              → 脚
        t < 0.50              → 腿
        其余                  → 躯干

分区确定了 → 每区一个语义色 → 平面着色渲染（自带明暗）+ 描边 = **彩色成品**。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mhsource as M      # noqa: E402
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


import mesh as ME         # noqa: E402

OUT = _out_dir('char')
SIZE = (744, 1754)

# 角色配方：变形名 → 权重
RECIPES = {
    '灰羽翎-体形': {'targets/macrodetails/asian-female-young': 1.0},
    '男性青年': {'targets/macrodetails/caucasian-male-young': 1.0},
    '儿童': {'targets/macrodetails/asian-female-child': 1.0},
}

PALETTES = {
    '夜蓝制服': dict(hair=(206, 212, 226), skin=(240, 222, 214), cloth=(56, 66, 106),
                     cloth2=(44, 52, 84), foot=(40, 42, 54)),
    '樱粉': dict(hair=(246, 228, 234), skin=(250, 232, 224), cloth=(206, 116, 148),
                 cloth2=(160, 80, 112), foot=(72, 40, 56)),
    '硝烟灰': dict(hair=(226, 224, 220), skin=(238, 226, 218), cloth=(120, 118, 112),
                   cloth2=(92, 90, 86), foot=(52, 52, 50)),
}


# ---------------------------------------------------------------------------
# 七头身 canon（从脚底 0 到头顶 1，单位 = 身高；1 头 = 1/7 = 0.1429）
#   脚底 0.000   脚踝 0.043(0.3头)   膝 0.286(2头)   胯 0.500(3.5头)
#   腰   0.571(4头)  胸 0.643(4.5头)  肩 0.786(5.5头)  下巴 0.857(6头)  头顶 1.000
# ---------------------------------------------------------------------------
CANON7 = {
    '脚底': 0.000, '脚踝': 0.043, '膝': 0.286, '胯': 0.500, '腰': 0.571,
    '胸': 0.643, '肩': 0.786, '下巴': 0.857, '头顶': 1.000,
}
# 横向阈值（|x-中轴| / 身高）
W_ARM = 0.105      # 超过它算胳膊
W_HAND = 0.200     # 超过它算手
# 脖子半宽。⚠️ 原来写 0.050 -> 总宽 0.10 倍身高 ≈ 1 个头宽，那是肩不是脖子！
#   真实女性颈径约 0.07 倍身高 -> 半宽 0.035。这里是**解剖定义**，
#   配合配方里的 neck-scale-horiz-decr 一起把脖子收细。
W_NECK = 0.035

# 分块图的配色：十块各一个明显不同的颜色，用来核验 canon 切得对不对
BLOCK_COLORS = {
    '头': (240, 222, 214), '脖子': (232, 196, 178),
    '胸': (86, 122, 178), '腰': (206, 138, 96), '胯': (150, 102, 168),
    '大腿': (96, 172, 122), '小腿': (78, 148, 178), '脚': (60, 60, 76),
    '胳膊': (222, 186, 96), '手': (206, 108, 120), '头发': (198, 204, 218),
}


def face_adjacency(F):
    """面邻接表：共享一条边的两个面互为邻居。"""
    from collections import defaultdict
    e = defaultdict(list)
    for i, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            e[(u, v) if u < v else (v, u)].append(i)
    adj = [[] for _ in range(len(F))]
    for fs in e.values():
        if len(fs) > 1:
            for i in fs:
                adj[i].extend(j for j in fs if j != i)
    return adj


def smooth_partition(part, adj, iters=3, need=3):
    """按网格邻接做多数表决，抹掉分区边界上的锯齿。

    ⚠️ 分区是按**面**重心算的，所以边界只能沿三角面走 —— 一定锯齿。
       按顶点分区也不行（一个面跨边界时仍只能取一个色）。
       正解是在**面邻接图**上平滑：某个面如果至少 need 个邻居都是别的区，
       就跟着改。need=3 能抹掉孤立三角形，又不会把细结构吃掉。
    """
    names = sorted(set(part.tolist()))
    idx = {n: i for i, n in enumerate(names)}
    lab = np.array([idx[p] for p in part], np.int32)
    for _ in range(iters):
        new = lab.copy()
        for i, a in enumerate(adj):
            if not a:
                continue
            u, c = np.unique(lab[a], return_counts=True)
            k = int(c.argmax())
            if c[k] >= need and u[k] != lab[i]:
                new[i] = u[k]
        lab = new
    return np.array([names[i] for i in lab], dtype=object)


def partition(V, F, sa, ua, da):
    """按**七头身 canon + 横向判据**分成十块。返回 (每面的区域名, 计数)

    十个区：头 脖子 手 胳膊 胸 腰 胯 大腿 小腿 脚（外加素体头发壳 = 头发）

    为什么不用"纯几何阈值"：上一版脖子只按 t 和 w<0.05 切，而 0.05 的半宽
    等于总宽 0.10 倍身高、约 1 个头宽 —— 那切出来的是肩，不是脖子，
    所以脖子怎么调都粗。脖子必须由**解剖宽度**定义（W_NECK）。
    """
    import collections
    c = V[F].mean(axis=1)
    y, x = c[:, ua], c[:, sa]
    h = float(y.max() - y.min())
    t = (y - y.min()) / max(h, 1e-9)                 # 0 = 脚底, 1 = 头顶
    xc = (x.max() + x.min()) / 2
    w = np.abs(x - xc) / max(h, 1e-9)                # 横向偏移（相对身高）
    z = c[:, da]

    part = np.full(len(c), '胸', dtype=object)       # 肩线下、腰线上的默认是胸
    part[t < CANON7['脚踝']] = '脚'
    part[(t >= CANON7['脚踝']) & (t < CANON7['膝'])] = '小腿'
    part[(t >= CANON7['膝']) & (t < CANON7['胯'])] = '大腿'
    part[(t >= CANON7['胯']) & (t < CANON7['腰'])] = '胯'
    part[(t >= CANON7['腰']) & (t < CANON7['胸'])] = '腰'
    # 脖子：肩线 -> 下巴，**且横向收窄**（这才是脖子的解剖定义）
    part[(t >= CANON7['肩']) & (t < CANON7['下巴']) & (w < W_NECK)] = '脖子'

    # 头区：素体头发壳和头在同一高度，要再按深度劈一刀
    HEAD = t >= CANON7['下巴']
    if HEAD.any():
        zt = z[HEAD]
        zc = float(np.median(zt))
        span = float(zt.max() - zt.min()) or 1.0
        front = z[HEAD] > (zc - 0.10 * span)
        sub = np.where(front, '头', '头发')
        sub = np.where(t[HEAD] > 0.945, '头发', sub)      # 头顶一律头发
        part[HEAD] = sub

    # 四肢放最后：横向判据最可靠，让它盖过高度判据
    below_head = t < CANON7['下巴']
    part[below_head & (w > W_ARM) & (t > CANON7['胯'])] = '胳膊'
    # 手：**不能再用横向阈值**。A-pose 下前臂一直斜向外，w>0.20 会把整条前臂
    # 都判成手（实测手占了肘以下全部，比上臂还大）。正解是沿"肩 -> 指尖"这条轴
    # 切远段：每侧各取一条轴（近端 = 最靠中轴的点，远端 = 最外的点），远端 22% 才是手。
    for side in (-1, 1):
        sel = below_head & (w > W_ARM) & (t > CANON7['胯']) & ((x - xc) * side > 0)
        if not sel.any():
            continue
        idx = np.nonzero(sel)[0]
        ac = c[idx]
        d = (ac[:, sa] - xc) * side
        p_near, p_far = ac[np.argmin(d)], ac[np.argmax(d)]
        ax = p_far - p_near
        nn = np.linalg.norm(ax)
        if nn < 1e-9:
            continue
        s = (ac - p_near) @ (ax / nn)
        part[idx[s >= s.max() * 0.78]] = '手'
        part[idx[s < s.max() * 0.78]] = '胳膊'
    return part, collections.Counter(part.tolist())


def slim_legs(V, F, part, sa, ua, t_hip=0.50, t_ankle=0.043,
              k_thigh=0.55, k_calf=0.60, k_foot=0.75, k_center=0.72):
    """把腿横向收细。返回新顶点表。

    为什么不用 MakeHuman 的变形目标：实测
    `l/r-upperleg-scale-horiz-decr`、`l/r-lowerleg-fat-decr`、
    `measure-thigh-circ-decr` 等 6 组配方量出来的腿宽**一模一样**
    （0.279 倍身高）—— 这些目标在这个基础网格上完全不生效，apply_targets
    静默跳过。所以改成直接改几何，效果可测。

    ⚠️ 两个量都要收，缺一个都不够：
       · k_center：把整条腿**连轴线一起**往全身中轴收。只收半宽是不够的 ——
         实测 k=0.55 只把小腿从 0.279 降到 0.245（12%），因为整块宽度主要是
         **两条腿轴线的间距**贡献的，不是半宽贡献的。
       · k_*：再按各自的腿轴收半宽，让腿本身变细。
    实测：腿并拢时整块 0.279 身高 / 每条 0.138（真实小腿 0.065 -> 粗 2.1 倍）。
    """
    Vn = V.copy()
    y = V[:, ua]
    h = float(y.max() - y.min())
    xc = float((V[:, sa].max() + V[:, sa].min()) / 2)

    for name, k in (('大腿', k_thigh), ('小腿', k_calf), ('脚', k_foot)):
        sel = np.nonzero(part == name)[0]
        if not len(sel):
            continue
        used = np.unique(F[sel])
        # (1) 连轴线一起往中轴收
        Vn[used, sa] = xc + (V[used, sa] - xc) * k_center
        # (2) 再按各自腿轴收半宽
        for sgn in (-1, 1):
            side = used[((V[used, sa] - xc) * sgn) > 0]
            if not len(side):
                continue
            tt = (V[side, ua] - y.min()) / max(h, 1e-9)
            lo_t, hi_t = tt.min(), tt.max()
            axis = np.zeros(len(side))
            nb = 8
            for b in range(nb):
                lo = lo_t + (hi_t - lo_t) * b / nb
                hi = lo_t + (hi_t - lo_t) * (b + 1) / nb
                m = (tt >= lo) & (tt <= hi)
                # 轴线要在 (1) 之后的位置上算
                axis[m] = (float(Vn[side][m, sa].mean()) if m.any()
                           else xc + (xc - xc) * k)
            Vn[side, sa] = axis + (Vn[side, sa] - axis) * k
    return Vn


def colorize(part, pal):
    """十块 -> 语义色。**人体阶段**用色只有皮肤/脚；衣服是后面单独一层。"""
    m = {'头': pal['skin'], '脖子': pal['skin'],
         '手': pal['skin'], '胳膊': pal['skin'],
         '胸': pal['cloth'], '腰': pal['cloth'], '胯': pal['cloth2'],
         '大腿': pal['cloth2'], '小腿': pal['cloth2'],
         '脚': pal['foot'], '头发': pal['hair']}
    return np.array([m.get(p, pal['cloth']) for p in part], float)


def colorize_blocks(part):
    """分块图：十块各一个明显不同的颜色，用来核验 canon 切得对不对。"""
    return np.array([BLOCK_COLORS.get(p, (255, 0, 255)) for p in part], float)


def main():
    t0 = time.time()
    b = M.load_base()
    V, F = b['coord'], b['faces']
    sa, ua, da = ME.guess_axes(V)
    kw = dict(up_axis=ua, side_axis=sa, depth_axis=da)
    print('基础网格 %d 顶点 / %d 三角面  身高 %.2f  轴(水平%d 竖直%d 深度%d)'
          % (len(V), len(F), M.height(V), sa, ua, da))

    import collections
    adj = face_adjacency(F)
    print('面邻接表建好：平均每面 %.2f 个邻居' % (sum(len(a) for a in adj) / len(adj)))
    tiles = []
    for rname, rec in RECIPES.items():
        V2 = M.apply_targets(V, b['root'], rec)
        part, cnt = partition(V2, F, sa, ua, da)
        part = smooth_partition(part, adj, iters=3)
        cnt = collections.Counter(part.tolist())
        print('\n--- %s   身高 %.2f（%+.1f%%）' % (
            rname, M.height(V2), (M.height(V2) / M.height(V) - 1) * 100))
        print('    分区: %s' % '  '.join('%s %d' % (k, v) for k, v in cnt.most_common()))
        ME.render_flat(V2, F, SIZE, outline=1, **kw).save(
            os.path.join(OUT, '%s_白膜.png' % rname))
        for pname, pal in PALETTES.items():
            img = ME.render_flat(V2, F, SIZE, bg=(250, 250, 252),
                                 face_rgb=colorize(part, pal),
                                 light=(-0.45, 0.55, -0.7), ambient=0.46,
                                 outline=1, **kw)
            img.save(os.path.join(OUT, '%s_%s.png' % (rname, pname)))
            if pname == '夜蓝制服':
                tiles.append((rname, img))
    from render import contact_sheet
    contact_sheet(tiles, os.path.join(OUT, 'char_sheet.png'), tile_w=340, cols=3)
    print('\n产物在 %s   总耗时 %.1f 秒' % (OUT, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
