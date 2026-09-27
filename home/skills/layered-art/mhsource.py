"""MakeHuman 数据源：直接读它的 .npz，**不驱动它的程序**。

MakeHuman 官方没有无头接口（CLI 只有 --version/--license 之类，没有 --script/--nogui），
`core` 还是命名空间包，`Human` 类长在 GUI 层（apps/human.py）。但它把数据全放在两个 npz 里：

    data/3dobjs/base.npz    coord(19158,3) 顶点 / fvert(18486,4) 四边形面
                            group + fgstr + fgidx  ← **身体部位分组**
    data/targets.npz        1280 个稀疏变形：<名字>.index(uint16 顶点号) + <名字>.vector(int16 位移)

所以套变形就是： coord[idx] += vector * weight * SCALE

⚠️ SCALE 是标定出来的：base 的坐标单位不是米（身高约 18 个单位），
   vector 是 int16，实际位移 = vector * SCALE。见 calibrate()。

好处：不依赖 GUI、不需要 X、跑一次 1~2 秒，而且是 CC0 数据。
"""
import os

import numpy as np

MH_DIR = os.environ.get('MAKEHUMAN_DIR') or ''

# int16 位移 → 世界单位。
# 权威来源：core/algos3d.py 第 168 行  `self.data = Target.npzfile[vname] * 1e-3`
# （写入端第 221 行 `np.round(self.data * 1e3)` → int16，两边对齐）
# ⚠️ 一开始我猜 1e-5，结果幅度最大的变形只占身高 0.63%（成人→婴儿怎么可能只差 0.63%），
#    改成 1e-3 后是 63%，才对。**猜出来的常数一定要用已知量反查。**
SCALE = 1e-3


def find_root(explicit=None):
    for c in (explicit, os.environ.get('MAKEHUMAN_DIR'), MH_DIR,
              # 常见安装位置（可用环境变量 MAKEHUMAN_DIR 覆盖）
              os.path.join(os.path.expanduser('~'), 'AppData', 'Local',
                           'makehuman-community', 'makehuman')):
        if c and os.path.exists(os.path.join(c, 'data', '3dobjs', 'base.npz')):
            return c
    raise FileNotFoundError('找不到 MakeHuman 数据目录，设 MAKEHUMAN_DIR 环境变量')


def faces_from_fvert(fvert, nfaces=None):
    """把 MakeHuman 的 fvert 变成三角面表。

    `fvert` 是四边形表，**三角形用「末位索引重复」编码**（d == c），
    所以统一拆两刀、再把退化三角丢掉就够了。base 和组件网格都适用。

    ⚠️ `nfaces` 看起来像"每个面的顶点数"，**其实不是** ——
       它是**按顶点**的（fvert 18486 行，但 nfaces 19158 行，跟 coord 对齐）。
       我一度拿它去筛面，直接 IndexError；就算行数对上也只会把 1~2 顶点的行
       误拆成跨模型的乱三角。**所以这里显式忽略它**，参数只为兼容调用方保留。
    """
    fvert = np.asarray(fvert)
    a, b, c, d = fvert[:, 0], fvert[:, 1], fvert[:, 2], fvert[:, 3]
    t = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)
    ok = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return t[ok]


def load_base(root=None):
    """返回 dict(coord=顶点, faces=三角面, quads=四边面, groups=部位名, gid=每面组号)"""
    root = find_root(root)
    z = np.load(os.path.join(root, 'data', '3dobjs', 'base.npz'), allow_pickle=True)
    coord = z['coord'].astype(np.float64).copy()
    fvert = z['fvert']                      # (M,4) 四边形
    nf = z['nfaces'] if 'nfaces' in z.files else None
    tri = faces_from_fvert(fvert, z['nfaces'])
    gid = z['group'] if 'group' in z.files else None
    tri_gid = gid if gid is not None else None
    names = {}
    try:
        fgstr = z['fgstr']
        fgidx = z['fgidx']
        names = {int(fgidx[i]): fgstr[i].decode('ascii', 'replace') for i in range(len(fgidx))}
    except Exception:
        pass
    return dict(root=root, coord=coord, quads=fvert, faces=tri, tri_gid=tri_gid,
                gid=gid, group_names=names, nvert=len(coord))


def list_targets(root=None, prefix=''):
    """列出变形名字（不含 .index/.vector 后缀）"""
    root = find_root(root) if root is None or isinstance(root, str) else root
    z = np.load(os.path.join(root, 'data', 'targets.npz'), allow_pickle=True)
    out = []
    for k in z.files:
        if k.endswith('.index'):
            n = k[:-6]
            if n.startswith(prefix):
                out.append(n)
    return sorted(out)


def apply_targets(coord, root, weights, scale=SCALE, verbose=False):
    """套变形。weights = {'targets/measure/height/...': 权重, ...}

    返回新的 coord（不就地改）。
    """
    z = np.load(os.path.join(root, 'data', 'targets.npz'), allow_pickle=True)
    out = coord.copy()
    for name, w in weights.items():
        if not w:
            continue
        ik, vk = name + '.index', name + '.vector'
        if ik not in z.files or vk not in z.files:
            if verbose:
                print('   ⚠ 没有变形 %s' % name)
            continue
        idx = z[ik].astype(np.int64)
        vec = z[vk].astype(np.float64)
        n = min(len(idx), len(vec))
        out[idx[:n]] += vec[:n] * float(w) * scale
        if verbose:
            print('   %-52s w=%+.3f  %d 顶点' % (name.replace('targets/', ''), w, n))
    return out


def height(coord):
    """身高（用 Y 轴跨度近似；MH 的坐标里身高是最长轴）"""
    ext = coord.max(0) - coord.min(0)
    return float(ext.max())


def calibrate(root=None, verbose=True):
    """标定 SCALE：看一个已知变形的实际位移量。

    MakeHuman 的 measure/height 类变形能把身高改到目标值，
    这里拿 bodyshapes 里幅度最大的几个看 displacement 量级，
    和 base 的身高比一下，判断 1e-5 这个固定点系数对不对。
    """
    b = load_base(root)
    z = np.load(os.path.join(b['root'], 'data', 'targets.npz'), allow_pickle=True)
    H = height(b['coord'])
    rows = []
    for k in z.files:
        if not k.endswith('.vector'):
            continue
        v = z[k]
        if v.size:
            rows.append((float(np.abs(v).max()), k[:-7]))
    rows.sort(reverse=True)
    if verbose:
        print('base 身高 %.4f 单位，%d 顶点 / %d 三角面' % (H, b['nvert'], len(b['faces'])))
        print('位移幅度最大的 5 个变形:')
        for mx, name in rows[:5]:
            print('   %-56s max|v|=%6d  → ×1e-5 = %.5f 单位 (身高的 %.3f%%)'
                  % (name.replace('targets/', ''), mx, mx * SCALE, mx * SCALE / H * 100))
    return H, rows


if __name__ == '__main__':
    H, rows = calibrate()
    print()
    tg = list_targets()
    print('可用变形 %d 个，顶层分类:' % len(tg))
    import collections
    c = collections.Counter(t.split('/')[1] for t in tg if t.count('/') > 1)
    for k, n in c.most_common(20):
        print('   %-24s %d' % (k, n))
