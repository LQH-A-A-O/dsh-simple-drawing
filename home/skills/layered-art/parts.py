"""组件分离：人物、头发、眉毛、衣服各画各的，最后拼起来。

MakeHuman 的架构本来就是组件化的：
    data/hair/*.npz          头发（10 个）
    data/eyebrows/*.npz      眉毛（12 个）
    data/eyes/*.mhpxy        眼球（2 个）
    data/clothes/*.npz       衣服（20 个）
    data/eyelashes/*.npz     睫毛（4 个）

每个组件都是独立网格，跟 base 用同一套坐标空间，所以**各自渲染再合成**就行。
这也正是用户要的："衣服头发单独绘制、人物单独绘制，最后拼起来"。

好处很实在：
    - 换发型只重画头发那层，身体不动
    - 头发被身体正确遮挡（按深度合成，不是贴图）
    - 每个组件可以有自己的配色、自己的明暗
"""
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mhsource as M      # noqa: E402
import mesh as ME         # noqa: E402

def _out_dir(name):
    """输出目录：环境变量 LAYERED_ART_OUT 优先，否则落在系统临时目录。

    ⚠️ 不要在这里写死自己机器的路径 —— 打包分发时会被拒绝，
       别人机器上也不存在。
    """
    import os as _os
    import tempfile as _tf
    base = _os.environ.get('LAYERED_ART_OUT') or _os.path.join(
        _tf.gettempdir(), 'layered-art')
    d = _os.path.join(base, name)
    _os.makedirs(d, exist_ok=True)
    return d


OUT = _out_dir('parts')
SIZE = (744, 1754)
# 日漫式平涂阴影的暗面亮度系数（亮面恒为 1.0）。皮肤约 0.80、头发更浅约 0.87 ——
# 头发暗面对比太强会显脏。
CEL_BODY, CEL_HAIR = 0.76, 0.86
# cel_thr：N·L 低于它才算暗面。0 表示"半球都算暗面"（阴影面积太大），
# 抬到 0.22 之后阴影只落在明显背光的面 -> 更像日漫那种有取舍的平涂阴影。
CEL_THR = 0.08

# 脖子变细 + 女性青年体型
RECIPE = {
    'targets/macrodetails/asian-female-young': 1.0,
    'targets/neck/neck-scale-horiz-decr': 0.62,      # ← 脖子横向收细（0.75 仍偏粗）
    'targets/neck/neck-scale-depth-decr': 0.35,
}

PAL = dict(hair=(206, 212, 226), skin=(240, 222, 214), cloth=(56, 66, 106),
           cloth2=(44, 52, 84), foot=(40, 42, 54), brow=(150, 156, 172))


def load_component(root, rel):
    """读组件网格（.npz 或 .mhpxy 旁的 npz）。返回 (coord, faces) 或 None"""
    p = os.path.join(root, 'data', rel)
    npz = p if p.endswith('.npz') else os.path.splitext(p)[0] + '.npz'
    if not os.path.exists(npz):
        return None
    z = np.load(npz, allow_pickle=True)
    if 'coord' not in z.files:
        return None
    V = z['coord'].astype(np.float64)
    F = M.faces_from_fvert(z['fvert'], z['nfaces']) if 'fvert' in z.files else None
    return dict(coord=V, faces=F, path=npz)


def fit_component(cV, base_raw, base_morph, k=3, chunk=128, inflate=1.0):
    """把组件装配到变形后的身体上。

    ⚠️ **必须用 k>=3 个邻近顶点加权插值，不能只取最近的 1 个。**
       实测：只取最近 1 个时，ob01 装配后头顶比身体矮 0.39、short01 反而高 1.00 ——
       因为 sian-female-young 把身体从 16.95 压到 15.28，**头部不是均匀缩放**
       （原始头顶 y=8.5 → 变形后 6.83），单点位移复现不出这种变化。
       取最近 3 个、按 1/d² 加权，等于对位移场做平滑插值，才跟得上。

    MakeHuman 1.3 的组件 npz 里**没有绑定表**（.mhclo 不存在），
    但组件坐标就在 base 的静止空间里，所以绑定可以自动算 —— 这就是 proxy fitting
    在做的事，只是绑定表由人工编写换成了最近邻插值。
    """
    disp = base_morph - base_raw
    n = len(cV)
    out = np.empty_like(cV)
    k = max(int(k), 1)
    for i in range(0, n, chunk):
        c = cV[i:i + chunk]
        d2 = ((c[:, None, :] - base_raw[None, :, :]) ** 2).sum(-1)      # (C,M)
        if k == 1:
            j = d2.argmin(1)
            out[i:i + chunk] = c + disp[j]
            continue
        idx = np.argpartition(d2, k, axis=1)[:, :k]                     # (C,k)
        dd = np.take_along_axis(d2, idx, 1)
        w = 1.0 / np.maximum(dd, 1e-9)
        w /= w.sum(1, keepdims=True)
        out[i:i + chunk] = c + (disp[idx] * w[:, :, None]).sum(1)
    if inflate != 1.0:
        c0 = base_morph.mean(0)
        out = c0 + (out - c0) * inflate
    return out


def list_component_dirs(root, kind):
    d = os.path.join(root, 'data', kind)
    out = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if os.path.isdir(os.path.join(d, n)):
                out.append(n)
    return out


def list_hair(root):
    """挑头发：优先**天然与头顶对齐**的（short04 上界 8.52 vs 头顶 8.50）。
    组件之间基准偏移不一致，而 MakeHuman 1.3 没有校正表。"""
    order = ['short04', 'short02', 'short03', 'bob02', 'short01', 'bob01']
    ds = list_component_dirs(root, 'hair')
    ds.sort(key=lambda x: next((i for i, k in enumerate(order) if k in x), 99))
    return [os.path.join('hair', d, d + '.npz') for d in ds]


def list_components(root, kind):
    d = os.path.join(root, 'data', kind)
    out = []
    for dp, dn, fn in os.walk(d):
        for f in fn:
            if f.endswith('.npz'):
                out.append(os.path.relpath(os.path.join(dp, f), os.path.join(root, 'data')))
    return sorted(out)


# ---------------------------------------------------------------- 层序表（学 Live2D）
# Live2D 的关键设计：**绘制顺序由人声明，不由深度算**。
# 我们从 3D 渲成 2D 部件，正好可以照搬这一条 —— 它一次解决头发壳盖住脸：
# 把头发按相对头中心的前后劈成后发/前发，后发垫在身体下面，前发盖在脸上面。
# (致命) 头必须自己占一层。
#   上一版整个身体放一层，后发(层0)被身体盖死 —— 头发两侧全丢，
#   实测左右对称度 上0.45 / 下0.09。头单独提到层20之后：
#   头比头发壳窄 -> 头发自然从头两侧露出来；脸又在层20压住头发内侧。
LAYER_ORDER = {
    '后发': 0,
    '躯干': 10,
    '头': 20,
    '眼球': 22,
    '睫毛': 24,
    '眉毛': 26,
    '前发': 30,
    '配饰': 40,
}


def align_to_head(cV, V2, ua, sa, da, width_k=1.16, top_margin=0.10):
    """把组件**自动对位**到头上。

    为什么必须这么做：MakeHuman 1.3 的 10 个发型组件**基准偏移完全不一致** ——
    实测顶部差从 -0.51 到 +1.00、横向宽从 1.45 到 2.42（头宽 2.04），
    而校正这份差异的 .mhclo 表**不存在**。所以不能靠挑一个刚好对的组件，
    得按「头顶对齐 + 宽度匹配头部 + 水平居中」把它规整上去。

    这样任何组件都能用，而且换发型不会跑位。
    """
    h = V2[V2[:, ua] > V2[:, ua].max() - 2.5]          # 头部顶点
    h_top, h_w = V2[:, ua].max(), h[:, sa].max() - h[:, sa].min()
    h_cx = (h[:, sa].max() + h[:, sa].min()) / 2
    h_cz = (h[:, da].max() + h[:, da].min()) / 2
    c_w = max(cV[:, sa].max() - cV[:, sa].min(), 1e-6)
    c_h = max(cV[:, ua].max() - cV[:, ua].min(), 1e-6)
    k = (h_w * width_k) / c_w                          # 宽度按头宽匹配
    cx = (cV[:, sa].max() + cV[:, sa].min()) / 2
    cz = (cV[:, da].max() + cV[:, da].min()) / 2
    out = cV.copy()
    out[:, sa] = (out[:, sa] - cx) * k + h_cx
    out[:, da] = (out[:, da] - cz) * k * 0.85 + h_cz   # 深度方向少放一点，别穿到脑后
    out[:, ua] = (out[:, ua] - cV[:, ua].min()) * (k * 0.9) + h_top + top_margin - c_h * k * 0.9
    return out


def split_by_depth(V, F, faces, axis, pivot):
    """把一个网格的面按重心深度在中轴之前还是之后劈成两半。

    axis  : 深度轴；+轴方向是朝向相机（跟 project/光照的约定一致）
    pivot : 分界深度（这里用头部中心的深度）
    返回 (后面的面, 前面的面)
    """
    zc = V[faces].mean(axis=1)[:, axis]
    back = zc <= pivot
    return faces[back], faces[~back]


def alpha_paste(base, layer_img, bg_color):
    """把一层贴上去：layer_img 里非背景色的像素才算有效。"""
    a = np.array(layer_img)
    m = (np.abs(a.astype(int) - np.array(bg_color)).sum(axis=2) > 18)
    out = np.array(base)
    out[m] = a[m]
    return Image.fromarray(out)


def main():
    t0 = time.time()
    b = M.load_base()
    V, F = b['coord'], b['faces']
    root = b['root']
    sa, ua, da = ME.guess_axes(V)
    kw = dict(up_axis=ua, side_axis=sa, depth_axis=da)
    print('base 身高 %.3f' % M.height(V))

    V2 = M.apply_targets(V, root, RECIPE, verbose=True)
    print('套完变形 身高 %.3f' % M.height(V2))
    _, _, TF = ME.project(V2, SIZE, ua, sa, da)      # 身体算一次，所有组件复用

    print('\n=== 可用组件 ===')
    for kind in ('hair', 'eyebrows', 'eyes', 'eyelashes', 'clothes', 'teeth'):
        cs = list_component_dirs(root, kind)
        print('   %-12s %d 个' % (kind, len(cs)))

    BG = (250, 250, 252)
    WHITE = (255, 255, 255)
    layers = {}          # 层名 -> 图

    # ---- 身体层
    import build_char as BC
    part, _ = BC.partition(V2, F, sa, ua, da)
    adj = BC.face_adjacency(F)
    part = BC.smooth_partition(part, adj, iters=3)
    # 素体自带一圈头发壳（会垂到肩上）。用了头发组件之后它必须**整块丢掉**：
    #   只改颜色是不够的 —— 它的背光面会渲成一片深灰压在脸上，
    #   而且它是独立壳，正好盖住真正的脸。丢掉之后头骨本身还在。
    nh = int((part == '头发').sum())
    if nh:
        hf = V2[F[part == '头发']]
        print('   丢掉素体头发壳 %d 面  (y %.2f~%.2f  z %.2f~%.2f)'
              % (nh, hf[:, :, ua].min(), hf[:, :, ua].max(),
                 hf[:, :, da].min(), hf[:, :, da].max()))
    part[part == '头发'] = '丢弃'
    # 身体劈成 躯干 / 头 两层：Live2D 层序要求头压在头发内侧之上
    is_head = (part == '头') | (part == '脖子')
    is_torso = (part != '丢弃') & ~is_head
    # 头用细色阶(5) -> 鼻梁/嘴唇/下巴这些基础网格自带的起伏才透得出来；
    # 躯干用粗色阶(3) -> 动漫式硬边。两者都不再逐层描边（描边一律留到整图那一次）。
    # ⚠️ 用户定调：**要二维平面，不要立体感**。
    #    所以 shade=False —— 关掉全部明暗，每个区就是一块纯色。
    #    后果要知道：脸上的鼻梁/嘴唇原来靠明暗透出来，压平之后就没了，
    #    这两样要改用**线**来画（见 pixel.py 的 line 系列）。
    for tag, sel, sm in (('躯干', is_torso, 5.0), ('头', is_head, 3.0)):
        # render_regions：区域的边界在 2D 掩膜上平滑过 -> 肩上那圈"毛刺"、
        # 下巴那团锯齿都没了。
        im = ME.render_regions(V2, F[sel], SIZE, BC.colorize(part[sel], PAL), bg=BG,
                               cel=CEL_BODY, cel_thr=CEL_THR, two_sided=False,
                               light=(-0.35, 0.50, 0.79),
                               shadow_smooth=1.0, shadow_edge=4.0,
                               smooth=sm, outline=None, fixed=TF, **kw)
        im.save(os.path.join(OUT, '01_%s.png' % tag))
        layers[tag] = im
    print('   身体劈开: 躯干 %d 面 / 头 %d 面' % (int((~is_head).sum()), int(is_head.sum())))

    # 头部中心深度（用来劈前后发）
    head_m = (part == '头')
    face_c = V2[F[head_m]] if head_m.any() else V2[F]
    hc = face_c.mean(axis=(0, 1))
    # (致命) 前/后发分界必须取 脸的前表面(高分位)，不能取均值/中位：
    #   阈值一旦落在头发两侧的深度上，左右就被分进不同层 -> 一边露一边秃。
    #   抬到"前15%"后两侧统一归后发(靠头更窄自然露出)，只有刘海算前发。
    z_split = float(np.percentile(face_c[:, :, da], 85))
    print('   脸中心 z=%.2f   前发表面 z=%.2f' % (hc[da], z_split))

    # ---- 头发层：劈成后发 / 前发
    # 用户要"像 testp 那样的头发、去掉双马尾" -> 挑长发垂肩的 long01。
    hairs = list_hair(root)
    prefer = [h for h in hairs if 'long01' in h]
    hairs = prefer + [h for h in hairs if h not in prefer]
    hc_used = None
    for h in hairs:
        c = load_component(root, h)
        if c is None or len(c['coord']) < 100:
            continue
        cV = fit_component(c['coord'], V, V2)
        cV = align_to_head(cV, V2, ua, sa, da)
        fb, ff = split_by_depth(cV, c['faces'], c['faces'], da, z_split)
        if len(fb) < 20 or len(ff) < 20:
            continue
        hair_col = tuple(int(v) for v in PAL['hair'])
        # (致命) 前后发的分界也是"按面深度阈值" -> 刘海下缘像撕开的纸。
        #   做法：把两个区当成两个标记色渲一次，render_regions 会把它们的
        #   共同边界在 2D 掩膜上平滑；再拿掩膜把一张整头发的图拆成两层。
        # (致命) split_by_depth 返回的是**面数组 (M,3)**（顶点索引三元组），
        #   不是面的序号！直接 lbl[fb]=... 会变成花式索引，把标签撒得到处都是
        #   （实测前发变成满脸 4.1% 的碎斑）。标签必须自己按深度算。
        zc = cV[c['faces']].mean(axis=1)[:, da]
        back_i = zc <= z_split
        lbl = np.where(back_i[:, None],
                       np.array([255, 0, 0], float),
                       np.array([0, 200, 0], float))
        _, masks = ME.render_regions(
            cV, c['faces'], SIZE, lbl, bg=WHITE, smooth=2.4, edge=2.5,
            shade=False, fixed=TF, return_mask=True, **kw)
        back_m, front_m = masks[(255, 0, 0)], masks[(0, 200, 0)]

        # 两档纯色：后发浅一档、前发深一档。压平之后层与层只靠**色差 + 描边**
        # 分开 —— 这正是 testp 那种低多边形平面色的做法（不靠明暗）。
        TONES = {'后发': tuple(int(v) for v in PAL['hair']),
                 '前发': tuple(int(v * 0.90) for v in PAL['hair'])}

        def draw(tag, keep):
            col = TONES[tag]
            im = ME.render_regions(
                cV, c['faces'], SIZE,
                np.tile(np.array(col, float), (len(c['faces']), 1)),
                bg=WHITE, cel=CEL_HAIR, cel_thr=CEL_THR, two_sided=False,
                light=(-0.35, 0.50, 0.79), spec=0.22, spec_pow=14,
                shadow_smooth=1.0, shadow_edge=4.0, smooth=1.4, outline=None,
                fixed=TF, **kw)
            a = np.asarray(im).copy()
            a[keep < 0.5] = WHITE          # 每个头发像素只归一边 -> 不留缝
            out_im = Image.fromarray(a)
            out_im.save(os.path.join(OUT, '02_%s.png' % tag))
            return out_im
        layers['后发'] = draw('后发', back_m >= front_m)
        layers['前发'] = draw('前发', front_m > back_m)
        hc_used = h
        print('   头发组件 %s：后发 %d 面 / 前发 %d 面'
              % (os.path.basename(h), len(fb), len(ff)))
        break
    if hc_used is None:
        print('   ⚠ 没有可用的头发组件')

    # ---- 五官层：组件天然就在正确位置（实测 eyrow001 y6.03~6.15 正好在眉线），
    #      所以只需要装配，**不需要 align_to_head**（那个是给偏移不一致的头发用的）
    FACE = [('eyebrows', 'eyebrow001', '眉毛', (108, 114, 132)),
            ('eyes', 'low-poly', '眼球', (74, 86, 112)),
            ('eyelashes', 'eyelashes01', '睫毛', (46, 44, 56))]
    for kind, name, lname, col in FACE:
        c = load_component(root, '%s/%s/%s.npz' % (kind, name, name))
        if c is None:
            print('   ⚠ 缺组件 %s/%s' % (kind, name)); continue
        cV = fit_component(c['coord'], V, V2)
        im = ME.render_flat(cV, c['faces'], SIZE, bg=WHITE, clay=col,
                            shade=False, fixed=TF, **kw)
        im.save(os.path.join(OUT, '02_%s.png' % lname))
        layers[lname] = im
        print('   %s ← %s/%s  (%d 顶点)' % (lname, kind, name, len(cV)))

    # ---- 按层序合成（这就是 Live2D 那条设计）
    order = sorted(layers.items(), key=lambda kv: LAYER_ORDER.get(kv[0], 50))
    print('\n=== 合成顺序 ===')
    for n, _ in order:
        print('   层 %-4d %s' % (LAYER_ORDER.get(n, 50), n))
    comp = Image.new('RGB', SIZE, BG)
    for n, im in order:
        comp = alpha_paste(comp, im, WHITE if n in ('后发', '前发') else BG)
    # (致命) 描边必须**整图只描一次**。逐层描边会把层与层的接缝描出来 ——
    #   头/躯干那条锯齿边就是这么跑到脸上的。
    comp.save(os.path.join(OUT, '03_合成_无线.png'))
    comp = ME.drop_specks(comp, BG)          # 丢掉两脚之间那个孤立碎块
    comp = ME.add_outline(comp, BG, 2)
    comp.save(os.path.join(OUT, '03_合成.png'))

    from render import contact_sheet
    tiles = [(n, im) for n, im in order] + [('合成', comp)]
    contact_sheet(tiles, os.path.join(OUT, 'parts_sheet.png'), tile_w=300, cols=len(tiles))
    print('\n总耗时 %.1f 秒 → %s' % (time.time() - t0, OUT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
