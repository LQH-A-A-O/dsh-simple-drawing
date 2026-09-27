"""网格 → 正交视图 → 平面着色渲染 / 剪影。

用途：把 MakeHuman（或任何）导出的 OBJ 变成"白膜"——
带隐藏面消除的平面着色渲染，以及可直接喂给 shapes.extract 的剪影。

只依赖 numpy + PIL，不装 3D 库。

约定
----
OBJ 常见是 Y 向上、Z 向前。正面视图 = 沿 -Z 看，屏幕 x = X，屏幕 y = -Y。
朝向由包围盒自动判断（最长轴当身高），可以用 axis 参数覆盖。
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw


def load_obj(path, scale_to=None):
    """只取 v / f，忽略材质法线纹理。f 支持 v、v/vt、v//vn、v/vt/vn。"""
    verts, faces = [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = []
                for tok in line.split()[1:]:
                    n = tok.split('/')[0]
                    if n:
                        i = int(n)
                        idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):        # 扇形三角化
                    faces.append((idx[0], idx[k], idx[k + 1]))
    V = np.array(verts, np.float64)
    F = np.array(faces, np.int32)
    if scale_to:
        c = (V.max(0) + V.min(0)) / 2
        s = scale_to / max(V.max(0) - V.min(0))
        V = (V - c) * s
    return V, F


def guess_axes(V):
    """返回 (水平轴, 竖直轴, 深度轴)。最长的当竖直轴。"""
    ext = V.max(0) - V.min(0)
    up = int(np.argmax(ext))
    rest = [i for i in range(3) if i != up]
    fwd = int(rest[np.argmin(ext[rest])])              # 最薄的当深度
    side = [i for i in rest if i != fwd][0]
    return side, up, fwd


def project(V, size, up_axis=1, side_axis=0, depth_axis=2, margin=0.04,
            mirror_x=False, flip_y=True, fixed=None):
    """正交投影到画布像素坐标。返回 (pts2d[N,2], depth[N], transform)

    ⚠️ 默认会**把传进来的顶点自动缩放填满画布** —— 对单独渲染一个组件
       （头发/衣服只有头部大小）是灾难：它会被放大到撑满整帧，看起来像盖住半个身子的大团。
       组件必须和身体用**同一套变换**，所以这里支持 ixed=(k, cx, cy) 传入已算好的变换。
    """
    W, H = size
    x, y, z = V[:, side_axis], V[:, up_axis], V[:, depth_axis]
    if mirror_x:
        x = -x
    if fixed is not None:
        k, cx, cy = fixed
    else:
        w, h = x.max() - x.min(), y.max() - y.min()
        k = min(W * (1 - 2 * margin) / max(w, 1e-9), H * (1 - 2 * margin) / max(h, 1e-9))
        cx = (x.max() + x.min()) / 2
        cy = (y.max() + y.min()) / 2
    px = W / 2 + (x - cx) * k
    py = H / 2 - (y - cy) * k if flip_y else H / 2 + (y - cy) * k
    return np.stack([px, py], 1), z.copy(), (k, cx, cy)


def render_flat(V, F, size, bg=(250, 250, 252), clay=(214, 216, 222),
                light=(-0.45, 0.55, -0.7), up_axis=1, side_axis=0, depth_axis=2,
                ambient=0.42, outline=None, face_rgb=None, shade=True,
                bands=0, spec=0.0, spec_pow=20.0, shade_face=False, **kw):
    """平面着色渲染（painter 算法按深度排序 + 每面一个亮度）。

    这就是"白膜"该有的样子：只看形状和体积，不带任何材质。

    face_rgb : (M,3) 每个面一个颜色。默认**不用光照**，直接按面填色；
               配 shade_face=True 可让面颜色再乘上光照强度（脸部立体感靠这个）。
               配合网格自带的部位分组，能得到"分块图" —— 再喂给 shapes.extract
               提结构，每个身体部位就是独立区域，可以分别上色。
    """
    P, Z, k = project(V, size, up_axis, side_axis, depth_axis, **kw)
    tri = P[F]                                          # (M,3,2)
    # (致命) 给了 face_rgb 就 inten=1 -> 整层一个平色块。
    #   头就是这么变成"一坨"的：没有鼻梁、没有嘴唇、没有下巴。
    #   要面颜色**也吃光照**就传 shade_face=True。
    if face_rgb is not None and not shade_face:
        inten = np.ones(len(F))
    else:
        # 面法线（用投影前的三维坐标算，光照才准）
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(ln, 1e-12)
        L = np.array(light, float)
        nl = np.linalg.norm(L)
        L = L / nl if nl > 1e-9 else np.zeros(3)     # light=(0,0,0) 时别除出 NaN
        zc = n[:, depth_axis]
        n = np.where((zc < 0)[:, None], -n, n)
        ndl = np.abs(n @ L)
        inten = np.clip(ambient + (1 - ambient) * ndl, 0, 1)
        # ---- cel 量化：动漫质感靠**硬边色阶**，不是平滑明暗。
        #      bands=0 是平滑；bands=3 就是经典的 底色/阴影/高光 三档。
        #      实测：头发是一整块的主因就是平滑明暗 —— 每一面都在渐变，看着像塑料。
        if bands and bands >= 2:
            inten = np.floor(inten * bands) / (bands - 1)
            inten = np.clip(inten, 0, 1)
        # ---- 高光带：视线与光源的中间向量（Blinn-Phong），做出头发那道亮带
        if spec:
            Vv = np.zeros(3); Vv[depth_axis] = 1.0        # 视线：+深度轴朝相机
            H = L + Vv
            hn = np.linalg.norm(H)
            if hn > 1e-9:
                H = H / hn
                inten = np.clip(inten + spec * np.power(np.abs(n @ H), spec_pow), 0, 1)
    # 按深度排序：远的先画
    order = np.argsort(Z[F].mean(axis=1))
    img = Image.new('RGB', size, bg)
    d = ImageDraw.Draw(img)
    # ⚠️ clay 是单个 RGB 三元组，必须广播成 (M,3)；直接 np.array 会得到 shape (3,)，
    #    循环到第 4 个面就 IndexError: index ... is out of bounds for axis 0 with size 3
    base = (np.asarray(face_rgb, float) if face_rgb is not None
            else np.tile(np.asarray(clay, float), (len(F), 1)))
    for i in order:
        t = tri[i]
        col = tuple(int(x) for x in np.clip(base[i] * (inten[i] if shade else 1.0), 0, 255))
        d.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=col)
    if outline:
        img = add_outline(img, bg, outline)
    return img


def _blur_u8(a, r):
    """uint8 高斯模糊（r 是像素半径）。返回 float32。"""
    if r <= 0:
        return np.asarray(a, np.float32)
    from PIL import ImageFilter
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(float(r)))).astype(np.float32)


def render_regions(V, F, size, face_rgb, bg=(255, 255, 255),
                   light=(-0.4, 0.5, -0.75), ambient=0.55,
                   up_axis=1, side_axis=0, depth_axis=2,
                   bands=0, spec=0.0, spec_pow=20.0, shade=True,
                   cel=0.0, cel_thr=0.0, shadow_smooth=2.2, shadow_edge=3.0,
                   smooth=1.8, edge=3.0, shade_smooth=0.8,
                   cull=False, two_sided=None,
                   outline=None, return_mask=False, **kw):
    """按区域上色，但**区域边界在 2D 图像空间里平滑**。

    为什么需要它
    ------------
    分区是按**面**判定的（面重心落在哪个区），所以边界只能沿着三角面走 ——
    一定是锯齿。实测症状：肩上一圈"毛刺"、下巴一团乱、刘海下缘像撕开的纸。
    在面邻接图上做多数表决（smooth_partition）只是治标：它能抹掉孤立三角形，
    但边界本身还是折线。

    这里的做法
    ----------
    1. 把每个区域单独渲成一张掩膜；
    2. 掩膜做高斯模糊，再用陡峭的 smoothstep 重新二值化
       -> 边界变成**平滑曲线**，而且自带抗锯齿；
    3. 光照单独渲一张强度图，只对各个区的颜色做统一调制，
       所以明暗不受分区影响（这也是手绘的层次：形状是一层，明暗是另一层）。

    face_rgb : (M,3) 每个面的颜色。**相同颜色的面自动归成一个区**，
               所以调用方直接传 colorize() 的结果即可，不用改接口。
    return_mask=True 时返回 (img, {颜色元组: 浮点掩膜})，
               掩膜可用来把一个渲染结果拆成两层（例如后发/前发）。
    """
    from PIL import ImageFilter
    W, H = size
    P2, Z, _ = project(V, size, up_axis, side_axis, depth_axis, **kw)
    tri = P2[F]
    keep = np.ones(len(F), bool)
    if cull:
        # 背面剔除：闭合壳（头发/衣服/眼睛，实测绕序一致）剔掉背面之后
        # 画家算法才稳定 —— 否则内表面会随机盖住外表面，渲出来一块块斑。
        a0, b0, c0 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n0 = np.cross(b0 - a0, c0 - a0)
        keep = n0[:, depth_axis] > 1e-9
    idx = np.nonzero(keep)[0]
    order = idx[np.argsort(Z[F[idx]].mean(axis=1))]    # 远的先画

    fr = np.asarray(face_rgb, float)
    cols, inv = np.unique(fr.round().astype(np.int32), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    fid = inv.astype(np.int32) + 1                  # 0 留给"没面"

    # ---- 1) 区域标签图
    limg = Image.new('I', (W, H), 0)
    d = ImageDraw.Draw(limg)
    for i in order:
        t = tri[i]
        d.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=int(fid[i]))
    lab = np.asarray(limg)

    # ---- 2) 光照强度图（每面一个灰度，painter 顺序）
    if shade:
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(b - a, c - a)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        L = np.array(light, float)
        nl = np.linalg.norm(L)
        L = L / nl if nl > 1e-9 else np.zeros(3)
        # 法线一律先翻到朝相机 —— 这样"单面光照"也能用（背面反正被前面的面挡住）。
        n = np.where((n[:, depth_axis] < 0)[:, None], -n, n)
        # two_sided 默认跟着 cull 走：剔了背面就是单面光照，没剔才需要 abs()
        ts = (not cull) if two_sided is None else two_sided
        ndl = np.abs(n @ L) if ts else np.maximum(n @ L, 0.0)
        if cel:
            # 日漫式平涂阴影：**只有亮/暗两档**，交界线由 N·L 的符号定。
            # 得到的是"一块阴影形状"，不是渐变 —— 鼻侧、下颌、刘海下都是这么来的。
            # ⚠️ 别用 ambient + 分段量化去做这件事：
            #    floor(inten*bands)/(bands-1) 只在 inten 铺满 0..1 时才分得出档，
            #    ambient=0.8 时 inten∈[0.8,1] -> 全部塌到 1，阴影**根本不出现**。
            inten = np.where(ndl > cel_thr, 1.0, float(cel))
        else:
            inten = np.clip(ambient + (1 - ambient) * ndl, 0, 1)
        if bands and bands >= 2 and not cel:
            inten = np.clip(np.floor(inten * bands) / (bands - 1), 0, 1)
        if spec:
            Vv = np.zeros(3); Vv[depth_axis] = 1.0
            Hv = L + Vv
            hn = np.linalg.norm(Hv)
            if hn > 1e-9:
                Hv = Hv / hn
                inten = np.clip(inten + spec * np.power(np.abs(n @ Hv), spec_pow), 0, 1)
        gimg = Image.new('L', (W, H), 0)
        dg = ImageDraw.Draw(gimg)
        if cel:
            gv = np.where(inten > 0.5, 255, 0).astype(np.uint8)   # 亮/暗**二值**
        else:
            gv = (inten * 255).astype(np.uint8)
        for i in order:
            t = tri[i]
            dg.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=int(gv[i]))
        g255 = np.asarray(gimg).astype(np.float32)
        if cel and shadow_smooth > 0:
            # 阴影形状也要在 2D 上平滑 —— 逐面二值化的交界只能沿三角面走，
            # 在密网格上会碎成一条条细片（不是"平涂的一块"）。
            # 跟分区掩膜同一套：模糊 -> 陡峭重二值化 -> 干净的平涂阴影。
            mm = _blur_u8(g255, shadow_smooth) / 255.0
            mm = np.clip((mm - 0.5) * shadow_edge + 0.5, 0, 1)
            g255 = np.where(mm > 0.5, 255.0, float(cel) * 255.0)
    else:
        g255 = np.full((H, W), 255.0, np.float32)

    cover = (lab > 0).astype(np.float32)
    if cel and shade:
        # 已经在 2D 上平滑+重二值化过，直接用；再走一遍模糊会把硬边糊掉。
        g = g255 / 255.0
    else:
        # 只在模型内部平滑明暗：直接模糊会被边界外的黑边拉暗（暗圈）
        num = _blur_u8(g255 * cover, shade_smooth)
        den = _blur_u8(cover * 255.0, shade_smooth)
        # ⚠️ num 和 den 都是 0..255 单位，相除**已经归一成 0..1** 了。
        #    这里要是再除一次 255，亮度只剩 0.0035 -> 取整全归零，整张图变纯黑。
        g = num / np.maximum(den, 1e-6)                 # 0..1

    # ---- 3) 逐区平滑掩膜 + 上色
    out = np.zeros((H, W, 3), np.float32)
    masks = {}
    for i, col in enumerate(cols):
        m = ((lab == i + 1) * 255).astype(np.uint8)
        mm = _blur_u8(m, smooth) / 255.0
        # 陡峭 smoothstep：保留硬边（动漫要硬边），但曲线本身是平滑的
        mm = np.clip((mm - 0.5) * edge + 0.5, 0, 1)
        masks[tuple(int(v) for v in col)] = mm
        out += mm[..., None] * col.astype(np.float32)[None, None, :]

    out *= g[..., None]
    arr = np.clip(out, 0, 255).astype(np.uint8)
    bgarr = np.asarray(bg, np.uint8).reshape(1, 1, 3)
    full = np.where((cover > 0)[..., None], arr, np.broadcast_to(bgarr, (H, W, 3)))
    img = Image.fromarray(full.copy())
    if outline:
        img = add_outline(img, bg, outline)
    if return_mask:
        return img, masks
    return img


def silhouette(V, F, size, **kw):
    """实心剪影（纯色），可直接喂给 shapes.extract 提结构。"""
    img = render_flat(V, F, size, bg=(255, 255, 255), clay=(0, 0, 0),
                      ambient=1.0, light=(0, 0, 0), **kw)
    return img


def drop_specks(img, bg, min_frac=0.01, min_px=64):
    """丢掉与主体分离的孤立小碎块。

    素体网格里有一小片几何落在两脚之间、和身体**不连通**，渲出来是个悬空色块；
    整图描边时描边宽度比它还大，于是被填成一个黑方块。
    这里按连通域面积过滤：只保留面积 >= 最大块 min_frac 的块（且不小于 min_px）。
    实测本图前景只有 2 块（主体 287372 px + 碎块 1640 px = 0.6%），判据很安全。
    """
    try:
        import cv2
    except ImportError:                     # 没装 cv2 就不做，不影响正常出图
        return img
    a = np.asarray(img.convert('RGB')).astype(np.int16)
    fg = (np.abs(a - np.asarray(bg, np.int16)).sum(2) > 24).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 2:
        return img
    big = int(stats[1:, 4].max())
    out = np.asarray(img.convert('RGB')).copy()
    bgarr = np.asarray(bg, np.uint8)
    dropped = 0
    for i in range(1, n):
        if stats[i, 4] < max(min_px, big * min_frac):
            out[lab == i] = bgarr
            dropped += 1
    return Image.fromarray(out)


def add_outline(img, bg, width=3):
    from PIL import ImageFilter
    a = np.array(img.convert('L'))
    fg = a < 240
    m = Image.fromarray((fg * 255).astype(np.uint8))
    dil = m.filter(ImageFilter.MaxFilter(width * 2 + 1))
    ring = (np.array(dil) > 127) & (~fg)
    out = np.array(img)
    out[ring] = (60, 58, 68)
    return Image.fromarray(out)


def info(V, F):
    return dict(verts=len(V), faces=len(F),
                bbox=tuple(round(float(x), 3) for x in (V.max(0) - V.min(0))))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('用法: python mesh.py <模型.obj> [-o 输出.png] [--size 1240x1754]')
        sys.exit(1)
    p = sys.argv[1]
    out = 'mesh_render.png'
    size = (1240, 1754)
    for i, a in enumerate(sys.argv):
        if a == '-o':
            out = sys.argv[i + 1]
        if a == '--size':
            w, h = sys.argv[i + 1].split('x')
            size = (int(w), int(h))
    V, F = load_obj(p)
    print(os.path.basename(p), info(V, F))
    sa, ua, da = guess_axes(V)
    print('轴判断: 水平=%d 竖直=%d 深度=%d' % (sa, ua, da))
    render_flat(V, F, size, up_axis=ua, side_axis=sa, depth_axis=da,
                outline=2).save(out)
    silhouette(V, F, size, up_axis=ua, side_axis=sa, depth_axis=da).save(
        os.path.splitext(out)[0] + '_sil.png')
    print('→', out)
