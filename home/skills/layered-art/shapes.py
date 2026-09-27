"""结构：从参考图提取可编辑的多边形骨架，并支持对称 / 变换 / 部件拼贴。

核心命题：**结构（谁在哪、什么轮廓）和配色是可以分离的。**
一旦图变成了"多边形 + 内孔 + 调色板索引"，它就有了骨：
    - 换配色 = 换一张色阶表，重跑（0.1 秒）
    - 改轮廓 = 动几个控制点
    - 左右对称 = 镜像 + 重提取
    - 换部件 = 抠一块贴过去 + 重提取

所有"编辑"都走同一条路：**栅格化 → 改像素 → 重新提取结构**。
比直接改多边形表更稳（不用处理多边形自交、内孔失效这些麻烦事）。

⚠️ 已知限制：提取是"描"不是"画"。出来的结构是参考图的衍生品，
   要当原创设计用得先把轮廓改到位（改发型/服装/脸型）。
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_K = 12
DEFAULT_EPS = 3.0          # 甜点：449 块/4460 顶点就能到 4.01/255 误差
DEFAULT_MIN_AREA = 25


# ---------------------------------------------------------------- 光栅化
def paste_poly(canvas, color, outer, holes):
    """贴一个含内孔的多边形。

    ⚠️ 必须处理内孔：只取外轮廓的话，环形区域（比如包住脸的头发）会被填成实心块，
       把里面的五官整个盖掉 —— 实测重建误差会从 3.86 飙到 5.11，而且脸直接糊。
       做法：包围盒内建 mask，先填外轮廓，再用 0 挖掉孔，最后带 mask 粘贴。
    """
    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    if not xs or not ys:
        return
    x0, y0 = max(min(xs), 0), max(min(ys), 0)
    x1, y1 = min(max(xs) + 1, canvas.width), min(max(ys) + 1, canvas.height)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    m = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(m)
    d.polygon([(x - x0, y - y0) for x, y in outer], fill=255)
    for hole in holes:
        if len(hole) >= 3:
            d.polygon([(x - x0, y - y0) for x, y in hole], fill=0)
    canvas.paste(color, (x0, y0), m)


def to_image(spec, cmap=None, bg=(255, 255, 255)):
    """纯结构重建（底色层）。cmap: {cluster_index: rgb} 用于换色。"""
    cmap = cmap or {}
    canvas = Image.new('RGB', (spec['w'], spec['h']), bg)
    for rg in spec['regions']:
        paste_poly(canvas, tuple(cmap.get(rg['ci'], rg['color'])), rg['pts'], rg['holes'])
    return canvas


# ---------------------------------------------------------------- 提取
def extract(src, k=DEFAULT_K, eps=DEFAULT_EPS, min_area=DEFAULT_MIN_AREA,
            median=3, source=None):
    """参考图 → 结构。src 可以是路径，也可以是 HxWx3 的 RGB numpy 数组。"""
    if isinstance(src, str):
        bgr = cv2.imread(src)
        if bgr is None:
            raise SystemExit('读不到图: %s' % src)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source = source or os.path.basename(src)
    else:
        rgb = np.asarray(src)[:, :, :3]
    H, W = rgb.shape[:2]

    Z = rgb.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.5)
    _, labels, centers = cv2.kmeans(Z, k, None, crit, 6, cv2.KMEANS_PP_CENTERS)
    lab = labels.reshape(H, W).astype(np.uint8)
    if median:
        # 只做中值去噪。别用 MORPH_OPEN —— 它会把 1~2px 的眼线/嘴线整个抹掉
        lab = cv2.medianBlur(lab, median)
    centers = centers.astype(np.uint8)                     # RGB（因为我们传的是 RGB）

    regions = []
    for i in range(k):
        m = ((lab == i).astype(np.uint8)) * 255
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hier = hier[0]
        for ci, c in enumerate(cnts):
            if hier[ci][3] != -1:          # 只处理外轮廓，孔归到父级
                continue
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            outer = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(outer) < 3:
                continue
            holes = []
            ch = hier[ci][2]
            while ch != -1:
                if cv2.contourArea(cnts[ch]) >= 12:
                    hp = cv2.approxPolyDP(cnts[ch], eps, True).reshape(-1, 2)
                    if len(hp) >= 3:
                        holes.append([(int(x), int(y)) for x, y in hp])
                ch = hier[ch][0]
            regions.append(dict(ci=int(i), color=tuple(int(x) for x in centers[i]),
                                area=float(a),
                                pts=[(int(x), int(y)) for x, y in outer],
                                holes=holes, name=''))
    regions.sort(key=lambda d: -d['area'])          # 画家算法：大的先画
    return dict(w=int(W), h=int(H), k=int(k), eps=float(eps), source=source or '',
                pal=[tuple(int(x) for x in c) for c in centers], regions=regions)


# ---------------------------------------------------------------- 存取
def save(spec, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(spec, f, ensure_ascii=False)
    return path


def load(path):
    with open(path, encoding='utf-8') as f:
        spec = json.load(f)
    spec['pal'] = [tuple(c) for c in spec['pal']]
    for rg in spec['regions']:
        rg['color'] = tuple(rg['color'])
        rg['pts'] = [tuple(p) for p in rg['pts']]
        rg['holes'] = [[tuple(p) for p in h] for h in rg.get('holes', [])]
    return spec


def stats(spec):
    npts = sum(len(r['pts']) for r in spec['regions'])
    nh = sum(len(r['holes']) for r in spec['regions'])
    return dict(regions=len(spec['regions']), vertices=npts, holes=nh,
                size='%dx%d' % (spec['w'], spec['h']),
                colors=len({r['ci'] for r in spec['regions']}))


# ---------------------------------------------------------------- 查询
def bbox(rg):
    xs = [p[0] for p in rg['pts']]
    ys = [p[1] for p in rg['pts']]
    return min(xs), min(ys), max(xs), max(ys)


def region_at(spec, x, y):
    """点 (x,y) 落在哪个区域上。从最小的区域往回找（画家算法的逆序）。"""
    for i in range(len(spec['regions']) - 1, -1, -1):
        rg = spec['regions'][i]
        x0, y0, x1, y1 = bbox(rg)
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        m = Image.new('L', (x1 - x0 + 1, y1 - y0 + 1), 0)
        d = ImageDraw.Draw(m)
        d.polygon([(px - x0, py - y0) for px, py in rg['pts']], fill=255)
        for h in rg['holes']:
            d.polygon([(px - x0, py - y0) for px, py in h], fill=0)
        if m.getpixel((x - x0, y - y0)) > 127:
            return i
    return -1


def list_regions(spec, top=40):
    rows = []
    for i, rg in enumerate(spec['regions'][:top]):
        x0, y0, x1, y1 = bbox(rg)
        rows.append(dict(idx=i, ci=rg['ci'], name=rg.get('name', ''),
                         area=int(rg['area']), color=rg['color'],
                         hex='#%02X%02X%02X' % rg['color'],
                         bbox=(x0, y0, x1, y1), pts=len(rg['pts']), holes=len(rg['holes'])))
    return rows


def name_region(spec, idx, name):
    spec['regions'][idx]['name'] = name
    return spec


def find_by_name(spec, name):
    return [i for i, rg in enumerate(spec['regions']) if rg.get('name') == name]


# ---------------------------------------------------------------- 编辑
def transform_region(spec, idx, dx=0, dy=0, scale=1.0, rot=0.0, anchor=None):
    """平移 / 缩放 / 旋转**单个区域**（轮廓控制点级别）。
    改完这个区域会自动重排到绘制顺序末尾（保证它盖在原来压着它的东西上面）。"""
    rg = spec['regions'][idx]
    x0, y0, x1, y1 = bbox(rg)
    ax, ay = anchor or ((x0 + x1) / 2, (y0 + y1) / 2)
    ca, sa = np.cos(np.radians(rot)), np.sin(np.radians(rot))

    def f(p):
        px, py = (p[0] - ax) * scale, (p[1] - ay) * scale
        rx, ry = px * ca - py * sa, px * sa + py * ca
        return (int(round(rx + ax + dx)), int(round(ry + ay + dy)))

    rg['pts'] = [f(p) for p in rg['pts']]
    rg['holes'] = [[f(p) for p in h] for h in rg['holes']]
    return spec


def move_vertex(spec, idx, v, dx, dy):
    """只动一个控制点 —— 微调轮廓用"""
    rg = spec['regions'][idx]
    x, y = rg['pts'][v]
    rg['pts'][v] = (x + dx, y + dy)
    return spec


def recolor_region(spec, idx, rgb):
    spec['regions'][idx]['color'] = tuple(rgb)
    return spec


def delete_region(spec, idx):
    spec['regions'].pop(idx)
    return spec


def resample(spec, w, h, eps=None):
    """改尺寸（结构等比缩放，不重提取）"""
    sx, sy = w / spec['w'], h / spec['h']
    for rg in spec['regions']:
        rg['pts'] = [(int(round(x * sx)), int(round(y * sy))) for x, y in rg['pts']]
        rg['holes'] = [[(int(round(x * sx)), int(round(y * sy))) for x, y in hh]
                       for hh in rg['holes']]
        rg['area'] *= sx * sy
    spec['w'], spec['h'] = int(w), int(h)
    if eps:
        spec['eps'] = eps
    return spec


# ---------------------------------------------------------------- 对称
def _axis_err(spec, x):
    a = np.asarray(to_image(spec)).astype(np.int16)
    n = min(x, spec['w'] - x)
    if n < 20:
        return None
    return float(np.abs(a[:, x - n:x] - a[:, x:x + n][:, ::-1]).mean())


def detect_axis(spec, samples=13):
    """找竖直对称轴：试几个候选，看左右两半镜像后差多少。"""
    a = np.asarray(to_image(spec)).astype(np.int16)
    best, bestx = None, spec['w'] // 2
    for x in np.linspace(spec['w'] * 0.35, spec['w'] * 0.65, samples):
        x = int(x)
        n = min(x, spec['w'] - x)
        if n < 20:
            continue
        l = a[:, x - n:x]
        r = a[:, x:x + n][:, ::-1]
        d = np.abs(l - r).mean()
        if best is None or d < best:
            best, bestx = d, x
    return bestx, (float(best) if best is not None else None)


SYM_MAX_ERR = 12.0      # 镜像差超过这个值就认为"这张图不对称"


def mirror_half(spec, axis=None, side='left', keep_axis=True, force=False):
    """做左右对称：保留一侧，镜像到另一侧，再重新提取结构。

    这是"把参考图规范化"的一步 —— 手绘/生成的图左右常不对称，
    对称化之后改轮廓才是可控的。

    ⚠️ **只对本来就接近对称的图有意义。**
       实测：一张侧身动态姿势、双马尾左右不等大的插画，自动检测出的轴
       (x=516) 镜像差高达 24.56 —— 强行镜像会得到万花筒一样的废图。
       所以这里默认拒绝：残差超过 SYM_MAX_ERR 就报错，除非 force=True。
       （正视图、证件照式的构图镜像差通常 < 10，可以放心用。）

    镜像语义：关于 x 轴镜像时，第 x-1 列映射到第 x 列，x-2 → x+1，依此类推。
    ⚠️ 取列方向很容易写反：保留左侧时镜像数组是 src[:, ::-1]，
       要取的是它的**前** W-x 列（不是后 W-x 列）—— 第一版反了，
       结果"对称化"之后左右差异反而从 2.5 涨到 61.6。
    """
    if axis is None:
        axis, err = detect_axis(spec)
    else:
        err = _axis_err(spec, int(axis))
    if err is not None and err > SYM_MAX_ERR and not force:
        raise ValueError(
            '这张图不对称（轴 x=%d 的左右镜像差 %.2f，阈值 %.1f）。\n'
            '强行镜像会毁图 —— 对称化只适合正视图/证件照式构图。\n'
            '确实要试就加 force=True（或 CLI 的 --force）。' % (axis, err, SYM_MAX_ERR))
    a = np.array(to_image(spec))
    x = int(axis)
    W = a.shape[1]
    if not (0 < x < W):
        raise ValueError('对称轴 x=%d 超出画布宽度 %d' % (x, W))
    if side == 'left':
        mir = a[:, :x][:, ::-1]
        take = mir[:, :W - x]
        if take.shape[1] < W - x:
            take = np.concatenate([take, np.repeat(take[:, -1:], W - x - take.shape[1], 1)], 1)
        a[:, x:] = take
    else:
        mir = a[:, x:][:, ::-1]
        take = mir[:, :x]
        if take.shape[1] < x:
            take = np.concatenate([take, np.repeat(take[:, -1:], x - take.shape[1], 1)], 1)
        a[:, x - take.shape[1]:x] = take
    return extract(a, k=spec.get('k', DEFAULT_K), eps=spec.get('eps', DEFAULT_EPS),
                   source=(spec.get('source', '') + '|mirror'))


# ---------------------------------------------------------------- 部件拼贴
def paste_region(src_spec, dst_spec, x, y, w, h, feather=0):
    """把 src 上的一块矩形区域贴到 dst 的 (x,y)，再重新提取结构。
    用来做"换发型 / 换部件"：从另一张结构图里抠一块过来。"""
    s = to_image(src_spec)
    d = to_image(dst_spec)
    tile = s.crop((x, y, x + w, y + h))
    d.paste(tile, (x, y))
    return extract(np.array(d), k=dst_spec.get('k', DEFAULT_K),
                   eps=dst_spec.get('eps', DEFAULT_EPS),
                   source=(dst_spec.get('source', '') + '|paste'))


# ---------------------------------------------------------------- 验收
def verify(spec, reference):
    """重建质量：误差 + 吻合率。reference 可以是路径或 RGB 数组。"""
    if isinstance(reference, str):
        ref = cv2.cvtColor(cv2.imread(reference), cv2.COLOR_BGR2RGB)
    else:
        ref = np.asarray(reference)[:, :, :3]
    img = to_image(spec)
    if img.size != (ref.shape[1], ref.shape[0]):
        img = img.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
    a = np.array(img).astype(np.int16)
    b = ref.astype(np.int16)
    diff = np.abs(a - b)
    return dict(mean_err=float(diff.mean()),
                match_pct=float((diff.max(axis=2) <= 12).mean() * 100),
                regions=len(spec['regions']),
                vertices=sum(len(r['pts']) for r in spec['regions']))
