# -*- coding: utf-8 -*-
"""Minecraft 皮肤：**逐面设计 -> 整合成方块人**

用户要的工作方式
----------------
"将这种设计分成单独的每一面做，最后整合成一块整体" ——
所以本模块的核心 API 是**按面**操作：一个部位（头/身/臂/腿）× 一个朝向
（前/后/左/右/上/下）= 贴图上一块固定矩形。逐面画完，再整合渲染。

⚠️ 不管用哪张参考图，**贴图布局是公开标准**，不要自己发明：
   64×64 贴图，每个部位 (w,h,d) 在基准点 (bx,by) 处按下面的顺序摆：
       top    (bx+d,     by)        w × d
       bottom (bx+d+w,   by)        w × d
       right  (bx,       by+d)      d × h
       front  (bx+d,     by+d)      w × h
       left   (bx+d+w,   by+d)      d × h
       back   (bx+d+w+d, by+d)      w × h
   实测头 (0,0) 8×8×8 -> top(8,0) front(8,8) back(24,8)，和标准一致。

渲染：把每个 texel 展开成一个小四边形，再拼成方块骨架。
这样能直接复用现有的 render_regions（画家算法 + 掩膜），
不需要写 UV 采样。
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw

__all__ = ['SKIN_SIZE', 'PARTS', 'FACES', 'new_skin', 'face_rect', 'paint',
           'paint_px', 'paste_face', 'preview_face', 'preview_layout',
           'build_textured', 'render', 'save_skin']

SKIN_SIZE = (64, 64)

# 部位 -> (宽, 高, 深) + 贴图基准点。尺寸就是《我的世界》玩家模型的尺寸，
# 单位是贴图象素。
PARTS = {
    '头':   dict(whd=(8, 8, 8),  base=(0, 0)),
    '身':   dict(whd=(8, 12, 4), base=(16, 16)),
    '右臂': dict(whd=(4, 12, 4), base=(40, 16)),
    '左臂': dict(whd=(4, 12, 4), base=(32, 48)),
    '右腿': dict(whd=(4, 12, 4), base=(0, 16)),
    '左腿': dict(whd=(4, 12, 4), base=(16, 48)),
}
FACES = ('top', 'bottom', 'right', 'front', 'left', 'back')


def new_skin(fill=(0, 0, 0, 0)):
    return Image.new('RGBA', SKIN_SIZE, fill)


def face_rect(part, face):
    """某个部位的某个面在贴图上的 (x, y, w, h)。"""
    w, h, d = PARTS[part]['whd']
    bx, by = PARTS[part]['base']
    table = {
        'top':    (bx + d,     by,     w, d),
        'bottom': (bx + d + w, by,     w, d),
        'right':  (bx,         by + d, d, h),
        'front':  (bx + d,     by + d, w, h),
        'left':   (bx + d + w, by + d, d, h),
        'back':   (bx + d + w + d, by + d, w, h),
    }
    return table[face]


def paint(skin, part, face, color):
    """整面填色。"""
    x, y, w, h = face_rect(part, face)
    ImageDraw.Draw(skin).rectangle([x, y, x + w - 1, y + h - 1], fill=color)
    return skin


def paint_px(skin, part, face, u, v, color):
    """在某个面上点一个像素。u/v 从该面左上角算（贴图坐标系）。"""
    x, y, w, h = face_rect(part, face)
    if 0 <= u < w and 0 <= v < h:
        skin.putpixel((x + u, y + v), color)
    return skin


def paste_face(skin, part, face, img):
    """把一张小图贴到某个面上；尺寸必须**恰好**等于该面的尺寸。

    ⚠️ 尺寸不匹配会直接报错，不做缩放 —— 缩放会让像素错位一格，
       在 8×8 的脸上就是灾难，而且很难看出来。
    """
    x, y, w, h = face_rect(part, face)
    img = img.convert('RGBA')
    if img.size != (w, h):
        raise ValueError('%s/%s 需要 %dx%d，给的是 %dx%d'
                         % (part, face, w, h, img.width, img.height))
    skin.alpha_composite(img, (x, y))
    return skin


# ---------------------------------------------------------------- 预览
def preview_face(skin, part, face, zoom=24, bg=(30, 32, 38)):
    """单独放大预览一个面 —— 逐面设计时的主要工作面。"""
    x, y, w, h = face_rect(part, face)
    tile = skin.crop((x, y, x + w, y + h))
    out = Image.new('RGB', (w * zoom, h * zoom), bg)
    out.paste(tile.convert('RGB'), (0, 0),
              tile.split()[-1])
    return out


def preview_layout(skin, zoom=8):
    """整张 64×64 贴图 + 每个面的名字标注（用来看画到哪了）。"""
    z = zoom
    out = skin.convert('RGBA').resize((64 * z, 64 * z), Image.NEAREST)
    board = Image.new('RGB', (out.width, out.height), (26, 28, 34))
    board.paste(out.convert('RGB'), (0, 0), out.split()[-1])
    d = ImageDraw.Draw(board)
    for part in PARTS:
        for f in FACES:
            x, y, w, h = face_rect(part, f)
            d.rectangle([x * z, y * z, (x + w) * z - 1, (y + h) * z - 1],
                        outline=(90, 190, 255), width=1)
            if w * z > 42 and h * z > 20:
                d.text((x * z + 3, y * z + 3), '%s%s' % (part, f[:2]),
                       fill=(255, 235, 120))
    return board


# ---------------------------------------------------------------- 整合渲染
def _texel_color(skin, x, y):
    r, g, b, a = skin.getpixel((x, y))
    if a == 0:
        return None
    return (r, g, b)


def build_textured(skin, unit=1.0, angle=0.0):
    """把皮肤展开成"每个 texel 一个四边形"的网格，并摆成方块人骨架。

    unit  : 一个贴图象素 = 多少世界单位（身高 32 像素 -> 32*unit）
    angle : 绕竖轴旋转的角度（度）。0 = 正视图；约 20 度就是常见的 3/4 视角。

    坐标约定：y 向上，0 = 脚底；x 右；z 朝相机。
    """
    th = math.radians(angle)
    ca, sa = math.cos(th), math.sin(th)

    def rot(p):
        x, y, z = p
        return (x * ca + z * sa, y, -x * sa + z * ca)

    V, F, C = [], [], []
    # 部位在骨架里的位置：(中心 x, 底部 y, 中心 z)
    # y 从脚底往上数：腿 12、身 12、头 8 = 32
    LAYOUT = {
        '头':   (0.0, 24.0, 0.0),
        '身':   (0.0, 12.0, 0.0),
        '右臂': (-6.0, 12.0, 0.0),
        '左臂': (6.0, 12.0, 0.0),
        '右腿': (-2.0, 0.0, 0.0),
        '左腿': (2.0, 0.0, 0.0),
    }
    # 每个面的外法线和面的两个切向轴（保证从外面看是逆时针）
    FACE_AXES = {
        'front':  ((0, 0, 1),  (1, 0, 0), (0, 1, 0)),
        'back':   ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        'right':  ((1, 0, 0),  (0, 0, -1), (0, 1, 0)),
        'left':   ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        'top':    ((0, 1, 0),  (1, 0, 0), (0, 0, -1)),
        'bottom': ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    }

    for part, meta in PARTS.items():
        w, h, d = meta['whd']
        cx, by, cz = LAYOUT[part]
        size = {'front': (w, h), 'back': (w, h), 'right': (d, h),
                'left': (d, h), 'top': (w, d), 'bottom': (w, d)}
        for face, (n, ua, va) in FACE_AXES.items():
            fw, fh = size[face]
            # 该面的中心（相对部位中心）
            off = {'front': (0, 0, d / 2), 'back': (0, 0, -d / 2),
                   'right': (w / 2, 0, 0), 'left': (-w / 2, 0, 0),
                   'top': (0, h / 2, 0), 'bottom': (0, -h / 2, 0)}[face]
            ox, oy, oz = off
            x0, y0 = face_rect(part, face)[0], face_rect(part, face)[1]
            for tv in range(fh):
                for tu in range(fw):
                    col = _texel_color(skin, x0 + tu, y0 + tv)
                    if col is None:
                        continue
                    # 该 texel 在面上的四角（左上、右上、右下、左下），单位=贴图象素
                    a = (-fw / 2 + tu, fh / 2 - tv)
                    corners = [(a[0], a[1]), (a[0] + 1, a[1]),
                               (a[0] + 1, a[1] - 1), (a[0], a[1] - 1)]
                    base = []
                    for cu, cv in corners:
                        # ⚠️ 全程用**贴图象素**为单位算，最后再统一乘 unit。
                        #    之前把"部位中心(象素)"和"偏移量×unit(世界单位)"直接相加，
                        #    量纲混了，渲出来是一堆斜条纹。
                        px = cx + ox + ua[0] * cu + va[0] * cv
                        py = by + h / 2 + oy + ua[1] * cu + va[1] * cv
                        pz = cz + oz + ua[2] * cu + va[2] * cv
                        base.append(rot((px * unit, py * unit, pz * unit)))
                    i0 = len(V)
                    V.extend(base)
                    F += [[i0, i0 + 1, i0 + 2], [i0, i0 + 2, i0 + 3]]
                    C += [col, col]
    return np.array(V, float), np.array(F, np.int32), np.array(C, float)


def render(skin, size=(760, 900), unit=22.0, angle=18.0, bg=(247, 250, 255),
           outline=None, up_axis=1, side_axis=0, depth_axis=2):
    """把皮肤整合渲染成图。返回 PIL Image。"""
    import mesh as ME
    V, F, C = build_textured(skin, unit=unit, angle=angle)
    if not len(F):
        return Image.new('RGB', size, bg)
    # 手工算投影（不用 project 的自动适配 —— 我们要固定的取景）
    y = V[:, up_axis]
    cy = (y.max() + y.min()) / 2
    k = size[1] * 0.92 / max(y.max() - y.min(), 1e-6)
    W, H = size
    xs = V[:, side_axis]
    zs = V[:, depth_axis]
    cx = (xs.max() + xs.min()) / 2
    P = np.stack([W / 2 + (xs - cx) * k, H / 2 - (y - cy) * k], 1)
    Z = zs.copy()
    tri = P[F]
    order = np.argsort(Z[F].mean(axis=1))          # 远的先画
    img = Image.new('RGB', size, bg)
    d = ImageDraw.Draw(img)
    for i in order:
        t = tri[i]
        d.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=tuple(int(v) for v in C[i]))
    if outline:
        img = ME.add_outline(img, bg, outline)
    return img


def save_skin(skin, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    skin.convert('RGBA').save(path)
    return path
