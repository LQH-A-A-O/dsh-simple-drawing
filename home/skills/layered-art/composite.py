# -*- coding: utf-8 -*-
"""高分辨率母版合成：**一个部件一块画布 -> 一张图**

用户的最终构想
--------------
"在无限大的画布里将画好的东西整合输出成一张图片（只看最后的图片），
 相当于细节到每一个像素点可看，但实际上是一张图片。"

这正是它成立的形态，但四条规则必须换掉（对比拼豆格阵）：

    部件画布     按需给，用**自然尺寸**（不再受"成品占几格"约束）
    摆放坐标     用**头高**为单位（不再要求整数格，高分辨率下亚像素无所谓）
    降采样       不降采样 —— 保持原生分辨率
    分隔手段     靠**明暗/描边**都行，分辨率够就不缺内部格子

⚠️ 新问题：**部件之间的缝**。规则是「画大一圈，靠重叠消灭缝隙」——
   重叠是安全的（后画的盖住先画的），**缝隙是致命的**（露出背景色）。
   所以每个部件都画得比它"应该"的尺寸大一点，宁重叠不留空。

关键合成点：**高分辨率是母版，拼豆滤镜是输出选项。**
同一份母版可以同时出精细版和拼豆版 —— "拼豆"从架构约束降级成输出滤镜。
"""
import os


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



import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = _out_dir('rig')
os.makedirs(OUT, exist_ok=True)

SS = 1                       # **不超采样** —— 抗锯齿的平滑边不讨喜，
                             # 保留硬像素锯齿。1 个像素就是 1 个像素。
C = {
    'skin': (243, 224, 210), 'skin2': (222, 194, 178), 'skin3': (198, 164, 148),
    'hair': (36, 33, 46), 'hair2': (66, 62, 84), 'hair3': (118, 114, 138),
    'iris': (72, 116, 186), 'iris2': (40, 72, 138), 'pupil': (20, 22, 34),
    'lash': (28, 26, 38), 'hilite': (252, 254, 255),
    'blouse': (241, 245, 253), 'blouse2': (206, 216, 234),
    'navy': (42, 52, 88), 'navy2': (28, 36, 64),
    'red': (194, 62, 78), 'shoe': (44, 44, 54),
    'line': (44, 40, 56),
}


def new(w, h):
    return Image.new('RGBA', (w * SS, h * SS), (0, 0, 0, 0))


def done(im, w, h):
    """**不做缩放** —— SS=1 时画布本来就是目标尺寸。

    ⚠️ 不要用 LANCZOS / BILINEAR 缩小。实测缩小会把硬边糊成灰渐变，
       像素锯齿那种"讨喜"的感觉全丢。要放大就用 NEAREST。
    """
    if SS == 1:
        return im
    return im.resize((w, h), Image.NEAREST)


# ================================================================ 眼睛（自己的画布）
def draw_eye(w=64, h=64):
    """眼睛：**在自己的画布上按自然尺寸画**，所以虹膜/瞳孔/高光/睫毛全都在。
    上一版它被降到 4×4 格，细节全没了 —— 这里不再降。"""
    im = new(w, h)
    d = ImageDraw.Draw(im)
    s = SS
    cx, cy = w * s / 2, h * s / 2
    # 眼白
    d.ellipse([cx - 0.40 * w * s, cy - 0.30 * h * s, cx + 0.40 * w * s, cy + 0.32 * h * s],
              fill=(250, 250, 253, 255))
    # 虹膜（外深内浅，做出立体感）
    d.ellipse([cx - 0.24 * w * s, cy - 0.24 * h * s, cx + 0.24 * w * s, cy + 0.24 * h * s],
              fill=C['iris'] + (255,))
    d.ellipse([cx - 0.19 * w * s, cy - 0.20 * h * s, cx + 0.19 * w * s, cy + 0.18 * h * s],
              fill=C['iris2'] + (255,))
    # 瞳孔
    d.ellipse([cx - 0.10 * w * s, cy - 0.12 * h * s, cx + 0.10 * w * s, cy + 0.11 * h * s],
              fill=C['pupil'] + (255,))
    # 主高光（左上）+ 副高光（右下）
    d.ellipse([cx - 0.17 * w * s, cy - 0.19 * h * s, cx - 0.06 * w * s, cy - 0.07 * h * s],
              fill=C['hilite'] + (255,))
    d.ellipse([cx + 0.06 * w * s, cy + 0.06 * h * s, cx + 0.13 * w * s, cy + 0.14 * h * s],
              fill=C['hilite'] + (215,))
    # 上眼睑（压在虹膜上）+ 睫毛
    d.polygon([(cx - 0.42 * w * s, cy - 0.18 * h * s), (cx + 0.42 * w * s, cy - 0.22 * h * s),
               (cx + 0.42 * w * s, cy - 0.34 * h * s), (cx - 0.42 * w * s, cy - 0.32 * h * s)],
              fill=C['lash'] + (255,))
    d.polygon([(cx + 0.30 * w * s, cy - 0.30 * h * s), (cx + 0.50 * w * s, cy - 0.42 * h * s),
               (cx + 0.50 * w * s, cy - 0.26 * h * s), (cx + 0.30 * w * s, cy - 0.20 * h * s)],
              fill=C['lash'] + (255,))
    # 下眼睑（细）
    d.line([(cx - 0.34 * w * s, cy + 0.26 * h * s), (cx + 0.34 * w * s, cy + 0.24 * h * s)],
           fill=C['skin3'] + (255,), width=max(1, int(0.035 * h * s)))
    return done(im, w, h)


# ================================================================ 脸（自己的画布）
def draw_face(w=200, h=240):
    im = new(w, h)
    d = ImageDraw.Draw(im)
    s = SS
    # 脸型（椭圆）
    d.ellipse([0.06 * w * s, 0.10 * h * s, 0.94 * w * s, 1.02 * h * s], fill=C['skin'] + (255,))
    # 下巴收窄一点
    d.polygon([(0.16 * w * s, 0.72 * h * s), (0.84 * w * s, 0.72 * h * s),
               (0.50 * w * s, 1.00 * h * s)], fill=C['skin'] + (255,))
    return done(im, w, h)


def draw_head(w=260, h=300):
    """头 + 头发。画得比脸大一圈，靠重叠盖住脸的外缘（消灭缝隙）。"""
    im = new(w, h)
    d = ImageDraw.Draw(im)
    s = SS
    # 后发（比头大一圈）
    d.ellipse([0.02 * w * s, 0.02 * h * s, 0.98 * w * s, 1.00 * h * s], fill=C['hair'] + (255,))
    # 两侧垂发
    d.polygon([(0.04 * w * s, 0.30 * h * s), (0.22 * w * s, 0.30 * h * s),
               (0.24 * w * s, 0.96 * h * s), (0.02 * w * s, 0.96 * h * s)],
              fill=C['hair'] + (255,))
    d.polygon([(0.78 * w * s, 0.30 * h * s), (0.96 * w * s, 0.30 * h * s),
               (0.98 * w * s, 0.96 * h * s), (0.76 * w * s, 0.96 * h * s)],
              fill=C['hair'] + (255,))
    # 刘海（斜向锯齿）
    pts = [(0.05 * w * s, 0.06 * h * s), (0.95 * w * s, 0.06 * h * s)]
    for i in range(6):
        x = 0.95 - i * 0.15
        pts += [(x * w * s, (0.30 + (i % 2) * 0.06) * h * s),
                ((x - 0.075) * w * s, (0.34 + (i % 2) * 0.06) * h * s)]
    d.polygon(pts, fill=C['hair'] + (255,))
    # 头发高光（斜向弧带）
    d.polygon([(0.16 * w * s, 0.14 * h * s), (0.62 * w * s, 0.10 * h * s),
               (0.58 * w * s, 0.17 * h * s), (0.14 * w * s, 0.21 * h * s)],
              fill=C['hair3'] + (150,))
    d.polygon([(0.20 * w * s, 0.22 * h * s), (0.56 * w * s, 0.18 * h * s),
               (0.54 * w * s, 0.23 * h * s), (0.18 * w * s, 0.27 * h * s)],
              fill=C['hair3'] + (95,))
    return done(im, w, h)


# ================================================================ 身体
def draw_body(w=560, h=1300, head_units=7.0):
    """7 头身。头高 = h/7。全部按头高定位（不再用格）。"""
    im = new(w, h)
    d = ImageDraw.Draw(im)
    s = SS
    HU = h / head_units                       # 1 头高 = 多少像素
    sole = h * 0.97
    cx = w / 2

    def Y(v):                                 # 头高坐标 -> 像素（y 向上）
        return (sole - v * HU) * s

    def X(v):
        return (cx + v * HU) * s

    R = {k: v for k, v in
         {'肩': 5.5, '腰': 4.0, '胯': 3.5, '膝': 2.0, '脚踝': 0.30}.items()}
    # 腿（两条，各自一个梯形，有粗细变化）
    for sgn in (-1, 1):
        hip_x, ank_x = sgn * 0.30, sgn * 0.34
        d.polygon([(X(hip_x - 0.17), Y(R['胯'])), (X(hip_x + 0.17), Y(R['胯'])),
                   (X(ank_x + 0.11), Y(R['脚踝'])), (X(ank_x - 0.11), Y(R['脚踝']))],
                  fill=C['skin'] + (255,))
        # 鞋
        d.polygon([(X(ank_x - 0.12), Y(R['脚踝'] + 0.02)), (X(ank_x + 0.12), Y(R['脚踝'] + 0.02)),
                   (X(ank_x + 0.13), Y(0.0)), (X(ank_x - 0.13), Y(0.0))],
                  fill=C['shoe'] + (255,))
    # 躯干（收腰）
    d.polygon([(X(-0.62), Y(R['肩'])), (X(0.62), Y(R['肩'])),
               (X(0.42), Y(R['腰'])), (X(0.50), Y(R['胯'])), (X(-0.50), Y(R['胯'])),
               (X(-0.42), Y(R['腰']))], fill=C['blouse'] + (255,))
    # 手臂
    for sgn in (-1, 1):
        d.polygon([(X(sgn * 0.60), Y(R['肩'] + 0.10)), (X(sgn * 0.78), Y(R['肩'] + 0.10)),
                   (X(sgn * 0.82), Y(R['腰'] + 0.10)), (X(sgn * 0.64), Y(R['腰'] + 0.10))],
                  fill=C['blouse'] + (255,))
        d.polygon([(X(sgn * 0.64), Y(R['腰'] + 0.15)), (X(sgn * 0.82), Y(R['腰'] + 0.15)),
                   (X(sgn * 0.80), Y(R['胯'] + 0.05)), (X(sgn * 0.62), Y(R['胯'] + 0.05))],
                  fill=C['skin'] + (255,))
    # 百褶裙（A 字）
    hem = R['胯'] - 0.75
    d.polygon([(X(-0.56), Y(R['胯'] + 0.15)), (X(0.56), Y(R['胯'] + 0.15)),
               (X(0.76), Y(hem)), (X(-0.76), Y(hem))], fill=C['navy2'] + (255,))
    for i in range(1, 7):
        t = i / 7.0
        x0 = -0.56 + 1.12 * t
        x1 = -0.76 + 1.52 * t
        d.line([(X(x0), Y(R['胯'] + 0.15)), (X(x1), Y(hem))],
               fill=C['navy'] + (255,), width=max(1, int(0.02 * HU * s)))
    # 水手领 + 领结
    d.polygon([(X(-0.60), Y(R['肩'] + 0.16)), (X(0.60), Y(R['肩'] + 0.16)),
               (X(0.34), Y(R['肩'] - 0.14)), (X(-0.34), Y(R['肩'] - 0.14))],
              fill=C['navy'] + (255,))
    d.polygon([(X(-0.10), Y(R['肩'] - 0.10)), (X(0.10), Y(R['肩'] - 0.10)),
               (X(0.0), Y(R['肩'] - 0.34))], fill=C['red'] + (255,))
    return done(im, w, h)


def bead(im, cell):
    """拼豆滤镜：按 cell×cell 取**众数**（不是平均）。母版 -> 拼豆版。"""
    a = np.asarray(im.convert('RGBA'))
    h, w = a.shape[:2]
    H, W = h // cell, w // cell
    a = a[:H * cell, :W * cell].reshape(H, cell, W, cell, 4).transpose(0, 2, 1, 3, 4)
    a = a.reshape(H, W, cell * cell, 4)
    out = np.zeros((H, W, 4), np.uint8)
    for i in range(H):
        for j in range(W):
            px = a[i, j].astype(np.int64)
            key = (px[:, 0] << 24) | (px[:, 1] << 16) | (px[:, 2] << 8) | px[:, 3]
            v, c = np.unique(key, return_counts=True)
            k = int(v[c.argmax()])
            out[i, j] = ((k >> 24) & 255, (k >> 16) & 255, (k >> 8) & 255, k & 255)
    return Image.fromarray(out, 'RGBA').resize((W * cell, H * cell), Image.NEAREST)


if __name__ == '__main__':
    W, H = 560, 1300
    canvas = Image.new('RGBA', (W, H), (249, 250, 253, 255))
    body = draw_body(W, H, 7.0)
    canvas.alpha_composite(body)

    # ---- 头：先放头发层，再放脸，再放眼睛（**顺序 + 重叠** 消灭缝隙）
    HU = H / 7.0
    sole = H * 0.97
    head_c = sole - 6.5 * HU                    # 头中心（头高 6.5 = 顶7 下巴6 的中点）
    hw, hh = int(HU * 1.30), int(HU * 1.42)     # 头部件画布（比头大一圈）
    head = draw_head(hw, hh)
    fx, fy = int(hw * 0.14), int(hh * 0.20)
    fw, fh = int(HU * 1.00), int(HU * 1.12)
    face = draw_face(fw, fh)
    ew, eh = int(HU * 0.34), int(HU * 0.30)
    eye = draw_eye(ew, eh)
    eye_r = eye.transpose(Image.FLIP_LEFT_RIGHT)     # 右眼镜像

    def put(layer, cxp, cyp):
        canvas.alpha_composite(layer, (int(cxp - layer.width / 2), int(cyp - layer.height / 2)))

    put(head, W / 2, head_c)                          # 1 头发
    put(face, W / 2, head_c + HU * 0.06)              # 2 脸（盖住头发中间）
    ey = head_c - HU * 0.04                           # 眼睛在脸中偏上
    put(eye, W / 2 - HU * 0.21, ey)                   # 3 左眼
    put(eye_r, W / 2 + HU * 0.21, ey)                 # 4 右眼

    out = canvas.convert('RGB')
    out.save(os.path.join(OUT, '母版_高分辨率.png'))
    print('母版 %dx%d ->' % out.size, os.path.join(OUT, '母版_高分辨率.png'))

    # 眼睛特写（证明"细节到每一个像素点可看"）
    ec = int(HU * 0.21)
    crop = out.crop((int(W / 2 - ec - ew), int(ey - eh), int(W / 2 - ec + ew), int(ey + eh)))
    crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST).save(
        os.path.join(OUT, '母版_眼睛特写.png'))

    # 拼豆滤镜：同一份母版出另一个版本
    b = bead(out, 8)
    b.save(os.path.join(OUT, '同母版_拼豆版.png'))
    print('拼豆版 %dx%d（cell=8，同一份母版）' % b.size)

    # 排版
    pad = 16
    th = 900
    def fit(im):
        k = th / im.height
        return im.resize((int(im.width * k), th), Image.NEAREST)
    items = [('母版 560x1300（每像素可看）', fit(out)),
             ('同一母版 + 拼豆滤镜 cell=8', fit(b.convert('RGB'))),
             ('眼睛特写 3x', Image.open(os.path.join(OUT, '母版_眼睛特写.png')).convert('RGB'))]
    Wt = sum(i.width for _, i in items) + pad * (len(items) + 1)
    Ht = th + 44
    sheet = Image.new('RGB', (Wt, Ht), (22, 24, 28))
    d = ImageDraw.Draw(sheet)
    x = pad
    for n, im in items:
        sheet.paste(im, (x, 34))
        d.text((x + 2, 14), n, fill=(205, 215, 230))
        x += im.width + pad
    sheet.save(os.path.join(OUT, '母版_与拼豆版.png'))
    print('->', os.path.join(OUT, '母版_与拼豆版.png'), sheet.size)
