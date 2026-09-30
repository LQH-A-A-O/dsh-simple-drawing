# -*- coding: utf-8 -*-
"""装配演示：眼睛 -> 脸 -> 整个人（三层嵌套）

验证用户的构想，并把"分辨率守恒"摆出来看：
    · 眼睛在**自己的画布**上画（5×5 成品格，但内部 8 倍 = 40×40，手很宽裕）
    · 降到成品格（5×5），再放进脸
    · 脸再放进身体
每一步都是**整数格偏移**，所以三层的格子严格对齐。
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


import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 不写绝对路径，打包才过得了净化检查
from assembly import Part, Assembler        # noqa: E402

OUT = _out_dir('rig')
os.makedirs(OUT, exist_ok=True)

BG = (248, 250, 253, 255)
C = {
    'skin': (243, 224, 210, 255), 'skin2': (219, 190, 174, 255),
    'hair': (36, 33, 46, 255), 'hair2': (74, 70, 96, 255), 'hair3': (128, 124, 150, 255),
    'iris': (58, 92, 150, 255), 'iris2': (32, 52, 92, 255),
    'eye': (26, 28, 42, 255), 'hilite': (250, 253, 255, 255),
    'mouth': (198, 116, 116, 255), 'blush': (246, 194, 192, 255),
    'blouse': (240, 244, 252, 255), 'navy': (40, 50, 86, 255),
    'navy2': (28, 36, 64, 255), 'red': (192, 62, 76, 255), 'shoe': (52, 52, 62, 255),
    'bg': BG,
}

# ================================================================ 1. 眼睛（自己的画布）
# 成品只占 5×5 格，但内部 8 倍 = 40×40 —— 手有空间，比例好控
EW, EH, ESUB = 4, 4, 8
eye = Part((EW, EH), sub=ESUB)
S = ESUB
eye.ellipse(EW * S / 2 - 0.5, EH * S / 2 - 0.5, 2.4 * S, 2.6 * S, C['eye'])      # 眼白/轮廓
eye.ellipse(EW * S / 2 - 0.5, EH * S / 2 - 0.2, 1.6 * S, 1.9 * S, C['iris'])     # 虹膜
eye.ellipse(EW * S / 2 - 0.5, EH * S / 2 + 0.1, 1.0 * S, 1.1 * S, C['iris2'])    # 瞳孔
eye.rect(int(1.2 * S), int(1.1 * S), int(1.7 * S), int(1.6 * S), C['hilite'])     # 高光
eye.rect(int(3.0 * S), int(2.6 * S), int(3.4 * S), int(2.9 * S), C['hilite'])     # 副高光
eye.rect(0, 0, EW * S - 1, int(0.7 * S), C['eye'])                                # 上眼睑（压在虹膜上）
eye_baked = eye.bake()
print('眼睛：成品 %dx%d 格，画布内部 %dx%d（%d 倍）'
      % (EW, EH, EW * S, EH * S, S))

# ================================================================ 2. 脸（自己的画布）
FW, FH, FSUB = 14, 16, 4
face = Part((FW, FH), sub=FSUB, bg=C['skin'])
F = FSUB
# 头发（上 + 两侧）
face.rect(0, 0, FW * F - 1, int(3.0 * F), C['hair'])
face.rect(0, 0, int(1.4 * F), int(13.0 * F), C['hair'])
face.rect(FW * F - int(1.4 * F), 0, FW * F - 1, int(13.0 * F), C['hair'])
# 刘海锯齿（斜向方块，保持方盒感）
for i in range(4):
    face.rect(int((2 + i * 4) * F), int(3.0 * F), int((3 + i * 4) * F), int((4.2 + i * 0.35) * F), C['hair'])
# 头发高光
for i in range(3):
    face.rect(int((2.5 + i) * F), int((1.0 + i) * F), int((4.0 + i) * F), int((1.5 + i) * F), C['hair3'])
# 腮红
face.rect(int(2.2 * F), int(13.5 * F), int(3.6 * F), int(14.6 * F), C['blush'])
# 眉
face.rect(int(3.0 * F), int(8.0 * F), int(6.4 * F), int(8.6 * F), C['hair2'])
# 嘴
face.rect(int(9.0 * F), int(16.0 * F), int(10.9 * F), int(16.6 * F), C['mouth'])

# ---- 把眼睛装进脸（整数格偏移），再镜像出右眼
# 脸宽 14 格，中轴在格线 7；左眼放第 1 格起，镜像后右眼落在第 9 格起
face.place(eye_baked, 1, 8, '左眼')
face.mirror_left_to_right()          # 眼睛、眉、腮红一起镜像
print('脸：成品 %dx%d 格；左眼在格 (3,9)，镜像后右眼在格 (%d,9)'
      % (FW, FH, FW - 1 - (3 + EW - 1)))
face_baked = face.bake()

# ================================================================ 3. 身体（总画布）
BODY_W, BODY_H = 34, 92
body = Assembler((BODY_W, BODY_H), bg=BG)
A = BODY_W // 2
sole = BODY_H - 3
# 7 头身：头 12 格（脸 20 宽比头略宽，作为"头发外扩"）
HC = 12
# ⚠️ 画布高必须 ≥ 7*HC + 上下留白。上一版给 78，而 7*12=84，
#    头顶算到行 -9（画布外），后面 slice 直接崩。
assert BODY_H >= 7 * HC + 6, '画布高 %d 装不下 7 头身 × 头 %d 格' % (BODY_H, HC)
ROW = {k: int(round(sole - v * HC)) for k, v in
       {'头顶': 7.0, '下巴': 6.0, '肩': 5.5, '腰': 4.0,
        '胯': 3.5, '膝': 2.0, '脚踝': 0.3}.items()}
print('身体：%dx%d 格，头 %d 格，头顶行 %d，脚底行 %d'
      % (BODY_W, BODY_H, HC, ROW['头顶'], sole))

skin = np.array([[C['skin']]], np.uint8)
blouse = np.array([[C['blouse']]], np.uint8)
navy = np.array([[C['navy2']]], np.uint8)
shoe = np.array([[C['shoe']]], np.uint8)


def band(y0, y1, x0, x1, rgb):
    """用 1x1 的部件铺一条带 —— 走 place() 通道，保证和别的部件同一套格。"""
    body.a[y0:y1 + 1, x0:x1 + 1] = rgb[0, 0]


# 腿（各 5 格宽）
band(ROW['胯'], ROW['脚踝'] - 1, A - 5, A - 1, navy)
band(ROW['胯'], ROW['脚踝'] - 1, A, A + 4, navy)
band(ROW['脚踝'], sole, A - 5, A - 1, shoe)
band(ROW['脚踝'], sole, A, A + 4, shoe)
# 躯干 16 格宽（肩宽 ≈ 1.7 头宽），手臂各 4 格宽贴在外侧
band(ROW['肩'], ROW['胯'] - 1, A - 8, A + 7, blouse)
band(ROW['肩'] + 1, ROW['腰'] + 2, A - 12, A - 9, blouse)
band(ROW['肩'] + 1, ROW['腰'] + 2, A + 8, A + 11, blouse)
band(ROW['腰'] + 3, ROW['胯'], A - 12, A - 9, skin)
band(ROW['腰'] + 3, ROW['胯'], A + 8, A + 11, skin)
# 裙（A 字，比躯干略宽）
band(ROW['胯'], ROW['胯'] + 7, A - 10, A + 9, navy)
# 领 + 领结
band(ROW['肩'], ROW['肩'] + 1, A - 8, A + 7, np.array([[C['navy']]], np.uint8))
band(ROW['肩'] + 2, ROW['肩'] + 3, A - 1, A, np.array([[C['red']]], np.uint8))
# 脖子
band(ROW['下巴'] - 1, ROW['肩'] - 1, A - 1, A, np.array([[C['skin2']]], np.uint8))

# ---- 把脸装进去（整数格偏移，取整）
hx = A - FW // 2
hy = ROW['头顶']
body.place(face_baked, hx, hy, '脸')
print('脸装配在格 (%d,%d)' % (hx, hy))

# 描边（按颜色差；这里部件少、尺寸够，效果比极小尺寸时好）
body.add_outline(k=0.74, bg=BG)
img_body = body.to_image(cell=2)

# ================================================================ 输出对照
def up(im, k):
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


# 眼睛三态：作者视图 -> 成品格 -> 放进脸里的实际样子
eye_auth = up(eye.preview(k=5), 1)
eye_final = up(Image.fromarray(eye_baked, 'RGBA'), 40 // max(EW, EH) * 5)
face_img = up(Image.fromarray(face_baked, 'RGBA'), 16)
body_img = up(img_body, 5)

# 对称性断言
L = body.a[:, :A].copy()
Rr = body.a[:, A:][:, ::-1].copy()
n = min(L.shape[1], Rr.shape[1])
sym = bool(np.array_equal(L[:, :n], Rr[:, :n]))
print('\n身体左右对称：%s' % ('是 ✓' if sym else '否 ✗'))

items = [('① 眼睛·作者画布 %dx%d（%d倍）' % (EW * S, EH * S, S), eye_auth),
         ('② 眼睛·降到成品 %dx%d 格' % (EW, EH), eye_final),
         ('③ 脸（眼睛已装进去）%dx%d 格' % (FW, FH), face_img),
         ('④ 整个人 %dx%d 格' % (BODY_W, BODY_H), body_img)]
pad = 14
W = sum(i.width for _, i in items) + pad * (len(items) + 1)
H = max(i.height for _, i in items) + 44
sheet = Image.new('RGB', (W, H), (22, 24, 28))
d = ImageDraw.Draw(sheet)
x = pad
for name, im in items:
    sheet.paste(im.convert('RGB'), (x, 34))
    d.text((x + 2, 14), name, fill=(205, 215, 230))
    x += im.width + pad
sheet.save(os.path.join(OUT, '装配_三层嵌套.png'))
print('->', os.path.join(OUT, '装配_三层嵌套.png'), sheet.size)
