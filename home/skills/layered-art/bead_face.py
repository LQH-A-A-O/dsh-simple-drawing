# -*- coding: utf-8 -*-
"""拼豆人脸：局部精修 + 镜像（验证用户的想法）

流程：
  1. 建格阵（约束由 Grid 的 assert 保证）
  2. 头高量化成 **偶数格**（12 格）—— 否则脸左右各 5.5 格，画不出对称
  3. 身体用格单位画（不是像素）
  4. **只画左眼**，然后 mirror_cols 镜像出右眼
  5. 数值断言：左右两半逐格相同
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
from bead import Grid, snap_even          # noqa: E402

OUT = _out_dir('rig')
os.makedirs(OUT, exist_ok=True)

CELL = 2
G = Grid((128, 192), cell=CELL)
A = G.axis                                # 中轴所在格线 = 32
CX = A - 0.5                              # 居中形状的格中心 = 31.5（保证镜像对称）

# ---- 量化：头高必须是偶数格
HEAD_CELLS = snap_even(22.0 / CELL, CELL)          # 22px -> 11 格 -> 进位 12 格
print('头高：目标 22px = 11 格（奇数，不可用）-> 量化成 %d 格 = %dpx'
      % (HEAD_CELLS, HEAD_CELLS * CELL))
HC = HEAD_CELLS
# CANON7 各高度（头高为单位）-> 格
LEV = {'头顶': 7.0, '下巴': 6.0, '肩': 5.5, '胸': 4.5, '腰': 4.0,
       '胯': 3.5, '膝': 2.0, '脚踝': 0.30}
SOLE_ROW = 92
ROW = {k: int(round(SOLE_ROW - v * HC)) for k, v in LEV.items()}
print('关键行（格）:', ROW)
assert ROW['头顶'] >= 2 and SOLE_ROW <= G.ch - 2, '构图超出格阵'

PAL = {
    'hair': (28, 28, 38, 255), 'hair2': (52, 52, 72, 255),
    'skin': (238, 216, 202, 255), 'skin2': (214, 184, 168, 255),
    'blouse': (240, 244, 252, 255), 'navy': (36, 46, 80, 255),
    'red': (186, 58, 72, 255), 'shoe': (32, 32, 40, 255),
    'eye': (34, 36, 54, 255), 'hilite': (245, 250, 255, 255),
    'mouth': (196, 118, 118, 255), 'bg': (247, 250, 255, 255),
}
G.a[:, :] = PAL['bg']

# ================================================================ 身体（格单位）
r_sh = 9.5            # 肩半宽（格）
r_waist = 6.0
r_hip = 7.0
r_thigh = 3.2
r_shin = 2.4
r_uarm = 1.9
r_farm = 1.6

# 后发
G.ellipse(CX, (ROW['头顶'] + ROW['下巴']) / 2, HC / 2 + 0.5, HC / 2 + 1.0, PAL['hair'])
# 腿
for s, cx in ((-1, CX - 3.2), (1, CX + 3.2)):
    G.capsule(cx, ROW['胯'], cx + s * 0.6, ROW['膝'], r_thigh, PAL['skin'])
    G.capsule(cx + s * 0.6, ROW['膝'], cx + s * 0.8, ROW['脚踝'], r_shin, PAL['skin'])
    G.rect(int(cx - r_shin), ROW['脚踝'], int(cx + r_shin), SOLE_ROW - 1, PAL['shoe'])
# 躯干（上衣直接盖到胯 —— 修掉上一版"上衣和裙子之间露一截皮肤"）
G.rect(int(CX - r_sh), ROW['肩'], int(CX + r_sh), ROW['胯'], PAL['blouse'])
# 收腰：腰那一行两侧各切掉几格
for dy in range(0, ROW['胯'] - ROW['腰'] + 1):
    y = ROW['腰'] + dy
    cut = int(round((r_sh - r_waist) * (1 - dy / max(1, ROW['胯'] - ROW['腰']))))
    G.rect(int(CX - r_sh), y, int(CX - r_sh) + cut, y, PAL['bg'])
    G.rect(int(CX + r_sh) - cut, y, int(CX + r_sh), y, PAL['bg'])
# 百褶裙
hem = ROW['胯'] + 8
G.rect(int(CX - r_hip), ROW['胯'] - 1, int(CX + r_hip), hem, PAL['navy'])
for i in range(1, 5):
    t = i / 5.0
    x = int(round((CX - r_hip) + (2 * r_hip) * t))
    G.rect(x, ROW['胯'], x, hem, PAL['navy'] if i % 2 else (52, 64, 104, 255))
# 手臂（垂在身侧）
for s in (-1, 1):
    x0 = CX + s * (r_sh + 0.5)
    G.capsule(x0, ROW['肩'] + 1, x0 + s * 1.5, ROW['腰'] + 2, r_uarm, PAL['blouse'])
    G.capsule(x0 + s * 1.5, ROW['腰'] + 2, x0 + s * 2.0, ROW['胯'] + 1, r_farm, PAL['skin'])
# 脖子 + 水手领
G.rect(int(CX - 1.5), ROW['下巴'] - 1, int(CX + 1.5), ROW['肩'], PAL['skin2'])
G.rect(int(CX - r_sh), ROW['肩'], int(CX + r_sh), ROW['肩'] + 2, PAL['navy'])
G.rect(int(CX - 1), ROW['肩'] + 2, int(CX + 1), ROW['肩'] + 4, PAL['red'])

# ================================================================ 脸
cy_head = (ROW['头顶'] + ROW['下巴']) / 2.0
G.ellipse(CX, cy_head, HC / 2 - 0.5, HC / 2 - 0.5, PAL['skin2'])
G.ellipse(CX, cy_head + 0.5, HC / 2 - 1.5, HC / 2 - 1.5, PAL['skin'])
# 刘海
G.rect(int(CX - HC / 2), ROW['头顶'] - 1, int(CX + HC / 2), ROW['头顶'] + 3, PAL['hair'])
G.rect(int(CX - HC / 2), ROW['头顶'] + 4, int(CX - HC / 2 + 1), ROW['头顶'] + 7, PAL['hair'])
G.rect(int(CX + HC / 2 - 1), ROW['头顶'] + 4, int(CX + HC / 2), ROW['头顶'] + 7, PAL['hair'])

# ---- 只画**左眼**（列 A-4..A-3），然后镜像出右眼
# ⚠️ 别贴着脸的边画 —— 脸在 26..37，眼放 27..28 会和两侧头发糊成一团，
#    在拼豆这个尺度上眼睛就"消失"了（实测踩过）。往里收 2 格。
y_eye = ROW['下巴'] - 4
ex0, ex1 = A - 4, A - 3
G.rect(ex0, y_eye, ex1, y_eye + 1, PAL['eye'])
G.put(ex1, y_eye, PAL['hilite'])              # 高光放外侧格，镜像后左右对称
print('左眼占列 %d~%d，行 %d~%d' % (ex0, ex1, y_eye, y_eye + 1))

before = G.a.copy()
G.mirror_cols(ex0, ex1)                        # <<< 镜像出右眼
print('镜像后：右眼应占列 %d~%d' % (G.cw - 1 - ex1, G.cw - 1 - ex0))

# 嘴（跨中轴，天然对称）
G.rect(A - 1, ROW['下巴'] - 2, A, ROW['下巴'] - 2, PAL['mouth'])
# 腮红
G.rect(A - 7, ROW['下巴'] - 3, A - 6, ROW['下巴'] - 2, (240, 190, 186, 255))

# ================================================================ 整体镜像
# ⚠️ 关键教训：**不要指望每个图元各自对称**。上面裙子用 int(CX-7)..int(CX+7)
#    得到 15 列（中心在 31，不是 31.5），鞋、领结、腰带都有同类半格误差。
#    在 128/192 这种格阵上，半格误差肉眼看得见（左右不对称）。
#    正解：**对称的部件随便画，最后整体镜像一次**，对称性由构造保证。
G.mirror_left_to_right()

# ---- 镜像之后再画**非对称**的东西（发夹、挎包、侧马尾都放这一层）
#      这一层不受镜像影响，所以"整体镜像"不会把所有角色都变成对称的
G.rect(A - 9, ROW['头顶'] + 3, A - 8, ROW['头顶'] + 4, (232, 96, 108, 255))
print('非对称件（左侧发夹）画在镜像之后，列 %d~%d' % (A - 9, A - 8))

# ================================================================ 断言
L = G.a[:, :A].copy()
Rr = G.a[:, A:][:, ::-1].copy()
# 故意非对称的左侧发夹要排除，否则它必然报不对称
clip_rows = slice(ROW['头顶'] + 3, ROW['头顶'] + 5)
L[clip_rows, :] = 0
Rr[clip_rows, :] = 0
sym = np.array_equal(L, Rr)
print('\n左右两半逐格相同（排除故意非对称的发夹）：%s' % ('是 ✓' if sym else '否 ✗'))
if not sym:
    diff = np.nonzero((L != Rr).any(2))
    print('  不同的格：%d 个，例如 %s' % (len(diff[0]), list(zip(diff[1][:5], diff[0][:5]))))
# 发夹确实存在，证明"镜像之后再画"这一层有效
clip = G.a[clip_rows, A - 9:A - 7]
print('发夹是否保留：%s（镜像没有把它抹掉）'
      % ('是 ✓' if (clip[:, :, 3] > 0).any() else '否 ✗'))
print('画布 %dx%d  格阵 %dx%d  cell=%d  中轴在列 %d 的格线上'
      % (G.px[0], G.px[1], G.cw, G.ch, CELL, A))

# ================================================================ 输出
full = G.to_image()
full.resize((full.width * 3, full.height * 3), Image.NEAREST).save(
    os.path.join(OUT, '拼豆_全身.png'))
face_box = (A - 8, ROW['头顶'] - 3, A + 7, ROW['下巴'] + 3)
G.zoom(face_box, k=14).save(os.path.join(OUT, '拼豆_脸_带格线.png'))
Image.fromarray(G.a[face_box[1]:face_box[3] + 1, face_box[0]:face_box[2] + 1], 'RGBA'
                ).resize(((face_box[2] - face_box[0] + 1) * 14,
                          (face_box[3] - face_box[1] + 1) * 14), Image.NEAREST
                         ).save(os.path.join(OUT, '拼豆_脸_无格线.png'))

# 并排：全身(3x) + 脸(无格线) + 脸(带格线)
a1 = Image.open(os.path.join(OUT, '拼豆_全身.png')).convert('RGB')
a2 = Image.open(os.path.join(OUT, '拼豆_脸_无格线.png')).convert('RGB')
a3 = Image.open(os.path.join(OUT, '拼豆_脸_带格线.png')).convert('RGB')
H = max(a1.height, a2.height, a3.height)
sheet = Image.new('RGB', (a1.width + a2.width + a3.width + 40, H + 40), (24, 26, 30))
d = ImageDraw.Draw(sheet)
x = 10
for n, im in (('全身 3x', a1), ('脸（局部精修）', a2), ('脸 + 格线（数格子用）', a3)):
    sheet.paste(im, (x, 30))
    d.text((x + 2, 10), n, fill=(200, 210, 225))
    x += im.width + 10
sheet.save(os.path.join(OUT, '拼豆_验收.png'))
print('\n产物 ->', OUT)
