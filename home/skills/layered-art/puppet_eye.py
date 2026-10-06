# -*- coding: utf-8 -*-
"""快速试眨眼：不跑整段动画，只出"睁 / 半闭 / 全闭"三态特写。

用途：调眨眼参数时**别每次重渲 42 帧**，先看这三张对不对。
眼睛外框给的是**紧贴眼睛**的框（比眼睛本身略大一点点就够）：
    ⚠️ 框太宽会把两侧头发也框进去，模糊底色时把头发糊成一片。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import puppet as PP       # noqa: E402

SRC = os.environ.get('PUPPET_SRC')
if not SRC:
    raise SystemExit('请用 PUPPET_SRC=<图片路径> 指定要做木偶的平面图')
def _out_dir(name):
    """输出目录：环境变量 LAYERED_ART_OUT 优先，否则落在系统临时目录。

    ⚠️ 不要在这里写死自己机器的路径 —— 打包分发时会被拒绝，
       别人机器上也不存在。
    """
    import tempfile
    base = os.environ.get('LAYERED_ART_OUT') or os.path.join(
        tempfile.gettempdir(), 'layered-art')
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    return d


OUT = os.environ.get('PUPPET_OUT') or _out_dir('puppet')
os.makedirs(OUT, exist_ok=True)

# 紧贴眼睛的框（从 115_头部.png 的网格上读的）
EYE = tuple(int(v) for v in (os.environ.get('EYE') or '447,183,503,269').split(','))
a0 = np.asarray(Image.open(SRC).convert('RGB')).astype(np.uint8)
sub = a0[EYE[1]:EYE[3], EYE[0]:EYE[2], :3].reshape(-1, 3)
lash = tuple(int(v) for v in sub[sub.mean(1).argmin()])
print('眼框 %s   睫毛色 %s' % (EYE, lash))

outs = [('睁', a0)]
for c, tag in ((0.45, '半闭'), (1.0, '全闭')):
    outs.append((tag, PP.blink(a0, EYE, lash=lash, close=c)))

box = (EYE[0] - 34, EYE[1] - 28, EYE[2] + 34, EYE[3] + 34)
k = 3
cw, ch = (box[2] - box[0]) * k, (box[3] - box[1]) * k
s = Image.new('RGB', (len(outs) * (cw + 8) + 8, ch + 26), (22, 24, 28))
d = ImageDraw.Draw(s)
for i, (n, fr) in enumerate(outs):
    s.paste(Image.fromarray(fr).crop(box).resize((cw, ch), Image.NEAREST),
            (8 + i * (cw + 8), 20))
    d.text((10 + i * (cw + 8), 5), n, fill=(205, 215, 230))
s.save(os.path.join(OUT, '眼睛三态.png'))
print('-> %s  %s' % (os.path.join(OUT, '眼睛三态.png'), s.size))
