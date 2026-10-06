# -*- coding: utf-8 -*-
"""立绘动画演示：抬手 / 放手 / 眨眼 / 眨眼单边 / 点头

用 363.png 那个角色的调色板（青绿发 + 粉红发带 + 淡紫衣 + 深蓝四肢）。
**全部坐标从骨架现取** —— 所以 `Clip` 里写的角度真的会让整张图动起来。
"""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anim as AN        # noqa: E402
import rig as RG         # noqa: E402


def _out_dir(name):
    import tempfile
    base = os.environ.get('LAYERED_ART_OUT') or os.path.join(
        tempfile.gettempdir(), 'layered-art')
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    return d


OUT = _out_dir('anim')
SIZE = (420, 900)
FPS = 12

# ================================================================ 时间轴
# 一个 3 秒的循环：站姿 -> 抬右手 -> 放下 -> 双击眨眼 -> 左手小抬 + 点头 -> 回站姿
c = AN.Clip(fps=FPS)
c.key(0.00, pose={}, face=AN.Face())
c.key(0.35, pose={'arm_R': 58, 'fore_R': -18}, face=AN.Face(brow=0.35))
c.key(0.70, pose={'arm_R': 62, 'fore_R': 24, 'tilt': -2},
      face=AN.Face(mouth=0.45, brow=0.55, blush=1.0))
c.key(1.00, pose={'arm_R': 20, 'fore_R': 8}, face=AN.Face())
c.key(1.20, pose={}, face=AN.Face(eye_open=0.0))            # 闭
c.key(1.32, pose={}, face=AN.Face(eye_open=1.0))            # 睁
c.key(1.44, pose={}, face=AN.Face(eye_open=0.0))            # 再闭
c.key(1.62, pose={}, face=AN.Face())                        # 睁
c.key(1.95, pose={'arm_L': 26, 'tilt': 3},
      face=AN.Face(eye_open_l=1.0, eye_open_r=0.0, mouth=0.3))   # 单眼眨
c.key(2.30, pose={'arm_L': 18, 'tilt': 3, 'arm_R': 12},
      face=AN.Face(brow=-0.4))
c.key(2.70, pose={'tilt': 1}, face=AN.Face())
c.key(3.00, pose={}, face=AN.Face())

DUR = 3.0
print('=== 时间轴 ===')
print('  %d 关键帧，%d fps，时长 %.1f 秒 -> %d 帧'
      % (len(c.keys), FPS, DUR, int(DUR * FPS)))
for t, p, f, e in c.keys:
    ang = ' '.join('%s=%g' % (k, v) for k, v in sorted(p.items())) or '—'
    print('   %4.2fs  角度 %-34s %s' % (t, ang, f))

# ================================================================ 逐帧渲染
frames = []
poses = c.frames(DUR)
for i, (pose, face) in enumerate(poses):
    r = RG.Rig()                       # 每帧都从干净模板起（pose 是绝对角度）
    # 呼吸：极轻微的上下浮动 + 肩线起伏（周期 1.5 秒）
    bob = 2.2 * (0.5 - 0.5 * abs(((i / float(FPS)) % 1.5) / 0.75 - 1.0))
    cam = AN.Cam(SIZE, head_units=r.t['head_count'])
    cam.bob = bob
    r.pose(pose)
    frames.append(AN.draw_char(r, face, cam=cam, size=SIZE,
                               head_units=r.t['head_count']))
print('\n渲染 %d 帧' % len(frames))

# ================================================================ 导出
gif = os.path.join(OUT, '动作_抬手眨眼.gif')
AN.export_gif(frames, gif, fps=FPS)
print('  GIF -> %s  (%.0f KB)' % (gif, os.path.getsize(gif) / 1024))

mp4 = os.path.join(OUT, '动作_抬手眨眼.mp4')
got = AN.export_mp4(frames, mp4, fps=FPS)
if got:
    print('  MP4 -> %s  (%.0f KB)' % (got, os.path.getsize(got) / 1024))
else:
    print('  MP4 跳过（没找到 ffmpeg，设 FFMPEG 环境变量可指定）')

# ================================================================ 联络图
picks = [0, 4, 8, 12, 15, 17, 22, 28, 34]
tiles = []
for i in picks:
    if i < len(frames):
        tiles.append(('%d (%.2fs)' % (i, i / float(FPS)), frames[i]))
pad = 6
tw = int(SIZE[0] * 0.55)
th = int(SIZE[1] * 0.55)
cols = len(tiles)
sheet = Image.new('RGB', (cols * (tw + pad) + pad, th + 26), (22, 24, 28))
d = ImageDraw.Draw(sheet)
for i, (name, fr) in enumerate(tiles):
    x = pad + i * (tw + pad)
    sheet.paste(fr.convert('RGB').resize((tw, th), Image.NEAREST), (x, 20))
    d.text((x + 2, 5), name, fill=(205, 215, 230))
sheet.save(os.path.join(OUT, '分镜抽帧.png'))
print('  分镜 -> %s' % os.path.join(OUT, '分镜抽帧.png'))

# 眼睛特写：睁 / 最闭 / 单眼眨 三态放大对比
# ⚠️ 别按固定帧号取 —— 眨眼的"最闭"那一帧落在关键帧附近，
#    固定取第 15 帧会取到过渡态（半睁），看着像没闭上。
closed_i = min(range(len(poses)), key=lambda i: poses[i][1].eye_open_l
               + poses[i][1].eye_open_r)
# 单眼眨：两只眼差别最大、**且至少一只确实是睁着的**。
# ⚠️ 只取 max(|l-r|) 会选中"两眼都闭"（差为 0 但和最小那帧重合），
#    所以必须加"至少一只 > 0.6"这个条件。
_cand = [(abs(f.eye_open_l - f.eye_open_r), i)
         for i, (p, f) in enumerate(poses)
         if max(f.eye_open_l, f.eye_open_r) > 0.6]
wink_i = max(_cand)[1] if _cand else 0
print('  最闭帧 = %d (眼开 %.2f)   单眼眨帧 = %d (%.2f/%.2f)'
      % (closed_i, poses[closed_i][1].eye_open_l,
         wink_i, poses[wink_i][1].eye_open_l, poses[wink_i][1].eye_open_r))

eye_tiles = []
for i, tag in ((0, '睁眼'), (closed_i, '闭眼'), (wink_i, '单眼眨')):
    if i < len(frames):
        fr = frames[i]
        cc = AN.Cam(SIZE, head_units=7.0)
        crop = fr.crop((int(cc.x(-1.0)), int(cc.y(7.15)),
                        int(cc.x(1.0)), int(cc.y(5.85))))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.NEAREST)
        eye_tiles.append(('%s  f%d  眼开%.0f%%'
                          % (tag, i, 100 * poses[i][1].eye_open_l), crop))
if eye_tiles:
    w = sum(t.width for _, t in eye_tiles) + 8 * (len(eye_tiles) + 1)
    h = max(t.height for _, t in eye_tiles) + 26
    s2 = Image.new('RGB', (w, h), (22, 24, 28))
    d2 = ImageDraw.Draw(s2)
    x = 8
    for n, t in eye_tiles:
        s2.paste(t.convert('RGB'), (x, 20))
        d2.text((x + 2, 5), n, fill=(205, 215, 230))
        x += t.width + 8
    s2.save(os.path.join(OUT, '表情对比.png'))
    print('  表情 -> %s' % os.path.join(OUT, '表情对比.png'))

print('\n产物 -> %s' % OUT)
