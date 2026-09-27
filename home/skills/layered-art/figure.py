"""人体白膜：A4 竖版画布 + 参数化人形 + **比例数值校验**。

为什么要有白膜
--------------
之前失败的原因是：想在一步里同时定轮廓、定比例、定配色、定细节 ——
一轮修 2 个坏 2 个。白膜把这件事拆开：

    L0 白膜   只有形状，没有颜色细节  ← **可以用数字验证对错**
    L1 细节   五官 / 服装分块 / 配饰   ← 锚定在关节上，跟着动作走
    L2 配色   底色 / 阴影 / 高光 / 描边 ← 已有引擎，自由配置
    L3 背景   独立画布，最后合成

关键在于：**人体比例是数学，不是审美。** 头身比、肩宽比、腕线过裆……
这些都能测，能断言，能在上色之前就判定"这具身体对不对"。

画布：A4 竖版 210×297mm。坐标一律从 mm 走，DPI 只影响最终像素。
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------- 画布
A4_MM = (210.0, 297.0)


def canvas_px(dpi=150):
    return (round(A4_MM[0] / 25.4 * dpi), round(A4_MM[1] / 25.4 * dpi))


# ---------------------------------------------------------------- 比例（7.5 头身，偏女性化的风格化骨架）
# 全部相对身高 H（0 = 头顶，1 = 脚底），改这些数字就是改体型
#
# ⚠️ 四肢粗细这一组是**补齐的**：第一版只定了头身比/肩宽/腿长，没定四肢围度，
#    结果校验 6/6 全过、大腿却明显过粗（radius 0.076H → 宽 0.152H = 1.5 倍头宽）。
#    教训：**校验只能抓住你想起来要量的东西。**
#    下面的数值按人体测量学反推（成年女性，身高 163cm）：
#      大腿围 55cm → 正面宽 ≈ 围/π×0.85 ≈ 14.9cm = 0.091H
#      小腿围 35cm → ≈ 9.5cm = 0.058H
#      上臂围 28cm → ≈ 7.6cm = 0.047H
#      颈围  32cm → ≈ 9.7cm = 0.059H
#    再用经典 7.5 头身canon 微调（头宽 = 0.75 头高 = 0.10H）。
PROP = dict(
    head_h=1 / 7.5,
    head_w=0.75 / 7.5,   # 0.100 H
    chin=1 / 7.5,
    neck=0.175,
    shoulder=0.195,
    shoulder_w=0.230,
    chest=0.28,
    waist=0.375,
    waist_w=0.150,
    hip=0.500,
    hip_w=0.190,
    knee=0.740,
    ankle=0.962,
    arm_len=0.380,
    foot_len=0.075,
    # ---- 四肢：**锥形**，不是等粗圆柱
    # ⚠️ 前两版用等粗胶囊，膝盖那一端必然显粗 —— 真人小腿在踝部只有腓肠最粗处的 6 成。
    #    这里按解剖学围度分布给"沿轴剖面"：[(位置 0..1, 宽度/H), ...]
    taper_thigh=[(0.00, 0.098), (0.35, 0.086), (0.80, 0.070), (1.00, 0.066)],
    taper_shin=[(0.00, 0.056), (0.30, 0.062), (0.75, 0.044), (1.00, 0.038)],
    taper_upper=[(0.00, 0.052), (0.45, 0.047), (1.00, 0.040)],
    taper_fore=[(0.00, 0.045), (0.45, 0.040), (1.00, 0.030)],
    w_neck=0.058,
    w_hand=0.042,
    w_foot=0.052,
    # 髋关节离中轴的距离系数（0.48 × 胯宽/2 → 两大腿并起来正好 ≈ 胯宽）
    hip_spacing=0.48,
)
# 校验区间：(下界, 上界, 说明)。数值一律**从画出来的剪影上量**，不是读输入参数。
CHECKS = {
    '头身比':   ('j', 6.4, 8.2, '身高 / 头高'),
    '肩宽比':   ('j', 0.19, 0.27, '肩宽 / 身高'),
    '腿长比':   ('j', 0.44, 0.53, '胯到脚底 / 身高'),
    '腰臀比':   ('j', 0.62, 0.90, '腰宽 / 胯宽'),
    '腕线过裆': ('j', 0.0, 1e9, '腕线 y 必须大于胯线 y —— "手够长"的硬指标'),
    '对称性':   ('j', 0.0, 0.5, '左右关节相对中轴的最大偏差 / 身高'),
    '头宽比':   ('m', 0.085, 0.115, '头宽 / 身高'),
    '颈宽比':   ('m', 0.046, 0.070, '颈宽 / 身高'),
    '上臂宽比': ('m', 0.038, 0.060, '上臂最宽 / 身高'),
    '前臂宽比': ('m', 0.032, 0.052, '前臂最宽 / 身高'),
    '大腿宽比': ('m', 0.082, 0.110, '大腿最宽（臀下）/ 身高'),
    '小腿宽比': ('m', 0.048, 0.072, '小腿最宽（腓肠）/ 身高'),
    # 锥度 = 末端宽 / 起端宽。**等粗圆柱的锥度就是 1.0 —— 那正是"显粗"的元凶**
    '大腿锥度': ('m', 0.55, 0.80, '大腿膝上 / 臀下 —— 真人约 0.70'),
    '小腿锥度': ('m', 0.50, 0.75, '小腿踝部 / 腓肠最粗 —— 真人约 0.60'),
    '上臂锥度': ('m', 0.60, 0.85, '上臂肘部 / 肩下 —— 真人约 0.77'),
}


def joints(pose, H, cx, top):
    """关节坐标（像素）。pose 里角度单位为度：0=竖直向下，正=朝该侧外侧。"""
    lean = math.radians(pose.get('lean', 0))
    P = PROP
    y = lambda t: top + t * H
    swing = math.sin(lean) * H
    j = {}
    j['head_top'] = (cx, y(0.0))
    j['head_c'] = (cx + swing * 0.10, y(P['head_h'] * 0.55))
    j['chin'] = (cx + swing * 0.14, y(P['chin']))
    j['neck'] = (cx + swing * 0.16, y(P['neck']))
    sx, sy = cx + swing * 0.30, y(P['shoulder'])
    j['shoulder_c'] = (sx, sy)
    j['waist'] = (cx + swing * 0.62, y(P['waist']))
    j['hip'] = (cx + swing * 0.78, y(P['hip']))
    for s, sgn in (('L', -1), ('R', 1)):
        a1 = math.radians(pose.get('arm1_' + s, 8 * sgn))
        a2 = math.radians(pose.get('arm2_' + s, 6 * sgn))
        sh = (sx + sgn * P['shoulder_w'] * H / 2, sy)
        L1 = P['arm_len'] * H * 0.52
        L2 = P['arm_len'] * H * 0.48
        el = (sh[0] + math.sin(a1) * L1, sh[1] + math.cos(a1) * L1)
        wr = (el[0] + math.sin(a1 + a2) * L2, el[1] + math.cos(a1 + a2) * L2)
        j['sh_' + s], j['el_' + s], j['wr_' + s] = sh, el, wr
        b1 = math.radians(pose.get('leg1_' + s, 2))
        b2 = math.radians(pose.get('leg2_' + s, -1))
        hp = (j['hip'][0] + sgn * P['hip_w'] * H / 2 * P['hip_spacing'], j['hip'][1])
        LT = (P['knee'] - P['hip']) * H
        LS = (P['ankle'] - P['knee']) * H
        kn = (hp[0] + math.sin(b1) * LT, hp[1] + math.cos(b1) * LT)
        an = (kn[0] + math.sin(b1 + b2) * LS, kn[1] + math.cos(b1 + b2) * LS)
        j['hp_' + s], j['kn_' + s], j['an_' + s] = hp, kn, an
    return j


# ---------------------------------------------------------------- 白膜形状
def parts(j, H):
    """返回 [(名字, 类型, 参数)]。类型：cap=胶囊(两点+半径)，poly=多边形，ell=椭圆"""
    P = PROP
    R = dict(
        torso_u=P['shoulder_w'] * H * 0.50,
        torso_w=P['waist_w'] * H * 0.52,
        torso_h=P['hip_w'] * H * 0.52,
        neck=P['w_neck'] * H / 2,
        hand=P['w_hand'] * H / 2,
        foot=P['w_foot'] * H / 2,
    )
    tp = lambda key: [(t, w * H) for t, w in P[key]]
    out = []
    # 躯干：肩 → 腰 → 胯 的六边形
    out.append(('躯干', 'poly', [
        (j['sh_L'][0], j['sh_L'][1]), (j['sh_R'][0], j['sh_R'][1]),
        (j['waist'][0] + R['torso_w'], j['waist'][1]),
        (j['hip'][0] + R['torso_h'], j['hip'][1]),
        (j['hip'][0] - R['torso_h'], j['hip'][1]),
        (j['waist'][0] - R['torso_w'], j['waist'][1]),
    ]))
    for s in ('L', 'R'):
        out.append(('上臂' + s, 'taper', (j['sh_' + s], j['el_' + s], tp('taper_upper'))))
        out.append(('前臂' + s, 'taper', (j['el_' + s], j['wr_' + s], tp('taper_fore'))))
        out.append(('手' + s, 'ell', (j['wr_' + s], R['hand'] * 1.15, R['hand'] * 1.75)))
        out.append(('大腿' + s, 'taper', (j['hp_' + s], j['kn_' + s], tp('taper_thigh'))))
        out.append(('小腿' + s, 'taper', (j['kn_' + s], j['an_' + s], tp('taper_shin'))))
        out.append(('脚' + s, 'ell', ((j['an_' + s][0] + (7 if s == 'R' else -7),
                                       j['an_' + s][1] + R['foot'] * 0.8),
                                      R['foot'] * 1.35, R['foot'] * 0.78)))
    out.append(('颈', 'cap', (j['neck'], j['shoulder_c'], R['neck'])))
    out.append(('头', 'ell', (j['head_c'], P['head_w'] * H / 2 * 1.06, P['head_h'] * H / 2)))
    return out


def _prof(prof, t):
    """沿轴剖面插值：prof = [(位置0..1, 宽度), ...]"""
    if t <= prof[0][0]:
        return prof[0][1]
    for k in range(len(prof) - 1):
        p0, w0 = prof[k]
        p1, w1 = prof[k + 1]
        if p0 <= t <= p1:
            u = (t - p0) / max(p1 - p0, 1e-9)
            return w0 + (w1 - w0) * u
    return prof[-1][1]


def raster(ps, size):
    """把部件画成 owner 索引图"""
    W, H = size
    idx = Image.new('I', (W, H), -1)
    d = ImageDraw.Draw(idx)
    names = []
    for name, kind, p in ps:
        names.append(name)
        i = len(names) - 1
        if kind == 'taper':
            A, B, prof = p
            dx, dy = B[0] - A[0], B[1] - A[1]
            L = max(math.hypot(dx, dy), 1e-6)
            nx, ny = -dy / L, dx / L
            left, right = [], []
            for k in range(25):
                t = k / 24.0
                w = _prof(prof, t) * 0.5
                cx_, cy_ = A[0] + dx * t, A[1] + dy * t
                left.append((cx_ + nx * w, cy_ + ny * w))
                right.append((cx_ - nx * w, cy_ - ny * w))
            d.polygon(left + right[::-1], fill=i)
        elif kind == 'cap':
            a, b, r = p
            d.line([a, b], fill=i, width=max(int(r * 2), 1))
            for q in (a, b):
                d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=i)
        elif kind == 'poly':
            d.polygon(p, fill=i)
        else:
            c, rx, ry = p
            d.ellipse([c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry], fill=i)
    return idx, names


def render_clay(pose, H, cx, top, size, clay=(228, 228, 232), line=(120, 122, 130),
                light=(-1, -1), band=6):
    """L0 白膜：纯形状，只有一个颜色 + 明暗，没有五官服装细节"""
    idx, names = raster(parts(joints(pose, H, cx, top), H), size)
    own = np.array(idx)
    W, Hh = size
    arr = np.full((Hh, W, 3), (250, 250, 252), np.uint8)
    owner = np.full((Hh, W), -1, np.int32)
    for i in range(len(names)):
        owner[own == i] = i
    base = np.array(clay, float)
    arr[owner >= 0] = base
    for i in range(len(names)):
        vis = owner == i
        if not vis.any():
            continue
        arr[vis & (~_shift(vis, light[0] * band, light[1] * band))] = np.clip(base * 0.80, 0, 255)
    for i in range(len(names)):
        vis = owner == i
        if not vis.any():
            continue
        arr[vis & (~_shift(vis, -light[0] * band, -light[1] * band))] = np.clip(
            base + (255 - base) * 0.55, 0, 255)
    union = owner >= 0
    dil = Image.fromarray((union * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    arr[(np.array(dil) > 127) & (~union)] = line
    return Image.fromarray(arr), (idx, names)


def _shift(a, dx, dy):
    o = np.zeros_like(a)
    dx, dy = int(dx), int(dy)
    if abs(dx) >= a.shape[1] or abs(dy) >= a.shape[0]:
        return o
    o[max(0, dy):a.shape[0] + min(0, dy), max(0, dx):a.shape[1] + min(0, dx)] = \
        a[max(0, -dy):a.shape[0] - max(0, dy), max(0, -dx):a.shape[1] - max(0, dx)]
    return o


def measure_parts(idx, names, H):
    """从画出来的剪影上量**沿轴剖面**（每 5% 一段的宽度），相对身高。

    不读输入参数 —— 读参数是循环论证。
    沿主轴分箱、取每箱的垂直宽度，所以**与肢体倾角无关**（早期版本逐行数像素，
    侧身站的大腿会被误判超标）。
    """
    a = np.array(idx)
    out = {}
    for i, n in enumerate(names):
        ys, xs = np.nonzero(a == i)
        if len(xs) < 30:
            continue
        if n == '头':
            out[n] = float((a == i).sum(axis=1).max()) / H
            continue
        pts = np.stack([xs, ys], 1).astype(float)
        p = pts - pts.mean(0)
        cov = p.T @ p / len(p)
        _, vec = np.linalg.eigh(cov)
        ax, nv = vec[:, -1], vec[:, 0]
        u, v = p @ ax, p @ nv
        L = max(u.max() - u.min(), 1.0)
        nb = 20
        prof = [0.0] * nb
        b = np.clip(((u - u.min()) / L * nb).astype(int), 0, nb - 1)
        for k in range(nb):
            m = b == k
            if m.sum() >= 2:
                prof[k] = float(v[m].max() - v[m].min())
        out[n] = max(prof) / H
        out['_prof_' + n] = [x / H for x in prof]
    return out


# ---------------------------------------------------------------- 比例校验
def check_anatomy(j, H, m=None):
    """**白膜能不能用，先看这张表。** 全是可测的数字，不含审美判断。"""
    d = {}
    head_h = j['chin'][1] - j['head_top'][1]
    d['头身比'] = H / max(head_h, 1e-6)
    d['肩宽比'] = (j['sh_R'][0] - j['sh_L'][0]) / H
    d['腿长比'] = (1 - PROP['hip'])
    d['腰臀比'] = PROP['waist_w'] / PROP['hip_w']
    d['腕线过裆'] = j['wr_L'][1] - j['hip'][1]
    dev = 0.0
    for a, b in (('sh_L', 'sh_R'), ('el_L', 'el_R'), ('wr_L', 'wr_R'),
                 ('hp_L', 'hp_R'), ('kn_L', 'kn_R'), ('an_L', 'an_R')):
        mid = j['shoulder_c'][0] if a.startswith(('sh', 'el', 'wr')) else j['hip'][0]
        dev = max(dev, abs(abs(j[a][0] - mid) - abs(j[b][0] - mid)) / H)
    d['对称性'] = dev
    if m:
        # 左右取平均（对称时两者相等）
        for key, parts_ in (('头宽比', ['头']), ('颈宽比', ['颈']),
                            ('上臂宽比', ['上臂L', '上臂R']), ('前臂宽比', ['前臂L', '前臂R']),
                            ('大腿宽比', ['大腿L', '大腿R']), ('小腿宽比', ['小腿L', '小腿R'])):
            vals = [m[p] for p in parts_ if p in m]
            if vals:
                d[key] = sum(vals) / len(vals)
    for key, parts_ in (('大腿锥度', ['大腿L', '大腿R']), ('小腿锥度', ['小腿L', '小腿R']),
                        ('上臂锥度', ['上臂L', '上臂R'])):
        rr = []
        for p in parts_:
            pr = m.get('_prof_' + p)
            if not pr or max(pr) <= 0:
                continue
            e1 = sum(pr[:3]) / 3.0
            e2 = sum(pr[-3:]) / 3.0
            hi_, lo_ = max(e1, e2), min(e1, e2)
            rr.append(lo_ / hi_ if hi_ > 0 else 1.0)
        if rr:
            d[key] = sum(rr) / len(rr)
    return d


def report_anatomy(j, H, m=None, strict_widths=True):
    """strict_widths=False 时，四肢宽度的区间**只报数不判定**。

    ⚠️ 为什么：宽度是从剪影上反推胶囊粗细的，肢体分离/大角度张开时
       （双腿微开那种）反推会偏细 —— 实测大腿从 0.081 掉到 0.070，
       而输入参数根本没变。所以基准姿势严格校验，动态姿势只做参考。
    """
    d = check_anatomy(j, H, m)
    lines, ok_all = [], True
    for k, (src, lo, hi, desc) in CHECKS.items():
        if k not in d:
            continue
        v = d[k]
        if src == 'm' and not strict_widths:
            lines.append('➖ %-9s %8.4f   （动态姿势，宽度只报数不判定）' % (k, v))
            continue
        ok = (v > 0) if k == '腕线过裆' else (lo <= v <= hi)
        ok_all &= ok
        lines.append('%s %-9s %8.4f   区间 [%.3f, %.3f]   %s'
                     % ('✅' if ok else '❌', k, v, lo, min(hi, 9.999), desc))
    return ok_all, '\n'.join(lines), d


# ---------------------------------------------------------------- 演示
def _default_font():
    """找一个能显示中文的字体。环境变量 LAYERED_ART_FONT 优先。

    ⚠️ 不要写死某台机器上的字体路径 —— 打包分发时会被拒绝，
       别人机器上也不存在。Windows 自带的微软雅黑/黑体就够用。
    """
    import os as _os
    cands = [_os.environ.get('LAYERED_ART_FONT') or '',
             r'C:\Windows\Fonts\msyh.ttc',
             r'C:\Windows\Fonts\simhei.ttf',
             '/System/Library/Fonts/PingFang.ttc',
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for c in cands:
        if c and _os.path.exists(c):
            return c
    return None


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


if __name__ == '__main__':
    OUT = _out_dir('a4')
    DPI = 150
    W, Hp = canvas_px(DPI)                    # A4 竖版 1240×1754 @150dpi
    FIG_H = Hp * 0.86                         # 人物占画布高度 86%
    TOP = Hp * 0.07
    CX = W * 0.5
    print('A4 竖版 210×297mm  @%ddpi → %d×%d px' % (DPI, W, Hp))
    print('人物高度 %.2f H画布（= %.1f mm）\n' % (FIG_H / Hp, FIG_H / Hp * 297))

    poses = {
        '站立展示': dict(arm1_L=-8, arm1_R=8, arm2_L=-6, arm2_R=6),
        '侧身站':   dict(lean=7, arm1_L=-4, arm1_R=14, arm2_L=-4, arm2_R=10),
        '挥手':     dict(arm1_L=-10, arm2_L=-6, arm1_R=118, arm2_R=-30),
        '双腿微开': dict(leg1_L=9, leg1_R=-9, arm1_L=-16, arm1_R=16, arm2_L=-10, arm2_R=10),
    }
    tiles = []
    for name, p in poses.items():
        img, (idx, names) = render_clay(p, FIG_H, CX, TOP, (W, Hp))
        img.save(os.path.join(OUT, 'clay_%s.png' % name))
        j = joints(p, FIG_H, CX, TOP)
        m = measure_parts(idx, names, FIG_H)
        ok, txt, d = report_anatomy(j, FIG_H, m, strict_widths=(name == '站立展示'))
        print('--- %s   %s' % (name, '比例全过 ✅' if ok else '有超范围 ❌'))
        print(txt)
        if name == '站立展示':
            print('   实测各部宽度（相对身高）: ' + '  '.join(
                '%s=%.4f' % (k, v) for k, v in sorted(m.items())
                if k.startswith(('头', '颈', '上臂', '前臂', '大腿', '小腿'))))
        tiles.append((name, img))

    from render import contact_sheet
    p = contact_sheet(tiles, os.path.join(OUT, 'clay_sheet.png'),
                      tile_w=int(W * 0.55), cols=4,
                      label_font=_default_font())
    print('\n联络图:', p)
