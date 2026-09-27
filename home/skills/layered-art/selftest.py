"""layered-art 自检。

自己造测试图（不依赖任何外部素材），把整条链路跑一遍：
    提取 → 重建验收 → 分层渲染 → 换配色 → 对称化 → 部件拼贴 → 单区域编辑 → 存读
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                       # noqa: E402
from PIL import Image, ImageDraw         # noqa: E402

import render as R                       # noqa: E402
import shapes as ST                      # noqa: E402

PASS, FAIL = [], []


def check(name, cond, extra=''):
    (PASS if cond else FAIL).append(name)
    print('  %s %-46s %s' % ('✅' if cond else '❌', name, extra))


def make_test_image(path, w=480, h=640):
    """造一张"平面色块 + 内孔"的测试图：脸(带内孔) + 头发 + 衣服 + 一只眼"""
    im = Image.new('RGB', (w, h), (250, 250, 252))
    d = ImageDraw.Draw(im)
    d.ellipse([80, 380, 400, 700], fill=(60, 72, 110))              # 衣服
    d.ellipse([130, 90, 350, 420], fill=(236, 214, 206))            # 脸
    d.polygon([(120, 200), (150, 70), (360, 70), (370, 210), (320, 150),
               (250, 200), (180, 150)], fill=(198, 204, 216))       # 头发
    d.ellipse([170, 250, 218, 300], fill=(252, 252, 254))           # 眼白
    d.ellipse([262, 250, 310, 300], fill=(252, 252, 254))
    d.ellipse([182, 262, 206, 296], fill=(104, 132, 164))           # 虹膜
    d.ellipse([274, 262, 298, 296], fill=(104, 132, 164))
    d.polygon([(160, 244), (228, 240), (226, 232), (162, 234)], fill=(44, 40, 52))
    d.polygon([(252, 240), (320, 244), (318, 234), (254, 232)], fill=(44, 40, 52))
    d.polygon([(236, 330), (244, 344), (230, 344)], fill=(70, 60, 70))   # 鼻
    d.polygon([(214, 366), (266, 366), (256, 376), (224, 376)], fill=(150, 96, 100))  # 嘴
    d.polygon([(200, 430), (280, 430), (250, 500), (230, 500)], fill=(232, 232, 238))  # 领口
    d.ellipse([40, 40, 120, 120], outline=(120, 140, 180), width=6)  # 背景花纹（细线，测内孔）
    im.save(path)
    return path


def main():
    D = os.path.join(os.path.dirname(HERE), '.tmp', 'layered-art-selftest')
    shutil.rmtree(D, ignore_errors=True)
    os.makedirs(D, exist_ok=True)
    ref = make_test_image(os.path.join(D, 'test.png'))
    print('自检目录: %s\n' % D)

    print('[1] 结构提取与重建验收')
    sp = ST.extract(ref, k=10, eps=2.0)
    st = ST.stats(sp)
    v = ST.verify(sp, ref)
    check('提取出区域', st['regions'] > 5, '%d 块' % st['regions'])
    check('有多边形顶点', st['vertices'] > 60, '%d 个' % st['vertices'])
    check('检出内孔', st['holes'] >= 1, '%d 个' % st['holes'])
    check('重建吻合率 ≥ 85%', v['match_pct'] >= 85.0,
          '误差 %.2f/255  吻合 %.1f%%' % (v['mean_err'], v['match_pct']))

    print('\n[2] 存 / 读 往返')
    pj = os.path.join(D, 'spec.json')
    ST.save(sp, pj)
    sp2 = ST.load(pj)
    check('存读后区域数不变', len(sp2['regions']) == len(sp['regions']))
    check('存读后颜色类型正确', isinstance(sp2['regions'][0]['color'], tuple))
    check('存读后坐标类型正确', isinstance(sp2['regions'][0]['pts'][0], tuple))

    print('\n[3] 分层渲染')
    d = R.render_layers(sp, light=(-1, -1), band=6, outline=2)
    names = list(d['layers'])
    check('四层都在', len(names) == 4, '、'.join(names))
    check('成品尺寸正确', d['final'].size == (sp['w'], sp['h']))
    a0 = np.array(d['layers']['L0 底色']).astype(int)
    a2 = np.array(d['layers']['L2 高光']).astype(int)
    check('阴影/高光真的改了像素', np.abs(a0 - a2).mean() > 0.5,
          '平均差 %.2f' % np.abs(a0 - a2).mean())
    d2 = R.render_layers(sp, light=(1, -1), band=6, outline=2)
    check('换光源方向结果不同',
          np.abs(np.array(d['final']).astype(int) - np.array(d2['final']).astype(int)).mean() > 0.5)

    print('\n[4] 换配色（结构与配色分离）')
    base = np.array(ST.to_image(sp)).astype(int)
    diffs = []
    for name, stops in R.PRESETS.items():
        if name == '原色':
            continue
        r = R.recolor(sp, stops)
        check('配色「%s」区域数不变' % name, len(r['regions']) == len(sp['regions']))
        img = np.array(ST.to_image(r)).astype(int)
        diffs.append(np.abs(base - img).mean())
    check('配色方案之间确有差异', min(diffs) > 5.0, '最小差异 %.1f/255' % min(diffs))
    rc = R.recolor(sp, R.PRESETS['樱花粉'])
    check('换色不动几何',
          all(rc['regions'][i]['pts'] == sp['regions'][i]['pts']
              for i in range(len(sp['regions']))))

    print('\n[5] 对称')
    ax, err = ST.detect_axis(sp)
    check('检测到对称轴', 0 < ax < sp['w'], 'x=%d 镜像差 %.2f' % (ax, err))
    sym = ST.mirror_half(sp, axis=ax, side='left')
    check('对称化后仍是合法结构', len(sym['regions']) > 3 and sym['w'] == sp['w'],
          '%d 块' % len(sym['regions']))
    a = np.array(ST.to_image(sym)).astype(int)
    n = min(ax, sym['w'] - ax)
    l = a[:, ax - n:ax]
    rr = a[:, ax:ax + n][:, ::-1]
    check('左右确实对称了', np.abs(l - rr).mean() < 12.0,
          '镜像差 %.2f（对称前 %.2f）' % (np.abs(l - rr).mean(), err))

    print('\n[6] 单区域编辑')
    big = max(range(len(sp['regions'])), key=lambda i: sp['regions'][i]['area'])
    x0, y0, x1, y1 = ST.bbox(sp['regions'][big])
    hit = ST.region_at(sp, (x0 + x1) // 2, (y0 + y1) // 2)
    check('region_at 命中区域', hit >= 0, 'idx=%d' % hit)
    ST.name_region(sp, big, '测试区')
    check('命名生效', ST.find_by_name(sp, '测试区') == [big], 'idx=%d' % big)
    before = list(sp['regions'][big]['pts'])
    ST.transform_region(sp, big, dx=7, dy=3, scale=1.05)
    check('变换改动了顶点', sp['regions'][big]['pts'] != before)
    ST.recolor_region(sp, big, (255, 0, 0))
    check('单区域改色生效', sp['regions'][big]['color'] == (255, 0, 0))

    print('\n[7] 部件拼贴（换部件）')
    before_n = len(sp['regions'])
    pasted = ST.paste_region(sp2, sp2, 40, 40, 220, 220)
    check('拼贴后重新提取出结构', len(pasted['regions']) > 3,
          '%d 块（原 %d）' % (len(pasted['regions']), before_n))

    print('\n[8] 改尺寸')
    r2 = ST.resample(ST.load(pj), 240, 320)
    check('缩放到 240x320', (r2['w'], r2['h']) == (240, 320))
    check('坐标已缩放', max(p[0] for p in r2['regions'][0]['pts']) <= 240)

    print('\n[9] 命令行入口')
    py = sys.executable
    cli_json = os.path.join(D, 'cli.json')
    cases = [
        ('extract', ['extract', ref, '-o', cli_json, '--k', '10']),
        ('render', ['render', cli_json, '-o', os.path.join(D, 'cli.png'),
                    '--preset', '夜色霓虹', '--outline', '2']),
        ('restyle', ['restyle', cli_json, '-o', os.path.join(D, 'rs')]),
        ('layers', ['layers', cli_json, '-o', os.path.join(D, 'lay.png')]),
        ('verify', ['verify', cli_json, ref]),
        ('info', ['info', cli_json]),
        ('region', ['region', cli_json, '--x', '240', '--y', '200']),
        ('mirror', ['mirror', cli_json, '-o', os.path.join(D, 'sym.json')]),
        ('edit', ['edit', cli_json, '-o', os.path.join(D, 'e.json'),
                  '--region', '0', '--scale', '1.1']),
    ]
    for name, args in cases:
        r = subprocess.run([py, os.path.join(HERE, 'art.py')] + args,
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=600)
        tail = ((r.stdout or '') + (r.stderr or '')).strip().splitlines()
        check('CLI %s' % name, r.returncode == 0, tail[-1][:60] if tail else '')
    check('CLI restyle 生成了联络图',
          os.path.exists(os.path.join(D, 'rs', 'restyle_sheet.png')))

    print('\n' + '=' * 62)
    print('通过 %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    for f in FAIL:
        print('   - ' + f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
