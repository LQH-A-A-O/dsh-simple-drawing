"""layered-art 命令行入口。

    python art.py extract ref.jpg -o spec.json [--k 12] [--eps 3]
    python art.py info spec.json
    python art.py render spec.json -o out.png [--preset 夜色霓虹] [--light -1,-1] [--outline 2]
    python art.py restyle spec.json -o dir/             # 所有预置配色各出一张 + 联络图
    python art.py layers spec.json -o layers.png        # 四层分解
    python art.py verify spec.json ref.jpg
    python art.py simplify ref.jpg -o sheet.png         # EPS 扫描：块数 vs 误差
    python art.py mirror spec.json -o sym.json [--axis N] [--side left]
    python art.py paste dst.json src.json --box 0,0,200,200 -o new.json
    python art.py region spec.json --x 300 --y 200      # 这一点落在哪个区域
    python art.py edit spec.json -o new.json [--region N] [--dx 0] [--scale 1.1]
                                        [--color 255,0,0] [--name 头发]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import render as R          # noqa: E402
import shapes as ST         # noqa: E402


def parse_rgb(s):
    if isinstance(s, (list, tuple)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(','))


def parse_pt(s):
    return tuple(int(x) for x in str(s).split(','))


def cmd_extract(a):
    sp = ST.extract(a.image, k=a.k, eps=a.eps, min_area=a.min_area)
    ST.save(sp, a.out)
    st = ST.stats(sp)
    print('提取完成 → %s' % a.out)
    print('  %s  区域 %d  顶点 %d  内孔 %d  实际用色 %d'
          % (st['size'], st['regions'], st['vertices'], st['holes'], st['colors']))
    if a.reference is not False:
        v = ST.verify(sp, a.image)
        print('  重建 误差 %.2f/255  吻合率 %.1f%%' % (v['mean_err'], v['match_pct']))
    return 0


def cmd_info(a):
    sp = ST.load(a.spec)
    st = ST.stats(sp)
    print('%s  %s  区域 %d  顶点 %d  内孔 %d'
          % (a.spec, st['size'], st['regions'], st['vertices'], st['holes']))
    print('\n调色板:')
    for i, c in enumerate(sp['pal']):
        n = sum(1 for r in sp['regions'] if r['ci'] == i)
        print('  #%02d  #%02X%02X%02X  %d 块' % (i, c[0], c[1], c[2], n))
    print('\n区域（按面积，前 %d 个）:' % a.top)
    print('%-5s %-10s %-14s %-8s %s' % ('idx', '名字', '颜色', '面积', 'bbox / 顶点/孔'))
    for r in ST.list_regions(sp, a.top):
        print('%-5d %-10s %-14s %-8d %s  %d/%d'
              % (r['idx'], r['name'] or '-', r['hex'], r['area'], r['bbox'],
                 r['pts'], r['holes']))
    return 0


def cmd_render(a):
    sp = ST.load(a.spec)
    if a.preset:
        if a.preset not in R.PRESETS:
            print('没有配色 %r。可用：%s' % (a.preset, '、'.join(R.PRESETS)))
            return 1
        sp = R.recolor(sp, R.PRESETS[a.preset])
    out = R.render_layers(sp, light=parse_pt(a.light), band=a.band,
                          outline=a.outline)['final']
    out.save(a.out)
    print('→ %s  (%dx%d)' % (a.out, out.width, out.height))
    return 0


def cmd_restyle(a):
    sp = ST.load(a.spec)
    os.makedirs(a.out, exist_ok=True)
    tiles = []
    for name, stops in R.PRESETS.items():
        img = R.render_layers(R.recolor(sp, stops), light=parse_pt(a.light),
                              band=a.band, outline=a.outline)['final']
        p = os.path.join(a.out, 'restyle_%s.png' % name)
        img.save(p)
        tiles.append((name, img))
    sheet = R.contact_sheet(tiles, os.path.join(a.out, 'restyle_sheet.png'))
    print('→ %d 套配色，联络图 %s' % (len(tiles), sheet))
    return 0


def cmd_layers(a):
    sp = ST.load(a.spec)
    d = R.render_layers(sp, light=parse_pt(a.light), band=a.band, outline=a.outline)
    tiles = [(k, v) for k, v in d['layers'].items()]
    p = R.contact_sheet(tiles, a.out, cols=len(tiles))
    print('→ %s' % p)
    return 0


def cmd_verify(a):
    sp = ST.load(a.spec)
    v = ST.verify(sp, a.reference)
    print('区域 %d  顶点 %d  误差 %.2f/255  吻合率 %.1f%%'
          % (v['regions'], v['vertices'], v['mean_err'], v['match_pct']))
    return 0 if v['match_pct'] >= a.min_match else 1


def cmd_simplify(a):
    from PIL import Image
    tiles, rows = [], []
    for eps in (1.2, 3, 6, 12, 24):
        sp = ST.extract(a.image, k=a.k, eps=eps, min_area=a.min_area)
        img = ST.to_image(sp)
        v = ST.verify(sp, a.image)
        rows.append((eps, len(sp['regions']),
                     sum(len(r['pts']) for r in sp['regions']), v['mean_err']))
        tiles.append(('EPS %.1f · %d块 · %.2f' % (eps, len(sp['regions']), v['mean_err']), img))
    print('%-8s %8s %10s %10s' % ('EPS', '块数', '顶点数', '重建误差'))
    for eps, n, npts, err in rows:
        print('%-8s %8d %10d %10.2f' % (eps, n, npts, err))
    p = R.contact_sheet(tiles, a.out)
    print('→ %s' % p)
    return 0


def cmd_mirror(a):
    sp = ST.load(a.spec)
    ax = a.axis if a.axis else None
    if ax is None:
        ax, err = ST.detect_axis(sp)
        print('自动检测对称轴 x=%d（左右镜像差 %.2f）' % (ax, err))
    try:
        out = ST.mirror_half(sp, axis=ax, side=a.side, force=a.force)
    except ValueError as e:
        print('❌ %s' % e)
        return 2
    ST.save(out, a.out)
    st = ST.stats(out)
    print('→ %s  区域 %d  顶点 %d' % (a.out, st['regions'], st['vertices']))
    return 0


def cmd_paste(a):
    dst = ST.load(a.dst)
    src = ST.load(a.src)
    x, y, w, h = parse_pt(a.box)
    out = ST.paste_region(src, dst, x, y, w, h)
    ST.save(out, a.out)
    st = ST.stats(out)
    print('→ %s  区域 %d  顶点 %d' % (a.out, st['regions'], st['vertices']))
    return 0


def cmd_region(a):
    sp = ST.load(a.spec)
    i = ST.region_at(sp, a.x, a.y)
    if i < 0:
        print('(%d,%d) 没落在任何区域上（可能是背景或孔）' % (a.x, a.y))
        return 1
    rg = sp['regions'][i]
    print('第 %d 块  #%02X%02X%02X  面积 %d  顶点 %d  孔 %d  名字 %s'
          % (i, rg['color'][0], rg['color'][1], rg['color'][2], rg['area'],
             len(rg['pts']), len(rg['holes']), rg.get('name') or '-'))
    return 0


def cmd_edit(a):
    sp = ST.load(a.spec)
    if a.region is not None:
        if a.name is not None:
            ST.name_region(sp, a.region, a.name)
        if a.color is not None:
            ST.recolor_region(sp, a.region, parse_rgb(a.color))
        if a.dx or a.dy or a.scale != 1.0 or a.rot:
            ST.transform_region(sp, a.region, dx=a.dx, dy=a.dy,
                                scale=a.scale, rot=a.rot)
    if a.resample:
        w, h = parse_pt(a.resample)
        ST.resample(sp, w, h)
    ST.save(sp, a.out)
    print('→ %s' % a.out)
    return 0


def main():
    ap = argparse.ArgumentParser(description='参考图 → 可编辑结构 → 分层重画')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('extract')
    p.add_argument('image'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--k', type=int, default=12); p.add_argument('--eps', type=float, default=3.0)
    p.add_argument('--min-area', type=int, default=25, dest='min_area')
    p.add_argument('--no-verify', action='store_false', dest='reference')
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser('info')
    p.add_argument('spec'); p.add_argument('--top', type=int, default=25)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('render')
    p.add_argument('spec'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--preset'); p.add_argument('--light', default='-1,-1')
    p.add_argument('--band', type=int, default=6); p.add_argument('--outline', type=int, default=0)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser('restyle')
    p.add_argument('spec'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--light', default='-1,-1')
    p.add_argument('--band', type=int, default=6); p.add_argument('--outline', type=int, default=0)
    p.set_defaults(func=cmd_restyle)

    p = sub.add_parser('layers')
    p.add_argument('spec'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--light', default='-1,-1')
    p.add_argument('--band', type=int, default=6); p.add_argument('--outline', type=int, default=2)
    p.set_defaults(func=cmd_layers)

    p = sub.add_parser('verify')
    p.add_argument('spec'); p.add_argument('reference')
    p.add_argument('--min-match', type=float, default=85.0, dest='min_match')
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser('simplify')
    p.add_argument('image'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--k', type=int, default=12); p.add_argument('--min-area', type=int, default=25,
                                                               dest='min_area')
    p.set_defaults(func=cmd_simplify)

    p = sub.add_parser('mirror')
    p.add_argument('spec'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--axis', type=int); p.add_argument('--side', choices=['left', 'right'],
                                                       default='left')
    p.add_argument('--force', action='store_true',
                   help='图不对称也强行镜像（通常得到万花筒，慎用）')
    p.set_defaults(func=cmd_mirror)

    p = sub.add_parser('paste')
    p.add_argument('dst'); p.add_argument('src')
    p.add_argument('--box', required=True, help='x,y,w,h')
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(func=cmd_paste)

    p = sub.add_parser('region')
    p.add_argument('spec'); p.add_argument('--x', type=int, required=True)
    p.add_argument('--y', type=int, required=True)
    p.set_defaults(func=cmd_region)

    p = sub.add_parser('edit')
    p.add_argument('spec'); p.add_argument('-o', '--out', required=True)
    p.add_argument('--region', type=int); p.add_argument('--name')
    p.add_argument('--color'); p.add_argument('--dx', type=int, default=0)
    p.add_argument('--dy', type=int, default=0); p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--rot', type=float, default=0.0)
    p.add_argument('--resample', help='w,h')
    p.set_defaults(func=cmd_edit)

    a = ap.parse_args()
    return a.func(a)


if __name__ == '__main__':
    sys.exit(main())
