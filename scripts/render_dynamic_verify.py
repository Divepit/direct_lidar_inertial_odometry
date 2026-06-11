#!/usr/bin/env python3
"""Render a dynamic-removal verification image from a DLIO run's saved outputs.

Produces <out_root>/human_removal_verify.png with two panels:
  * top-down static map (grey) with the removed dynamic points (red) and the
    robot path (blue) -- removed points should trace people in open areas while
    walls stay grey;
  * height profile (range vs z) of the removed points -- humans show as ~0..2 m
    vertical columns.

Usage:
    render_dynamic_verify.py [<out_root>]
      <out_root> contains  maps/{clean_map.pcd,dynamic_points.pcd}  and
      run_stats/run_stats.csv. Defaults to the a2 replay output dir.
"""
import os
import sys
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_OUT = '/home/tutuna/colcon_ws/src/summerschool_2026/outdoor_turcan_dual_lidar/replay_output'

_TYPE_MAP = {('F', 4): '<f4', ('F', 8): '<f8',
             ('U', 1): '<u1', ('U', 2): '<u2', ('U', 4): '<u4', ('U', 8): '<u8',
             ('I', 1): '<i1', ('I', 2): '<i2', ('I', 4): '<i4', ('I', 8): '<i8'}


def read_pcd_xyz(path):
    """Read x,y,z from a binary PCD by parsing its header (format-agnostic)."""
    with open(path, 'rb') as f:
        fields = sizes = types = counts = None
        npoints = 0
        data_fmt = None
        while True:
            raw = f.readline()
            if not raw:
                raise RuntimeError('unexpected EOF in PCD header: ' + path)
            line = raw.decode('ascii', 'replace').strip()
            if line.startswith('#') or not line:
                continue
            tok = line.split()
            key = tok[0].upper()
            if key == 'FIELDS':
                fields = tok[1:]
            elif key == 'SIZE':
                sizes = [int(x) for x in tok[1:]]
            elif key == 'TYPE':
                types = tok[1:]
            elif key == 'COUNT':
                counts = [int(x) for x in tok[1:]]
            elif key == 'POINTS':
                npoints = int(tok[1])
            elif key == 'DATA':
                data_fmt = tok[1].lower()
                break
        if counts is None:
            counts = [1] * len(fields)
        if data_fmt != 'binary':
            raise RuntimeError(f'only binary PCD supported (got {data_fmt}): {path}')
        names, formats = [], []
        for fn, sz, ty, ct in zip(fields, sizes, types, counts):
            np_ty = _TYPE_MAP[(ty.upper(), sz)]
            for c in range(ct):
                names.append(fn if ct == 1 else f'{fn}_{c}')
                formats.append(np_ty)
        dt = np.dtype({'names': names, 'formats': formats})
        body = f.read(npoints * dt.itemsize)
        arr = np.frombuffer(body, dtype=dt, count=npoints)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def subsample(a, n, seed=0):
    if len(a) <= n:
        return a
    idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
    return a[idx]


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    static_p = os.path.join(out_root, 'maps', 'clean_map.pcd')
    dyn_p = os.path.join(out_root, 'maps', 'dynamic_points.pcd')
    csv_p = os.path.join(out_root, 'run_stats', 'run_stats.csv')
    out_png = os.path.join(out_root, 'human_removal_verify.png')

    if not os.path.exists(static_p):
        print('[verify] no static map at', static_p, '- skipping'); return 1
    static = read_pcd_xyz(static_p)
    dyn = read_pcd_xyz(dyn_p) if os.path.exists(dyn_p) else np.empty((0, 3))

    # frame: robot path bbox padded, else static-map percentiles
    if os.path.exists(csv_p):
        rs = np.genfromtxt(csv_p, delimiter=',', names=True, invalid_raise=False)
        path = np.stack([rs['p_x_m'], rs['p_y_m']], axis=1)
        path = path[np.isfinite(path).all(axis=1)]
    else:
        path = np.empty((0, 2))
    if len(path):
        cx, cy = path[:, 0].mean(), path[:, 1].mean()
        half = float(np.percentile(np.abs(path - [cx, cy]), 99)) + 14.0
    else:
        cx, cy = np.median(static[:, 0]), np.median(static[:, 1])
        half = 22.0
    half = float(np.clip(half, 12.0, 45.0))
    xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)

    s = subsample(static[(np.abs(static[:, 0] - cx) < half) & (np.abs(static[:, 1] - cy) < half)], 1_500_000)
    d = dyn[(np.abs(dyn[:, 0] - cx) < half) & (np.abs(dyn[:, 1] - cy) < half)] if len(dyn) else dyn

    fig = plt.figure(figsize=(22, 11))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(s[:, 0], s[:, 1], c='0.82', s=0.2, linewidths=0, rasterized=True)
    if len(d):
        ax1.scatter(d[:, 0], d[:, 1], c='red', s=2.5, linewidths=0, rasterized=True,
                    label=f'removed dynamic ({len(dyn):,})')
    if len(path):
        ax1.plot(path[:, 0], path[:, 1], 'b-', lw=1.0, label='robot path')
    ax1.set_aspect('equal'); ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
    ax1.legend(loc='upper left'); ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
    ax1.set_title('Removed dynamic points (red) over static map (grey)\nrobot path blue')

    ax2 = fig.add_subplot(1, 2, 2)
    if len(d):
        r = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        ax2.scatter(r, d[:, 2], s=2, c=d[:, 2], cmap='turbo', linewidths=0)
    ax2.set_ylim(-2.5, 3.5); ax2.grid(alpha=0.3)
    ax2.set_xlabel('range from origin [m]'); ax2.set_ylabel('height z [m]')
    ax2.set_title('Removed-point height profile (humans ~0..2 m vertical columns)')

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print('[verify] wrote', out_png, f'(static={len(static):,} removed={len(dyn):,})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
