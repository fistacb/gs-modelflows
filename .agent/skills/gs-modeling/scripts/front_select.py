"""Point selection by what the design actually shows, plus edge-aware colour denoise.

Use these instead of ad-hoc colour/coordinate masks when recolouring parts to match the design:
- front_visibility(): per-splat contribution to an orthographic FRONT view (camera at -Y), computed with
  front-to-back alpha compositing. Points hidden behind other surfaces or seen only from the side get ~0.
- design_lookup(): design pixel colour/alpha under each splat (silhouette-bbox mapping of model x,z).
- denoise_colours(): edge-aware kNN colour smoothing (graininess), keeps boundaries between parts.

CLI (inspection):  python front_select.py --ply X.ply --design 01_input/front_rgba.png --out qa/x_front
  writes <out>_vis.png (front visibility map) and <out>.json (summary).

Model frame: Z-up, -Y forward, image left = model -x.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

C0 = 0.28209479


def _arr(d, k):
    return np.asarray(d[k], np.float64)


def colours(d):
    return np.clip(np.stack([_arr(d, f'f_dc_{i}') for i in range(3)], 1) * C0 + 0.5, 0, 1)


def set_colours(d, rgb):
    for i in range(3):
        d[f'f_dc_{i}'] = ((rgb[:, i] - 0.5) / C0).astype(d[f'f_dc_{i}'].dtype)


def opacity(d):
    return 1 / (1 + np.exp(-_arr(d, 'opacity')))


def _bbox(d, op):
    x, z = _arr(d, 'x'), _arr(d, 'z'); s = op > 0.5
    return np.percentile(x[s], [0.05, 99.95]), np.percentile(z[s], [0.05, 99.95])


def front_visibility(d, res: int = 640, max_r: int = 3) -> np.ndarray:
    """Per-splat weight in [0,1]: max over its footprint pixels of alpha * transmittance in front of it."""
    x, y, z = _arr(d, 'x'), _arr(d, 'y'), _arr(d, 'z'); op = opacity(d)
    (x0, x1), (z0, z1) = _bbox(d, op)
    span = max(x1 - x0, z1 - z0) * 1.04; cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    px = span / res
    u = (x - cx) / px + res / 2; v = (cz - z) / px + res / 2
    sig = np.exp(np.max(np.stack([_arr(d, f'scale_{i}') for i in range(3)], 1), 1)) / px
    sig = np.clip(sig, 0.35, max_r)
    rr = np.ceil(np.minimum(2 * sig, max_r)).astype(int)
    idx, pix, a = [], [], []
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            m = (np.abs(dx) <= rr) & (np.abs(dy) <= rr)
            pu, pv = ui[m] + dx, vi[m] + dy
            ok = (pu >= 0) & (pu < res) & (pv >= 0) & (pv < res)
            k = np.where(m)[0][ok]
            d2 = (pu[ok] - u[k]) ** 2 + (pv[ok] - v[k]) ** 2
            al = op[k] * np.exp(-0.5 * d2 / sig[k] ** 2)
            keep = al > 0.01
            idx.append(k[keep]); pix.append((pv[ok] * res + pu[ok])[keep]); a.append(al[keep])
    idx, pix, a = np.concatenate(idx), np.concatenate(pix), np.clip(np.concatenate(a), 0, 0.99)
    order = np.lexsort((y[idx], pix))            # per pixel, front (small y) first
    idx, pix, a = idx[order], pix[order], a[order]
    la = np.log1p(-a)
    cs = np.cumsum(la)
    start = np.r_[0, np.where(np.diff(pix) != 0)[0] + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(pix)]))
    before = cs - la - np.where(first > 0, cs[first - 1], 0.0)   # sum of log(1-a) of splats in front
    w = a * np.exp(before)
    vis = np.zeros(len(x)); np.maximum.at(vis, idx, w)
    return vis


def design_lookup(d, path: str):
    """Design rgb (N,3), alpha (N,) under each splat, by mapping model silhouette bbox onto design alpha bbox."""
    im = np.asarray(Image.open(path).convert('RGBA')).astype(np.float64) / 255
    ys, xs = np.where(im[..., 3] > 0.5)
    x, z = _arr(d, 'x'), _arr(d, 'z'); (x0, x1), (z0, z1) = _bbox(d, opacity(d))
    u = xs.min() + (x - x0) / (x1 - x0) * (xs.max() - xs.min())
    v = ys.max() - (z - z0) / (z1 - z0) * (ys.max() - ys.min())
    ui = np.clip(np.round(u).astype(int), 0, im.shape[1] - 1); vi = np.clip(np.round(v).astype(int), 0, im.shape[0] - 1)
    return im[vi, ui, :3], im[vi, ui, 3]


def denoise_colours(d, mask=None, k: int = 16, sigma_c: float = 0.06, strength: float = 0.6, keep_detail: float = 0.0):
    """Edge-aware smoothing: neighbours weighted by spatial rank and colour similarity (exp(-dc^2/2sigma_c^2)).
    Boundaries between differently coloured parts are kept because dissimilar neighbours get ~0 weight."""
    P = np.stack([_arr(d, 'x'), _arr(d, 'y'), _arr(d, 'z')], 1); rgb = colours(d); op = opacity(d)
    sel = np.arange(len(P)) if mask is None else np.where(mask)[0]
    _, nn = cKDTree(P).query(P[sel], k=k)
    dc = np.linalg.norm(rgb[nn] - rgb[sel][:, None], axis=2)
    w = np.exp(-0.5 * (dc / sigma_c) ** 2) * op[nn]
    sm = (w[..., None] * rgb[nn]).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None]
    new = rgb.copy()
    new[sel] = rgb[sel] * (1 - strength) + sm * strength
    if keep_detail > 0:
        new[sel] += keep_detail * (rgb[sel] - sm)
    set_colours(d, np.clip(new, 0, 1))
    return float(np.mean(np.linalg.norm(new[sel] - rgb[sel], axis=1)))


def colour_noise(d, mask=None, k: int = 12) -> float:
    """Median local colour deviation from neighbour median (grain measure, lower = smoother)."""
    P = np.stack([_arr(d, 'x'), _arr(d, 'y'), _arr(d, 'z')], 1); rgb = colours(d)
    sel = np.arange(len(P)) if mask is None else np.where(mask)[0]
    _, nn = cKDTree(P).query(P[sel], k=k)
    return float(np.median(np.linalg.norm(rgb[sel] - np.median(rgb[nn], 1), axis=1)))


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ply_io import read_ply
    ap = argparse.ArgumentParser(); ap.add_argument('--ply', required=True); ap.add_argument('--design')
    ap.add_argument('--out', required=True); ap.add_argument('--res', type=int, default=640)
    a = ap.parse_args()
    d, _ = read_ply(a.ply); vis = front_visibility(d, a.res)
    rep = {'n': len(vis), 'front_visible_gt_0.2': int((vis > 0.2).sum()), 'colour_noise': colour_noise(d)}
    # visibility map rendered as point image (visible points white)
    x, z = _arr(d, 'x'), _arr(d, 'z'); (x0, x1), (z0, z1) = _bbox(d, opacity(d))
    W = a.res; img = np.zeros((W, W, 3))
    s = max(x1 - x0, z1 - z0) * 1.04
    ui = np.clip(((x - (x0 + x1) / 2) / s * W + W / 2).astype(int), 0, W - 1)
    vi = np.clip((((z0 + z1) / 2 - z) / s * W + W / 2).astype(int), 0, W - 1)
    rgb = colours(d); o = np.argsort(vis)
    img[vi[o], ui[o]] = rgb[o] * (0.25 + 0.75 * vis[o, None])
    Image.fromarray((img * 255).astype(np.uint8)).save(a.out + '_vis.png')
    json.dump(rep, open(a.out + '.json', 'w'), indent=1); print(json.dumps(rep))


if __name__ == '__main__':
    main()
