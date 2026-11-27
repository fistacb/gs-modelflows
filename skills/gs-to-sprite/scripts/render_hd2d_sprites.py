#!/usr/bin/env python3
"""Generic 3DGS -> 2D / HD-2D Game Sprite Sheet & Transparent GIF Generator.

Supports both:
  1. Rigged GPU-skinned 3DGS animations (`--anim-json path/to/anim.json`)
  2. Static 3DGS `.ply` models (`--ply path/to/model.ply`)

Pipeline:
  - Renders dual-background (#000000 and #ffffff) passes via Spark GPU (`capture.mjs`)
  - Recovers fringe-free RGBA via exact difference matting
  - Crops with a shared tight bounding box so ground contact & scale stay locked
  - Downsamples with premultiplied alpha, binary crisp alpha, tinted 1px outline,
    and highlight-preserving global palette quantization
  - Exports:
      * Game-ready transparent spritesheet PNG (frames x directions)
      * Transparent animated GIFs (primary direction & multi-direction lineup)
      * Optional HD-2D diorama scene preview GIF
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SH_C0 = 0.28209479177387814


def _read_binary_ply(path: Path) -> tuple[bytes, np.ndarray]:
    """Minimal self-contained binary_little_endian PLY reader (preserves exact header)."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        marker = b"end_header\r\n"
        idx = raw.find(marker)
    if idx < 0:
        raise ValueError(f"Invalid PLY header: {path}")
    header_bytes = raw[: idx + len(marker)]
    header_text = header_bytes.decode("ascii", errors="ignore")
    n_verts = 0
    fields = []
    type_map = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4", "short": "<i2", "ushort": "<u2",
    }
    in_vertex = False
    for line in header_text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = (len(parts) >= 3 and parts[1] == "vertex")
            if in_vertex:
                n_verts = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3:
            fields.append((parts[2], type_map.get(parts[1], "<f4")))
    dtype = np.dtype(fields)
    data = np.frombuffer(raw, dtype=dtype, count=n_verts, offset=len(header_bytes)).copy()
    return header_bytes, data


def _write_position_encoded_ply(src_path: Path, dst_path: Path, lo: np.ndarray, span: np.ndarray) -> None:
    """Copy a 3DGS PLY replacing f_dc_0..2 with normalized 3D coordinates (for single-pass Spark depth/normal maps)."""
    header_bytes, data = _read_binary_ply(src_path)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    norm_rgb = np.clip((xyz - lo[None, :]) / np.maximum(span[None, :], 1e-5), 0.0, 1.0) * 0.90 + 0.05
    fdc = (norm_rgb - 0.5) / SH_C0
    data["f_dc_0"] = fdc[:, 0]
    data["f_dc_1"] = fdc[:, 1]
    data["f_dc_2"] = fdc[:, 2]
    # Zero out higher-order SH if present so encoded position RGB is strictly view-independent
    for name in data.dtype.names or ():
        if name.startswith("f_rest_"):
            data[name] = 0.0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(header_bytes + data.tobytes())


def extract_depth_and_normal_maps(
    pos_path: Path,
    rgba_full: Image.Image,
    bbox: tuple[int, int, int, int],
    sprite_alpha_mask: np.ndarray,
    lo: np.ndarray,
    span: np.ndarray,
    center: np.ndarray,
    half_extent: float,
    yaw_deg: float,
    pitch_deg: float,
    canvas_size: int,
    char_height_ratio: float = 0.92,
) -> tuple[Image.Image, Image.Image]:
    """Recover pixel-exact camera-space depth map & smooth view-space normal map aligned 1:1 with the sprite."""
    bx0, by0, bx1, by1 = bbox
    pos_crop = np.asarray(Image.open(pos_path).convert("RGB").crop((bx0, by0, bx1, by1)), dtype=np.float32) / 255.0
    a_crop = np.asarray(rgba_full.crop((bx0, by0, bx1, by1)), dtype=np.float32)[:, :, 3] / 255.0

    # Unpremultiply position RGB using exact alpha from black/white matting
    valid = a_crop > 0.08
    pos_rgb = np.zeros_like(pos_crop)
    pos_rgb[valid] = np.clip(pos_crop[valid] / np.maximum(a_crop[valid, None], 0.08), 0.0, 1.0)
    xyz_px = lo[None, None, :] + np.clip((pos_rgb - 0.05) / 0.90, 0.0, 1.0) * span[None, None, :]

    # Camera unit direction pointing TOWARD camera (exact match to capture.js setupCameraAndRenderTarget)
    az = math.radians(yaw_deg)
    el = math.radians(pitch_deg)
    cam_dir = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)], dtype=np.float32)

    rel = xyz_px - center[None, None, :]
    # Closeness in [-half_extent, +half_extent] -> normalized depth in [0.05, 0.98] (bright = closer to camera)
    closeness = np.sum(rel * cam_dir[None, None, :], axis=-1)
    depth_norm = np.clip(0.5 + 0.46 * (closeness / max(half_extent, 1e-4)), 0.04, 0.98)
    depth_norm = np.where(valid, depth_norm, 0.0)

    # Smooth depth inside valid mask before computing view-space normal gradients (eliminates splat grain!)
    d_im = Image.fromarray((depth_norm * 255.0).astype(np.uint8), mode="L")
    w_im = Image.fromarray((valid.astype(np.float32) * 255.0).astype(np.uint8), mode="L")
    d_blur = np.asarray(d_im.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32) / 255.0
    w_blur = np.asarray(w_im.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32) / 255.0
    d_smooth = np.where(valid, d_blur / np.maximum(w_blur, 0.15), 0.0)

    # View-space normal from depth gradient (OpenGL/Unity/Godot convention: +X right, +Y up, +Z toward camera)
    # Note: image row y increases DOWNWARD, so d(depth)/d(y_up) = -d(depth)/d(row)
    gy, gx = np.gradient(d_smooth)
    strength = max(float(max(bx1 - bx0, by1 - by0)) * 0.085, 14.0)
    nx = -gx * strength
    ny = gy * strength
    nz = np.ones_like(nx, dtype=np.float32)
    nlen = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-6)
    nx /= nlen
    ny /= nlen
    nz /= nlen
    nrm_rgb = np.stack([0.5 + 0.5 * nx, 0.5 + 0.5 * ny, 0.5 + 0.5 * nz], axis=-1)
    nrm_rgb = np.where(valid[:, :, None], nrm_rgb, np.array([0.5, 0.5, 1.0], dtype=np.float32))

    # Downsample into exact same sprite canvas placement as stylize_hd2d_sprite()
    cw, ch = bx1 - bx0, by1 - by0
    target_h = max(16, int(round(canvas_size * char_height_ratio)))
    scale = target_h / max(ch, 1)
    target_w = max(8, int(round(cw * scale)))
    if target_w > canvas_size - 4:
        target_w = canvas_size - 4
        target_h = max(8, int(round(ch * (target_w / max(cw, 1)))))
    ox = (canvas_size - target_w) // 2
    oy = canvas_size - target_h - 3

    # Premultiplied downsample for depth so background 0 doesn't darken rim pixels
    d_premul = np.dstack([d_smooth * valid.astype(np.float32), valid.astype(np.float32)])
    d_small = np.asarray(
        Image.fromarray((np.clip(d_premul[:, :, 0], 0, 1) * 255).astype(np.uint8), mode="L").resize((target_w, target_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    a_small = np.asarray(
        Image.fromarray((valid.astype(np.uint8) * 255), mode="L").resize((target_w, target_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    d_unp = np.where(a_small > 0.05, np.clip(d_small / np.maximum(a_small, 0.05), 0.04, 0.99), 0.0)

    n_small = np.asarray(
        Image.fromarray((np.clip(nrm_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB").resize((target_w, target_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    # Place onto canvas and dilate 1px into outline pixels so outline also receives valid depth/normal!
    c_depth = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    c_nrm = np.full((canvas_size, canvas_size, 3), [0.5, 0.5, 1.0], dtype=np.float32)
    c_has = np.zeros((canvas_size, canvas_size), dtype=bool)
    c_depth[oy : oy + target_h, ox : ox + target_w] = d_unp
    c_nrm[oy : oy + target_h, ox : ox + target_w] = n_small
    c_has[oy : oy + target_h, ox : ox + target_w] = (a_small > 0.15)

    # Fill 1px outline border pixels from 4-neighbors
    need_fill = (sprite_alpha_mask > 0) & (~c_has)
    if need_fill.any():
        pd = np.pad(c_depth, 1, mode="edge")
        pn = np.pad(c_nrm, ((1, 1), (1, 1), (0, 0)), mode="edge")
        ph = np.pad(c_has, 1, mode="constant", constant_values=False)
        acc_d = np.zeros_like(c_depth)
        acc_n = np.zeros_like(c_nrm)
        cnt = np.zeros_like(c_depth)
        for sl_y, sl_x in ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)), (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None))):
            m = ph[sl_y, sl_x]
            acc_d += np.where(m, pd[sl_y, sl_x], 0.0)
            acc_n += np.where(m[:, :, None], pn[sl_y, sl_x], 0.0)
            cnt += m.astype(np.float32)
        ok = need_fill & (cnt > 0)
        c_depth[ok] = acc_d[ok] / cnt[ok]
        c_nrm[ok] = acc_n[ok] / cnt[ok, None]

    # Renormalize normal vectors inside sprite mask
    nv = c_nrm * 2.0 - 1.0
    nlen2 = np.maximum(np.linalg.norm(nv, axis=-1, keepdims=True), 1e-6)
    c_nrm = np.clip(0.5 + 0.5 * (nv / nlen2), 0.0, 1.0)

    fg = sprite_alpha_mask > 0
    d_u8 = np.where(fg, (np.clip(c_depth, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), 0)
    n_u8 = np.where(fg[:, :, None], (c_nrm * 255.0 + 0.5).astype(np.uint8), np.array([128, 128, 255], dtype=np.uint8))
    a_u8 = sprite_alpha_mask.astype(np.uint8)

    depth_im = Image.fromarray(np.dstack([d_u8, d_u8, d_u8, a_u8]), mode="RGBA")
    normal_im = Image.fromarray(np.dstack([n_u8, a_u8]), mode="RGBA")
    return depth_im, normal_im


def build_relight_demo_gif(
    albedo_frames: list[Image.Image],
    depth_frames: list[Image.Image],
    normal_frames: list[Image.Image],
    out_path: Path,
    duration_ms: int = 100,
) -> None:
    """Render a 2D real-time dynamic lantern relighting demo GIF using ONLY albedo + depth + normal maps!"""
    sz = albedo_frames[0].size[0]
    scale = 3 if sz <= 96 else (2 if sz <= 160 else 1)
    n_walk = len(albedo_frames)
    n_total = max(n_walk * 3, 24)
    yy, xx = np.mgrid[0:sz, 0:sz].astype(np.float32)
    u = (xx / max(sz - 1, 1)) * 2.0 - 1.0  # -1..+1 (+X right)
    v = 1.0 - (yy / max(sz - 1, 1)) * 2.0  # -1..+1 (+Y up)

    frames = []
    for fi in range(n_total):
        idx = fi % n_walk
        alb = np.asarray(albedo_frames[idx].convert("RGBA"), dtype=np.float32) / 255.0
        dep = np.asarray(depth_frames[idx].convert("RGBA"), dtype=np.float32)[:, :, 0] / 255.0
        nrm = (np.asarray(normal_frames[idx].convert("RGBA"), dtype=np.float32)[:, :, :3] / 255.0) * 2.0 - 1.0
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        fg = alb[:, :, 3] > 0.5

        # Orbiting warm golden lantern in 3D view-space around the sprite
        ang = (fi / n_total) * 2.0 * math.pi
        lx = 0.85 * math.cos(ang)
        ly = 0.25 + 0.28 * math.sin(ang * 2.0)
        lz = 0.55 + 0.45 * math.sin(ang)  # moves from in front (+1.0) to behind (+0.1)

        # Surface 3D position in normalized view space
        pz = (dep - 0.5) * 0.9
        dx, dy, dz = lx - u, ly - v, lz - pz
        dist2 = dx * dx + dy * dy + dz * dz + 0.18
        inv_d = 1.0 / np.sqrt(dist2)
        L_dir = np.stack([dx * inv_d, dy * inv_d, dz * inv_d], axis=-1)

        ndotl = np.clip(np.sum(nrm * L_dir, axis=-1), 0.0, 1.0)
        atten = 1.15 / (0.45 + dist2 * 1.1)
        # Subtle rim glow when lantern moves behind the character
        rim = (1.0 - np.clip(nrm[:, :, 2], 0.0, 1.0)) ** 2.2 * np.clip(0.8 - lz, 0.0, 1.0) * 1.4

        # Cool moonlight ambient + warm golden lantern point light
        amb = np.array([0.28, 0.34, 0.52], dtype=np.float32)
        light_col = np.array([1.35, 0.92, 0.48], dtype=np.float32)
        lit = alb[:, :, :3] * (amb[None, None, :] + light_col[None, None, :] * (ndotl * atten + rim)[:, :, None])

        # Dark twilight backdrop + glowing lantern orb
        bg = np.full((sz, sz, 3), [0.10, 0.12, 0.18], dtype=np.float32)
        orb_d = np.hypot(u - lx, v - ly)
        orb_glow = np.exp(-(orb_d / 0.16) ** 2)[:, :, None] * np.array([1.0, 0.72, 0.32], dtype=np.float32)[None, None, :]
        orb_core = (orb_d < 0.045)[:, :, None] * np.array([1.0, 0.96, 0.80], dtype=np.float32)[None, None, :]

        # Depth-sort lantern orb: if lantern is behind sprite center (lz < 0.42), draw behind sprite; else in front!
        if lz < 0.42:
            comp = np.where(fg[:, :, None], lit, bg + orb_glow + orb_core)
        else:
            comp = np.where(fg[:, :, None], lit, bg) + orb_glow * 0.45 + orb_core

        im = Image.fromarray((np.clip(comp, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        im_up = im.resize((sz * scale, sz * scale), Image.Resampling.NEAREST)
        frames.append(im_up.convert("P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.NONE))

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def default_spark_dir() -> Path:
    env_dir = os.environ.get("GS_SPARK_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "capture.mjs").exists():
            return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "renderer" / "spark_demo"
        if (cand / "capture.mjs").exists():
            return cand
    return Path("renderer/spark_demo")


def get_font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def parse_directions(spec: str) -> list[tuple[str, float]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, deg = item.split(":", 1)
            out.append((name.strip(), float(deg.strip())))
        else:
            deg = float(item)
            out.append((f"d{int(round(deg))%360:03d}", deg))
    if not out:
        raise ValueError("No valid directions specified.")
    return out


def extract_rgba_from_bw(black_path: Path, white_path: Path) -> Image.Image:
    """Exact difference matting from black (#000000) and white (#ffffff) renders."""
    b = np.asarray(Image.open(black_path).convert("RGB"), dtype=np.float32) / 255.0
    w = np.asarray(Image.open(white_path).convert("RGB"), dtype=np.float32) / 255.0
    diff = np.clip(w - b, 0.0, 1.0)
    alpha = np.clip(1.0 - diff.mean(axis=-1), 0.0, 1.0)
    rgb = np.zeros_like(b)
    mask = alpha > 1e-3
    rgb[mask] = np.clip(b[mask] / alpha[mask, None], 0.0, 1.0)
    alpha[alpha < 0.02] = 0.0
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGBA")


def compute_global_bbox(rgba_list: list[Image.Image], alpha_thresh: int = 32):
    x0, y0, x1, y1 = 10**9, 10**9, -1, -1
    for im in rgba_list:
        a = np.asarray(im)[:, :, 3]
        ys, xs = np.where(a >= alpha_thresh)
        if len(xs) == 0:
            continue
        x0 = min(x0, int(xs.min()))
        y0 = min(y0, int(ys.min()))
        x1 = max(x1, int(xs.max()) + 1)
        y1 = max(y1, int(ys.max()) + 1)
    if x1 <= x0 or y1 <= y0:
        w, h = rgba_list[0].size
        return 0, 0, w, h
    return x0, y0, x1, y1


def stylize_hd2d_sprite(
    rgba_crop: Image.Image,
    canvas_size: int = 160,
    char_height_ratio: float = 0.92,
    alpha_cutoff: int = 122,
    outline: bool = True,
) -> Image.Image:
    """Convert high-res RGBA render into a crisp 2D / HD-2D pixel sprite."""
    rgb = rgba_crop.convert("RGB")
    alpha = rgba_crop.getchannel("A")
    rgb = ImageEnhance.Color(rgb).enhance(1.18)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.10)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.8, percent=120, threshold=3))
    enhanced = Image.merge("RGBA", (*rgb.split(), alpha))

    cw, ch = enhanced.size
    target_h = max(16, int(round(canvas_size * char_height_ratio)))
    scale = target_h / max(ch, 1)
    target_w = max(8, int(round(cw * scale)))
    if target_w > canvas_size - 4:
        target_w = canvas_size - 4
        target_h = max(8, int(round(ch * (target_w / max(cw, 1)))))

    arr = np.asarray(enhanced, dtype=np.float32) / 255.0
    a_f = arr[:, :, 3:4]
    premul = np.concatenate([arr[:, :, :3] * a_f, a_f], axis=-1)
    premul_im = Image.fromarray((np.clip(premul, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGBA")
    small_premul = premul_im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    sp = np.asarray(small_premul, dtype=np.float32) / 255.0
    sa = sp[:, :, 3]
    srgb = np.zeros_like(sp[:, :, :3])
    nz = sa > 1e-3
    srgb[nz] = np.clip(sp[:, :, :3][nz] / sa[nz, None], 0.0, 1.0)

    solid = (sa * 255.0) >= alpha_cutoff
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[:-2, 1:-1].astype(np.int32)
        + padded[2:, 1:-1].astype(np.int32)
        + padded[1:-1, :-2].astype(np.int32)
        + padded[1:-1, 2:].astype(np.int32)
    )
    solid = solid & (neighbors >= 1)

    canvas_rgb = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    canvas_solid = np.zeros((canvas_size, canvas_size), dtype=bool)
    ox = (canvas_size - target_w) // 2
    oy = canvas_size - target_h - 3
    canvas_rgb[oy : oy + target_h, ox : ox + target_w] = srgb
    canvas_solid[oy : oy + target_h, ox : ox + target_w] = solid

    out_rgb = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    out_a = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    out_rgb[canvas_solid] = canvas_rgb[canvas_solid]
    out_a[canvas_solid] = 255

    if outline:
        p_s = np.pad(canvas_solid, 1, mode="constant", constant_values=False)
        p_r = np.pad(canvas_rgb, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0.0)
        up, dn, lt, rt = p_s[:-2, 1:-1], p_s[2:, 1:-1], p_s[1:-1, :-2], p_s[1:-1, 2:]
        border = (~canvas_solid) & (up | dn | lt | rt)
        acc = np.zeros_like(canvas_rgb)
        cnt = np.zeros((canvas_size, canvas_size, 1), dtype=np.float32)
        for m, shifted in (
            (up, p_r[:-2, 1:-1]),
            (dn, p_r[2:, 1:-1]),
            (lt, p_r[1:-1, :-2]),
            (rt, p_r[1:-1, 2:]),
        ):
            acc += np.where(m[:, :, None], shifted, 0.0)
            cnt += m[:, :, None].astype(np.float32)
        avg_col = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0)
        outline_col = np.clip(avg_col * 0.28 + np.array([0.055, 0.045, 0.075], dtype=np.float32), 0.0, 1.0)
        out_rgb[border] = outline_col[border]
        out_a[border] = 255

    rgba_u8 = np.dstack([(np.clip(out_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), out_a])
    return Image.fromarray(rgba_u8, mode="RGBA")


def quantize_sprite_sequence(sprites: list[Image.Image], num_colors: int = 96) -> list[Image.Image]:
    """Build ONE shared palette across all sprites, preserving bright facial/highlight colors."""
    fg_pixels = []
    for sp in sprites:
        arr = np.asarray(sp)
        fg = arr[arr[:, :, 3] > 0][:, :3]
        if len(fg) > 0:
            fg_pixels.append(fg)
    if not fg_pixels:
        return sprites
    all_fg = np.concatenate(fg_pixels, axis=0)
    lum = 0.299 * all_fg[:, 0] + 0.587 * all_fg[:, 1] + 0.114 * all_fg[:, 2]
    bright = all_fg[lum > 140]
    combined = np.concatenate([all_fg, bright, bright, bright], axis=0) if len(bright) > 0 else all_fg
    side = int(math.ceil(math.sqrt(len(combined))))
    pad_n = side * side - len(combined)
    if pad_n > 0:
        combined = np.concatenate([combined, combined[:pad_n]], axis=0)
    sample_im = Image.fromarray(combined.reshape(side, side, 3).astype(np.uint8), mode="RGB")
    pal_img = sample_im.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

    out = []
    for sp in sprites:
        q = sp.convert("RGB").quantize(palette=pal_img, dither=Image.Dither.NONE).convert("RGB")
        out.append(Image.merge("RGBA", (*q.split(), sp.getchannel("A"))))
    return out


def save_transparent_gif(frames_rgba: list[Image.Image], path: Path, duration_ms: int = 95):
    w, h = frames_rgba[0].size
    strip = Image.new("RGB", (w * len(frames_rgba), h), (20, 18, 24))
    for i, fr in enumerate(frames_rgba):
        strip.paste(fr.convert("RGB"), (i * w, 0), fr.getchannel("A"))
    pal_img = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = (pal_img.getpalette() + [0] * 768)[:768]
    pal[255 * 3 : 255 * 3 + 3] = [255, 0, 255]

    gif_frames = []
    for fr in frames_rgba:
        q = fr.convert("RGB").quantize(palette=pal_img, dither=Image.Dither.NONE)
        q_arr = np.asarray(q, dtype=np.uint8).copy()
        a_arr = np.asarray(fr.getchannel("A"), dtype=np.uint8)
        q_arr[a_arr < 128] = 255
        p_im = Image.fromarray(q_arr, mode="P")
        p_im.putpalette(pal)
        gif_frames.append(p_im)

    gif_frames[0].save(
        path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        transparency=255,
        optimize=False,
    )


def build_hd2d_backdrop(width: int, height: int, scroll_px: float = 0.0) -> Image.Image:
    """Procedural warm HD-2D cobblestone + bokeh lantern diorama backdrop."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    horizon = 0.56
    top_col = np.array([22, 26, 44], dtype=np.float32)
    mid_col = np.array([64, 48, 52], dtype=np.float32)
    bot_col = np.array([34, 30, 36], dtype=np.float32)

    t_top = np.clip(y / horizon, 0.0, 1.0)
    bg = top_col * (1.0 - t_top[:, :, None]) + mid_col * t_top[:, :, None]

    ground_mask = y >= horizon
    g_t = np.clip((y - horizon) / (1.0 - horizon), 0.0, 1.0)
    ground_base = np.array([76, 66, 60], dtype=np.float32) * (1.0 - 0.45 * g_t[:, :, None]) + bot_col * (0.45 * g_t[:, :, None])

    persp_z = 1.0 / (0.18 + g_t * 1.6)
    u_world = (x - 0.5) * persp_z * 6.0 - scroll_px * 0.045
    v_world = persp_z * 2.2
    row_id = np.floor(v_world)
    u_shift = u_world + 0.5 * (row_id % 2)
    col_id = np.floor(u_shift)
    fu, fv = u_shift - col_id, v_world - row_id
    grout = (fu < 0.07) | (fu > 0.93) | (fv < 0.09) | (fv > 0.91)
    hval = np.sin(col_id * 12.9898 + row_id * 78.233) * 43758.5453
    stone_var = (hval - np.floor(hval)) * 18.0 - 9.0
    ground = np.where(grout[:, :, None], (ground_base + stone_var[:, :, None]) * 0.62, ground_base + stone_var[:, :, None])

    img = Image.fromarray(np.clip(np.where(ground_mask[:, :, None], ground, bg), 0, 255).astype(np.uint8), mode="RGB")
    bg_layer = img.copy()
    draw = ImageDraw.Draw(bg_layer, "RGBA")
    for bx, bw, bh in [(10, 95, 195), (115, 80, 165), (210, 110, 205), (335, 90, 175), (430, 95, 190)]:
        sx = int((bx + scroll_px * 0.35) % (width + 120)) - 60
        draw.rectangle([sx, int(height * horizon) - bh, sx + bw, int(height * horizon) + 6], fill=(26, 24, 38, 220))
        for wy in range(int(height * horizon) - bh + 20, int(height * horizon) - 25, 32):
            for wx in range(sx + 16, sx + bw - 18, 24):
                draw.rectangle([wx, wy, wx + 12, wy + 18], fill=(255, 198, 108, 170))
    for lx, ly, lr, col in [
        (85, 130, 28, (255, 196, 110, 90)),
        (250, 105, 36, (255, 220, 145, 75)),
        (415, 140, 26, (255, 180, 95, 95)),
        (165, 175, 18, (140, 195, 255, 60)),
    ]:
        sx = int((lx + scroll_px * 0.5) % (width + 80)) - 40
        draw.ellipse([sx - lr, ly - lr, sx + lr, ly + lr], fill=col)

    blurred_top = bg_layer.filter(ImageFilter.GaussianBlur(radius=6.5))
    dof_weight = np.clip(np.abs(y - 0.78) / 0.28, 0.0, 1.0) ** 1.4
    dof_weight = np.where(y < horizon, 1.0, dof_weight)
    blended = np.asarray(img, dtype=np.float32) * (1.0 - dof_weight[:, :, None]) + np.asarray(blurred_top, dtype=np.float32) * dof_weight[:, :, None]
    glow = np.exp(-(((x - 0.78) ** 2) / 0.18 + ((y - 0.25) ** 2) / 0.20))
    blended += glow[:, :, None] * np.array([58.0, 38.0, 16.0], dtype=np.float32)
    vig = 1.0 - 0.42 * (((x - 0.5) * 1.35) ** 2 + ((y - 0.5) * 1.25) ** 2)
    blended *= np.clip(vig, 0.45, 1.0)[:, :, None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 2D / HD-2D game sprites and transparent GIFs from 3DGS.")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--anim-json", type=Path, help="Spark GPU anim.json manifest from gs-modeling.")
    src_group.add_argument("--ply", type=Path, help="Static 3DGS .ply file (Spark coordinate convention).")
    parser.add_argument("--spark-dir", type=Path, default=default_spark_dir(), help="Path to renderer/spark_demo.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for spritesheet and GIFs.")
    parser.add_argument("--size", type=int, default=160, help="Square sprite canvas size in px (e.g. 64, 80, 128, 160, 224).")
    parser.add_argument("--colors", type=int, default=0, help="Unified palette size (0 = auto from --size).")
    parser.add_argument("--directions", type=str, default="front:0,diag:40,side:90,back:180", help="Comma-separated name:yaw_deg list.")
    parser.add_argument("--frame-prefix", type=str, default="loco-", help="Filter anim.json frames by PNG filename prefix.")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of animation frames to sample.")
    parser.add_argument("--reverse-frames", action="store_true", help="Reverse frame playback order.")
    parser.add_argument("--fps", type=float, default=10.5, help="Playback FPS for exported GIFs.")
    parser.add_argument("--fov", type=float, default=11.0, help="Camera vertical FOV (narrow = orthographic-like HD-2D).")
    parser.add_argument("--pitch", type=float, default=13.0, help="Camera elevation pitch in degrees.")
    parser.add_argument("--distance", type=float, default=7.1, help="Camera orbit distance.")
    parser.add_argument("--target", type=str, default="0.0,-0.55,0.0", help="Camera look-at target x,y,z.")
    parser.add_argument("--render-res", type=int, default=512, help="Intermediate Spark GPU render resolution.")
    parser.add_argument("--no-outline", action="store_true", help="Disable 1px tinted dark outline.")
    parser.add_argument("--scene-preview", action="store_true", help="Also generate an HD-2D diorama scene preview GIF.")
    parser.add_argument("--export-maps", action="store_true", help="Also export pixel-aligned depth & view-space normal sprite sheets + 2D relighting demo GIF.")
    parser.add_argument("--job-name", type=str, default="sprite_job", help="Temporary folder stem inside spark_demo.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    spark_dir = args.spark_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    directions = parse_directions(args.directions)
    target = [float(v) for v in args.target.split(",")]
    duration_ms = max(20, int(round(1000.0 / max(args.fps, 1.0))))
    num_colors = args.colors if args.colors > 0 else (128 if args.size >= 200 else 96 if args.size >= 128 else 64)

    base_cam = {
        "width": args.render_res,
        "height": args.render_res,
        "target": target,
        "pitch": args.pitch,
        "fov": args.fov,
        "distance": args.distance,
        "front": 0,
        "side": 90,
    }

    render_dir_name = f"renders_{args.job_name}"
    render_dir = spark_dir / render_dir_name
    job_frames_dir = spark_dir / "frames" / args.job_name
    job_frames_dir.mkdir(parents=True, exist_ok=True)

    pos_lo = pos_span = pos_center = None
    pos_half_extent = 0.5

    if args.anim_json:
        src_anim = json.loads(args.anim_json.read_text(encoding="utf-8"))
        matched = [f for f in src_anim["frames"] if f["png"].startswith(args.frame_prefix)]
        src_frames = matched[: args.num_frames] if matched else src_anim["frames"][: args.num_frames]
        n_frames = len(src_frames)
        anim_frames = []
        for dname, yaw in directions:
            for fi, wf in enumerate(src_frames):
                for bg_name, bg_hex in (("blk", "#000000"), ("wht", "#ffffff")):
                    cam = dict(base_cam, background=bg_hex, front=yaw)
                    anim_frames.append({
                        "png": f"{dname}-{fi:02d}-{bg_name}.png",
                        "camera": cam,
                        "yaw": yaw,
                        "bones": wf["bones"],
                        "extraPly": wf.get("extraPly", []),
                        "extraTransforms": wf.get("extraTransforms"),
                    })
        manifest = {
            "basePly": src_anim["basePly"],
            "skinBin": src_anim["skinBin"],
            "numBones": src_anim["numBones"],
            "defaultCamera": dict(base_cam, background="#000000"),
            "frames": anim_frames,
        }
        manifest_rel = f"frames/{args.job_name}/anim.json"
        (spark_dir / manifest_rel).write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["node", "capture.mjs", "anim", render_dir_name, manifest_rel], cwd=str(spark_dir), check=True)

        if args.export_maps:
            base_ply_abs = spark_dir / src_anim["basePly"]
            _, base_data = _read_binary_ply(base_ply_abs)
            b_xyz = np.column_stack([base_data["x"], base_data["y"], base_data["z"]]).astype(np.float32)
            pos_lo = np.percentile(b_xyz, 0.2, axis=0) - 0.05
            pos_hi = np.percentile(b_xyz, 99.8, axis=0) + 0.05
            pos_span = np.maximum(pos_hi - pos_lo, 1e-3)
            pos_center = 0.5 * (pos_lo + pos_hi)
            # Use horizontal/depth radius (X/Z in Spark coords) so depth uses full 8-bit contrast front-to-back!
            pos_half_extent = float(max(0.5 * max(pos_span[0], pos_span[2]) * 0.85, 0.14))

            pos_base_rel = f"frames/{args.job_name}/body_pos.ply"
            _write_position_encoded_ply(base_ply_abs, spark_dir / pos_base_rel, pos_lo, pos_span)
            extra_pos_map = {}
            pos_frames = []
            for dname, yaw in directions:
                for fi, wf in enumerate(src_frames):
                    pos_extras = []
                    for ep in wf.get("extraPly", []):
                        if ep not in extra_pos_map:
                            ep_dst = f"frames/{args.job_name}/extra_pos_{len(extra_pos_map)}.ply"
                            _write_position_encoded_ply(spark_dir / ep, spark_dir / ep_dst, pos_lo, pos_span)
                            extra_pos_map[ep] = ep_dst
                        pos_extras.append(extra_pos_map[ep])
                    pos_frames.append({
                        "png": f"{dname}-{fi:02d}-pos.png",
                        "camera": dict(base_cam, background="#000000", front=yaw),
                        "yaw": yaw,
                        "bones": wf["bones"],
                        "extraPly": pos_extras,
                        "extraTransforms": wf.get("extraTransforms"),
                    })
            pos_manifest = {
                "basePly": pos_base_rel,
                "skinBin": src_anim["skinBin"],
                "numBones": src_anim["numBones"],
                "defaultCamera": dict(base_cam, background="#000000"),
                "frames": pos_frames,
            }
            pos_manifest_rel = f"frames/{args.job_name}/anim_pos.json"
            (spark_dir / pos_manifest_rel).write_text(json.dumps(pos_manifest), encoding="utf-8")
            subprocess.run(["node", "capture.mjs", "anim", render_dir_name, pos_manifest_rel], cwd=str(spark_dir), check=True)
    else:
        # Static PLY: stage into spark_demo/frames/<job>/static.ply and render single frame per direction
        n_frames = 1
        staged = job_frames_dir / "static.ply"
        shutil.copy2(args.ply, staged)
        if args.export_maps:
            _, st_data = _read_binary_ply(staged)
            s_xyz = np.column_stack([st_data["x"], st_data["y"], st_data["z"]]).astype(np.float32)
            pos_lo = np.percentile(s_xyz, 0.2, axis=0) - 0.05
            pos_hi = np.percentile(s_xyz, 99.8, axis=0) + 0.05
            pos_span = np.maximum(pos_hi - pos_lo, 1e-3)
            pos_center = 0.5 * (pos_lo + pos_hi)
            # Use horizontal/depth radius (X/Z in Spark coords) so depth uses full 8-bit contrast front-to-back!
            pos_half_extent = float(max(0.5 * max(pos_span[0], pos_span[2]) * 0.85, 0.14))
            _write_position_encoded_ply(staged, job_frames_dir / "static_pos.ply", pos_lo, pos_span)
        for dname, yaw in directions:
            passes = [("blk", "#000000", f"frames/{args.job_name}/static.ply"), ("wht", "#ffffff", f"frames/{args.job_name}/static.ply")]
            if args.export_maps:
                passes.append(("pos", "#000000", f"frames/{args.job_name}/static_pos.ply"))
            for bg_name, bg_hex, ply_rel in passes:
                cam = dict(base_cam, background=bg_hex, front=yaw)
                cam_rel = f"frames/{args.job_name}/cam_{dname}_{bg_name}.json"
                man_rel = f"frames/{args.job_name}/man_{dname}_{bg_name}.json"
                (spark_dir / cam_rel).write_text(json.dumps(cam), encoding="utf-8")
                (spark_dir / man_rel).write_text(json.dumps({
                    "static": [],
                    "review": [],
                    "frames": [{"blenderFrame": 0, "ply": [ply_rel], "png": f"{dname}-00-{bg_name}.png"}],
                }), encoding="utf-8")
                subprocess.run(["node", "capture.mjs", "frames", render_dir_name, cam_rel, man_rel], cwd=str(spark_dir), check=True)

    frame_order = list(range(n_frames - 1, -1, -1)) if args.reverse_frames else list(range(n_frames))
    raw_by_dir: dict[str, list[Image.Image]] = {}
    all_raw: list[Image.Image] = []
    for dname, _ in directions:
        lst = []
        for fi in frame_order:
            im = extract_rgba_from_bw(render_dir / f"{dname}-{fi:02d}-blk.png", render_dir / f"{dname}-{fi:02d}-wht.png")
            lst.append(im)
            all_raw.append(im)
        raw_by_dir[dname] = lst

    bx0, by0, bx1, by1 = compute_global_bbox(all_raw)
    pad = 10
    bx0, by0 = max(0, bx0 - pad), max(0, by0 - pad)
    bx1, by1 = min(args.render_res, bx1 + pad), min(args.render_res, by1 + pad)

    unquant = [
        stylize_hd2d_sprite(im.crop((bx0, by0, bx1, by1)), canvas_size=args.size, outline=not args.no_outline)
        for dname, _ in directions
        for im in raw_by_dir[dname]
    ]
    quant = quantize_sprite_sequence(unquant, num_colors=num_colors)
    sprites: dict[str, list[Image.Image]] = {}
    idx = 0
    for dname, _ in directions:
        sprites[dname] = quant[idx : idx + n_frames]
        idx += n_frames

    # 1. Spritesheet PNG
    sheet = Image.new("RGBA", (args.size * n_frames, args.size * len(directions)), (0, 0, 0, 0))
    for r, (dname, _) in enumerate(directions):
        for c, sp in enumerate(sprites[dname]):
            sheet.paste(sp, (c * args.size, r * args.size))
    sheet_path = out_dir / f"spritesheet_{args.size}px.png"
    sheet.save(sheet_path)

    # 1b. Optional pixel-aligned Depth & View-Space Normal maps + 2D dynamic relighting demo GIF
    if args.export_maps and pos_lo is not None:
        depth_sheet = Image.new("RGBA", (args.size * n_frames, args.size * len(directions)), (0, 0, 0, 0))
        normal_sheet = Image.new("RGBA", (args.size * n_frames, args.size * len(directions)), (0, 0, 0, 0))
        depth_by_dir: dict[str, list[Image.Image]] = {}
        normal_by_dir: dict[str, list[Image.Image]] = {}
        for r, (dname, yaw) in enumerate(directions):
            d_list, n_list = [], []
            for c, fi in enumerate(frame_order):
                sp_alpha = np.asarray(sprites[dname][c])[:, :, 3]
                d_im, n_im = extract_depth_and_normal_maps(
                    render_dir / f"{dname}-{fi:02d}-pos.png",
                    raw_by_dir[dname][c],
                    (bx0, by0, bx1, by1),
                    sp_alpha,
                    pos_lo, pos_span, pos_center, pos_half_extent,
                    yaw, args.pitch, args.size,
                )
                depth_sheet.paste(d_im, (c * args.size, r * args.size))
                normal_sheet.paste(n_im, (c * args.size, r * args.size))
                d_list.append(d_im)
                n_list.append(n_im)
            depth_by_dir[dname] = d_list
            normal_by_dir[dname] = n_list

        depth_sheet.save(out_dir / f"spritesheet_{args.size}px_depth.png")
        normal_sheet.save(out_dir / f"spritesheet_{args.size}px_normal.png")
        maps_meta = {
            "albedo": f"spritesheet_{args.size}px.png",
            "depth": f"spritesheet_{args.size}px_depth.png",
            "normal": f"spritesheet_{args.size}px_normal.png",
            "tile_size_px": args.size,
            "num_frames": n_frames,
            "directions": [{"name": dn, "yaw_deg": float(yw)} for dn, yw in directions],
            "depth_convention": "linear_camera_space (255=closest, 1..254=front_to_back, alpha=0 outside)",
            "depth_half_extent_world": round(pos_half_extent, 5),
            "normal_convention": "view_space_opengl_yup (R=+X_right, G=+Y_up, B=+Z_camera)",
            "pitch_deg": args.pitch,
            "fov_deg": args.fov,
        }
        (out_dir / f"maps_meta_{args.size}px.json").write_text(json.dumps(maps_meta, indent=2), encoding="utf-8")
        pdir_m = "diag" if "diag" in sprites else directions[0][0]
        build_relight_demo_gif(
            sprites[pdir_m], depth_by_dir[pdir_m], normal_by_dir[pdir_m],
            out_dir / f"sprite_relight_demo_{args.size}px.gif",
            duration_ms=duration_ms,
        )

    # 2. Transparent GIFs
    primary_dir = "diag" if "diag" in sprites else directions[0][0]
    scale_vis = 2 if args.size <= 224 else 1
    primary_gif = out_dir / f"sprite_{primary_dir}_{args.size}px_transparent.gif"
    save_transparent_gif(
        [sp.resize((args.size * scale_vis, args.size * scale_vis), Image.Resampling.NEAREST) for sp in sprites[primary_dir]],
        primary_gif,
        duration_ms=duration_ms,
    )

    lineup_scale = 2 if args.size <= 160 else 1
    lineup_frames = []
    for fi in range(n_frames):
        row = Image.new("RGBA", (args.size * len(directions) * lineup_scale, args.size * lineup_scale), (0, 0, 0, 0))
        for di, (dname, _) in enumerate(directions):
            row.paste(sprites[dname][fi].resize((args.size * lineup_scale, args.size * lineup_scale), Image.Resampling.NEAREST), (di * args.size * lineup_scale, 0))
        lineup_frames.append(row)
    lineup_gif = out_dir / f"sprite_all_dirs_{args.size}px_transparent.gif"
    save_transparent_gif(lineup_frames, lineup_gif, duration_ms=duration_ms)

    # 3. Optional HD-2D diorama scene preview GIF
    if args.scene_preview:
        scene_w, scene_h = (640, 480) if args.size >= 200 else (540, 360)
        sp_scale = 2 if args.size <= 224 else 1
        scene_frames = []
        for fi in range(n_frames):
            backdrop = build_hd2d_backdrop(scene_w, scene_h, scroll_px=fi * 8.0).convert("RGBA")
            shadow = Image.new("RGBA", (scene_w, scene_h), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(shadow)
            cx, foot_y = scene_w // 2, int(scene_h * 0.91)
            sh_rx, sh_ry = int(58 * (scene_w / 540.0)), int(12 * (scene_h / 360.0))
            sdraw.ellipse([cx - sh_rx, foot_y - sh_ry, cx + sh_rx, foot_y + sh_ry], fill=(12, 8, 16, 140))
            backdrop = Image.alpha_composite(backdrop, shadow.filter(ImageFilter.GaussianBlur(radius=5.0)))
            sp_big = sprites[primary_dir][fi].resize((args.size * sp_scale, args.size * sp_scale), Image.Resampling.NEAREST)
            backdrop.alpha_composite(sp_big, (cx - sp_big.width // 2, foot_y - sp_big.height + int(12 * (scene_h / 360.0))))
            scene_frames.append(backdrop.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG))
        scene_gif = out_dir / f"hd2d_scene_preview_{args.size}px.gif"
        scene_frames[0].save(scene_gif, save_all=True, append_images=scene_frames[1:], duration=duration_ms, loop=0, optimize=False)

    print(f"[gs-to-sprite] Completed in {time.perf_counter() - t0:.2f}s -> {out_dir}")


if __name__ == "__main__":
    main()
