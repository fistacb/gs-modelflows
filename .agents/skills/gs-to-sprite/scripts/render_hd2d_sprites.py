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
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def default_spark_dir() -> Path:
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
    else:
        # Static PLY: stage into spark_demo/frames/<job>/static.ply and render single frame per direction
        n_frames = 1
        staged = job_frames_dir / "static.ply"
        shutil.copy2(args.ply, staged)
        # Render via anim mode with 0 bones? Use capture.mjs draw mode per direction/bg via minimal camera.json
        for dname, yaw in directions:
            for bg_name, bg_hex in (("blk", "#000000"), ("wht", "#ffffff")):
                cam = dict(base_cam, background=bg_hex, front=yaw)
                cam_rel = f"frames/{args.job_name}/cam_{dname}_{bg_name}.json"
                man_rel = f"frames/{args.job_name}/man_{dname}_{bg_name}.json"
                (spark_dir / cam_rel).write_text(json.dumps(cam), encoding="utf-8")
                (spark_dir / man_rel).write_text(json.dumps({
                    "static": [],
                    "review": [],
                    "frames": [{"blenderFrame": 0, "ply": [f"frames/{args.job_name}/static.ply"], "png": f"{dname}-00-{bg_name}.png"}],
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
