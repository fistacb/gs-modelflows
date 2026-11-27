"""Batch-generate 3DGS static_prop v1_prop/ packages from a single multi-object prop sheet image.

Pipeline phases (resumable, --phase split|gen|package|all):
  1. split   : Auto-segment non-overlapping props (black, white, or alpha background) via connected
               components, extract clean neighbor-free RGBA squares, write manifest.json & input_sheet.png.
  2. gen     : Submit each prop to local ComfyUI / TripoSplat API (with optional auto-managed ComfyUI
               and proactive process recycling every N props to avoid low-VRAM aimdo crashes).
  3. package : Convert each raw TripoSplat PLY -> Blender Z-up, clean floaters, scale to target_height_m,
               ground bottom-center at Z=0 via build_static_prop.py, and render an upright Spark 3DGS
               contact sheet (props_3dgs_sheet.png).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPTS_DIR = Path(__file__).resolve().parent
TEMPLATE_JSON = SCRIPTS_DIR.parent / "resources" / "triposplat_api_workflow.template.json"
BUILD_PROP_PY = SCRIPTS_DIR / "build_static_prop.py"

sys.path.insert(0, str(SCRIPTS_DIR))
from ply_io import read_ply, write_ply


def http_json(url: str, data: dict | None = None, timeout: int = 30) -> dict:
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        txt = resp.read().decode("utf-8")
        return json.loads(txt) if txt.strip() else {}


def upload_image(api_url: str, image_path: Path, upload_name: str) -> str:
    boundary = f"----gsbatch-{uuid.uuid4().hex}"
    file_bytes = image_path.read_bytes()
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="image"; filename="{upload_name}"\r\n'.encode("utf-8"),
        b"Content-Type: image/png\r\n\r\n",
        file_bytes,
        b"\r\n",
        f"--{boundary}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{api_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    name = res.get("name", upload_name)
    subfolder = res.get("subfolder", "")
    return f"{subfolder}/{name}" if subfolder else name


def detect_bg_mode(im_rgba: Image.Image) -> str:
    arr = np.asarray(im_rgba)
    if arr[:, :, 3].min() < 200:
        return "alpha"
    border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]], axis=0)
    mean_lum = float(border.mean())
    return "black" if mean_lum < 128.0 else "white"


def split_prop_sheet(
    sheet_path: Path,
    out_root: Path,
    bg_mode: str = "auto",
    min_area: int = 450,
    min_dim: int = 22,
    canvas_size: int = 768,
    default_height_m: float = 1.0,
    default_gaussians: int = 65000,
    row_band_px: int = 110,
) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    im_rgba = Image.open(sheet_path).convert("RGBA")
    arr_rgba = np.asarray(im_rgba)
    arr_rgb = arr_rgba[:, :, :3]

    mode = detect_bg_mode(im_rgba) if bg_mode == "auto" else bg_mode
    if mode == "alpha":
        fg = (arr_rgba[:, :, 3] > 32).astype(np.uint8)
        fg_soft = arr_rgba[:, :, 3] > 10
    elif mode == "black":
        fg = (arr_rgb.max(axis=-1) > 26).astype(np.uint8)
        fg_soft = arr_rgb.max(axis=-1) > 10
    else:  # white
        fg = (arr_rgb.min(axis=-1) < 230).astype(np.uint8)
        fg_soft = arr_rgb.min(axis=-1) < 245

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_o = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_o, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < min_dim or h < min_dim:
            continue
        comps.append((y // max(row_band_px, 1), x, i, x, y, w, h, area))
    comps.sort()
    if not comps:
        raise RuntimeError(f"No prop objects detected in {sheet_path} (bg_mode={mode})")

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    manifest: list[dict] = []
    tiles: list[Image.Image] = []

    for idx, (_, _, cid, x, y, w, h, area) in enumerate(comps, 1):
        name = f"prop_{idx:02d}"
        prop_dir = out_root / name
        in_dir = prop_dir / "01_input"
        in_dir.mkdir(parents=True, exist_ok=True)

        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(arr_rgb.shape[1], x + w + pad), min(arr_rgb.shape[0], y + h + pad)

        sub_lbl = labels[y0:y1, x0:x1]
        sub_rgb = arr_rgb[y0:y1, x0:x1]
        sub_soft = fg_soft[y0:y1, x0:x1]

        own = (sub_lbl == cid).astype(np.uint8)
        other = ((sub_lbl > 0) & (sub_lbl != cid)).astype(np.uint8)
        own_d = cv2.dilate(own, k5, iterations=1)
        alpha_bin = ((own > 0) | ((own_d > 0) & sub_soft & (other == 0))).astype(np.uint8) * 255
        if mode == "alpha":
            alpha = np.where(alpha_bin > 0, arr_rgba[y0:y1, x0:x1, 3], 0).astype(np.uint8)
        else:
            alpha = cv2.GaussianBlur(alpha_bin, (3, 3), 0.7)
            alpha = np.where(own > 0, 255, alpha)

        crop_pil = Image.fromarray(np.dstack([sub_rgb, alpha]), mode="RGBA")
        usable = int(canvas_size * 0.78)
        cw_c, ch_c = crop_pil.size
        sc = usable / max(cw_c, ch_c)
        nw, nh = max(1, int(round(cw_c * sc))), max(1, int(round(ch_c * sc)))
        resized = crop_pil.resize((nw, nh), Image.Resampling.LANCZOS)
        sq = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        sq.paste(resized, ((canvas_size - nw) // 2, (canvas_size - nh) // 2), resized)

        out_png = in_dir / "front_rgba.png"
        sq.save(out_png)

        th = Image.new("RGB", (180, 180), (20, 24, 34))
        sq_th = sq.resize((164, 164), Image.Resampling.LANCZOS)
        th.paste(sq_th, (8, 8), sq_th)
        dr = ImageDraw.Draw(th)
        dr.rectangle([0, 158, 180, 180], fill=(10, 12, 18))
        dr.text((5, 163), f"{name} ({w}x{h})", fill=(210, 245, 220))
        tiles.append(th)

        manifest.append({
            "name": name,
            "category": "prop",
            "target_height_m": default_height_m,
            "num_gaussians": default_gaussians,
            "input_png": str(out_png),
            "crop_wh": [int(w), int(h)],
        })

    cols = min(8, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    in_sheet = Image.new("RGB", (cols * 180, rows * 180), (12, 14, 20))
    for i, th in enumerate(tiles):
        in_sheet.paste(th, ((i % cols) * 180, (i // cols) * 180))
    in_sheet.save(out_root / "input_sheet.png")
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split {len(manifest)} props (bg={mode}) -> {out_root / 'input_sheet.png'}", flush=True)
    return manifest


def start_comfyui(args: argparse.Namespace) -> subprocess.Popen:
    api_parsed = urllib.parse.urlparse(args.api_url)
    host = api_parsed.hostname or "127.0.0.1"
    port = str(api_parsed.port or 8188)
    cmd = [args.comfy_python, args.comfy_main, "--listen", host, "--port", port]
    if args.comfy_base_dir:
        cmd += ["--base-directory", args.comfy_base_dir]
    if args.comfy_extra_paths:
        cmd += ["--extra-model-paths-config", args.comfy_extra_paths]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
        try:
            http_json(f"{args.api_url}/system_stats")
            return proc
        except Exception:
            time.sleep(2)
    proc.terminate()
    raise RuntimeError(f"ComfyUI did not become ready at {args.api_url} within 180s")


def run_triposplat_batch(manifest: list[dict], out_root: Path, args: argparse.Namespace) -> None:
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    api = args.api_url.rstrip("/")
    managed = bool(args.comfy_python and args.comfy_main)
    t_all = time.time()
    proc: subprocess.Popen | None = None
    jobs_in_proc = 0

    try:
        if not managed:
            for _ in range(60):
                try:
                    http_json(f"{api}/system_stats")
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError(f"ComfyUI not reachable at {api}")

        for idx, item in enumerate(manifest, 1):
            name = item["name"]
            raw_dir = out_root / name / "02_triposplat"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_ply = raw_dir / f"raw_s{args.seed}.ply"

            if raw_ply.exists():
                print(f"[{idx:02d}/{len(manifest):02d}] {name:24s} SKIP ({raw_ply.stat().st_size/1024/1024:.1f} MB)", flush=True)
                continue

            if managed and proc is not None and args.recycle_every > 0 and jobs_in_proc >= args.recycle_every:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                proc = None
                jobs_in_proc = 0

            for attempt in range(3):
                if managed and (proc is None or proc.poll() is not None):
                    print("  Starting managed ComfyUI ...", flush=True)
                    proc = start_comfyui(args)
                    jobs_in_proc = 0

                t0 = time.time()
                try:
                    comfy_img = upload_image(api, Path(item["input_png"]), f"batch_{name}.png")
                    graph = json.loads(json.dumps(template))
                    graph["6"]["inputs"]["seed"] = args.seed
                    graph["55"]["inputs"]["num_gaussians"] = int(item.get("num_gaussians", args.num_gaussians))
                    graph["99"]["inputs"]["image"] = comfy_img
                    graph["51"]["inputs"]["filename_prefix"] = f"3d/batch_{name}_s{args.seed}"

                    queued = http_json(f"{api}/prompt", {"prompt": graph})
                    pid = queued["prompt_id"]
                    for _ in range(240):
                        if managed and proc is not None and proc.poll() is not None:
                            raise ConnectionError("Managed ComfyUI exited during generation")
                        hist = http_json(f"{api}/history/{pid}")
                        res = hist.get(pid)
                        if res:
                            st = res.get("status", {})
                            if st.get("status_str") != "success":
                                raise RuntimeError(f"TripoSplat failed on {name}: {st}")
                            outs = []
                            for no in res.get("outputs", {}).values():
                                for vs in no.values():
                                    if isinstance(vs, list):
                                        outs.extend(vs)
                            ply_it = next(x for x in outs if isinstance(x, dict) and str(x.get("filename", "")).endswith(".ply"))
                            qs = urllib.parse.urlencode(ply_it)
                            with urllib.request.urlopen(f"{api}/view?{qs}", timeout=60) as r:
                                raw_ply.write_bytes(r.read())
                            break
                        time.sleep(1.5)
                    else:
                        raise RuntimeError(f"Timeout waiting for {name}")
                    try:
                        http_json(f"{api}/free", {"unload_models": True, "free_memory": True})
                    except Exception:
                        pass
                    jobs_in_proc += 1
                    print(
                        f"[{idx:02d}/{len(manifest):02d}] {name:24s} gen={time.time()-t0:4.1f}s "
                        f"({raw_ply.stat().st_size/1024/1024:.1f} MB) total={(time.time()-t_all)/60:.1f}m",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    print(f"  WARN {name} attempt {attempt+1}: {str(exc)[:180]}", flush=True)
                    if managed and proc is not None and proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()
                    proc = None
                    jobs_in_proc = 0
                    time.sleep(2)
                    if attempt == 2:
                        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


def run_package_batch(manifest: list[dict], out_root: Path, seed: int) -> None:
    t0 = time.time()
    for idx, item in enumerate(manifest, 1):
        name = item["name"]
        raw_ply = out_root / name / "02_triposplat" / f"raw_s{seed}.ply"
        v1_dir = out_root / name / "v1_prop"
        cmd = [
            sys.executable, str(BUILD_PROP_PY),
            "--ply", str(raw_ply),
            "--out-dir", str(v1_dir),
            "--name", name,
            "--category", str(item.get("category", "prop")),
            "--target-height-m", str(item.get("target_height_m", 1.0)),
            "--from-triposplat",
            "--clean",
        ]
        if item.get("yaw_deg"):
            cmd += ["--yaw-deg", str(item["yaw_deg"])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise RuntimeError(f"build_static_prop failed on {name}")
        print(f"[{idx:02d}/{len(manifest):02d}] {name:24s} v1_prop OK ({time.time()-t0:.1f}s)", flush=True)


def blender_to_spark_unit(data: np.ndarray) -> np.ndarray:
    x, y, z = data["x"].copy(), data["y"].copy(), data["z"].copy()
    op = 1.0 / (1.0 + np.exp(-np.clip(data["opacity"], -20.0, 20.0)))
    m = op >= 0.15
    if m.sum() < 50:
        m = np.ones(len(data), dtype=bool)
    lo = np.array([x[m].min(), y[m].min(), z[m].min()], np.float32)
    hi = np.array([x[m].max(), y[m].max(), z[m].max()], np.float32)
    ctr = 0.5 * (lo + hi)
    sc = 1.0 / float(max((hi - lo).max(), 1e-4))
    out = data.copy()
    xn, yn, zn = (x - ctr[0]) * sc, (y - ctr[1]) * sc, (z - ctr[2]) * sc
    out["x"], out["y"], out["z"] = xn, -zn, yn
    ls = math.log(sc)
    for si in range(3):
        out[f"scale_{si}"] = data[f"scale_{si}"] + ls
    s = math.sqrt(0.5)
    w0, x0, y0, z0 = data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]
    out["rot_0"] = s * (w0 + x0)
    out["rot_1"] = s * (x0 - w0)
    out["rot_2"] = s * (y0 + z0)
    out["rot_3"] = s * (z0 - y0)
    return out


def render_contact_sheet(manifest: list[dict], out_root: Path, spark_dir: Path) -> None:
    tag = f"batch_{out_root.name}"
    tmp_dir = spark_dir / "frames" / tag
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seq = []
    for idx, item in enumerate(manifest):
        d, _ = read_ply(out_root / item["name"] / "v1_prop" / "prop.ply")
        ds = blender_to_spark_unit(d)
        rel = f"frames/{tag}/p_{idx:03d}.ply"
        write_ply(spark_dir / rel, ds)
        seq.append({"blenderFrame": idx, "ply": [rel], "png": f"p_{idx:03d}.png"})

    cam = {
        "width": 256, "height": 256, "background": "#141822",
        "target": [0.0, 0.0, 0.0], "distance": 2.35,
        "pitch": 16.0, "fov": 30.0, "front": -24.0, "side": 66.0,
    }
    (tmp_dir / "cam.json").write_text(json.dumps(cam), encoding="utf-8")
    man_rel = f"frames/{tag}/man.json"
    (spark_dir / man_rel).write_text(json.dumps({"static": [], "review": [], "frames": seq}), encoding="utf-8")

    res = subprocess.run(
        ["node", "capture.mjs", tag, f"{tag}_out", f"frames/{tag}/cam.json", man_rel],
        cwd=str(spark_dir), capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError("Spark contact sheet render failed")

    tile = 256
    cols = min(8, len(manifest))
    rows = math.ceil(len(manifest) / cols)
    sheet = Image.new("RGB", (cols * tile, rows * tile), (16, 20, 28))
    for idx, item in enumerate(manifest):
        im = Image.open(spark_dir / f"{tag}_out" / f"p_{idx:03d}.png").convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, tile - 22, tile, tile], fill=(10, 12, 18))
        dr.text((6, tile - 17), f"{item['name']} ({float(item.get('target_height_m', 1.0)):.2f}m)", fill=(220, 245, 230))
        sheet.paste(im, ((idx % cols) * tile, (idx // cols) * tile))

    out_sheet = out_root / "props_3dgs_sheet.png"
    sheet.save(out_sheet)
    print(f"Saved {len(manifest)}-prop 3DGS showcase sheet -> {out_sheet}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-split a prop sheet and generate v1_prop/ 3DGS assets.")
    ap.add_argument("--sheet", type=Path, help="Input multi-prop sheet image (required for split/all).")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root directory for the prop pack.")
    ap.add_argument("--phase", choices=("split", "gen", "package", "all"), default="all")
    ap.add_argument("--bg", choices=("auto", "black", "white", "alpha"), default="auto")
    ap.add_argument("--min-area", type=int, default=450)
    ap.add_argument("--min-dim", type=int, default=22)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--num-gaussians", type=int, default=65000)
    ap.add_argument("--default-height-m", type=float, default=1.0)
    ap.add_argument("--api-url", default="http://127.0.0.1:8188")
    ap.add_argument("--template", type=Path, default=TEMPLATE_JSON)
    ap.add_argument("--comfy-python", default=None, help="Optional Python exe to auto-manage ComfyUI lifecycle.")
    ap.add_argument("--comfy-main", default=None, help="Optional ComfyUI/main.py path for auto-managed mode.")
    ap.add_argument("--comfy-base-dir", default=None)
    ap.add_argument("--comfy-extra-paths", default=None)
    ap.add_argument("--recycle-every", type=int, default=2, help="Recycle managed ComfyUI every N props (0=off).")
    ap.add_argument("--spark-dir", type=Path, default=None, help="Optional spark_demo dir to render props_3dgs_sheet.png.")
    args = ap.parse_args()

    manifest_path = args.out_root / "manifest.json"
    if args.phase in ("split", "all") and not manifest_path.exists():
        if not args.sheet:
            ap.error("--sheet is required when manifest.json does not yet exist.")
        manifest = split_prop_sheet(
            args.sheet, args.out_root, bg_mode=args.bg, min_area=args.min_area,
            min_dim=args.min_dim, default_height_m=args.default_height_m,
            default_gaussians=args.num_gaussians,
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.phase in ("gen", "all"):
        run_triposplat_batch(manifest, args.out_root, args)
    if args.phase in ("package", "all"):
        run_package_batch(manifest, args.out_root, args.seed)
        if args.spark_dir:
            render_contact_sheet(manifest, args.out_root, args.spark_dir)


if __name__ == "__main__":
    main()
