#!/usr/bin/env python3
"""Automated Preflight & QA Validator for 3DGS Rigs (Humanoid & Creature).

Computes objective, numerical PASS / WARN / FAIL diagnostics without requiring GPU rendering:
  1. Geometry sanity: NaN/Inf, floating blobs ratio, oversized splats, ground contact & height.
  2. Skeleton & harmonic skinning: NaN/weight-sum, L/R pair symmetry, ear/wing face invasion, low-density joints.
  3. Pre-vs-Post Surface Protrusion / Recess (signed offset along gaze/normal in mm):
     - Eyeball front cap vs original PLY eye surface (+mm = protruding, -mm = sunken/recessed)
     - Eyeball lateral/vertical shift vs original eye cluster center
     - Eyelid float vs original face & eyeball poke-through during blink
     - Mouth patch float / muzzle burial vs original muzzle surface
     - Residual eye-contour & double-mouth risk splat counts

Usage:
  python validate_rig_qa.py --orig-ply <raw_norm.ply> --skeleton <rig_skeleton.json> [--out-json qa_report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parent.parent.parent / "shared"
for p in (str(SHARED_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from harmonic_dqs import SH_C0, compute_harmonic_skin_weights, _is_legacy_humanoid_skeleton
from creature_face_rig import (
    fit_creature_face,
    carve_creature_sockets,
    build_creature_eyeball,
    build_creature_eyelid_shell,
    build_creature_mouth_patch,
)


def read_ply(path: Path) -> tuple[np.ndarray, list[str]]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii")
            header.append(line)
            if line.strip() == "end_header":
                break
        n = 0
        props: list[tuple[str, str]] = []
        type_map = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
        for ln in header:
            if ln.startswith("element vertex"):
                n = int(ln.split()[2])
            elif ln.startswith("property"):
                parts = ln.split()
                props.append((parts[2], type_map[parts[1]]))
        data = np.fromfile(f, dtype=np.dtype(props), count=n)
    return data, header


class CheckReport:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, key: str, status: str, summary: str, metrics: dict[str, Any] | None = None, hint: str = "") -> None:
        self.items.append({
            "key": key,
            "status": status,  # PASS | WARN | FAIL | INFO
            "summary": summary,
            "metrics": metrics or {},
            "hint": hint,
        })

    def counts(self) -> dict[str, int]:
        c = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
        for it in self.items:
            c[it["status"]] = c.get(it["status"], 0) + 1
        return c

    def print_table(self) -> None:
        for it in self.items:
            tag = f"[{it['status']:4s}]"
            line = f"{tag} {it['key']:32s} : {it['summary']}"
            if it["hint"] and it["status"] in ("WARN", "FAIL"):
                line += f"  -> {it['hint']}"
            print(line)
        c = self.counts()
        print(f"\nSummary: PASS={c['PASS']}, WARN={c['WARN']}, FAIL={c['FAIL']}, INFO={c['INFO']}")


def check_geometry(data: np.ndarray, report: CheckReport) -> float:
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    scales = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-np.clip(data["opacity"], -20, 20)))

    # 1. NaN / Inf
    bad = int((~np.isfinite(xyz)).any(axis=1).sum() + (~np.isfinite(scales)).any(axis=1).sum())
    report.add(
        "geometry.finite",
        "PASS" if bad == 0 else "FAIL",
        f"non-finite splats={bad} / {len(data)}",
        {"non_finite": bad, "total": len(data)},
    )

    m_vis = op > 0.20
    vxyz = xyz[m_vis] if np.any(m_vis) else xyz
    z_min = float(np.percentile(vxyz[:, 2], 0.5))
    z_max = float(np.percentile(vxyz[:, 2], 99.5))
    height = max(z_max - z_min, 1e-4)

    # 2. Ground & normalized height
    g_ok = abs(z_min) <= 0.035 and 0.65 <= height <= 1.50
    report.add(
        "geometry.ground_and_height",
        "PASS" if g_ok else "WARN",
        f"z_min={z_min:+.4f}, height={height:.3f} (expected z_min~0, height~1.0)",
        {"z_min": z_min, "z_max": z_max, "height": height},
        hint="Run normalization / grounding if asset is intended for standard scale",
    )

    # 3. Floating blobs (connected components on subsampled visible points for speed)
    rng = np.random.default_rng(0)
    idx_sub = rng.choice(len(vxyz), size=min(len(vxyz), 18000), replace=False)
    sub = vxyz[idx_sub]
    tree = cKDTree(sub)
    adj = tree.sparse_distance_matrix(tree, max_distance=max(height * 0.035, 0.025), output_type="coo_matrix")
    n_c, lbls = connected_components(adj, directed=False)
    sizes = np.bincount(lbls)
    main_ratio = float(sizes.max()) / max(len(sub), 1)
    float_pct = (1.0 - main_ratio) * 100.0
    st = "PASS" if float_pct <= 2.0 else ("WARN" if float_pct <= 5.0 else "FAIL")
    report.add(
        "geometry.floating_blobs",
        st,
        f"detached={float_pct:.2f}% across {n_c} components (limit <= 2.0%)",
        {"detached_pct": round(float_pct, 3), "components": int(n_c)},
        hint="Remove detached floating splats before rigging",
    )

    # 4. Oversized splats
    max_s = np.exp(np.clip(scales.max(axis=1), -15, 5))
    over_pct = float((max_s > height * 0.15).mean()) * 100.0
    report.add(
        "geometry.oversized_splats",
        "PASS" if over_pct <= 0.5 else "WARN",
        f"oversized(>{height*0.15:.3f})={over_pct:.2f}% (limit <= 0.5%)",
        {"oversized_pct": round(over_pct, 3)},
    )
    return height


def check_skeleton_and_weights(data: np.ndarray, skel: dict[str, Any], report: CheckReport) -> tuple[list[str], np.ndarray]:
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    bones = skel["bones"]
    is_legacy = _is_legacy_humanoid_skeleton(bones)
    skin_bones, W = compute_harmonic_skin_weights(xyz, bones, np.ones(len(xyz), dtype=bool))

    has_nan = bool(np.isnan(W).any() or np.isinf(W).any())
    row_err = float(np.max(np.abs(W.sum(axis=1) - 1.0))) if len(W) else 0.0
    report.add(
        "rig.weights_valid",
        "PASS" if (not has_nan and row_err < 1e-3) else "FAIL",
        f"mode={'legacy_v20' if is_legacy else 'generic_v21'}, skin_bones={len(skin_bones)}, NaN={has_nan}, max_sum_err={row_err:.2e}",
        {"is_legacy": is_legacy, "skin_bones": len(skin_bones), "has_nan": has_nan, "max_sum_err": row_err},
    )

    dom = np.argmax(W, axis=1)
    counts = {b: int((dom == i).sum()) for i, b in enumerate(skin_bones)}

    # Check L/R symmetry pairs
    checked_pairs = set()
    worst_ratio = 1.0
    worst_pair = ""
    pair_metrics = {}
    for b in skin_bones:
        if "_L" in b:
            partner = b.replace("_L", "_R")
            if partner in counts and (b, partner) not in checked_pairs:
                checked_pairs.add((b, partner))
                cL, cR = counts[b], counts[partner]
                ratio = (max(cL, cR) + 1) / (min(cL, cR) + 1)
                pair_metrics[f"{b}/{partner}"] = {"L": cL, "R": cR, "ratio": round(ratio, 2)}
                if ratio > worst_ratio:
                    worst_ratio, worst_pair = ratio, f"{b}({cL}) vs {partner}({cR})"
    if pair_metrics:
        st = "PASS" if worst_ratio <= 2.2 else ("WARN" if worst_ratio <= 3.5 else "FAIL")
        report.add(
            "rig.bilateral_symmetry",
            st,
            f"worst L/R splat ratio={worst_ratio:.2f}x at {worst_pair} (limit <= 2.2x)",
            pair_metrics,
            hint="Check for missing/asymmetric limb or wing (consider bilateral mirroring)",
        )

    # Check head face invasion by ears/wings
    if "head" in bones and "head" in counts:
        hh = np.asarray(bones["head"]["head"], dtype=np.float32)
        ht = np.asarray(bones["head"]["tail"], dtype=np.float32)
        h_ctr = 0.5 * (hh + ht)
        m_front_face = (
            (xyz[:, 1] < h_ctr[1] - 0.02)
            & (np.abs(xyz[:, 0] - h_ctr[0]) < 0.14)
            & (xyz[:, 2] >= min(hh[2], ht[2]) - 0.03)
            & (xyz[:, 2] <= max(hh[2], ht[2]) + 0.05)
        )
        invaders: dict[str, int] = {}
        for i, b in enumerate(skin_bones):
            if b.startswith(("ear_", "wing_")):
                n_inv = int((m_front_face & (dom == i)).sum())
                if n_inv > 0:
                    invaders[b] = n_inv
        tot_inv = sum(invaders.values())
        st = "PASS" if tot_inv <= 20 else ("WARN" if tot_inv <= 80 else "FAIL")
        report.add(
            "rig.head_face_invasion",
            st,
            f"ear/wing splats inside front face={tot_inv} ({invaders}) (limit <= 20)",
            {"total": tot_inv, "by_bone": invaders},
            hint="Tighten ear/wing region_box or increase head influence_radius_scale",
        )

    return skin_bones, W


def check_creature_face_protrusion_and_integrity(
    orig_data: np.ndarray,
    skel: dict[str, Any],
    report: CheckReport,
    recess_override_m: float | None = None,
) -> None:
    """Compare fitted creature face parts (eyeballs, eyelids, mouth) against the ORIGINAL uncarved PLY."""
    bones_copy = {k: dict(v) for k, v in skel["bones"].items()}
    face_meta = fit_creature_face(orig_data, bones_copy)

    # Optional override for synthetic regression testing (e.g. simulate +10mm protrusion or -7mm recess)
    if recess_override_m is not None:
        for side in ("eye_L", "eye_R"):
            em = face_meta[side]
            surf = np.asarray(em["surface_center"], dtype=np.float32)
            g_hat = np.asarray(em["gaze_dir"], dtype=np.float32)
            r_eye = float(em["eyeball_radius"])
            em["eyeball_center"] = (surf - g_hat * (r_eye + recess_override_m)).tolist()

    carved = carve_creature_sockets(orig_data, face_meta)

    orig_xyz = np.column_stack([orig_data["x"], orig_data["y"], orig_data["z"]]).astype(np.float32)
    orig_rgb = np.clip(0.5 + SH_C0 * np.column_stack([orig_data["f_dc_0"], orig_data["f_dc_1"], orig_data["f_dc_2"]]), 0.0, 1.0)
    orig_op = 1.0 / (1.0 + np.exp(-np.clip(orig_data["opacity"], -20, 20)))
    m_orig_vis = orig_op > 0.25
    orig_vis_xyz = orig_xyz[m_orig_vis]
    orig_vis_rgb = orig_rgb[m_orig_vis]
    orig_tree = cKDTree(orig_vis_xyz)

    carved_xyz = np.column_stack([carved["x"], carved["y"], carved["z"]]).astype(np.float32)
    carved_rgb = np.clip(0.5 + SH_C0 * np.column_stack([carved["f_dc_0"], carved["f_dc_1"], carved["f_dc_2"]]), 0.0, 1.0)
    carved_op = 1.0 / (1.0 + np.exp(-np.clip(carved["opacity"], -20, 20)))
    fur_rgb = np.asarray(face_meta["fur_rgb"], dtype=np.float32)

    # 1. Bilateral eye balance
    sL = np.asarray(face_meta["eye_L"]["surface_center"], dtype=np.float32)
    sR = np.asarray(face_meta["eye_R"]["surface_center"], dtype=np.float32)
    d_eye = float(face_meta["d_eye"])
    dz_ratio = abs(float(sL[2] - sR[2])) / max(d_eye, 1e-4)
    rL, rR = float(face_meta["eye_L"]["eyeball_radius"]), float(face_meta["eye_R"]["eyeball_radius"])
    r_ratio = max(rL, rR) / max(min(rL, rR), 1e-4)
    report.add(
        "face.eye_bilateral_balance",
        "PASS" if (dz_ratio <= 0.06 and r_ratio <= 1.35) else "WARN",
        f"dz/D_eye={dz_ratio:.3f} (<=0.06), r_ratio={r_ratio:.2f} (<=1.35), D_eye={d_eye:.4f}",
        {"dz_ratio": round(dz_ratio, 4), "r_ratio": round(r_ratio, 3), "d_eye": round(d_eye, 4)},
    )

    # 2. Per-eye protrusion / recess & lateral shift vs ORIGINAL PLY
    for side in ("eye_L", "eye_R"):
        em = face_meta[side]
        sc = np.asarray(em["surface_center"], dtype=np.float32)
        eb_ctr = np.asarray(em["eyeball_center"], dtype=np.float32)
        r_eye = float(em["eyeball_radius"])
        g_hat = np.asarray(em["gaze_dir"], dtype=np.float32)
        er = np.asarray(em["exclusion_radius"], dtype=np.float32)
        R_align = (
            Rotation.from_euler("z", float(em["cheek_yaw_deg"]), degrees=True).as_matrix()
            @ Rotation.from_euler("x", float(em["cheek_pitch_deg"]), degrees=True).as_matrix()
        ).astype(np.float32)

        # Build eyeball in world space & isolate front iris cap (dot(n, g_hat) > 0.75)
        eb_ply = build_creature_eyeball(em, face_meta["iris_palette"], orig_data.dtype)
        eb_xyz = np.column_stack([eb_ply["x"], eb_ply["y"], eb_ply["z"]]).astype(np.float32)
        eb_nrm = eb_xyz / np.maximum(np.linalg.norm(eb_xyz, axis=1, keepdims=True), 1e-8)
        m_cap = (eb_nrm @ g_hat) > 0.78
        cap_w = eb_ctr[None, :] + eb_xyz[m_cap]

        # Signed offset against ORIGINAL uncarved PLY surface along outward gaze vector g_hat
        _, nn_idx = orig_tree.query(cap_w, k=1)
        nearest_orig = orig_vis_xyz[nn_idx]
        signed_mm = ((cap_w - nearest_orig) @ g_hat) * 1000.0
        p50_mm = float(np.percentile(signed_mm, 50))
        p95_mm = float(np.percentile(signed_mm, 95))

        if p95_mm > 4.5:
            st, hint = ("FAIL" if p95_mm > 7.5 else "WARN"), "Eyeball protrudes past original PLY surface"
        elif p50_mm < -4.0:
            st, hint = ("FAIL" if p50_mm < -7.0 else "WARN"), "Eyeball sits too deep (sunken) behind original PLY"
        else:
            st, hint = "PASS", ""
        report.add(
            f"face.{side}_depth_offset",
            st,
            f"p50={p50_mm:+.2f}mm, p95={p95_mm:+.2f}mm vs orig PLY (range [-4.0, +4.5]mm)",
            {"p50_mm": round(p50_mm, 2), "p95_mm": round(p95_mm, 2)},
            hint=hint,
        )

        # Lateral/vertical shift between iris pole and original dark eye cluster
        d_fur_o = np.linalg.norm(orig_vis_rgb - fur_rgb[None, :], axis=1)
        lum_o = 0.299 * orig_vis_rgb[:, 0] + 0.587 * orig_vis_rgb[:, 1] + 0.114 * orig_vis_rgb[:, 2]
        m_orig_eye = (np.linalg.norm(orig_vis_xyz - sc[None, :], axis=1) < max(r_eye * 2.2, 0.045)) & (d_fur_o > 0.26) & (lum_o < 0.64)
        if int(m_orig_eye.sum()) >= 10:
            orig_eye_ctr = np.median(orig_vis_xyz[m_orig_eye], axis=0)
            iris_pole = eb_ctr + g_hat * r_eye
            loc_diff_mm = ((iris_pole - orig_eye_ctr) @ R_align) * 1000.0
            lat_mm = float(np.hypot(loc_diff_mm[0], loc_diff_mm[2]))
            report.add(
                f"face.{side}_lateral_shift",
                "PASS" if lat_mm <= 5.5 else "WARN",
                f"dX={loc_diff_mm[0]:+.1f}mm, dZ={loc_diff_mm[2]:+.1f}mm (radial={lat_mm:.1f}mm, limit <= 5.5mm)",
                {"dx_mm": round(float(loc_diff_mm[0]), 2), "dz_mm": round(float(loc_diff_mm[2]), 2), "radial_mm": round(lat_mm, 2)},
                hint="Eyeball center drifted laterally/vertically from original painted eye",
            )

        # Residual dark eye contour left on carved body below dz_brow_floor
        loc_c = (carved_xyz - sc[None, :]) @ R_align
        d_fur_c = np.linalg.norm(carved_rgb - fur_rgb[None, :], axis=1)
        lum_c = 0.299 * carved_rgb[:, 0] + 0.587 * carved_rgb[:, 1] + 0.114 * carved_rgb[:, 2]
        dz_bf = float(em["dz_brow_floor"])
        m_res = (
            (carved_op > 0.25)
            & (lum_c < 0.44)
            & (d_fur_c > 0.22)
            & (np.abs(loc_c[:, 0]) < er[0] * 1.8)
            & (np.abs(loc_c[:, 1]) < er[1] * 1.4)
            & (loc_c[:, 2] > -er[2] * 1.2)
            & (loc_c[:, 2] <= dz_bf)
        )
        n_res = int(m_res.sum())
        report.add(
            f"face.{side}_residual_dark",
            "PASS" if n_res <= 15 else "WARN",
            f"leftover dark splats={n_res} below dz_brow_floor (has_brow={em['has_brow']}, limit <= 15)",
            {"residual_dark": n_res, "has_brow": bool(em["has_brow"])},
            hint="Static eye contour remnant may show as double-line when blinking",
        )

        # Eyelid ribbon presence & float check at blink=0.0 and blink=1.0
        lid0 = build_creature_eyelid_shell(em, face_meta["fur_rgb"], face_meta["lash_rgb"], 0.0, orig_data.dtype, body_xyz=carved_xyz, body_rgb=carved_rgb, always_draw_lash=True)
        lid1 = build_creature_eyelid_shell(em, face_meta["fur_rgb"], face_meta["lash_rgb"], 1.0, orig_data.dtype, body_xyz=carved_xyz, body_rgb=carved_rgb, always_draw_lash=True)
        lid_ok = len(lid0) >= 80 and len(lid1) >= 400
        l1_xyz = np.column_stack([lid1["x"], lid1["y"], lid1["z"]]).astype(np.float32) if len(lid1) else np.zeros((0, 3), dtype=np.float32)
        float_mm = 0.0
        if len(l1_xyz):
            _, nn_l = orig_tree.query(l1_xyz, k=1)
            float_mm = float(np.percentile(((l1_xyz - orig_vis_xyz[nn_l]) @ g_hat) * 1000.0, 90))
        st_lid = "FAIL" if not lid_ok else ("PASS" if float_mm <= 9.0 else "WARN")
        report.add(
            f"face.{side}_eyelid_shell",
            st_lid,
            f"n(blink=0)={len(lid0)}, n(blink=1)={len(lid1)}, float_p90={float_mm:+.1f}mm (limit <= 9.0mm)",
            {"n_open": len(lid0), "n_closed": len(lid1), "float_p90_mm": round(float_mm, 2)},
            hint="Eyelid shell missing or floating too far off face surface",
        )

    # 3. Mouth double-mouth risk & muzzle surface clearance
    mc = np.asarray(face_meta["mouth_center"], dtype=np.float32)
    np_pos = np.asarray(face_meta["nose_pos"], dtype=np.float32)
    mw = float(face_meta["mouth_width"])
    d_fur_c = np.linalg.norm(carved_rgb - fur_rgb[None, :], axis=1)
    lum_c = 0.299 * carved_rgb[:, 0] + 0.587 * carved_rgb[:, 1] + 0.114 * carved_rgb[:, 2]
    m_res_mouth = (
        (carved_op > 0.20)
        & (np.abs(carved_xyz[:, 0] - mc[0]) < mw * 0.50)
        & (carved_xyz[:, 1] < mc[1] + 0.025)
        & (carved_xyz[:, 2] > mc[2] - 0.020)
        & (carved_xyz[:, 2] < np_pos[2] - 0.010)
        & (d_fur_c > 0.16)
        & (lum_c < 0.72)
    )
    n_res_m = int(m_res_mouth.sum())
    report.add(
        "face.mouth_double_risk",
        "PASS" if n_res_m <= 5 else "WARN",
        f"residual static mouth splats={n_res_m} after inpaint (limit <= 5)",
        {"residual_mouth_splats": n_res_m},
        hint="Static mouth remnant may appear as a double-mouth when opening",
    )

    mp1 = build_creature_mouth_patch(face_meta, 1.0, orig_data.dtype, body_xyz=carved_xyz, always_draw_closed=True)
    mp_xyz = np.column_stack([mp1["x"], mp1["y"], mp1["z"]]).astype(np.float32)
    _, nn_m = orig_tree.query(mp_xyz, k=1)
    # Outward direction at muzzle is -Y
    m_off_mm = ((mp_xyz - orig_vis_xyz[nn_m])[:, 1] * -1.0) * 1000.0
    mp50 = float(np.percentile(m_off_mm, 50))
    st_m = "PASS" if (1.0 <= mp50 <= 8.0) else "WARN"
    report.add(
        "face.mouth_surface_offset",
        st_m,
        f"p50={mp50:+.2f}mm in front of orig muzzle (range [+1.0, +8.0]mm)",
        {"p50_mm": round(mp50, 2)},
        hint="Mouth patch may be buried inside muzzle or floating too far forward",
    )


def _extract_fg_rgb_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized RGB [0,1] and boolean foreground mask from RGBA or solid-background image."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    if float((alpha < 0.5).mean()) > 0.05:
        mask = alpha > 0.5
    else:
        # Solid background: estimate bg color from 4 corners (8x8 patches)
        h, w = rgb.shape[:2]
        p = max(4, min(h, w) // 24)
        corners = np.concatenate([
            rgb[:p, :p].reshape(-1, 3),
            rgb[:p, -p:].reshape(-1, 3),
            rgb[-p:, :p].reshape(-1, 3),
            rgb[-p:, -p:].reshape(-1, 3),
        ], axis=0)
        bg = np.median(corners, axis=0)
        dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
        mask = dist > 0.08
    return rgb, mask


def _normalize_to_canvas(rgb: np.ndarray, mask: np.ndarray, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Crop foreground bbox and fit centered into a square canvas preserving aspect ratio."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((size, size, 3), dtype=np.float32), np.zeros((size, size), dtype=bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_m = mask[y0:y1, x0:x1]
    ch, cw = crop_m.shape
    pad = int(size * 0.06)
    avail = size - 2 * pad
    scale = min(avail / max(cw, 1), avail / max(ch, 1))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))

    im_rgb = Image.fromarray((np.clip(crop_rgb, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    im_m = Image.fromarray((crop_m.astype(np.uint8) * 255)).resize((nw, nh), Image.NEAREST)

    out_rgb = np.zeros((size, size, 3), dtype=np.float32)
    out_m = np.zeros((size, size), dtype=bool)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    out_rgb[oy : oy + nh, ox : ox + nw] = np.asarray(im_rgb, dtype=np.float32) / 255.0
    out_m[oy : oy + nh, ox : ox + nw] = np.asarray(im_m) > 127
    return out_rgb, out_m


def check_design_consistency(ref_image: Path, front_render: Path, report: CheckReport) -> None:
    """Compare original 2D reference image against front 3DGS render (silhouette IoU, color histogram, symmetry)."""
    ref_rgb, ref_m = _extract_fg_rgb_and_mask(ref_image)
    ren_rgb, ren_m = _extract_fg_rgb_and_mask(front_render)

    c_ref_rgb, c_ref_m = _normalize_to_canvas(ref_rgb, ref_m, size=256)
    c_ren_rgb, c_ren_m = _normalize_to_canvas(ren_rgb, ren_m, size=256)

    # 1. Silhouette IoU & Dice
    inter = int((c_ref_m & c_ren_m).sum())
    union = int((c_ref_m | c_ren_m).sum())
    iou = float(inter) / max(union, 1)
    dice = 2.0 * float(inter) / max(int(c_ref_m.sum()) + int(c_ren_m.sum()), 1)
    st_iou = "PASS" if iou >= 0.70 else ("WARN" if iou >= 0.55 else "FAIL")
    report.add(
        "design.silhouette_iou",
        st_iou,
        f"IoU={iou:.3f}, Dice={dice:.3f} vs {ref_image.name} (PASS >= 0.70)",
        {"iou": round(iou, 4), "dice": round(dice, 4)},
        hint="Front silhouette deviates from reference 2D artwork (check missing wings/ears or yaw angle)",
    )

    # 2. Foreground color histogram correlation & mean RGB drift
    ref_fg = c_ref_rgb[c_ref_m]
    ren_fg = c_ren_rgb[c_ren_m]
    if len(ref_fg) >= 20 and len(ren_fg) >= 20:
        h_ref = np.concatenate([np.histogram(ref_fg[:, ch], bins=16, range=(0.0, 1.0), density=True)[0] for ch in range(3)])
        h_ren = np.concatenate([np.histogram(ren_fg[:, ch], bins=16, range=(0.0, 1.0), density=True)[0] for ch in range(3)])
        corr = float(np.corrcoef(h_ref, h_ren)[0, 1]) if (h_ref.std() > 1e-6 and h_ren.std() > 1e-6) else 0.0
        mean_diff = float(np.linalg.norm(ref_fg.mean(axis=0) - ren_fg.mean(axis=0)))
        st_col = "PASS" if (corr >= 0.65 and mean_diff <= 0.18) else ("WARN" if corr >= 0.45 else "FAIL")
        report.add(
            "design.color_consistency",
            st_col,
            f"hist_corr={corr:.3f} (>=0.65), mean_dRGB={mean_diff:.3f} (<=0.18)",
            {"hist_corr": round(corr, 4), "mean_drgb": round(mean_diff, 4)},
            hint="Rendered color distribution shifted significantly from reference artwork",
        )

    # 3. Front render L/R silhouette symmetry
    mid = c_ren_m.shape[1] // 2
    area_l = int(c_ren_m[:, :mid].sum())
    area_r = int(c_ren_m[:, mid:].sum())
    sym_ratio = (max(area_l, area_r) + 1) / (min(area_l, area_r) + 1)
    report.add(
        "design.front_symmetry",
        "PASS" if sym_ratio <= 1.45 else "WARN",
        f"front L/R silhouette area ratio={sym_ratio:.2f}x (limit <= 1.45x)",
        {"area_left": area_l, "area_right": area_r, "ratio": round(sym_ratio, 3)},
        hint="Front silhouette looks lopsided (check yaw alignment or asymmetric limbs/wings)",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Automated 3DGS Rig & Surface Protrusion/Recess QA Validator")
    ap.add_argument("--orig-ply", required=True, help="Original (uncarved) normalized 3DGS PLY")
    ap.add_argument("--skeleton", default="", help="Rig skeleton JSON (v2.0 humanoid or v2.1 modular/creature)")
    ap.add_argument("--creature-face", action="store_true", help="Run creature face protrusion/recess & integrity checks")
    ap.add_argument("--ref-image", default="", help="Optional original 2D reference image for design consistency checks")
    ap.add_argument("--front-render", default="", help="Optional front 3DGS render image to compare against --ref-image")
    ap.add_argument("--simulate-recess-mm", type=float, default=None, help="Optional synthetic offset in mm for self-testing (-10=protrude +10mm, +7=recess 7mm)")
    ap.add_argument("--out-json", default="", help="Optional path to save machine-readable QA report JSON")
    args = ap.parse_args()

    orig_data, _ = read_ply(Path(args.orig_ply))
    report = CheckReport()
    check_geometry(orig_data, report)

    if args.skeleton:
        skel = json.loads(Path(args.skeleton).read_text(encoding="utf-8"))
        check_skeleton_and_weights(orig_data, skel, report)
        if args.creature_face:
            ov_m = (args.simulate_recess_mm / 1000.0) if args.simulate_recess_mm is not None else None
            check_creature_face_protrusion_and_integrity(orig_data, skel, report, recess_override_m=ov_m)

    if args.ref_image and args.front_render:
        check_design_consistency(Path(args.ref_image), Path(args.front_render), report)

    report.print_table()
    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({"counts": report.counts(), "checks": report.items}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON report -> {out_p}")


if __name__ == "__main__":
    main()
