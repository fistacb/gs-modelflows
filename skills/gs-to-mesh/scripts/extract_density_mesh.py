#!/usr/bin/env python3
"""Extract a watertight surface mesh from a 3D Gaussian Splatting (.ply) file.

Evaluates a 3D Gaussian density field on a regular voxel grid, fills interior volume
using 3D morphological closing + 3-axis scanline interior fill + hole filling,
smooths the binary solid, and runs Marching Cubes to produce a clean mesh (.npz / .obj).

Backend selection:
  - If `GS_COMFYUI_DIR` (or `--comfyui-dir`) points to a local ComfyUI installation
    containing `comfy_extras.nodes_gaussian_splat._splat_density`, uses that GPU kernel.
  - Otherwise falls back to a self-contained PyTorch batched Gaussian density splatting
    evaluator on CUDA (or CPU).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import torch


def find_shared_dir() -> Path:
    here = Path(__file__).resolve()
    if (here.parent / "ply_io.py").exists():
        return here.parent
    for parent in here.parents:
        cand = parent / "shared"
        if (cand / "ply_io.py").exists():
            return cand
    raise FileNotFoundError("Could not locate ply_io.py alongside script or in shared/")


sys.path.insert(0, str(find_shared_dir()))
from ply_io import read_ply  # noqa: E402


# Optional coordinate preset: raw TripoSplat -> Blender Z-up
RAW_TO_BLENDER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def fill_volume(surface: np.ndarray, radius: int, scanline_fill: bool = False) -> tuple[np.ndarray, list[int]]:
    structure = ndi.generate_binary_structure(3, 1)
    closed = ndi.binary_closing(surface, structure=structure, iterations=max(1, radius)) if radius > 0 else surface.copy()
    if scanline_fill:
        interior = closed.copy()
        for axis in range(3):
            forward = np.maximum.accumulate(closed, axis=axis)
            backward = np.flip(np.maximum.accumulate(np.flip(closed, axis=axis), axis=axis), axis=axis)
            interior = (forward & backward) if axis == 0 else (interior & forward & backward)
        filled = ndi.binary_fill_holes(closed | interior)
    else:
        # Concave-safe fill: seals enclosed interior cavities without filling concave exterior pockets
        # (e.g. hooded faces, bangs, cape interiors, or outstretched arms).
        filled = ndi.binary_fill_holes(closed)
    labels, _ = ndi.label(filled)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if len(counts) <= 1 or counts.max() == 0:
        raise RuntimeError("Density volume is empty at the chosen --threshold. Try lowering --threshold.")
    return (labels == counts.argmax()), sorted(counts[counts > 0].tolist(), reverse=True)[:16]


def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (N, 4) in wxyz order to rotation matrices (N, 3, 3)."""
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def evaluate_density_fallback(
    positions: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    rotations_wxyz: torch.Tensor,
    grid_res: int,
    padding: int,
    device: torch.device,
    chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Self-contained PyTorch ANISOTROPIC oriented-ellipsoid Gaussian density splatting evaluator.

    Evaluates the true 3DGS Mahalanobis distance delta^T (R diag(1/s^2) R^T) delta so flat surface
    splats stay sleek and flush along the surface instead of ballooning into bumpy spheres!
    """
    import math

    p_min = positions.min(dim=0).values
    p_max = positions.max(dim=0).values
    extent = float((p_max - p_min).max().item())
    usable = max(16, grid_res - 2 * padding)
    voxel = extent / usable
    origin = p_min - padding * voxel
    shape = tuple(int(math.ceil(float((p_max[i] - p_min[i]).item()) / voxel)) + 2 * padding + 1 for i in range(3))

    field = torch.zeros(shape, dtype=torch.float32, device=device)
    # Clamp minimum axis gently to 0.72*voxel so razor-thin surfels are captured cleanly without spherical bloating,
    # and cap runaway oversize splats at 4.5*voxel so stray giant splats never create huge warts.
    scl = torch.clamp(scales, min=0.72 * voxel, max=4.5 * voxel)
    inv_s = 1.0 / scl  # (N, 3)
    R = _quat_wxyz_to_rotmat(rotations_wxyz)  # (N, 3, 3), columns are local principal axes
    # Transform matrix M = diag(1/s) @ R^T -> local_scaled = M @ delta -> d2 = ||local_scaled||^2
    M = R.transpose(1, 2) * inv_s[:, :, None]  # (N, 3, 3)

    rad_vox = torch.clamp((2.2 * scl.max(dim=1).values / voxel).ceil().to(torch.int64), min=1, max=6)
    coords = (positions - origin[None, :]) / voxel

    n = positions.shape[0]
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        c_b = coords[start:end]
        op_b = opacity[start:end]
        M_b = M[start:end]  # (B, 3, 3)
        r_max = int(rad_vox[start:end].max().item())
        offs = torch.arange(-r_max, r_max + 1, device=device)
        ox, oy, oz = torch.meshgrid(offs, offs, offs, indexing="ij")
        dgrid = torch.stack([ox.ravel(), oy.ravel(), oz.ravel()], dim=-1).to(torch.float32)  # (K, 3)

        base = c_b.round().to(torch.int64)  # (B, 3)
        vox_idx = base[:, None, :] + dgrid[None, :, :].to(torch.int64)  # (B, K, 3)
        delta = (vox_idx.to(torch.float32) - c_b[:, None, :]) * voxel  # (B, K, 3)
        # Mahalanobis squared distance: ||M_b @ delta||^2
        u_loc = torch.einsum("bij,bkj->bki", M_b, delta)
        maha2 = (u_loc * u_loc).sum(dim=-1)
        val = op_b[:, None] * torch.exp(-0.5 * maha2)

        valid = (
            (vox_idx[:, :, 0] >= 0) & (vox_idx[:, :, 0] < shape[0])
            & (vox_idx[:, :, 1] >= 0) & (vox_idx[:, :, 1] < shape[1])
            & (vox_idx[:, :, 2] >= 0) & (vox_idx[:, :, 2] < shape[2])
            & (val > 1e-3)
        )
        if valid.any():
            idx_v = vox_idx[valid]
            flat = idx_v[:, 0] * (shape[1] * shape[2]) + idx_v[:, 1] * shape[2] + idx_v[:, 2]
            field.view(-1).index_add_(0, flat, val[valid])

    return field.cpu().numpy(), origin.cpu().numpy().astype(np.float32), float(voxel)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a watertight mesh from a 3DGS .ply density field.")
    parser.add_argument("--ply", type=Path, required=True, help="Input 3DGS .ply file.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path (vertices, faces).")
    parser.add_argument("--out-obj", type=Path, default=None, help="Optional Wavefront .obj output path.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--comfyui-dir", type=Path, default=Path(os.environ["GS_COMFYUI_DIR"]) if os.environ.get("GS_COMFYUI_DIR") else None)
    parser.add_argument("--threshold", type=float, default=0.30, help="Density iso-threshold.")
    parser.add_argument("--closing", type=int, default=2, help="3D 6-connected binary closing iterations.")
    parser.add_argument("--scanline-fill", action="store_true", help="Enable legacy 3-axis scanline interior fill (can fill concave pockets like hoods).")
    parser.add_argument("--grid-res", type=int, default=420, help="Voxel grid resolution along longest axis.")
    parser.add_argument("--padding", type=int, default=6, help="Voxel grid border padding.")
    parser.add_argument("--min-opacity", type=float, default=0.20, help="Drop wispy halo splats below this sigmoid opacity.")
    parser.add_argument("--min-scale", type=float, default=0.0016, help="Minimum surfel scale to close thin gaps.")
    parser.add_argument("--smooth-sigma", type=float, default=0.90, help="Gaussian smoothing sigma before Marching Cubes.")
    parser.add_argument("--coord-preset", choices=("blender", "raw_to_blender"), default="blender", help="Coordinate transform applied to output vertices.")
    args = parser.parse_args()

    splats, _ = read_ply(args.ply)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(splats["opacity"], -20.0, 20.0)))
    kept = opacity >= args.min_opacity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_np = np.column_stack([splats["x"], splats["y"], splats["z"]]).astype(np.float32)[kept]
    scl_np = np.maximum(np.exp(np.column_stack([splats["scale_0"], splats["scale_1"], splats["scale_2"]]).astype(np.float32)[kept]), args.min_scale)
    rot_wxyz_np = np.column_stack([splats["rot_0"], splats["rot_1"], splats["rot_2"], splats["rot_3"]]).astype(np.float32)[kept]
    rot_xyzw_np = np.column_stack([splats["rot_3"], splats["rot_0"], splats["rot_1"], splats["rot_2"]]).astype(np.float32)[kept]
    op_np = opacity.astype(np.float32)[kept]

    positions = torch.as_tensor(pos_np, dtype=torch.float32, device=device)
    scales = torch.as_tensor(scl_np, dtype=torch.float32, device=device)
    rotations_wxyz = torch.as_tensor(rot_wxyz_np, dtype=torch.float32, device=device)
    rotations = torch.as_tensor(rot_xyzw_np, dtype=torch.float32, device=device)
    op_t = torch.as_tensor(op_np, dtype=torch.float32, device=device)

    backend = "pytorch_anisotropic"
    field = None
    origin = None
    voxel = None

    if args.comfyui_dir and (args.comfyui_dir / "comfy_extras" / "nodes_gaussian_splat.py").exists():
        sys.path.insert(0, str(args.comfyui_dir))
        try:
            from comfy_extras.nodes_gaussian_splat import _splat_density

            density, color, weights, origin_val, voxel_val = _splat_density(
                positions, op_t, scales, rotations, torch.zeros_like(positions), args.grid_res, args.padding, device, chunk=1200
            )
            del color, weights
            field = density.cpu().numpy()
            del density
            origin = np.asarray(origin_val, dtype=np.float32)
            voxel = float(voxel_val)
            backend = "comfyui_splat_density"
        except Exception as exc:
            print(f"[gs-to-mesh] ComfyUI _splat_density unavailable ({exc}); using PyTorch anisotropic fallback.", flush=True)

    if field is None:
        field, origin, voxel = evaluate_density_fallback(positions, op_t, scales, rotations_wxyz, args.grid_res, args.padding, device)

    print(f"[gs-to-mesh] density ({backend}): shape={field.shape} voxel={voxel:.6f} max={float(field.max()):.4f}", flush=True)
    solid, components = fill_volume(field > args.threshold, args.closing, scanline_fill=args.scanline_fill)
    soft = ndi.gaussian_filter(solid.astype(np.float32), args.smooth_sigma)
    vertices, triangles, _, _ = marching_cubes(
        soft, 0.50, spacing=(voxel, voxel, voxel), allow_degenerate=False, gradient_direction="ascent"
    )
    vertices = vertices + origin[None, :]
    if args.coord_preset == "raw_to_blender":
        vertices = vertices @ RAW_TO_BLENDER.T

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, vertices=vertices.astype(np.float32), faces=triangles.astype(np.int32))
    if args.out_obj:
        write_obj(args.out_obj, vertices, triangles)

    report = {
        "backend": backend,
        "input_ply": str(args.ply),
        "kept_splats": int(kept.sum()),
        "threshold": args.threshold,
        "closing_steps": args.closing,
        "voxel": voxel,
        "shape": list(field.shape),
        "voxels_before_component_filter": components,
        "vertices": int(len(vertices)),
        "faces": int(len(triangles)),
        "device": str(device),
    }
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    import math

    main()
