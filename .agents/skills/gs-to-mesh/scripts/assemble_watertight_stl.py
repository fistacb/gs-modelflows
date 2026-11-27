#!/usr/bin/env python3
"""Blender CLI script: Voxel-union extracted 3DGS mesh (+ optional extras & base) into a watertight STL/GLB.

Run with:
    blender -b --python skills/gs-to-mesh/scripts/assemble_watertight_stl.py -- \
        --body-npz work/char/body_mesh.npz \
        --out-stl work/char/model_25p4mm.stl \
        --add-base --base-diameter-mm 25.4 \
        --render-qa-dir work/char/qa_mesh
"""
import argparse
import json
import sys
from math import cos, radians, sin
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
import numpy as np


def parse_cli_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Assemble watertight STL/GLB from extracted 3DGS mesh.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--body-npz", type=Path, help="Input .npz with 'vertices' and 'faces'.")
    src.add_argument("--body-obj", type=Path, help="Input Wavefront .obj mesh.")
    parser.add_argument("--extra-mesh", type=Path, action="append", default=[], help="Optional extra .obj/.stl/.npz prop meshes to union.")
    parser.add_argument("--out-stl", type=Path, required=True, help="Output watertight .stl path.")
    parser.add_argument("--out-glb", type=Path, default=None, help="Optional .glb export path.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON validation report path.")
    parser.add_argument("--voxel-size", type=float, default=0.0024, help="Blender VOXEL remesh size in model units.")
    parser.add_argument("--smooth-iters", type=int, default=6, help="Volume-preserving Taubin smooth iterations on body before base union (0 to disable).")
    parser.add_argument("--post-smooth-iters", type=int, default=1, help="Light Taubin smooth iterations after voxel remesh to remove stair-stepping.")
    parser.add_argument("--add-base", action="store_true", help="Attach a chamfered circular miniature base under the feet.")
    parser.add_argument("--base-radius-ratio", type=float, default=0.38, help="Circular base radius relative to model height.")
    parser.add_argument("--base-thickness-ratio", type=float, default=0.065, help="Circular base thickness relative to model height.")
    parser.add_argument("--base-diameter-mm", type=float, default=None, help="Scale final model so circular base diameter equals this value in mm (e.g. 25.4).")
    parser.add_argument("--target-height-mm", type=float, default=None, help="Scale final model so total Z height equals this value in mm (used if --base-diameter-mm is omitted).")
    parser.add_argument("--bottom-clip", type=float, default=0.0013, help="Flat bottom clip margin before Z=0 grounding.")
    parser.add_argument("--render-qa-dir", type=Path, default=None, help="Optional directory to render 6-view clay comparison PNGs.")
    return parser.parse_args(argv)


def mesh_add(name: str, vertices: np.ndarray, faces: np.ndarray):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist() if isinstance(faces, np.ndarray) else faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def load_mesh_object(path: Path, name: str):
    suf = path.suffix.lower()
    if suf == ".npz":
        data = np.load(path)
        return mesh_add(name, data["vertices"], data["faces"])
    if suf == ".obj":
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(path))
        added = list(set(bpy.data.objects) - before)
        if not added:
            raise RuntimeError(f"Failed to import OBJ: {path}")
        return added[0]
    if suf == ".stl":
        before = set(bpy.data.objects)
        bpy.ops.wm.stl_import(filepath=str(path))
        added = list(set(bpy.data.objects) - before)
        if not added:
            raise RuntimeError(f"Failed to import STL: {path}")
        return added[0]
    raise ValueError(f"Unsupported mesh format: {path}")


def add_circular_base(body_obj, radius_ratio: float, thickness_ratio: float):
    coords = np.empty(len(body_obj.data.vertices) * 3, dtype=np.float32)
    body_obj.data.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)
    min_z = float(coords[:, 2].min())
    height = float(coords[:, 2].max() - min_z)
    cx = float(0.5 * (coords[:, 0].min() + coords[:, 0].max()))
    cy = float(0.5 * (coords[:, 1].min() + coords[:, 1].max()))

    radius = max(0.05, height * radius_ratio)
    thickness = max(0.01, height * thickness_ratio)
    # Slight overlap into feet so voxel remesh fuses solidly
    overlap = thickness * 0.25
    bpy.ops.mesh.primitive_cone_add(
        vertices=64,
        radius1=radius,
        radius2=radius * 0.94,
        depth=thickness,
        location=(cx, cy, min_z - thickness * 0.5 + overlap),
    )
    base_obj = bpy.context.active_object
    base_obj.name = "CircularBase"
    return base_obj


def taubin_smooth_mesh(obj, iterations: int = 8, lam: float = 0.5, mu: float = -0.53) -> None:
    """Volume-preserving Taubin smoothing (alternating shrink/inflate Laplacian passes)."""
    if iterations <= 0:
        return
    mesh = obj.data
    n_verts = len(mesh.vertices)
    n_edges = len(mesh.edges)
    if n_verts == 0 or n_edges == 0:
        return
    v = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    e = np.empty(n_edges * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    u0, u1 = e[:, 0], e[:, 1]
    deg = np.zeros(n_verts, dtype=np.float32)
    np.add.at(deg, u0, 1.0)
    np.add.at(deg, u1, 1.0)
    deg = np.maximum(deg, 1.0)[:, None]
    for _ in range(iterations):
        for factor in (lam, mu):
            acc = np.zeros_like(v)
            np.add.at(acc, u0, v[u1])
            np.add.at(acc, u1, v[u0])
            v = v + factor * (acc / deg - v)
    mesh.vertices.foreach_set("co", v.ravel())
    mesh.update()


def cleanup_components(obj) -> tuple[list[int], int, int]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    unseen = set(bm.verts)
    groups = []
    while unseen:
        start = unseen.pop()
        group = [start]
        pending = [start]
        while pending:
            vert = pending.pop()
            for edge in vert.link_edges:
                v = edge.other_vert(vert)
                if v in unseen:
                    unseen.remove(v)
                    group.append(v)
                    pending.append(v)
        groups.append(group)
    groups.sort(key=len, reverse=True)
    counts = [len(g) for g in groups]
    removable = [v for g in groups[1:] if len(g) <= max(100, int(counts[0] * 0.002)) for v in g]
    if removable:
        bmesh.ops.delete(bm, geom=removable, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bad_edges = sum(not e.is_manifold for e in bm.edges)
    remaining_groups = sum(1 for g in groups if len(g) > max(100, int(counts[0] * 0.002))) or 1
    bm.to_mesh(obj.data)
    bm.free()
    return counts, remaining_groups, bad_edges


def render_clay_qa(scene, model, out_dir: Path, angles=(0, 45, 90, 180, 225, 270)):
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    scene.world = bpy.data.worlds.new("Studio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.045, 0.053, 0.073, 1.0)

    material = bpy.data.materials.new("Clay")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.36, 0.40, 0.48, 1.0)
    model.data.materials.clear()
    model.data.materials.append(material)
    for face in model.data.polygons:
        face.use_smooth = True

    coords = np.empty(len(model.data.vertices) * 3, dtype=np.float32)
    model.data.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)
    center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    extent = float(np.ptp(coords, axis=0).max())
    look = Vector(center.tolist())

    for name, offset, power, size in [
        ("Key", (-2.0, -3.0, 2.8), 450, 2.0),
        ("Fill", (2.0, -1.5, 2.0), 210, 2.4),
        ("Rim", (0.4, 2.0, 2.4), 300, 1.8),
    ]:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = power * (extent ** 2)
        data.size = size * extent
        lamp = bpy.data.objects.new(name, data)
        scene.collection.objects.link(lamp)
        lamp.location = look + Vector(offset) * extent
        lamp.rotation_euler = (look - lamp.location).to_track_quat("-Z", "Y").to_euler()

    cam_obj = bpy.data.objects.new("QA_Camera", bpy.data.cameras.new("QA_Camera"))
    scene.collection.objects.link(cam_obj)
    cam_obj.data.type = "ORTHO"
    cam_obj.data.ortho_scale = extent * 1.32
    scene.camera = cam_obj

    dist = extent * 2.8
    elev = extent * 0.45
    for angle in angles:
        rad = radians(angle)
        cam_obj.location = look + Vector((-sin(rad) * dist, -cos(rad) * dist, elev))
        cam_obj.rotation_euler = (look - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out_dir / f"clay-{angle:03d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    args = parse_cli_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene

    body = load_mesh_object(args.body_npz or args.body_obj, "ExtractedBody")
    taubin_smooth_mesh(body, iterations=args.smooth_iters)
    parts = [body]
    for i, ep in enumerate(args.extra_mesh):
        parts.append(load_mesh_object(ep, f"ExtraProp_{i}"))
    if args.add_base:
        parts.append(add_circular_base(body, args.base_radius_ratio, args.base_thickness_ratio))

    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = body
    if len(parts) > 1:
        bpy.ops.object.join()

    modifier = body.modifiers.new("SolidUnion", "REMESH")
    modifier.mode = "VOXEL"
    modifier.voxel_size = args.voxel_size
    modifier.adaptivity = 0.0
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    taubin_smooth_mesh(body, iterations=args.post_smooth_iters)

    counts, remaining_groups, bad_edges = cleanup_components(body)
    assert bad_edges == 0, f"Found {bad_edges} non-manifold edges after voxel union!"

    coords = np.empty(len(body.data.vertices) * 3, dtype=np.float32)
    body.data.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)

    # Flatten bottom slightly and ground at Z = 0
    bottom = float(coords[:, 2].min() + args.bottom_clip)
    coords[:, 2] = np.maximum(coords[:, 2], bottom) - bottom

    # Scale to real-world millimeters
    if args.base_diameter_mm is not None:
        z_band = max(0.02, float(coords[:, 2].max()) * 0.04)
        lowest = coords[:, 2] < z_band
        base_width = float(np.ptp(coords[lowest, 0]))
        factor = args.base_diameter_mm / max(base_width, 1e-6)
    elif args.target_height_mm is not None:
        factor = args.target_height_mm / max(float(coords[:, 2].max()), 1e-6)
    else:
        factor = 1.0

    scaled = coords * factor
    body.data.vertices.foreach_set("co", scaled.ravel())
    body.data.update()

    args.out_stl.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.stl_export(
        filepath=str(args.out_stl),
        export_selected_objects=True,
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
    )
    if args.out_glb:
        args.out_glb.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(filepath=str(args.out_glb), use_selection=True, export_format="GLB")

    dims_mm = np.ptp(scaled, axis=0).tolist()
    report = {
        "faces": len(body.data.polygons),
        "vertices": len(body.data.vertices),
        "components_before_cleanup": counts[:16],
        "components_after_cleanup": remaining_groups,
        "nonmanifold_edges": bad_edges,
        "min_z_mm": float(scaled[:, 2].min()),
        "overall_dimensions_mm": [round(float(v), 3) for v in dims_mm],
        "merge_voxel_mm": round(float(args.voxel_size * factor), 4),
        "stl_file": str(args.out_stl),
        "glb_file": str(args.out_glb) if args.out_glb else None,
    }
    report_path = args.report or args.out_stl.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if args.render_qa_dir:
        render_clay_qa(scene, body, args.render_qa_dir)


if __name__ == "__main__":
    main()
