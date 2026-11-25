"""Render a 3DGS .ply in Blender (headless) as a multi-view contact sheet using Geometry Nodes.

Usage:
  blender -b --python render_ply_blender.py -- --ply input.ply --out preview.png [--face-center x,y,z] [--face-size 0.35] [--skeleton skel.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector, Quaternion, Matrix

SH_C0 = 0.28209479177387814

PLY_TYPE_MAP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def read_ply(path: str):
    with open(path, "rb") as f:
        vertex_count = None
        in_vertex = False
        props = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = (parts[1] == "vertex")
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                props.append((parts[2], PLY_TYPE_MAP[parts[1]]))
        dtype = np.dtype(props)
        return np.fromfile(f, dtype=dtype, count=vertex_count)


def quat_wxyz_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Convert normalized (N,4) [w,x,y,z] quaternions to XYZ Euler radians."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    ry = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = np.arctan2(siny_cosp, cosy_cosp)
    return np.column_stack([rx, ry, rz]).astype(np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def build_splat_object(ply_path: str, opacity_thresh: float = 0.12, scale_mult: float = 1.8):
    data = read_ply(ply_path)
    op_raw = data["opacity"].astype(np.float32)
    if op_raw.min() < 0.0 or op_raw.max() > 1.0:
        op = 1.0 / (1.0 + np.exp(-np.clip(op_raw, -20.0, 20.0)))
    else:
        op = op_raw

    keep = op >= opacity_thresh
    d = data[keep]
    n = len(d)
    print(f"[BlenderRender] Loaded {len(data)} splats, keeping {n} (opacity >= {opacity_thresh})")

    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    sc_raw = np.column_stack([d["scale_0"], d["scale_1"], d["scale_2"]]).astype(np.float32)
    sc = np.exp(np.clip(sc_raw, -12.0, 2.0)) if sc_raw.min() < 0.0 else sc_raw
    sc = np.clip(sc * scale_mult, 1e-4, 0.25)

    q = np.column_stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]]).astype(np.float32)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(qn > 1e-8, q / np.maximum(qn, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    eul = quat_wxyz_to_euler_xyz(q)

    fdc = np.column_stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]]).astype(np.float32)
    rgb_srgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    rgb_lin = srgb_to_linear(rgb_srgb)
    rgba = np.column_stack([rgb_lin, np.ones((n, 1), dtype=np.float32)]).astype(np.float32)

    mesh = bpy.data.meshes.new("SplatPoints")
    mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", xyz.ravel())

    attr_scale = mesh.attributes.new(name="gs_scale", type="FLOAT_VECTOR", domain="POINT")
    attr_scale.data.foreach_set("vector", sc.ravel())

    attr_rot = mesh.attributes.new(name="gs_rot", type="FLOAT_VECTOR", domain="POINT")
    attr_rot.data.foreach_set("vector", eul.ravel())

    attr_col = mesh.attributes.new(name="gs_color", type="FLOAT_COLOR", domain="POINT")
    attr_col.data.foreach_set("color", rgba.ravel())

    mesh.update()
    obj = bpy.data.objects.new("SplatObject", mesh)
    bpy.context.collection.objects.link(obj)

    # Material using stored attribute gs_color on instancer
    mat = bpy.data.materials.new("SplatMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    emit_node = nt.nodes.new("ShaderNodeEmission")
    attr_node = nt.nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = "gs_color"
    attr_node.attribute_type = "INSTANCER"
    nt.links.new(attr_node.outputs["Color"], emit_node.inputs["Color"])
    nt.links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    # Geometry Nodes: Instance on Points with low-poly icosphere
    mod = obj.modifiers.new("SplatGN", "NODES")
    gn = bpy.data.node_groups.new("SplatGNTree", "GeometryNodeTree")
    mod.node_group = gn
    gn.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    gn.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    n_in = gn.nodes.new("NodeGroupInput")
    n_out = gn.nodes.new("NodeGroupOutput")
    n_ico = gn.nodes.new("GeometryNodeMeshIcoSphere")
    n_ico.inputs["Radius"].default_value = 1.0
    n_ico.inputs["Subdivisions"].default_value = 1

    n_setmat = gn.nodes.new("GeometryNodeSetMaterial")
    n_setmat.inputs["Material"].default_value = mat
    gn.links.new(n_ico.outputs["Mesh"], n_setmat.inputs["Geometry"])

    n_scale = gn.nodes.new("GeometryNodeInputNamedAttribute")
    n_scale.data_type = "FLOAT_VECTOR"
    n_scale.inputs["Name"].default_value = "gs_scale"

    n_rot = gn.nodes.new("GeometryNodeInputNamedAttribute")
    n_rot.data_type = "FLOAT_VECTOR"
    n_rot.inputs["Name"].default_value = "gs_rot"

    n_inst = gn.nodes.new("GeometryNodeInstanceOnPoints")
    gn.links.new(n_in.outputs["Geometry"], n_inst.inputs["Points"])
    gn.links.new(n_setmat.outputs["Geometry"], n_inst.inputs["Instance"])
    gn.links.new(n_rot.outputs["Attribute"], n_inst.inputs["Rotation"])
    gn.links.new(n_scale.outputs["Attribute"], n_inst.inputs["Scale"])
    gn.links.new(n_inst.outputs["Instances"], n_out.inputs["Geometry"])

    return obj, xyz


def setup_scene(res: int = 512):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # Use EEVEE Next (or EEVEE) for fast GPU/CPU headless rendering
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.14, 0.16, 0.20, 1.0)
        bg.inputs[1].default_value = 1.0

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return scene, cam_obj


def point_camera(cam_obj, eye: Vector, target: Vector, up: Vector, ortho_scale: float):
    direction = (target - eye).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    # Align camera up with desired up
    cam_obj.location = eye
    cam_obj.rotation_euler = rot_quat.to_euler()
    cam_obj.data.ortho_scale = ortho_scale


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--opacity-thresh", type=float, default=0.12)
    ap.add_argument("--scale-mult", type=float, default=1.6)
    ap.add_argument("--up-axis", choices=["Z", "Y"], default="Z")
    ap.add_argument("--front-axis", choices=["-Y", "+Y", "+Z", "-Z", "+X", "-X"], default="-Y")
    ap.add_argument("--face-center", default=None, help="x,y,z for close-up view")
    ap.add_argument("--face-scale", type=float, default=0.42)
    args = ap.parse_args(argv)

    scene, cam = setup_scene(args.res)
    _, xyz = build_splat_object(args.ply, args.opacity_thresh, args.scale_mult)

    p02 = np.percentile(xyz, 2, axis=0)
    p98 = np.percentile(xyz, 98, axis=0)
    center = Vector(((p02 + p98) * 0.5).tolist())
    extent = p98 - p02
    max_ext = float(np.max(extent))
    dist = max_ext * 3.0
    ortho = max_ext * 1.18

    axis_vec = {
        "+X": Vector((1, 0, 0)), "-X": Vector((-1, 0, 0)),
        "+Y": Vector((0, 1, 0)), "-Y": Vector((0, -1, 0)),
        "+Z": Vector((0, 0, 1)), "-Z": Vector((0, 0, -1)),
    }
    fwd = axis_vec[args.front_axis]
    up = Vector((0, 0, 1)) if args.up_axis == "Z" else Vector((0, 1, 0))
    right = fwd.cross(up).normalized()

    if args.face_center:
        fc = Vector([float(v) for v in args.face_center.split(",")])
    else:
        # Default upper-center (head area)
        fc = center + up * (float(extent[2 if args.up_axis == "Z" else 1]) * 0.28)

    views = [
        ("front", center + fwd * dist, center, ortho),
        ("side", center + right * dist, center, ortho),
        ("back", center - fwd * dist, center, ortho),
        ("face", fc + fwd * dist, fc, max_ext * args.face_scale),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_imgs = []
    for name, eye, tgt, osc in views:
        tmp = out_path.parent / f"{out_path.stem}_{name}.png"
        # Look-at matrix with explicit up vector
        z_ax = (eye - tgt).normalized()
        x_ax = up.cross(z_ax).normalized()
        y_ax = z_ax.cross(x_ax).normalized()
        rot_m = Matrix((
            (x_ax.x, y_ax.x, z_ax.x, eye.x),
            (x_ax.y, y_ax.y, z_ax.y, eye.y),
            (x_ax.z, y_ax.z, z_ax.z, eye.z),
            (0.0, 0.0, 0.0, 1.0),
        ))
        cam.matrix_world = rot_m
        cam.data.ortho_scale = osc
        scene.render.filepath = str(tmp)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(str(tmp))
        w, h = img.size
        buf = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        panel_imgs.append(buf)
        bpy.data.images.remove(img)
        tmp.unlink(missing_ok=True)

    # Assemble 2x2 contact sheet (top: front, side; bottom: back, face)
    # Note: Blender image pixels are bottom-to-top
    top = np.concatenate([panel_imgs[0], panel_imgs[1]], axis=1)
    bot = np.concatenate([panel_imgs[2], panel_imgs[3]], axis=1)
    grid = np.concatenate([bot, top], axis=0)
    gh, gw, _ = grid.shape
    out_img = bpy.data.images.new("Grid", width=gw, height=gh, alpha=True)
    out_img.pixels.foreach_set(grid.ravel())
    out_img.filepath_raw = str(out_path)
    out_img.file_format = "PNG"
    out_img.save()
    print(f"[BlenderRender] Saved 2x2 contact sheet: {out_path} ({gw}x{gh})")


if __name__ == "__main__":
    main()
