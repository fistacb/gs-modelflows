"""Render X-ray 3-view (Front 0 deg / Oblique 45 deg / Side 90 deg) rig preview:
Overlays rig_skeleton.json bones (TRUE X-ray always-on-top via 2-pass alpha composite) +
translucent wireframe rig_cages.obj on master.ply in Blender.
Works with ANY bone count or skeleton hierarchy (zero hardcoded bone names/counts).
"""
import json
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector, Matrix

# Reuse splat loader from render_ply_blender.py
sys.path.insert(0, str(Path(__file__).parent))
from render_ply_blender import build_splat_object, setup_scene


def add_cage_overlay(obj_path: Path):
    if not obj_path.exists():
        return None
    verts, faces = [], []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f":
            faces.append(tuple(int(p.split("/")[0]) - 1 for p in parts[1:]))

    if not verts or not faces:
        return None
    mesh = bpy.data.meshes.new("RigCages")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("RigCages", mesh)
    bpy.context.collection.objects.link(obj)

    mod = obj.modifiers.new("Wire", "WIREFRAME")
    mod.thickness = 0.0032
    mod.use_replace = True

    mat = bpy.data.materials.new("CageMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.15, 0.85, 1.0, 1.0)
    emit.inputs["Strength"].default_value = 1.5
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def add_skeleton_overlay(skel_path: Path):
    """Build bone shaft + joint sphere objects for ALL bones in rig_skeleton.json (bone-count agnostic)."""
    skel = json.loads(skel_path.read_text(encoding="utf-8"))

    mat_bone = bpy.data.materials.new("BoneMat")
    mat_bone.use_nodes = True
    nt = mat_bone.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (1.0, 0.86, 0.10, 1.0)  # bright yellow shafts
    emit.inputs["Strength"].default_value = 2.2
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    mat_joint = bpy.data.materials.new("JointMat")
    mat_joint.use_nodes = True
    nt2 = mat_joint.node_tree
    nt2.nodes.clear()
    out2 = nt2.nodes.new("ShaderNodeOutputMaterial")
    emit2 = nt2.nodes.new("ShaderNodeEmission")
    emit2.inputs["Color"].default_value = (1.0, 0.25, 0.10, 1.0)  # vivid orange-red joints
    emit2.inputs["Strength"].default_value = 2.4
    nt2.links.new(emit2.outputs["Emission"], out2.inputs["Surface"])

    bone_objs = []
    for bname, b in skel["bones"].items():
        h = Vector(b["head"])
        t = Vector(b["tail"])
        # Joint sphere at head (smaller for eyes so they don't clutter face)
        j_rad = 0.006 if bname.startswith("eye_") else 0.0105
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=j_rad, location=h)
        sp = bpy.context.active_object
        sp.name = f"Joint_{bname}"
        sp.data.materials.append(mat_joint)
        bone_objs.append(sp)

        dv = t - h
        length = dv.length
        if length > 1e-4:
            mid = (h + t) * 0.5
            s_rad = 0.0025 if bname.startswith("eye_") else 0.0042
            bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=s_rad, depth=length, location=mid)
            cyl = bpy.context.active_object
            cyl.name = f"Bone_{bname}"
            cyl.rotation_mode = "QUATERNION"
            cyl.rotation_quaternion = dv.normalized().to_track_quat("Z", "Y")
            cyl.data.materials.append(mat_bone)
            bone_objs.append(cyl)
    return bone_objs


def render_rgba(scene, tmp_path: Path) -> np.ndarray:
    scene.render.filepath = str(tmp_path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(tmp_path))
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    bpy.data.images.remove(img)
    tmp_path.unlink(missing_ok=True)
    return arr


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        raise SystemExit("Usage: blender -b --python render_rig_preview_blender.py -- <package_v1_dir> [out_png]")
    ver_dir = Path(argv[0])
    out_path = Path(argv[1]) if len(argv) >= 2 else (ver_dir / "preview_rig.png")

    scene, cam = setup_scene(512)
    splat_obj, xyz = build_splat_object(str(ver_dir / "master.ply"), opacity_thresh=0.12, scale_mult=1.5)
    cage_obj = add_cage_overlay(ver_dir / "rig_cages.obj")
    bone_objs = add_skeleton_overlay(ver_dir / "rig_skeleton.json")
    base_objs = [o for o in [splat_obj, cage_obj] if o is not None]

    z_mid = float(0.5 * (xyz[:, 2].min() + xyz[:, 2].max())) if len(xyz) else 0.50
    h_span = float(max(xyz[:, 2].max() - xyz[:, 2].min(), 0.2)) if len(xyz) else 1.0
    center = Vector((0.0, 0.0, z_mid))
    up = Vector((0.0, 0.0, 1.0))

    # Mandatory 3-view X-ray: Front 0 deg, Oblique 45 deg, Side 90 deg
    views = [
        ("front0", Vector((0.0, -3.0, z_mid)), center, h_span * 1.18),
        ("oblique45", Vector((2.1, -2.1, z_mid + 0.15 * h_span)), center, h_span * 1.24),
        ("side90", Vector((3.0, 0.0, z_mid)), center, h_span * 1.18),
    ]

    panels = []
    for name, eye, tgt, osc in views:
        z_ax = (eye - tgt).normalized()
        x_ax = up.cross(z_ax).normalized()
        y_ax = z_ax.cross(x_ax).normalized()
        cam.matrix_world = Matrix((
            (x_ax.x, y_ax.x, z_ax.x, eye.x),
            (x_ax.y, y_ax.y, z_ax.y, eye.y),
            (x_ax.z, y_ax.z, z_ax.z, eye.z),
            (0.0, 0.0, 0.0, 1.0),
        ))
        cam.data.ortho_scale = osc

        # Pass 1: Character + Cages (bones hidden)
        for o in base_objs:
            o.hide_render = False
        for o in bone_objs:
            o.hide_render = True
        scene.render.film_transparent = False
        base_rgba = render_rgba(scene, ver_dir / f"_tmp_base_{name}.png")

        # Pass 2: True X-ray Bones (base hidden, transparent background)
        for o in base_objs:
            o.hide_render = True
        for o in bone_objs:
            o.hide_render = False
        scene.render.film_transparent = True
        bone_rgba = render_rgba(scene, ver_dir / f"_tmp_bone_{name}.png")

        # Alpha-over bones on top of character (88% opacity so silhouette underneath remains readable)
        a = np.clip(bone_rgba[:, :, 3:4] * 0.88, 0.0, 1.0)
        comp = base_rgba.copy()
        comp[:, :, :3] = base_rgba[:, :, :3] * (1.0 - a) + bone_rgba[:, :, :3] * a
        comp[:, :, 3] = 1.0
        panels.append(comp)

    comb = np.concatenate(panels, axis=1)
    gh, gw, _ = comb.shape
    out_img = bpy.data.images.new("RigPreviewXray3View", width=gw, height=gh, alpha=True)
    out_img.pixels.foreach_set(comb.ravel())
    out_img.filepath_raw = str(out_path)
    out_img.file_format = "PNG"
    out_img.save()
    print(f"Saved X-ray 3-view rig preview (0/45/90 deg): {out_path} ({gw}x{gh})")


if __name__ == "__main__":
    main()
