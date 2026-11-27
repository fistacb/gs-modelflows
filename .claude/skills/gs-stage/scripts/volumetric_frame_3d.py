"""True 3D Volumetric Map Frame Engine (Multi-Layer Nav-Span + 3D A* + Slope-Aware Walk Planner).

Every (ix, iy) column stores stacked walkable spans:
    spans[iy][ix] = [{"z": float, "type": str, "layer": str, "thick": float, "headroom": float}, ...]
Adjacency between neighbor columns (ix, iy, s_a) -> (nx, ny, s_b) is ONLY allowed when:
    abs(z_b - z_a) <= max_step_height  AND  headroom >= min_headroom
This enables true 3D underpass/overpass vertical crossings without layer teleportation.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def point_in_oriented_box(x: np.ndarray, y: np.ndarray, cx: float, cy: float, hx: float, hy: float, yaw_deg: float = 0.0):
    yr = math.radians(-yaw_deg)
    c, s = math.cos(yr), math.sin(yr)
    dx, dy = x - cx, y - cy
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return (np.abs(lx) <= hx) & (np.abs(ly) <= hy), lx, ly


class VolumetricFrame3D:
    """Compiles a 3D volumetric frame JSON into multi-layer walkable spans + 3D solid obstacles."""

    def __init__(self, frame: dict, props_root: Path | None = None, grid_res: float = 0.15, min_headroom: float = 1.55):
        self.frame = frame
        self.props_root = Path(props_root) if props_root else None
        b = frame["bounds"]
        self.x_min, self.x_max = float(b["x_min"]), float(b["x_max"])
        self.y_min, self.y_max = float(b["y_min"]), float(b["y_max"])
        self.res = float(grid_res)
        self.min_headroom = float(min_headroom)
        self.nx = int(round((self.x_max - self.x_min) / self.res)) + 1
        self.ny = int(round((self.y_max - self.y_min) / self.res)) + 1
        self.xs = np.linspace(self.x_min, self.x_max, self.nx, dtype=np.float32)
        self.ys = np.linspace(self.y_min, self.y_max, self.ny, dtype=np.float32)
        xx, yy = np.meshgrid(self.xs, self.ys)

        self.spans: list[list[list[dict]]] = [[[] for _ in range(self.nx)] for _ in range(self.ny)]
        self.solid_boxes: list[dict] = []

        # 1. Base ground layer
        z0 = self._eval_ground_z(xx, yy)
        g_type = self._eval_ground_type(xx, yy)
        for iy in range(self.ny):
            for ix in range(self.nx):
                gt = g_type[iy][ix]
                if gt != "water":
                    self.spans[iy][ix].append({"z": float(z0[iy, ix]), "type": gt, "layer": "ground", "thick": 0.25})

        # 2. Volumetric 3D structural elements (platforms, overpasses, ramps, pillars)
        for vol in frame.get("volumes", []):
            vkind = vol["kind"]
            if vkind in ("platform", "overpass"):
                m, lx, _ = point_in_oriented_box(xx, yy, vol["cx"], vol["cy"], vol["hx"], vol["hy"], vol.get("yaw_deg", 0.0))
                z_top = float(vol["z_top"])
                thick = float(vol.get("thickness", 0.30))
                stype = vol.get("surface_type", "ruin_stone")
                lid = vol["id"]
                ys_m, xs_m = np.where(m)
                for iy, ix in zip(ys_m, xs_m):
                    camber = float(vol.get("arch_camber", 0.0)) * (1.0 - (lx[iy, ix] / max(vol["hx"], 1e-4)) ** 2)
                    self.spans[iy][ix].append({"z": z_top + camber, "type": stype, "layer": lid, "thick": thick})
            elif vkind == "ramp":
                m, lx, _ = point_in_oriented_box(xx, yy, vol["cx"], vol["cy"], vol["hx"], vol["hy"], vol.get("yaw_deg", 0.0))
                z0_r, z1_r = float(vol["z_start"]), float(vol["z_end"])
                thick = float(vol.get("thickness", 0.28))
                lid = vol["id"]
                ys_m, xs_m = np.where(m)
                for iy, ix in zip(ys_m, xs_m):
                    u = float(np.clip((lx[iy, ix] + vol["hx"]) / (2.0 * vol["hx"]), 0.0, 1.0))
                    zr = z0_r * (1.0 - u) + z1_r * u
                    self.spans[iy][ix].append({"z": zr, "type": "stairs", "layer": lid, "thick": thick})
            elif vkind == "pillar":
                self.solid_boxes.append({
                    "cx": float(vol["cx"]), "cy": float(vol["cy"]),
                    "r": float(vol["radius"]), "z_min": float(vol["z_min"]), "z_max": float(vol["z_max"]),
                })

        # 3. Prop 3D cylindrical obstacles (only block spans whose Z intersects [prop_z, prop_z + prop_h])
        walk_through_defaults = {
            "19_fern_bush", "20_leafy_bush", "29_flower_patch", "31_flower_bush",
            "22_red_mushrooms_small", "02_stone_arch", "17_hanging_vines", "25_vine_arch",
        }
        for p in frame.get("props", []):
            if p.get("walk_through", False) or p["asset"] in walk_through_defaults:
                continue
            px, py = float(p["x"]), float(p["y"])
            pz = self.query_layer_z(px, py, p.get("layer", "ground")) + float(p.get("z_offset", 0.0))
            sc = float(p.get("scale", 1.0))
            r_col, h_col = 0.40 * sc, 1.5 * sc
            if self.props_root is not None:
                meta_p = self.props_root / p["asset"] / "v1_prop" / "meta.json"
                if meta_p.exists():
                    m = json.loads(meta_p.read_text(encoding="utf-8"))
                    r_col = float(m.get("footprint_radius_m", 0.5)) * sc * 0.48
                    h_col = float(m.get("dimensions_m", {}).get("z", 1.5)) * sc
            self.solid_boxes.append({"cx": px, "cy": py, "r": r_col, "z_min": pz - 0.1, "z_max": pz + h_col})

        # 4. Sort spans by Z, compute headroom to next layer above, filter blocked/low-headroom spans
        for iy in range(self.ny):
            for ix in range(self.nx):
                wx, wy = float(self.xs[ix]), float(self.ys[iy])
                cell = sorted(self.spans[iy][ix], key=lambda s: s["z"])
                merged = []
                for sp in cell:
                    if merged and abs(sp["z"] - merged[-1]["z"]) < 0.22:
                        merged[-1] = sp
                    else:
                        merged.append(sp)
                valid = []
                for si, sp in enumerate(merged):
                    z_s = sp["z"]
                    headroom = 99.0
                    if si + 1 < len(merged):
                        headroom = (merged[si + 1]["z"] - merged[si + 1]["thick"]) - z_s
                    sp["headroom"] = headroom
                    if headroom < self.min_headroom:
                        continue
                    if ix < 3 or ix >= self.nx - 3 or iy < 3 or iy >= self.ny - 3:
                        continue
                    blocked = False
                    for ob in self.solid_boxes:
                        if ob["z_max"] > z_s + 0.15 and ob["z_min"] < z_s + 1.45:
                            if math.hypot(wx - ob["cx"], wy - ob["cy"]) < ob["r"] + 0.22:
                                blocked = True
                                break
                    if not blocked:
                        valid.append(sp)
                self.spans[iy][ix] = valid

    def _eval_ground_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = 0.03 * np.sin(x * 0.7) * np.cos(y * 0.8)
        for h in self.frame.get("ground", {}).get("hills", []):
            r2 = ((x - h["cx"]) / h["rx"]) ** 2 + ((y - h["cy"]) / h["ry"]) ** 2
            z = z + h["h"] * np.exp(-r2)
        return z.astype(np.float32)

    def _eval_ground_type(self, x: np.ndarray, y: np.ndarray) -> list[list[str]]:
        ny, nx = x.shape
        out = [["forest_floor" for _ in range(nx)] for _ in range(ny)]
        for reg in self.frame.get("ground", {}).get("regions", []):
            rk = reg["kind"]
            if rk == "box":
                m, _, _ = point_in_oriented_box(x, y, reg["cx"], reg["cy"], reg["hx"], reg["hy"], reg.get("yaw_deg", 0.0))
            elif rk == "circle":
                m = np.hypot((x - reg["cx"]) / reg["rx"], (y - reg["cy"]) / reg["ry"]) <= 1.0
            else:
                continue
            ys_m, xs_m = np.where(m)
            for iy, ix in zip(ys_m, xs_m):
                out[iy][ix] = reg["surface_type"]
        return out

    def query_layer_z(self, x: float, y: float, layer_id: str) -> float:
        ix = int(np.clip(round((x - self.x_min) / self.res), 0, self.nx - 1))
        iy = int(np.clip(round((y - self.y_min) / self.res), 0, self.ny - 1))
        for sp in self.spans[iy][ix]:
            if sp["layer"] == layer_id:
                return float(sp["z"])
        for vol in self.frame.get("volumes", []):
            if vol["id"] == layer_id:
                return float(vol.get("z_top", vol.get("z_end", 0.0)))
        return float(self._eval_ground_z(np.array([[x]], np.float32), np.array([[y]], np.float32))[0, 0])

    def _nearest_node(self, x: float, y: float, target_z: float) -> tuple[int, int, int]:
        ix0 = int(np.clip(round((x - self.x_min) / self.res), 0, self.nx - 1))
        iy0 = int(np.clip(round((y - self.y_min) / self.res), 0, self.ny - 1))
        best, best_d = None, 1e9
        for iy in range(max(0, iy0 - 8), min(self.ny, iy0 + 9)):
            for ix in range(max(0, ix0 - 8), min(self.nx, ix0 + 9)):
                for si, sp in enumerate(self.spans[iy][ix]):
                    d = math.hypot((ix - ix0) * self.res, (iy - iy0) * self.res) + abs(sp["z"] - target_z) * 2.5
                    if d < best_d:
                        best_d, best = d, (ix, iy, si)
        if best is None:
            raise RuntimeError(f"No walkable 3D span near ({x}, {y}, z={target_z})")
        return best

    def astar_3d(
        self,
        start_xyz: tuple[float, float, float],
        goal_xyz: tuple[float, float, float],
        max_step_height: float = 0.36,
    ) -> list[tuple[float, float, float, str, str]]:
        s_node = self._nearest_node(*start_xyz)
        g_node = self._nearest_node(*goal_xyz)
        gx, gy, gs = g_node
        gz = self.spans[gy][gx][gs]["z"]

        pq = [(0.0, s_node)]
        g_cost = {s_node: 0.0}
        parent = {}
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]

        while pq:
            _, cur = heapq.heappop(pq)
            if cur == g_node:
                break
            cx, cy, cs = cur
            cz = self.spans[cy][cx][cs]["z"]
            cg = g_cost[cur]
            for dx, dy, step_d in nbrs:
                nx_, ny_ = cx + dx, cy + dy
                if not (0 <= nx_ < self.nx and 0 <= ny_ < self.ny):
                    continue
                for ns, sp in enumerate(self.spans[ny_][nx_]):
                    dz = abs(sp["z"] - cz)
                    if dz > max_step_height:
                        continue
                    mul = 0.78 if sp["type"] in ("stone_path", "ruin_stone", "stairs") else 1.15
                    ng = cg + (step_d * self.res + dz * 0.45) * mul
                    nxt = (nx_, ny_, ns)
                    if ng < g_cost.get(nxt, 1e9):
                        g_cost[nxt] = ng
                        parent[nxt] = cur
                        h = math.hypot((nx_ - gx) * self.res, (ny_ - gy) * self.res) * 0.75 + abs(sp["z"] - gz) * 0.4
                        heapq.heappush(pq, (ng + h, nxt))

        if g_node not in parent and g_node != s_node:
            raise RuntimeError(f"No 3D walkable path from {start_xyz} to {goal_xyz}")
        out = []
        cur = g_node
        while cur != s_node:
            cx, cy, cs = cur
            sp = self.spans[cy][cx][cs]
            out.append((float(self.xs[cx]), float(self.ys[cy]), float(sp["z"]), sp["type"], sp["layer"]))
            cur = parent[cur]
        cx, cy, cs = s_node
        sp = self.spans[cy][cx][cs]
        out.append((float(self.xs[cx]), float(self.ys[cy]), float(sp["z"]), sp["type"], sp["layer"]))
        out.reverse()
        return out

    def plan_3d_trajectory(self, waypoints_xyz: list[tuple[float, float, float]], n_frames: int = 64) -> list[dict]:
        raw = []
        for i in range(len(waypoints_xyz) - 1):
            seg = self.astar_3d(waypoints_xyz[i], waypoints_xyz[i + 1])
            if raw:
                seg = seg[1:]
            raw.extend(seg)

        pts = np.array([[p[0], p[1], p[2]] for p in raw], dtype=np.float32)
        stypes = [p[3] for p in raw]
        layers = [p[4] for p in raw]

        for _ in range(3):
            ps = pts.copy()
            ps[1:-1] = 0.25 * pts[:-2] + 0.50 * pts[1:-1] + 0.25 * pts[2:]
            pts = ps

        ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(ds)])
        total_len = float(s_cum[-1])
        s_samp = np.linspace(0.0, total_len, n_frames)
        sx = np.interp(s_samp, s_cum, pts[:, 0])
        sy = np.interp(s_samp, s_cum, pts[:, 1])
        sz = np.interp(s_samp, s_cum, pts[:, 2])

        traj = []
        stride_cycle_m = 0.78
        for i in range(n_frames):
            i0, i1 = max(0, i - 1), min(n_frames - 1, i + 1)
            tx, ty = float(sx[i1] - sx[i0]), float(sy[i1] - sy[i0])
            horiz = max(math.hypot(tx, ty), 1e-5)
            yaw_deg = math.degrees(math.atan2(tx, -ty))
            dz_ds = float(sz[i1] - sz[i0]) / horiz
            slope_deg = math.degrees(math.atan(dz_ds))
            src_i = int(np.clip(np.searchsorted(s_cum, s_samp[i]), 0, len(stypes) - 1))
            st, ly = stypes[src_i], layers[src_i]
            is_stairs = (st == "stairs")
            traj.append({
                "frame": i,
                "x": float(sx[i]), "y": float(sy[i]), "z": float(sz[i]),
                "yaw_deg": yaw_deg, "slope_deg": slope_deg,
                "surface_type": st, "layer": ly,
                "gait_phase": (float(s_samp[i]) / stride_cycle_m) * 2.0 * math.pi,
                "torso_pitch_deg": float(np.clip(slope_deg * 0.42 + (4.0 if is_stairs else 0.0), -10.0, 14.0)),
                "knee_lift_mul": float(np.clip(1.0 + max(0.0, dz_ds) * 1.15 + (0.30 if is_stairs else 0.0), 0.85, 1.70)),
                "stride_mul": float(np.clip(1.0 - abs(dz_ds) * 0.45, 0.70, 1.0)),
                "arc_s": float(s_samp[i]),
            })
        return traj

    def export_nav_json(self, out_path: Path) -> None:
        """Export engine-agnostic multi-layer 3D walkable spans for 2D/HD-2D or 3D game runtimes."""
        cells = []
        overlap = 0
        for iy in range(self.ny):
            for ix in range(self.nx):
                if self.spans[iy][ix]:
                    if len(self.spans[iy][ix]) >= 2:
                        overlap += 1
                    cells.append({
                        "ix": ix, "iy": iy,
                        "x": round(float(self.xs[ix]), 3), "y": round(float(self.ys[iy]), 3),
                        "spans": [
                            {"z": round(sp["z"], 3), "type": sp["type"], "layer": sp["layer"], "headroom": round(sp["headroom"], 2)}
                            for sp in self.spans[iy][ix]
                        ],
                    })
        payload = {
            "bounds": {"x_min": self.x_min, "x_max": self.x_max, "y_min": self.y_min, "y_max": self.y_max},
            "grid_res_m": self.res,
            "nx": self.nx, "ny": self.ny,
            "overlap_cells": overlap,
            "walkable_cells": cells,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_multilayer_preview(self, traj: list[dict] | None, out_png: Path) -> None:
        W, H = 560, 440
        l0_img = np.full((self.ny, self.nx, 3), [22, 26, 36], dtype=np.uint8)
        l1_img = np.full((self.ny, self.nx, 3), [22, 26, 36], dtype=np.uint8)
        multi_cnt = 0
        for iy in range(self.ny):
            ry = self.ny - 1 - iy
            for ix in range(self.nx):
                sps = self.spans[iy][ix]
                if len(sps) >= 2:
                    multi_cnt += 1
                for sp in sps:
                    if sp["z"] < 1.0:
                        c = [142, 140, 128] if sp["type"] == "stone_path" else ([200, 165, 92] if sp["type"] == "stairs" else [68, 98, 72])
                        l0_img[ry, ix] = c
                    else:
                        c = [168, 178, 174] if sp["type"] == "ruin_stone" else [205, 170, 96]
                        l1_img[ry, ix] = c

        p0 = Image.fromarray(l0_img).resize((W, H), Image.Resampling.NEAREST)
        p1 = Image.fromarray(l1_img).resize((W, H), Image.Resampling.NEAREST)

        def w2p(wx, wy):
            return (wx - self.x_min) / (self.x_max - self.x_min) * (W - 1), (self.y_max - wy) / (self.y_max - self.y_min) * (H - 1)

        d0, d1 = ImageDraw.Draw(p0, "RGBA"), ImageDraw.Draw(p1, "RGBA")
        if traj:
            for i in range(len(traj) - 1):
                a, b = traj[i], traj[i + 1]
                pa, pb = w2p(a["x"], a["y"]), w2p(b["x"], b["y"])
                if a["z"] < 1.05:
                    d0.line([pa, pb], fill=(95, 225, 255, 240), width=4)
                else:
                    d1.line([pa, pb], fill=(255, 195, 85, 240), width=4)

        d0.rectangle([8, 8, 395, 36], fill=(12, 16, 24, 215))
        d0.text((16, 15), f"LAYER 0 (Z=0.0m Ground + Underpass)  [Overlap cells: {multi_cnt}]", fill=(180, 240, 255, 255))
        d1.rectangle([8, 8, 395, 36], fill=(12, 16, 24, 215))
        d1.text((16, 15), "LAYER 1 (Z>=1.0m Elevated Platforms / Overpass)", fill=(255, 225, 155, 255))

        prof_h = 170 if traj else 0
        full = Image.new("RGB", (W * 2, H + prof_h), (14, 18, 24))
        full.paste(p0, (0, 0))
        full.paste(p1, (W, 0))
        if traj:
            dr = ImageDraw.Draw(full)
            dr.text((18, H + 8), "3D Multi-Layer Elevation Profile Z(s)", fill=(220, 238, 232))
            gx0, gx1, gy0, gy1 = 48, W * 2 - 24, H + 34, H + prof_h - 22
            dr.rectangle([gx0, gy0, gx1, gy1], outline=(70, 82, 98), width=1)
            zs = [t["z"] for t in traj]
            zmin_p, zmax_p = min(zs) - 0.1, max(zs) + 0.25
            g_pts = []
            for i, t in enumerate(traj):
                px = gx0 + (gx1 - gx0) * (i / max(len(traj) - 1, 1))
                py = gy1 - (gy1 - gy0) * ((t["z"] - zmin_p) / max(zmax_p - zmin_p, 1e-4))
                g_pts.append((px, py))
                c = (205, 170, 96) if t["surface_type"] == "stairs" else ((155, 185, 178) if t["z"] > 1.2 else (85, 175, 195))
                dr.line([(px, gy1), (px, py)], fill=c, width=3)
            for i in range(len(g_pts) - 1):
                dr.line([g_pts[i], g_pts[i + 1]], fill=(245, 250, 240), width=3)
            dr.text((8, gy0 - 4), f"{zmax_p:.1f}m", fill=(180, 200, 210))
            dr.text((8, gy1 - 10), f"{zmin_p:.1f}m", fill=(180, 200, 210))
        full.save(out_png)
