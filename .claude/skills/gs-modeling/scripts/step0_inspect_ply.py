"""Step 0c: Inspect generated 3DGS PLY for quality gate (attributes, ranges, NaNs, bbox, orientation)."""
import json
import sys
from pathlib import Path
import numpy as np
from ply_io import read_ply


def stats(arr: np.ndarray) -> dict:
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }


def main():
    ply_path = Path(sys.argv[1])
    data, _ = read_ply(ply_path)
    names = list(data.dtype.names)

    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    missing = [k for k in required if k not in names]
    f_rest = [k for k in names if k.startswith("f_rest_")]
    sh_degree = {0: 0, 9: 1, 24: 2, 45: 3}.get(len(f_rest), -1)

    report = {
        "file": str(ply_path),
        "num_gaussians": len(data),
        "properties": names,
        "missing_required": missing,
        "f_rest_count": len(f_rest),
        "sh_degree": sh_degree,
    }

    if not missing:
        xyz = np.column_stack([data["x"], data["y"], data["z"]])
        op = data["opacity"]
        sc = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]])
        rot = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]])
        fdc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]])
        rot_norm = np.linalg.norm(rot, axis=1)

        # Heuristic: opacity is logit if negative values or > 1 exist; scale is log if negative values exist
        opacity_space = "logit (raw)" if (op.min() < 0.0 or op.max() > 1.0) else "linear [0,1]"
        scale_space = "log (raw)" if sc.min() < 0.0 else "linear (>0)"

        # Compute robust bbox (1%-99%) to ignore floaters
        p01 = np.percentile(xyz, 1, axis=0)
        p99 = np.percentile(xyz, 99, axis=0)
        ext = p99 - p01
        report["schema"] = {
            "opacity_space": opacity_space,
            "scale_space": scale_space,
        }
        report["bbox_full_min"] = xyz.min(axis=0).tolist()
        report["bbox_full_max"] = xyz.max(axis=0).tolist()
        report["bbox_p01"] = p01.tolist()
        report["bbox_p99"] = p99.tolist()
        report["extent_p01_p99"] = ext.tolist()
        report["dominant_height_axis"] = ["X", "Y", "Z"][int(np.argmax(ext))]
        report["opacity"] = stats(op)
        report["scale_all"] = stats(sc)
        report["rot_norm"] = stats(rot_norm)
        report["f_dc"] = stats(fdc)

        # Gate verdict
        nan_total = sum(int(np.isnan(data[k]).sum() + np.isinf(data[k]).sum()) for k in required)
        report["quality_gate"] = {
            "pass": (len(data) >= 10000) and (len(missing) == 0) and (nan_total == 0),
            "nan_or_inf_total": nan_total,
        }
    else:
        report["quality_gate"] = {"pass": False, "reason": f"Missing required properties: {missing}"}

    out_json = ply_path.with_suffix(".inspect.json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
