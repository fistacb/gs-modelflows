"""Step 0b: Submit TripoSplat 65536 workflow to local ComfyUI API (port 8194) and download PLY."""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from config_loader import load_config


def http_json(url: str, data: dict | None = None):
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--template", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["work_dir"])
    c_cfg = cfg["comfyui"]
    API = c_cfg["api_url"]
    points = args.points or int(c_cfg["num_gaussians"])
    seed = args.seed if args.seed is not None else int(c_cfg["seed"])
    template = Path(args.template or c_cfg["template_prompt_json"])
    if not template.exists():
        raise FileNotFoundError(f"Set comfyui.template_prompt_json or --template to a valid ComfyUI TripoSplat API prompt JSON: {template}")
    stem = f"step0_raw_{points}_s{seed}"

    graph = json.loads(template.read_text(encoding="utf-8"))
    graph["6"]["inputs"]["seed"] = seed
    graph["55"]["inputs"]["num_gaussians"] = points
    graph["99"]["inputs"]["image"] = "front_refined.png"
    graph["51"]["inputs"]["filename_prefix"] = f"3d/{stem}"

    wf_path = root / "workflows" / f"{stem}_prompt.json"
    wf_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    # Wait for ComfyUI server to be ready
    for i in range(60):
        try:
            http_json(f"{API}/system_stats")
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("ComfyUI server did not become ready on port 8194 within 120s")

    queued = http_json(f"{API}/prompt", {"prompt": graph})
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {queued}")
    print(f"Queued prompt_id={prompt_id} (points={points}, seed={seed})", flush=True)

    t0 = time.time()
    for _ in range(360):
        hist = http_json(f"{API}/history/{prompt_id}")
        result = hist.get(prompt_id)
        if result:
            (root / "workflows" / f"{stem}_job.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            status = result.get("status", {})
            if status.get("status_str") != "success":
                raise RuntimeError(f"ComfyUI job failed: {status}")
            outputs = []
            for node_out in result.get("outputs", {}).values():
                for items in node_out.values():
                    if isinstance(items, list):
                        outputs.extend(items)
            ply_item = next((x for x in outputs if isinstance(x, dict) and str(x.get("filename", "")).endswith(".ply")), None)
            if not ply_item:
                raise RuntimeError(f"No .ply found in outputs: {outputs}")
            qs = urllib.parse.urlencode(ply_item)
            out_ply = root / "outputs" / f"{stem}.ply"
            with urllib.request.urlopen(f"{API}/view?{qs}", timeout=60) as r:
                out_ply.write_bytes(r.read())
            print(f"Done in {time.time() - t0:.1f}s -> {out_ply} ({out_ply.stat().st_size:,} bytes)")
            return
        time.sleep(3)
    raise RuntimeError("Timed out waiting for TripoSplat job")


if __name__ == "__main__":
    main()
