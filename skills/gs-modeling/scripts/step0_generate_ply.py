"""Step 0b: Upload input image and submit TripoSplat workflow to local ComfyUI API, then download PLY."""
import argparse
import json
import mimetypes
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from config_loader import load_config

BUNDLED_TEMPLATE = Path(__file__).resolve().parent.parent / "resources" / "triposplat_api_workflow.template.json"


def http_json(url: str, data: dict | None = None):
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_image_to_comfyui(api_url: str, image_path: Path) -> str:
    """Upload a local PNG/JPG to ComfyUI /upload/image so LoadImage can read it regardless of work_dir."""
    boundary = f"----gsmodelflows-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    file_bytes = image_path.read_bytes()
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode("utf-8"),
        f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
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
    name = res.get("name", image_path.name)
    subfolder = res.get("subfolder", "")
    return f"{subfolder}/{name}" if subfolder else name


def main():
    ap = argparse.ArgumentParser(description="Submit TripoSplat workflow to ComfyUI API and download PLY.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None, help="Local input PNG to upload to ComfyUI (default: work_dir/input/front_refined.png).")
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--template", default=None, help="ComfyUI API prompt JSON (default: bundled triposplat_api_workflow.template.json).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["work_dir"])
    c_cfg = cfg["comfyui"]
    API = c_cfg["api_url"].rstrip("/")
    points = args.points or int(c_cfg["num_gaussians"])
    seed = args.seed if args.seed is not None else int(c_cfg["seed"])

    tmpl_raw = args.template or c_cfg.get("template_prompt_json") or ""
    template = Path(tmpl_raw) if tmpl_raw else BUNDLED_TEMPLATE
    if not template.exists():
        raise FileNotFoundError(
            f"ComfyUI TripoSplat API prompt JSON not found: {template} "
            f"(bundled default: {BUNDLED_TEMPLATE})"
        )
    stem = f"step0_raw_{points}_s{seed}"
    (root / "workflows").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)

    # Wait for ComfyUI server to be ready
    for _ in range(60):
        try:
            http_json(f"{API}/system_stats")
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError(f"ComfyUI server did not become ready at {API} within 120s")

    # Upload local input image to ComfyUI if present
    local_img = Path(args.image) if args.image else (root / "input" / "front_refined.png")
    if local_img.exists():
        comfy_image_name = upload_image_to_comfyui(API, local_img)
        print(f"Uploaded {local_img} -> ComfyUI input '{comfy_image_name}'", flush=True)
    else:
        comfy_image_name = local_img.name
        print(f"Warning: local image {local_img} not found; assuming '{comfy_image_name}' already exists in ComfyUI input.", flush=True)

    graph = json.loads(template.read_text(encoding="utf-8"))
    graph["6"]["inputs"]["seed"] = seed
    graph["55"]["inputs"]["num_gaussians"] = points
    graph["99"]["inputs"]["image"] = comfy_image_name
    graph["51"]["inputs"]["filename_prefix"] = f"3d/{stem}"

    wf_path = root / "workflows" / f"{stem}_prompt.json"
    wf_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    queued = http_json(f"{API}/prompt", {"prompt": graph})
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to queue prompt: {queued}")
    print(f"Queued prompt_id={prompt_id} (points={points}, seed={seed}, api={API})", flush=True)

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
