import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';

const root = new URL('./', import.meta.url);
const args = process.argv.slice(2);
const review = args[0] === 'review';
const orbit = args[0] === 'orbit';
const anim = args[0] === 'anim';
const stem = args[1] ?? 'baseline-65536';
const destination = review ? `evidence/${stem}` : orbit ? `renders-orbit-${stem}` : (args[1] ?? 'renders');
assert(/^[\w-]+$/.test(stem) && /^[\w/-]+$/.test(destination));
const animSpec = anim ? JSON.parse(await readFile(new URL(args[2] ?? 'frames/anim.json', root))) : null;
const camera = (review && !args[2]) || anim ? null : JSON.parse(await readFile(new URL(args[2] ?? 'camera.json', root)));
const sequence = review || orbit || anim ? null : JSON.parse(await readFile(new URL(args[3] ?? 'frames/manifest.json', root)));
const shaders = ['Pass.js', 'spark.module.min.js', 'three.module.js', 'three.core.js'];
const allowed = new Set(['/capture.html', '/capture.js', ...shaders.map(s => `/vendor/${s}`)]);
const baseline = `/assets/${stem}.ply`;
const staticPaths = sequence ? sequence.static.map(path => `/${path}`) : [];
const framePaths = entry => entry.ply.map(path => `/${path}`);
if (anim) {
  const animPaths = [
    `/${animSpec.basePly}`,
    `/${animSpec.skinBin}`,
    ...animSpec.frames.flatMap(f => (f.extraPly || []).map(p => `/${p}`)),
  ];
  for (const path of animPaths) {
    assert(/^\/(assets|frames)\/[\w/.-]+\.(ply|bin)$/.test(path), `Invalid anim path: ${path}`);
    allowed.add(path);
  }
} else {
  for (const path of sequence ? [...staticPaths, ...sequence.frames.flatMap(framePaths)] : [baseline]) {
    assert(/^\/(assets|frames)\/[\w/-]+\.ply$/.test(path));
    allowed.add(path);
  }
}

const server = createServer(async (request, response) => {
  const path = request.url;
  if (!allowed.has(path)) return response.writeHead(404).end();
  try {
    const bytes = await readFile(new URL(path.slice(1), root));
    const type = (path.endsWith('.ply') || path.endsWith('.bin')) ? 'application/octet-stream' : path.endsWith('.js') ? 'text/javascript' : 'text/html';
    response.writeHead(200, { 'Content-Type': type }).end(bytes);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
await mkdir(new URL('temp/', root), { recursive: true });
const profile = await mkdtemp(fileURLToPath(new URL('temp/chrome-', root)));
const useSoftware = process.env.SPARK_SOFTWARE === '1';
const gpuFlags = useSoftware
  ? ['--headless', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
  : ['--headless=new', '--use-gl=angle', '--use-angle=d3d11', '--enable-webgl', '--ignore-gpu-blocklist'];
const chrome = spawn(process.env.SPARK_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe', [
  ...gpuFlags, '--disable-background-networking',
  '--disable-extensions', '--no-first-run', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
  `http://127.0.0.1:${server.address().port}/capture.html`,
], { stdio: 'ignore', windowsHide: true });
let socket;
try {
  let port;
  for (let i = 0; i < 100; i++) {
    try {
      port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0].trim();
      if (/^\d+$/.test(port)) break;
      port = undefined;
    } catch (error) {
      if (error.code !== 'ENOENT' && error.code !== 'EBUSY') throw error;
    }
    await delay(150);
  }
  assert(port, 'Chrome did not start');
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let id = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    pending.get(message.id)?.(message);
  });
  async function evaluate(expression) {
    const key = ++id;
    const message = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(Error(`Browser timeout: ${expression.slice(0, 180)}`)), 60000);
      pending.set(key, answer => { clearTimeout(timer); resolve(answer); });
      socket.send(JSON.stringify({ id: key, method: 'Runtime.evaluate', params: { expression, awaitPromise: true, returnByValue: true } }));
    });
    pending.delete(key);
    if (message.error) throw Error(JSON.stringify(message.error));
    const result = message.result;
    if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
    return result.result.value;
  }
  for (let i = 0; i < 100; i++) {
    if (await evaluate('Boolean(window.capture)')) break;
    await delay(150);
  }
  assert(await evaluate('Boolean(window.capture)'), 'Spark did not initialize');
  let options = camera;
  let bounds;
  if (review) {
    bounds = await evaluate(`capture.bounds(${JSON.stringify(baseline)})`);
    const target = bounds.min.map((lo, i) => (lo + bounds.max[i]) / 2);
    const extent = Math.max(...bounds.max.map((hi, i) => hi - bounds.min[i]));
    options ??= { width: 640, height: 640, background: '#252028', target,
      pitch: 0, fov: 3, distance: extent * 1.16 / (2 * Math.tan(Math.PI / 120)), front: 0, side: 90 };
  }
  const out = new URL(`${destination}/`, root);
  await mkdir(out, { recursive: true });
  let pendingWrite = Promise.resolve();
  async function capture(paths, angle, filename, camOverride = null) {
    const cam = camOverride ? { ...options, ...camOverride } : options;
    const params = [paths, staticPaths, cam, camOverride?.front ?? angle].map(item => JSON.stringify(item));
    const png = await evaluate(`capture.draw(${params.join(',')})`);
    assert(png?.length > 4000, `Empty capture ${filename}`);
    await pendingWrite;
    pendingWrite = writeFile(new URL(filename, out), Buffer.from(png, 'base64'));
  }
  if (anim) {
    const info = await evaluate(
      `capture.initSkinned(${JSON.stringify(`/${animSpec.basePly}`)}, ${JSON.stringify(`/${animSpec.skinBin}`)}, ${JSON.stringify(animSpec.numBones)})`,
    );
    const fmt = animSpec.format || 'png';
    for (const frame of animSpec.frames) {
      const cam = frame.camera || animSpec.defaultCamera;
      const yaw = frame.yaw ?? cam.front;
      const extraPaths = (frame.extraPly || []).map(p => `/${p}`);
      const extraTransforms = frame.extraTransforms || null;
      const params = [frame.bones, extraPaths, extraTransforms, cam, yaw, fmt].map(item => JSON.stringify(item));
      const img = await evaluate(`capture.drawSkinned(${params.join(',')})`);
      assert(img?.length > 2000, `Empty anim capture ${frame.png}`);
      await pendingWrite;
      pendingWrite = writeFile(new URL(frame.png, out), Buffer.from(img, 'base64'));
    }
    await pendingWrite;
    await writeFile(new URL('manifest.json', out), JSON.stringify({ info, count: animSpec.frames.length }, null, 2));
  } else if (review) {
    for (const angle of [0, 45, 90, 135, 180, 225, 270, 315]) {
      await capture([baseline], options.front + angle, `angle-${String(angle).padStart(3, '0')}.png`);
    }
    await pendingWrite;
    await writeFile(new URL('camera.json', out), JSON.stringify({ ...options, bounds }, null, 2));
  } else if (orbit) {
    const duration = 4;
    const fps = 24;
    const count = duration * fps;
    const frames = [];
    for (let index = 0; index < count; index++) {
      const number = index + 1;
      const yaw = options.front + (360 * index) / count;
      const png = `frame-${String(number).padStart(4, '0')}.png`;
      await capture([baseline], yaw, png);
      frames.push({ index: number, seconds: index / fps, yaw, png });
    }
    await pendingWrite;
    const manifest = { camera: options, asset: baseline, fps, frames };
    await writeFile(new URL('manifest.json', out), JSON.stringify(manifest, null, 2));
  } else {
    for (const frame of sequence.frames) await capture(framePaths(frame), options.front, frame.png, frame.camera || null);
    const shots = { front: options.front, side: options.side, back: options.front + 180 };
    for (const frame of sequence.frames.filter(f => sequence.review.includes(f.blenderFrame))) {
      for (const [name, angle] of Object.entries(shots)) await capture(framePaths(frame), angle, `${name}-${frame.png}`);
    }
    await pendingWrite;
    await writeFile(new URL('manifest.json', out), JSON.stringify({ camera: options, ...sequence }, null, 2));
  }
  console.log(destination);
} finally {
  socket?.close();
  chrome.kill();
  server.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
