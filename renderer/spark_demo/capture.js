import * as THREE from 'three';
import { SparkRenderer, SplatMesh, SplatSkinning } from '@sparkjsdev/spark';

const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
document.body.append(renderer.domElement);
const spark = new SparkRenderer({ renderer, enableLod: false, sortRadial: false, autoUpdate: false });
scene.add(spark);
const root = new THREE.Group();
scene.add(root);
const loaded = new Map();
const camera = new THREE.PerspectiveCamera(20, 1, 0.01, 1000);
let warmedUp = false;

// Reusable scratch objects for GPU SplatSkinning
const tmpQ = new THREE.Quaternion();
const tmpP = new THREE.Vector3();
const idQ = new THREE.Quaternion(0, 0, 0, 1);
const zeroP = new THREE.Vector3(0, 0, 0);
let skinnedMesh = null;
let skinnedPath = null;
let skinning = null;

async function add(path) {
  if (!loaded.has(path)) {
    const model = new SplatMesh({ url: path, lod: false });
    await model.initialized;
    if (!model.numSplats) throw new Error(`No splats in ${path}`);
    root.add(model);
    loaded.set(path, model);
  }
  return loaded.get(path);
}

function setupCameraAndRenderTarget(options, yaw) {
  const { width, height, background, target, distance, pitch, fov } = options;
  if (renderer.domElement.width !== width || renderer.domElement.height !== height) {
    renderer.setSize(width, height);
    warmedUp = false;
  }
  renderer.setClearColor(background);
  root.rotation.x = Math.PI;
  root.position.set(-target[0], target[1], target[2]);
  root.updateMatrixWorld(true);
  camera.fov = fov;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  const az = THREE.MathUtils.degToRad(yaw);
  const el = THREE.MathUtils.degToRad(pitch);
  camera.position.set(
    distance * Math.sin(az) * Math.cos(el),
    distance * Math.sin(el),
    distance * Math.cos(az) * Math.cos(el),
  );
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
}

async function renderFrame(format = 'png') {
  if (!warmedUp) {
    renderer.render(scene, camera);
    warmedUp = true;
  }
  await spark.update({ scene, camera });
  renderer.render(scene, camera);
  const mime = format === 'jpeg' ? 'image/jpeg' : 'image/png';
  const dataUrl = format === 'jpeg'
    ? renderer.domElement.toDataURL(mime, 0.95)
    : renderer.domElement.toDataURL(mime);
  return dataUrl.split(',')[1];
}

window.capture = {
  async bounds(path) {
    const model = await add(path);
    const box = model.getBoundingBox(true);
    return { min: box.min.toArray(), max: box.max.toArray(), count: model.numSplats };
  },
  async draw(paths, staticPaths, options, yaw, format = 'png') {
    const wanted = new Set([...paths, ...staticPaths]);
    for (const [name, model] of loaded) {
      if (!wanted.has(name)) {
        root.remove(model);
        model.dispose();
        loaded.delete(name);
      }
    }
    for (const path of wanted) await add(path);
    setupCameraAndRenderTarget(options, yaw);
    return renderFrame(format);
  },
  async initSkinned(plyPath, skinBinPath, numBones) {
    if (skinnedMesh && skinnedPath === plyPath) {
      return { count: skinnedMesh.numSplats, numBones };
    }
    if (skinnedMesh) {
      root.remove(skinnedMesh);
      skinnedMesh.dispose();
      skinnedMesh = null;
      skinning = null;
    }
    const model = new SplatMesh({ url: plyPath, lod: false });
    await model.initialized;
    if (!model.numSplats) throw new Error(`No splats in ${plyPath}`);
    const resp = await fetch(skinBinPath);
    if (!resp.ok) throw new Error(`Failed to fetch ${skinBinPath}: ${resp.status}`);
    const buf = await resp.arrayBuffer();
    const skinArr = new Uint16Array(buf);
    if (skinArr.length < model.numSplats * 4) {
      throw new Error(`skinBin too small: ${skinArr.length} < ${model.numSplats * 4}`);
    }
    const sk = new SplatSkinning({ mesh: model, numSplats: model.numSplats, numBones, mode: 'dual_quaternion' });
    sk.skinData.set(skinArr.subarray(0, model.numSplats * 4));
    sk.skinTexture.needsUpdate = true;
    for (let b = 0; b < numBones; b++) {
      sk.setRestQuatPos(b, idQ, zeroP);
    }
    model.skinning = sk;
    model.updateGenerator();
    root.add(model);
    skinnedMesh = model;
    skinnedPath = plyPath;
    skinning = sk;
    warmedUp = false;
    return { count: model.numSplats, numBones };
  },
  async drawSkinned(bonesFlat, extraPaths, extraTransforms, options, yaw, format = 'png') {
    if (!skinning) throw new Error('initSkinned must be called before drawSkinned');
    const nb = Math.floor(bonesFlat.length / 7);
    for (let b = 0; b < nb; b++) {
      const o = b * 7;
      tmpQ.set(bonesFlat[o], bonesFlat[o + 1], bonesFlat[o + 2], bonesFlat[o + 3]);
      tmpP.set(bonesFlat[o + 4], bonesFlat[o + 5], bonesFlat[o + 6]);
      skinning.setBoneQuatPos(b, tmpQ, tmpP);
    }
    skinning.updateBones();

    const wanted = new Set(extraPaths || []);
    for (const [name, model] of loaded) {
      if (!wanted.has(name)) {
        root.remove(model);
        model.dispose();
        loaded.delete(name);
      }
    }
    const pathsArr = extraPaths || [];
    for (let i = 0; i < pathsArr.length; i++) {
      const model = await add(pathsArr[i]);
      const tr = extraTransforms && extraTransforms[i];
      if (tr) {
        model.quaternion.set(tr[0], tr[1], tr[2], tr[3]);
        model.position.set(tr[4], tr[5], tr[6]);
      } else {
        model.quaternion.set(0, 0, 0, 1);
        model.position.set(0, 0, 0);
      }
      model.updateMatrixWorld();
    }
    setupCameraAndRenderTarget(options, yaw);
    return renderFrame(format);
  },
};
