"use client";

import { ChangeEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from "three-mesh-bvh";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

type SpinPlane = "horizontal" | "vertical";
type SpinDirection = "ccw" | "cw";
type EditorTool = "orbit" | "armor-move" | "armor-rotate" | "spinner-move";
type Stage = "editor" | "running" | "result";

type WeaponConfig = {
  rpm: number;
  toothCount: number;
  toothWidth: number;
  radius: number;
  momentOfInertia: number;
  closingSpeed: number;
  moveDirection: number;
  phase: number;
  spinPlane: SpinPlane;
  spinDirection: SpinDirection;
  startX: number;
  startY: number;
  startZ: number;
};

type ArmorTransform = {
  x: number;
  y: number;
  z: number;
  rx: number;
  ry: number;
  rz: number;
};

type SimFrame = {
  t: number;
  center: THREE.Vector3;
  angle: number;
  tracerPointCount: number;
  pathLength: number;
};

type SimResult = {
  frames: SimFrame[];
  tracerPoints: THREE.Vector3[];
  pathLength: number;
  toothIndex: number | null;
  sampleIndex: number | null;
  hit: boolean;
  entry: THREE.Vector3 | null;
  exit: THREE.Vector3 | null;
  physicsDuration: number;
};

type ThreeState = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  composer: EffectComposer;
  orbit: OrbitControls;
  transform: TransformControls;
  transformHelper: THREE.Object3D;
  armorRoot: THREE.Group;
  armorMesh: THREE.Mesh;
  spinner: THREE.Group;
  directionArrow: THREE.ArrowHelper;
  tracerMesh: THREE.Mesh;
};

const DEFAULT_WEAPON: WeaponConfig = {
  rpm: 2400,
  toothCount: 2,
  toothWidth: 7,
  radius: 110,
  momentOfInertia: 0.045,
  closingSpeed: 3,
  moveDirection: 0,
  phase: 18,
  spinPlane: "horizontal",
  spinDirection: "ccw",
  startX: -230,
  startY: 10,
  startZ: 28,
};

const DEFAULT_ARMOR: ArmorTransform = { x: 20, y: 0, z: 0, rx: 0, ry: 0, rz: 0 };
// The tooth is treated as an effectively sharp blade at the configured
// weapon radius. Its glow supplies visibility without adding bulky geometry.
const TOOTH_ROD_RADIUS = 0.25;
const DEG = Math.PI / 180;

// Accelerate repeated inside/outside tests on uploaded meshes.
(THREE.Mesh.prototype as unknown as { raycast: typeof acceleratedRaycast }).raycast = acceleratedRaycast;
(THREE.BufferGeometry.prototype as unknown as { computeBoundsTree: typeof computeBoundsTree }).computeBoundsTree = computeBoundsTree;
(THREE.BufferGeometry.prototype as unknown as { disposeBoundsTree: typeof disposeBoundsTree }).disposeBoundsTree = disposeBoundsTree;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeStlUnitsToMillimeters(geometry: THREE.BufferGeometry) {
  geometry.computeBoundingBox();
  const rawSize = geometry.boundingBox?.getSize(new THREE.Vector3()) ?? new THREE.Vector3();
  const largestDimension = Math.max(rawSize.x, rawSize.y, rawSize.z);

  // STL is unitless. Combat-robot armor measuring less than 2 raw units across
  // is overwhelmingly likely to have been exported in meters (for example,
  // 0.2 means 200 mm), not to be a sub-2 mm armor plate.
  if (largestDimension > 0 && largestDimension < 2) {
    geometry.scale(1000, 1000, 1000);
    geometry.computeBoundingBox();
    return {
      size: geometry.boundingBox?.getSize(new THREE.Vector3()) ?? rawSize.multiplyScalar(1000),
      note: "meters detected · converted to mm ×1000",
    };
  }

  return { size: rawSize, note: "STL units treated as mm" };
}

function makeDemoPlateGeometry() {
  // A representative wedge-style armor plate. The spinner approaches along
  // +X, so the leading edge is low and the plate rises as X increases.
  const thickness = 30;
  const width = 250;
  const slopeLength = 100;
  const incline = -45 * DEG;
  const geometry = new THREE.BoxGeometry(slopeLength, width, thickness);
  geometry.rotateY(incline);

  // Rest the low, spinner-facing edge on the XY floor.
  geometry.computeBoundingBox();
  geometry.translate(0, 0, -(geometry.boundingBox?.min.z ?? 0));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return geometry;
}

function baseQuaternion(config: WeaponConfig) {
  if (config.spinPlane === "horizontal") return new THREE.Quaternion();
  const heading = config.moveDirection * DEG;
  const xAxis = new THREE.Vector3(Math.cos(heading), Math.sin(heading), 0).normalize();
  const yAxis = new THREE.Vector3(0, 0, 1);
  const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize();
  const matrix = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
  return new THREE.Quaternion().setFromRotationMatrix(matrix);
}

function toothPoint(
  config: WeaponConfig,
  center: THREE.Vector3,
  angle: number,
  toothIndex: number,
  rodOffset: number,
) {
  const theta = angle + (toothIndex * Math.PI * 2) / config.toothCount;
  return new THREE.Vector3(
    config.radius * Math.cos(theta),
    config.radius * Math.sin(theta),
    rodOffset,
  )
    .applyQuaternion(baseQuaternion(config))
    .add(center);
}

function segmentSurfaceHit(mesh: THREE.Mesh, from: THREE.Vector3, to: THREE.Vector3) {
  const delta = to.clone().sub(from);
  const length = delta.length();
  if (length < 1e-7) return null;
  const raycaster = new THREE.Raycaster(from, delta.normalize(), 0, length + 0.001);
  (raycaster as THREE.Raycaster & { firstHitOnly?: boolean }).firstHitOnly = true;
  const hit = raycaster.intersectObject(mesh, false)[0];
  return hit && hit.distance <= length + 0.001 ? hit.point.clone() : null;
}

function pointInsideMesh(mesh: THREE.Mesh, point: THREE.Vector3, worldBox: THREE.Box3) {
  if (!worldBox.containsPoint(point)) return false;
  const direction = new THREE.Vector3(1, 0.37139, 0.21713).normalize();
  const origin = point.clone().addScaledVector(direction, 0.0001);
  const raycaster = new THREE.Raycaster(origin, direction, 0, 100000);
  (raycaster as THREE.Raycaster & { firstHitOnly?: boolean }).firstHitOnly = false;
  const hits = raycaster.intersectObject(mesh, false);
  if (!hits.length) return false;
  let unique = 0;
  let last = -Infinity;
  for (const hit of hits) {
    if (Math.abs(hit.distance - last) > 0.001) {
      unique += 1;
      last = hit.distance;
    }
  }
  return unique % 2 === 1;
}

function toothOverlapSegments(
  config: WeaponConfig,
  mesh: THREE.Mesh,
  worldBox: THREE.Box3,
  center: THREE.Vector3,
  angle: number,
  toothIndex: number,
) {
  const denseOffsets = Array.from(
    { length: 17 },
    (_, index) => -config.toothWidth / 2 + (config.toothWidth * index) / 16,
  );
  const engagementOffsets = Array.from(
    { length: 7 },
    (_, index) => -config.toothWidth / 2 + (config.toothWidth * index) / 6,
  );
  const offsets = [...denseOffsets, ...engagementOffsets]
    .sort((a, b) => a - b)
    .filter((offset, index, all) => index === 0 || Math.abs(offset - all[index - 1]) > 1e-6);
  const sampleCount = offsets.length;
  const inside = offsets.map((offset) =>
    pointInsideMesh(mesh, toothPoint(config, center, angle, toothIndex, offset), worldBox),
  );

  const boundary = (outsideOffset: number, insideOffset: number) => {
    let outside = outsideOffset;
    let inner = insideOffset;
    for (let iteration = 0; iteration < 6; iteration += 1) {
      const midpoint = (outside + inner) / 2;
      const point = toothPoint(config, center, angle, toothIndex, midpoint);
      if (pointInsideMesh(mesh, point, worldBox)) inner = midpoint;
      else outside = midpoint;
    }
    return (outside + inner) / 2;
  };

  const segments: THREE.Vector3[] = [];
  let startOffset: number | null = null;
  for (let index = 0; index < sampleCount; index += 1) {
    if (inside[index] && startOffset === null) {
      startOffset = index === 0
        ? offsets[index]
        : boundary(offsets[index - 1], offsets[index]);
    }
    if (inside[index] && (index === sampleCount - 1 || !inside[index + 1])) {
      const endOffset = index === sampleCount - 1
        ? offsets[index]
        : boundary(offsets[index + 1], offsets[index]);
      if (startOffset !== null && endOffset - startOffset > 0.01) {
        segments.push(
          toothPoint(config, center, angle, toothIndex, startOffset),
          toothPoint(config, center, angle, toothIndex, endOffset),
        );
      }
      startOffset = null;
    }
  }
  return segments;
}

function buildTracerGeometry(points: THREE.Vector3[]) {
  const positions: number[] = [];
  const indices: number[] = [];
  const halfThickness = 1.2;

  for (let pointIndex = 0; pointIndex + 1 < points.length; pointIndex += 2) {
    const start = points[pointIndex];
    const end = points[pointIndex + 1];
    const direction = end.clone().sub(start);
    if (direction.lengthSq() < 1e-6) continue;
    direction.normalize();
    const reference = Math.abs(direction.z) < 0.9
      ? new THREE.Vector3(0, 0, 1)
      : new THREE.Vector3(0, 1, 0);
    const side = new THREE.Vector3().crossVectors(direction, reference).normalize().multiplyScalar(halfThickness);
    const up = new THREE.Vector3().crossVectors(direction, side).normalize().multiplyScalar(halfThickness);
    const corners = [
      start.clone().sub(side).sub(up),
      start.clone().add(side).sub(up),
      start.clone().add(side).add(up),
      start.clone().sub(side).add(up),
      end.clone().sub(side).sub(up),
      end.clone().add(side).sub(up),
      end.clone().add(side).add(up),
      end.clone().sub(side).add(up),
    ];
    const base = positions.length / 3;
    corners.forEach((corner) => positions.push(corner.x, corner.y, corner.z));
    indices.push(
      base, base + 2, base + 1, base, base + 3, base + 2,
      base + 4, base + 5, base + 6, base + 4, base + 6, base + 7,
      base, base + 1, base + 5, base, base + 5, base + 4,
      base + 1, base + 2, base + 6, base + 1, base + 6, base + 5,
      base + 2, base + 3, base + 7, base + 2, base + 7, base + 6,
      base + 3, base, base + 4, base + 3, base + 4, base + 7,
    );
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  if (positions.length) {
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
  }
  return geometry;
}

function createSpinner(config: WeaponConfig) {
  const group = new THREE.Group();
  group.name = "spinner";

  const hubMaterial = new THREE.MeshStandardMaterial({
    color: 0x66ffbd,
    emissive: 0x15ff8a,
    emissiveIntensity: 3.2,
    metalness: 0.45,
    roughness: 0.22,
  });
  hubMaterial.fog = false;
  const ghostMaterial = new THREE.MeshBasicMaterial({
    color: 0x21ff9a,
    transparent: true,
    opacity: 0.34,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  ghostMaterial.fog = false;
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(8, 8, 12, 24), hubMaterial);
  hub.rotation.x = Math.PI / 2;
  group.add(hub);

  for (let i = 0; i < config.toothCount; i += 1) {
    const theta = (i * Math.PI * 2) / config.toothCount;
    const toothGroup = new THREE.Group();
    toothGroup.rotation.z = theta;

    const arm = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(1, config.radius - 8), 2.2, 2.2),
      ghostMaterial,
    );
    arm.position.x = (config.radius + 8) / 2;
    toothGroup.add(arm);

    const rod = new THREE.Mesh(
      new THREE.BoxGeometry(TOOTH_ROD_RADIUS * 2, TOOTH_ROD_RADIUS * 2, config.toothWidth),
      hubMaterial,
    );
    rod.position.x = config.radius;
    toothGroup.add(rod);

    const halo = new THREE.Mesh(
      new THREE.BoxGeometry(
        TOOTH_ROD_RADIUS * 3.2,
        TOOTH_ROD_RADIUS * 3.2,
        config.toothWidth + 3,
      ),
      ghostMaterial,
    );
    halo.position.x = config.radius;
    toothGroup.add(halo);
    group.add(toothGroup);
  }

  return group;
}

function createDirectionArrow(config: WeaponConfig) {
  const heading = config.moveDirection * DEG;
  const direction = new THREE.Vector3(Math.cos(heading), Math.sin(heading), 0).normalize();
  const arrow = new THREE.ArrowHelper(
    direction,
    new THREE.Vector3(config.startX, config.startY, config.startZ),
    Math.max(80, config.radius * 0.9),
    0x35ffbd,
    22,
    12,
  );
  arrow.name = "travel-direction";
  arrow.line.material.fog = false;
  arrow.cone.material.fog = false;
  return arrow;
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments || child instanceof THREE.Line) {
      child.geometry?.dispose();
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => material?.dispose());
    }
  });
}

function frameObjects(state: ThreeState, objects: THREE.Object3D[]) {
  const box = new THREE.Box3();
  objects.forEach((object) => {
    object.updateMatrixWorld(true);
    box.union(new THREE.Box3().setFromObject(object));
  });
  if (box.isEmpty()) return;

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 10);
  const verticalHalfFov = THREE.MathUtils.degToRad(state.camera.fov * 0.5);
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * state.camera.aspect);
  const limitingHalfFov = Math.min(verticalHalfFov, horizontalHalfFov);
  const distance = (radius / Math.sin(limitingHalfFov)) * 1.12;
  const viewDirection = state.camera.position
    .clone()
    .sub(state.orbit.target)
    .normalize();

  state.orbit.target.copy(center);
  state.camera.position.copy(center).addScaledVector(viewDirection, distance);
  state.camera.near = Math.max(0.01, distance / 1000);
  state.camera.far = Math.max(5000, distance * 30);
  // Fog is decorative, so scale it with the current shot instead of letting a
  // physically larger STL disappear into the background when the camera backs up.
  if (state.scene.fog instanceof THREE.FogExp2) {
    state.scene.fog.density = clamp(0.7 / distance, 0.00004, 0.0014);
  }
  state.camera.updateProjectionMatrix();
  state.orbit.update();
}

function buildArmorObject(geometry: THREE.BufferGeometry) {
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  (geometry as THREE.BufferGeometry & { computeBoundsTree?: () => void }).computeBoundsTree?.();

  const root = new THREE.Group();
  root.name = "armor";
  const bodyMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x087dff,
    emissive: 0x004dff,
    emissiveIntensity: 0.7,
    transparent: true,
    opacity: 0.12,
    roughness: 0.08,
    metalness: 0.18,
    transmission: 0.08,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  bodyMaterial.fog = false;
  const mesh = new THREE.Mesh(geometry, bodyMaterial);
  mesh.name = "armor-solid";
  root.add(mesh);

  const wireMaterial = new THREE.MeshBasicMaterial({
    color: 0x0b78ff,
    wireframe: true,
    transparent: true,
    opacity: 0.2,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  wireMaterial.fog = false;
  const triangleMesh = new THREE.Mesh(geometry, wireMaterial);
  root.add(triangleMesh);

  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0x36c8ff,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
  });
  edgeMaterial.fog = false;
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry, 18),
    edgeMaterial,
  );
  root.add(edges);
  return { root, mesh };
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit,
  integer = false,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  integer?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (inputRef.current) inputRef.current.value = String(value);
  }, [value]);

  const commit = (input: HTMLInputElement) => {
    const parsed = Number(input.value);
    if (!Number.isFinite(parsed) || input.value.trim() === "") {
      input.value = String(value);
      return;
    }
    let next = integer ? Math.round(parsed) : parsed;
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    input.value = String(next);
    onChange(next);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commit(event.currentTarget);
    } else if (event.key === "Escape") {
      event.currentTarget.value = String(value);
      event.currentTarget.blur();
    }
  };

  return (
    <label className="number-field">
      <span>{label}</span>
      <span className="number-input-wrap">
        <input
          type="number"
          ref={inputRef}
          defaultValue={value}
          min={min}
          max={max}
          step={step}
          title="Press Enter to apply"
          onKeyDown={handleKeyDown}
          onBlur={(event) => { event.currentTarget.value = String(value); }}
        />
        {unit && <b>{unit}</b>}
      </span>
    </label>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`metric ${accent ? "metric-accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function Home() {
  const [config, setConfig] = useState<WeaponConfig>(DEFAULT_WEAPON);
  const [armorTransform, setArmorTransform] = useState<ArmorTransform>(DEFAULT_ARMOR);
  const [stage, setStage] = useState<Stage>("editor");
  const [editorTool, setEditorTool] = useState<EditorTool>("orbit");
  const [armorName, setArmorName] = useState("Demo wedge plate");
  const [meshStatus, setMeshStatus] = useState("Demo solid loaded");
  const [livePathLength, setLivePathLength] = useState(0);
  const [result, setResult] = useState<SimResult | null>(null);
  const [status, setStatus] = useState("Position the scene, then arm the simulation.");

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const threeRef = useRef<ThreeState | null>(null);
  const configRef = useRef(config);
  const stageRef = useRef(stage);
  const resultRef = useRef<SimResult | null>(null);
  const playbackRef = useRef<{ active: boolean; started: number; duration: number; lastIndex: number }>({
    active: false,
    started: 0,
    duration: 4.8,
    lastIndex: -1,
  });

  useEffect(() => { configRef.current = config; }, [config]);
  useEffect(() => { stageRef.current = stage; }, [stage]);
  useEffect(() => { resultRef.current = result; }, [result]);

  const derived = useMemo(() => {
    const omega = (config.rpm * Math.PI * 2) / 60;
    const tipMps = (config.radius / 1000) * omega;
    const energy = 0.5 * config.momentOfInertia * omega * omega;
    const biteMm = config.rpm > 0 && config.toothCount > 0
      ? (config.closingSpeed * 1000 * 60) / (config.rpm * config.toothCount)
      : 0;
    return {
      omega,
      tipMps,
      energy,
      biteMm,
    };
  }, [config]);

  const patchConfig = useCallback(<K extends keyof WeaponConfig>(key: K, value: WeaponConfig[K]) => {
    setConfig((previous) => ({ ...previous, [key]: value }));
  }, []);

  const patchArmor = useCallback(<K extends keyof ArmorTransform>(key: K, value: ArmorTransform[K]) => {
    setArmorTransform((previous) => ({ ...previous, [key]: value }));
  }, []);

  const applySpinnerPose = useCallback((center: THREE.Vector3, angle: number) => {
    const state = threeRef.current;
    if (!state) return;
    state.spinner.position.copy(center);
    const spin = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle);
    state.spinner.quaternion.copy(baseQuaternion(configRef.current)).multiply(spin);
  }, []);

  const setTracerGeometry = useCallback((points: THREE.Vector3[]) => {
    const state = threeRef.current;
    if (!state) return;
    state.tracerMesh.geometry.dispose();
    state.tracerMesh.geometry = buildTracerGeometry(points);
    state.tracerMesh.visible = points.length > 1;
  }, []);

  useEffect(() => {
    const host = viewportRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070315);
    scene.fog = new THREE.FogExp2(0x08021c, 0.0018);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 5000);
    camera.position.set(330, 310, 250);
    camera.up.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    host.appendChild(renderer.domElement);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(1, 1), 1.28, 0.72, 0.18));

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.065;
    orbit.target.set(0, 0, 28);

    const grid = new THREE.GridHelper(900, 36, 0x7f32ff, 0x21448b);
    grid.rotation.x = Math.PI / 2;
    grid.material.transparent = true;
    grid.material.opacity = 0.22;
    scene.add(grid);

    const starGeometry = new THREE.BufferGeometry();
    const stars = new Float32Array(420 * 3);
    for (let i = 0; i < 420; i += 1) {
      stars[i * 3] = (Math.random() - 0.5) * 1300;
      stars[i * 3 + 1] = (Math.random() - 0.5) * 1300;
      stars[i * 3 + 2] = Math.random() * 500 + 20;
    }
    starGeometry.setAttribute("position", new THREE.BufferAttribute(stars, 3));
    const starMaterial = new THREE.PointsMaterial({
      color: 0xa86cff,
      size: 1.25,
      transparent: true,
      opacity: 0.55,
      depthTest: false,
      depthWrite: false,
      fog: false,
    });
    const starField = new THREE.Points(starGeometry, starMaterial);
    starField.renderOrder = -100;
    scene.add(starField);

    scene.add(new THREE.AmbientLight(0x577dff, 1.2));
    const key = new THREE.DirectionalLight(0x45e9ff, 3.4);
    key.position.set(-180, -120, 280);
    scene.add(key);
    const rim = new THREE.PointLight(0xb12eff, 450, 700);
    rim.position.set(120, 180, 180);
    scene.add(rim);

    const armorBuilt = buildArmorObject(makeDemoPlateGeometry());
    scene.add(armorBuilt.root);
    const spinner = createSpinner(configRef.current);
    scene.add(spinner);
    const directionArrow = createDirectionArrow(configRef.current);
    scene.add(directionArrow);

    const tracerMaterial = new THREE.MeshBasicMaterial({
      color: 0x35ff92,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    tracerMaterial.fog = false;
    const tracerMesh = new THREE.Mesh(
      new THREE.BufferGeometry(),
      tracerMaterial,
    );
    tracerMesh.visible = false;
    tracerMesh.renderOrder = 20;
    scene.add(tracerMesh);

    const transform = new TransformControls(camera, renderer.domElement);
    transform.setSize(0.82);
    const transformHelper = transform.getHelper();
    transformHelper.visible = false;
    scene.add(transformHelper);
    transform.addEventListener("dragging-changed", (event) => {
      orbit.enabled = !(event.value as boolean);
    });
    transform.addEventListener("objectChange", () => {
      const object = transform.object;
      if (!object || stageRef.current !== "editor") return;
      if (object.name === "armor") {
        setArmorTransform({
          x: Number(object.position.x.toFixed(1)),
          y: Number(object.position.y.toFixed(1)),
          z: Number(object.position.z.toFixed(1)),
          rx: Number((object.rotation.x / DEG).toFixed(1)),
          ry: Number((object.rotation.y / DEG).toFixed(1)),
          rz: Number((object.rotation.z / DEG).toFixed(1)),
        });
      }
    });
    transform.addEventListener("mouseUp", () => {
      const object = transform.object;
      if (object?.name === "spinner" && stageRef.current === "editor") {
        setConfig((previous) => ({
          ...previous,
          startX: Number(object.position.x.toFixed(1)),
          startY: Number(object.position.y.toFixed(1)),
          startZ: Number(object.position.z.toFixed(1)),
        }));
      }
    });

    const state: ThreeState = {
      scene,
      camera,
      renderer,
      composer,
      orbit,
      transform,
      transformHelper,
      armorRoot: armorBuilt.root,
      armorMesh: armorBuilt.mesh,
      spinner,
      directionArrow,
      tracerMesh,
    };
    threeRef.current = state;

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    let raf = 0;
    const animate = (now: number) => {
      raf = requestAnimationFrame(animate);
      orbit.update();

      const playback = playbackRef.current;
      const activeResult = resultRef.current;
      if (playback.active && activeResult?.frames.length) {
        const progress = clamp((now - playback.started) / (playback.duration * 1000), 0, 1);
        const index = Math.min(activeResult.frames.length - 1, Math.floor(progress * (activeResult.frames.length - 1)));
        const frame = activeResult.frames[index];
        if (playback.lastIndex !== index) {
          playback.lastIndex = index;
          applySpinnerPose(frame.center, frame.angle);
          setTracerGeometry(activeResult.tracerPoints.slice(0, frame.tracerPointCount));
          setLivePathLength(frame.pathLength);
        }
        if (progress >= 1) {
          playback.active = false;
          setStage("result");
          setLivePathLength(activeResult.pathLength);
          setTracerGeometry(activeResult.tracerPoints);
          state.spinner.visible = false;
          setStatus(activeResult.hit
            ? `Tooth ${Number(activeResult.toothIndex) + 1} cleared the armor. Path locked.`
            : "No tooth intersected the armor in this pass.");
        }
      }

      composer.render();
    };
    raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      transform.dispose();
      orbit.dispose();
      composer.dispose();
      renderer.dispose();
      disposeObject(scene);
      host.replaceChildren();
      threeRef.current = null;
    };
  }, [applySpinnerPose, setTracerGeometry]);

  useEffect(() => {
    const state = threeRef.current;
    if (!state) return;
    state.directionArrow.visible = stage === "editor";
    if (stage !== "editor") return;
    state.scene.remove(state.spinner);
    disposeObject(state.spinner);
    const spinner = createSpinner(config);
    state.spinner = spinner;
    state.scene.add(spinner);
    applySpinnerPose(
      new THREE.Vector3(config.startX, config.startY, config.startZ),
      config.phase * DEG,
    );
    const heading = config.moveDirection * DEG;
    state.directionArrow.position.set(config.startX, config.startY, config.startZ);
    state.directionArrow.setDirection(new THREE.Vector3(Math.cos(heading), Math.sin(heading), 0).normalize());
    state.directionArrow.setLength(Math.max(80, config.radius * 0.9), 22, 12);
    state.directionArrow.visible = stage === "editor";
    setTracerGeometry([]);
  }, [config, stage, applySpinnerPose, setTracerGeometry]);

  useEffect(() => {
    const state = threeRef.current;
    if (!state) return;
    state.armorRoot.position.set(armorTransform.x, armorTransform.y, armorTransform.z);
    state.armorRoot.rotation.set(armorTransform.rx * DEG, armorTransform.ry * DEG, armorTransform.rz * DEG);
    state.armorRoot.updateMatrixWorld(true);
  }, [armorTransform]);

  useEffect(() => {
    const state = threeRef.current;
    if (!state) return;
    state.transform.detach();
    state.transformHelper.visible = false;
    state.orbit.enabled = true;
    if (stage !== "editor" || editorTool === "orbit") return;
    if (editorTool === "armor-move") {
      state.transform.setMode("translate");
      state.transform.attach(state.armorRoot);
    } else if (editorTool === "armor-rotate") {
      state.transform.setMode("rotate");
      state.transform.attach(state.armorRoot);
    } else {
      state.transform.setMode("translate");
      state.transform.attach(state.spinner);
    }
    state.transformHelper.visible = true;
  }, [editorTool, stage]);

  const replaceArmorGeometry = useCallback((geometry: THREE.BufferGeometry, name: string, statusText: string) => {
    const state = threeRef.current;
    if (!state) return;
    geometry.computeBoundingBox();
    const bounds = geometry.boundingBox;
    if (!bounds || bounds.isEmpty()) throw new Error("Empty STL bounds");
    const center = bounds.getCenter(new THREE.Vector3());
    // STL coordinates are millimeters. Center the part in XY and place its
    // lowest point on the Z=0 floor instead of burying half of it below it.
    geometry.translate(-center.x, -center.y, -bounds.min.z);
    geometry.computeBoundingBox();
    const oldRoot = state.armorRoot;
    state.scene.remove(oldRoot);
    disposeObject(oldRoot);
    const built = buildArmorObject(geometry);
    built.root.position.set(DEFAULT_ARMOR.x, DEFAULT_ARMOR.y, DEFAULT_ARMOR.z);
    state.armorRoot = built.root;
    state.armorMesh = built.mesh;
    state.scene.add(built.root);
    setArmorTransform(DEFAULT_ARMOR);
    setArmorName(name);
    setMeshStatus(statusText);
    setEditorTool("orbit");
    // Keep the configured spinner in shot as well as the uploaded armor. The
    // two objects retain their real millimeter scale and world coordinates.
    frameObjects(state, [built.root, state.spinner, state.directionArrow]);
    setStatus("Armor placed on the XY floor and framed. Position it with the gizmo or coordinate controls.");
  }, []);

  const uploadStl = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setMeshStatus("Reading mesh…");
      const buffer = await file.arrayBuffer();
      const geometry = new STLLoader().parse(buffer);
      const positions = geometry.getAttribute("position");
      const triangles = Math.floor((positions?.count ?? 0) / 3);
      if (!triangles) throw new Error("No triangles found");
      for (let index = 0; index < positions.count; index += 1) {
        if (!Number.isFinite(positions.getX(index)) || !Number.isFinite(positions.getY(index)) || !Number.isFinite(positions.getZ(index))) {
          throw new Error("Invalid vertex coordinates");
        }
      }
      const normalized = normalizeStlUnitsToMillimeters(geometry);
      const dimensions = normalized.size;
      const sizeLabel = `${dimensions.x.toFixed(1)} × ${dimensions.y.toFixed(1)} × ${dimensions.z.toFixed(1)} mm`;
      replaceArmorGeometry(
        geometry,
        file.name,
        `${triangles.toLocaleString()} triangles · ${sizeLabel} · ${normalized.note}`,
      );
    } catch {
      setMeshStatus("Could not read this STL");
      setStatus("Try a binary or ASCII STL containing a closed solid.");
    } finally {
      event.target.value = "";
    }
  }, [replaceArmorGeometry]);

  const restoreDemo = useCallback(() => {
    replaceArmorGeometry(makeDemoPlateGeometry(), "Demo wedge plate", "250 mm wide × 30 mm thick · 45° half-height ramp");
  }, [replaceArmorGeometry]);

  const frameScene = useCallback(() => {
    const state = threeRef.current;
    if (!state) return;
    frameObjects(state, [state.armorRoot, state.spinner, state.directionArrow]);
    setStatus("Camera framed around both the armor and spinner.");
  }, []);

  const calculateSimulation = useCallback((): SimResult | null => {
    const state = threeRef.current;
    if (!state || config.rpm <= 0 || config.closingSpeed <= 0 || config.radius <= 0) return null;

    state.transform.detach();
    state.transformHelper.visible = false;
    state.armorRoot.updateMatrixWorld(true);
    state.armorMesh.updateMatrixWorld(true);
    const worldBox = new THREE.Box3().setFromObject(state.armorMesh).expandByScalar(0.001);
    const heading = config.moveDirection * DEG;
    const direction = new THREE.Vector3(Math.cos(heading), Math.sin(heading), 0).normalize();
    const start = new THREE.Vector3(config.startX, config.startY, config.startZ);
    const speed = config.closingSpeed * 1000;
    const omega = (config.rpm * Math.PI * 2) / 60 * (config.spinDirection === "ccw" ? 1 : -1);
    const phase = config.phase * DEG;
    const corners = [
      new THREE.Vector3(worldBox.min.x, worldBox.min.y, worldBox.min.z),
      new THREE.Vector3(worldBox.min.x, worldBox.min.y, worldBox.max.z),
      new THREE.Vector3(worldBox.min.x, worldBox.max.y, worldBox.min.z),
      new THREE.Vector3(worldBox.min.x, worldBox.max.y, worldBox.max.z),
      new THREE.Vector3(worldBox.max.x, worldBox.min.y, worldBox.min.z),
      new THREE.Vector3(worldBox.max.x, worldBox.min.y, worldBox.max.z),
      new THREE.Vector3(worldBox.max.x, worldBox.max.y, worldBox.min.z),
      new THREE.Vector3(worldBox.max.x, worldBox.max.y, worldBox.max.z),
    ];
    const maxProjection = Math.max(...corners.map((corner) => corner.dot(direction)));
    const requiredTravel = Math.max(280, maxProjection - start.dot(direction) + config.radius + 140);
    const endTime = clamp(requiredTravel / speed, 0.08, 4);
    const tipSpeedMm = Math.abs(omega) * config.radius;
    let dt = Math.min(1 / 720, 0.8 / Math.max(1, tipSpeedMm + speed));
    const maxSteps = 32000;
    if (endTime / dt > maxSteps) dt = endTime / maxSteps;
    const steps = Math.ceil(endTime / dt);
    const frameStride = Math.max(1, Math.floor(steps / 850));
    const visualStride = Math.max(1, Math.floor(steps / 1200));
    const rodSamples = Array.from({ length: 7 }, (_, index) =>
      -config.toothWidth / 2 + (config.toothWidth * index) / 6,
    );
    const previousPoints: THREE.Vector3[][] = Array.from({ length: config.toothCount }, () =>
      rodSamples.map(() => new THREE.Vector3()),
    );
    const previousInside: boolean[][] = Array.from({ length: config.toothCount }, () =>
      rodSamples.map(() => false),
    );

    const frames: SimFrame[] = [];
    const tracerPoints: THREE.Vector3[] = [];
    let pathLength = 0;
    let trackedTooth: number | null = null;
    let trackedSample: number | null = null;
    let entry: THREE.Vector3 | null = null;
    let exit: THREE.Vector3 | null = null;
    let lastTrackedPoint: THREE.Vector3 | null = null;
    let lastTrackedInside = false;
    let finished = false;
    let finalTime = endTime;

    for (let step = 0; step <= steps && !finished; step += 1) {
      const t = Math.min(endTime, step * dt);
      const center = start.clone().addScaledVector(direction, speed * t);
      const angle = phase + omega * t;

      if (trackedTooth === null) {
        let earliest: { tooth: number; sample: number; point: THREE.Vector3; current: THREE.Vector3; score: number } | null = null;
        for (let tooth = 0; tooth < config.toothCount; tooth += 1) {
          for (let sample = 0; sample < rodSamples.length; sample += 1) {
            const point = toothPoint(config, center, angle, tooth, rodSamples[sample]);
            const inside = pointInsideMesh(state.armorMesh, point, worldBox);
            if (step > 0 && inside && !previousInside[tooth][sample]) {
              const hit = segmentSurfaceHit(state.armorMesh, previousPoints[tooth][sample], point) ?? point.clone();
              const segmentLength = Math.max(1e-9, previousPoints[tooth][sample].distanceTo(point));
              const score = previousPoints[tooth][sample].distanceTo(hit) / segmentLength;
              if (!earliest || score < earliest.score) earliest = { tooth, sample, point: hit, current: point.clone(), score };
            }
            previousPoints[tooth][sample].copy(point);
            previousInside[tooth][sample] = inside;
          }
        }
        if (earliest) {
          trackedTooth = earliest.tooth;
          trackedSample = earliest.sample;
          entry = earliest.point.clone();
          lastTrackedPoint = earliest.current.clone();
          pathLength += entry.distanceTo(lastTrackedPoint);
          lastTrackedInside = true;
          tracerPoints.push(...toothOverlapSegments(
            config,
            state.armorMesh,
            worldBox,
            center,
            angle,
            trackedTooth,
          ));
        }
      } else if (trackedSample !== null && lastTrackedPoint) {
        const current = toothPoint(config, center, angle, trackedTooth, rodSamples[trackedSample]);
        const inside = pointInsideMesh(state.armorMesh, current, worldBox);
        if (lastTrackedInside && inside) {
          pathLength += lastTrackedPoint.distanceTo(current);
        } else if (lastTrackedInside && !inside) {
          const surfaceExit = segmentSurfaceHit(state.armorMesh, lastTrackedPoint, current) ?? current.clone();
          pathLength += lastTrackedPoint.distanceTo(surfaceExit);
          exit = surfaceExit;
          finished = true;
          finalTime = t;
        }
        if (step % visualStride === 0 || finished) {
          tracerPoints.push(...toothOverlapSegments(
            config,
            state.armorMesh,
            worldBox,
            center,
            angle,
            trackedTooth,
          ));
        }
        lastTrackedPoint.copy(current);
        lastTrackedInside = inside;
      }

      if (step % frameStride === 0 || trackedTooth !== null || finished || step === steps) {
        const previousFrame = frames[frames.length - 1];
        if (!previousFrame || t - previousFrame.t >= endTime / 900 || finished) {
          frames.push({
            t,
            center: center.clone(),
            angle,
            tracerPointCount: tracerPoints.length,
            pathLength,
          });
        }
      }
    }

    if (!frames.length || frames[frames.length - 1].t < finalTime) {
      const center = start.clone().addScaledVector(direction, speed * finalTime);
      frames.push({
        t: finalTime,
        center,
        angle: phase + omega * finalTime,
        tracerPointCount: tracerPoints.length,
        pathLength,
      });
    }

    return {
      frames,
      tracerPoints,
      pathLength,
      toothIndex: trackedTooth,
      sampleIndex: trackedSample,
      hit: Boolean(entry && exit),
      entry,
      exit,
      physicsDuration: finalTime,
    };
  }, [config]);

  const runSimulation = useCallback(() => {
    setStatus("Scanning every tooth for first entry…");
    setLivePathLength(0);
    setTracerGeometry([]);
    const state = threeRef.current;
    if (state) state.spinner.visible = true;
    const nextResult = calculateSimulation();
    if (!nextResult) {
      setStatus("Enter a positive RPM, radius, and closing speed.");
      return;
    }
    setResult(nextResult);
    resultRef.current = nextResult;
    setStage("running");
    const duration = clamp(3.6 + nextResult.physicsDuration * 5, 3.8, 7.2);
    playbackRef.current = { active: true, started: performance.now(), duration, lastIndex: -1 };
    setStatus(nextResult.hit ? "First tooth acquired. Tracking through armor…" : "Running geometric pass…");
  }, [calculateSimulation, setTracerGeometry]);

  const replay = useCallback(() => {
    if (!result) return;
    setTracerGeometry([]);
    const state = threeRef.current;
    if (state) state.spinner.visible = true;
    setLivePathLength(0);
    setStage("running");
    playbackRef.current = {
      active: true,
      started: performance.now(),
      duration: clamp(3.6 + result.physicsDuration * 5, 3.8, 7.2),
      lastIndex: -1,
    };
    setStatus("Replaying first-tooth engagement…");
  }, [result, setTracerGeometry]);

  const returnToEditor = useCallback(() => {
    playbackRef.current.active = false;
    setStage("editor");
    setResult(null);
    setLivePathLength(0);
    setTracerGeometry([]);
    const state = threeRef.current;
    if (state) state.spinner.visible = true;
    setStatus("Scenario unlocked. Adjust the geometry and run again.");
  }, [setTracerGeometry]);

  return (
    <main className="app-shell">
      <div className="scanlines" aria-hidden="true" />
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><span /><span /></div>
          <div>
            <p>CR // BITE LAB</p>
            <h1>Combat Robot Bite Visualizer</h1>
          </div>
        </div>
        <div className="topbar-status">
          <span className={`status-dot ${stage === "running" ? "is-live" : ""}`} />
          {stage === "editor" ? "SCENARIO EDITOR" : stage === "running" ? "SIMULATION LIVE" : "PATH CAPTURED"}
        </div>
      </header>

      <section className="metric-rail" aria-label="Live calculated metrics">
        <Metric label="RPM" value={Math.round(config.rpm).toLocaleString()} />
        <Metric label="TIP SPEED" value={`${derived.tipMps.toFixed(1)} m/s`} />
        <Metric label="CLOSING SPEED" value={`${config.closingSpeed.toFixed(1)} m/s`} />
        <Metric label="STORED ENERGY" value={`${derived.energy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J`} />
        <Metric label="THEORETICAL MAXIMUM BITE DEPTH" value={`${derived.biteMm.toFixed(1)} mm`} />
        <Metric label="IMPACT PATH" value={`${livePathLength.toFixed(1)} mm`} accent />
      </section>

      <div className="workspace">
        <aside className={`control-panel left-panel ${stage !== "editor" ? "panel-locked" : ""}`}>
          <div className="panel-heading">
            <span>01</span>
            <div><p>ARMOR TARGET</p><h2>Mesh setup</h2></div>
          </div>

          <button className="upload-card" onClick={() => fileRef.current?.click()} disabled={stage !== "editor"}>
            <span className="upload-icon">↥</span>
            <span><strong>Load armor STL</strong><small>{armorName}</small></span>
          </button>
          <input ref={fileRef} type="file" accept=".stl" hidden onChange={uploadStl} />
          <div className="mesh-status"><i /> {meshStatus}</div>
          <div className="mesh-actions">
            <button className="text-button" onClick={restoreDemo} disabled={stage !== "editor"}>↺ Restore demo</button>
            <button className="text-button" onClick={frameScene}>⌖ Frame scene</button>
          </div>

          <div className="tool-strip" aria-label="3D editor tool">
            {([
              ["orbit", "Orbit"],
              ["armor-move", "Move"],
              ["armor-rotate", "Rotate"],
              ["spinner-move", "Spinner"],
            ] as [EditorTool, string][]).map(([tool, label]) => (
              <button key={tool} className={editorTool === tool ? "active" : ""} onClick={() => setEditorTool(tool)} disabled={stage !== "editor"}>{label}</button>
            ))}
          </div>

          <div className="field-section">
            <h3>Armor position <span>mm</span></h3>
            <div className="triple-grid">
              <NumberField label="X" value={armorTransform.x} onChange={(v) => patchArmor("x", v)} step={1} />
              <NumberField label="Y" value={armorTransform.y} onChange={(v) => patchArmor("y", v)} step={1} />
              <NumberField label="Z" value={armorTransform.z} onChange={(v) => patchArmor("z", v)} step={1} />
            </div>
          </div>
          <div className="field-section">
            <h3>Armor rotation <span>degrees</span></h3>
            <div className="triple-grid">
              <NumberField label="X" value={armorTransform.rx} onChange={(v) => patchArmor("rx", v)} step={1} />
              <NumberField label="Y" value={armorTransform.ry} onChange={(v) => patchArmor("ry", v)} step={1} />
              <NumberField label="Z" value={armorTransform.rz} onChange={(v) => patchArmor("rz", v)} step={1} />
            </div>
          </div>

          <div className="coordinate-note">
            <span className="axis x">X</span><span className="axis y">Y</span><span className="axis z">Z</span>
            <p>XY is the floor plane. Z is up. STL units are interpreted as millimeters.</p>
          </div>
        </aside>

        <section className="viewport-card">
          <div ref={viewportRef} className="viewport" aria-label="Interactive three-dimensional bite simulation" />
          <div className="viewport-corners" aria-hidden="true"><i /><i /><i /><i /></div>
          <div className="view-badge">LIVE GEOMETRY // MM</div>
          <div className="view-help">
            <span>DRAG TO {editorTool === "orbit" ? "ORBIT" : editorTool.replace("-", " ").toUpperCase()}</span>
            <span>SCROLL TO ZOOM</span>
          </div>

          {stage === "running" && (
            <div className="running-overlay">
              <div className="radar-ring"><span /></div>
              <p>TRACKING FIRST TOOTH</p>
            </div>
          )}

          {stage === "result" && (
            <div className="result-controls">
              <span>{result?.hit ? "TRACER CAPTURED · ORBIT TO INSPECT" : "NO INTERSECTION"}</span>
              <button className="secondary-action" onClick={returnToEditor}>Edit scenario</button>
              <button className="primary-action small" onClick={replay}>Replay</button>
            </div>
          )}

          <div className="status-console"><span>&gt;_</span>{status}</div>
        </section>

        <aside className={`control-panel right-panel ${stage !== "editor" ? "panel-locked" : ""}`}>
          <div className="panel-heading">
            <span>02</span>
            <div><p>WEAPON SYSTEM</p><h2>Spinner setup</h2></div>
          </div>

          <div className="segmented">
            <button className={config.spinPlane === "horizontal" ? "active" : ""} onClick={() => patchConfig("spinPlane", "horizontal")} disabled={stage !== "editor"}>Horizontal</button>
            <button className={config.spinPlane === "vertical" ? "active" : ""} onClick={() => patchConfig("spinPlane", "vertical")} disabled={stage !== "editor"}>Vertical</button>
          </div>

          <div className="field-grid">
            <NumberField label="Weapon radius" value={config.radius} onChange={(v) => patchConfig("radius", clamp(v, 10, 500))} min={10} max={500} unit="mm" />
            <NumberField label="Tooth count" value={config.toothCount} onChange={(v) => patchConfig("toothCount", v)} min={1} max={8} integer />
            <NumberField label="Tooth width" value={config.toothWidth} onChange={(v) => patchConfig("toothWidth", clamp(v, 2, 200))} min={2} max={200} unit="mm" />
            <NumberField label="RPM" value={config.rpm} onChange={(v) => patchConfig("rpm", clamp(v, 1, 30000))} min={1} max={30000} step={100} />
            <NumberField label="Closing speed" value={config.closingSpeed} onChange={(v) => patchConfig("closingSpeed", clamp(v, 0.1, 20))} min={0.1} max={20} step={0.1} unit="m/s" />
            <NumberField label="Moment of inertia" value={config.momentOfInertia} onChange={(v) => patchConfig("momentOfInertia", clamp(v, 0.0001, 100))} min={0.0001} step={0.001} unit="kg·m²" />
            <NumberField label="Starting phase" value={config.phase} onChange={(v) => patchConfig("phase", v % 360)} step={1} unit="°" />
            <NumberField label="Move direction" value={config.moveDirection} onChange={(v) => patchConfig("moveDirection", v % 360)} step={1} unit="°" />
          </div>

          <div className="spin-toggle">
            <span>Direction of spin</span>
            <div className="segmented compact">
              <button className={config.spinDirection === "ccw" ? "active" : ""} onClick={() => patchConfig("spinDirection", "ccw")} disabled={stage !== "editor"}>↺ CCW</button>
              <button className={config.spinDirection === "cw" ? "active" : ""} onClick={() => patchConfig("spinDirection", "cw")} disabled={stage !== "editor"}>↻ CW</button>
            </div>
          </div>

          <div className="field-section start-position">
            <h3>Spinner start <span>mm</span></h3>
            <div className="triple-grid">
              <NumberField label="X" value={config.startX} onChange={(v) => patchConfig("startX", v)} />
              <NumberField label="Y" value={config.startY} onChange={(v) => patchConfig("startY", v)} />
              <NumberField label="Z" value={config.startZ} onChange={(v) => patchConfig("startZ", v)} />
            </div>
          </div>

          <div className="formula-card">
            <div><span>BITE</span><strong>{derived.biteMm.toFixed(2)} mm</strong></div>
            <p>Theoretical maximum assuming the armor intrudes into the weapon circle just as a tooth passes.</p>
          </div>

          <button className="primary-action" onClick={runSimulation} disabled={stage !== "editor"}>
            <span>▶</span> Run bite simulation
          </button>
          <p className="disclaimer">GEOMETRIC VISUALIZATION ONLY · NO FORCE OR PENETRATION MODEL</p>
        </aside>
      </div>
    </main>
  );
}
