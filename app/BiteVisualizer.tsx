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

type ContactKinematics = {
  normalVelocity: number;
  tangentialVelocity: number;
  impactAngle: number;
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
  contactKinematics: ContactKinematics | null;
};

type MetricSnapshot = {
  rpm: number;
  tipMps: number;
  closingSpeed: number;
  energy: number;
  biteMm: number;
  pathLength: number;
  normalVelocity: number | null;
  tangentialVelocity: number | null;
  impactAngle: number | null;
};

type ArmorSource =
  | { kind: "demo" }
  | { kind: "stl"; buffer: ArrayBuffer };

type HistoryEntry = {
  id: string;
  createdAt: number;
  config: WeaponConfig;
  armorTransform: ArmorTransform;
  armorName: string;
  meshStatus: string;
  armorSource: ArmorSource;
  distanceOffset: number;
  metrics: MetricSnapshot;
  hit: boolean;
};

type MetricDeltas = Record<keyof MetricSnapshot, number | null>;

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
const HISTORY_DB_NAME = "combat-robot-bite-visualizer";
const HISTORY_STORE_NAME = "simulation-history";

function openHistoryDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(HISTORY_DB_NAME, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(HISTORY_STORE_NAME)) {
        database.createObjectStore(HISTORY_STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readSimulationHistory() {
  const database = await openHistoryDatabase();
  return new Promise<HistoryEntry[]>((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE_NAME, "readonly");
    const request = transaction.objectStore(HISTORY_STORE_NAME).getAll();
    request.onsuccess = () => resolve((request.result as HistoryEntry[]).sort((a, b) => b.createdAt - a.createdAt));
    request.onerror = () => reject(request.error);
    transaction.oncomplete = () => database.close();
  });
}

async function writeHistoryEntry(entry: HistoryEntry) {
  const database = await openHistoryDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE_NAME, "readwrite");
    transaction.objectStore(HISTORY_STORE_NAME).put(entry);
    transaction.oncomplete = () => { database.close(); resolve(); };
    transaction.onerror = () => { database.close(); reject(transaction.error); };
  });
}

async function removeHistoryEntry(id: string) {
  const database = await openHistoryDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE_NAME, "readwrite");
    transaction.objectStore(HISTORY_STORE_NAME).delete(id);
    transaction.oncomplete = () => { database.close(); resolve(); };
    transaction.onerror = () => { database.close(); reject(transaction.error); };
  });
}

async function removeAllHistory() {
  const database = await openHistoryDatabase();
  return new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(HISTORY_STORE_NAME, "readwrite");
    transaction.objectStore(HISTORY_STORE_NAME).clear();
    transaction.oncomplete = () => { database.close(); resolve(); };
    transaction.onerror = () => { database.close(); reject(transaction.error); };
  });
}

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

function normalizeDegrees(value: number) {
  return ((value % 360) + 360) % 360;
}

function closestPointOnSegment(point: THREE.Vector3, start: THREE.Vector3, end: THREE.Vector3) {
  const delta = end.clone().sub(start);
  const denominator = delta.lengthSq();
  if (denominator < 1e-12) return start.clone();
  const t = clamp(point.clone().sub(start).dot(delta) / denominator, 0, 1);
  return start.clone().addScaledVector(delta, t);
}

function clipSegmentToZRange(start: THREE.Vector3, end: THREE.Vector3, minZ: number, maxZ: number) {
  let low = 0;
  let high = 1;
  const dz = end.z - start.z;
  if (Math.abs(dz) < 1e-9) {
    return start.z >= minZ && start.z <= maxZ ? [start.clone(), end.clone()] as const : null;
  }
  const tMin = (minZ - start.z) / dz;
  const tMax = (maxZ - start.z) / dz;
  low = Math.max(low, Math.min(tMin, tMax));
  high = Math.min(high, Math.max(tMin, tMax));
  if (low > high) return null;
  const delta = end.clone().sub(start);
  return [
    start.clone().addScaledVector(delta, low),
    start.clone().addScaledVector(delta, high),
  ] as const;
}

/** Find the nearest point on the STL's intersection with the spinner plane. */
function nearestPointOnArmorSlice(
  mesh: THREE.Mesh,
  plane: THREE.Plane,
  reference: THREE.Vector3,
  zRange?: { min: number; max: number },
) {
  const geometry = mesh.geometry;
  const positions = geometry.getAttribute("position");
  if (!positions) return null;
  const index = geometry.index;
  const vertex = (positionIndex: number) => new THREE.Vector3(
    positions.getX(positionIndex),
    positions.getY(positionIndex),
    positions.getZ(positionIndex),
  ).applyMatrix4(mesh.matrixWorld);
  const triangleCount = Math.floor((index?.count ?? positions.count) / 3);
  const epsilon = 1e-5;
  let nearest: THREE.Vector3 | null = null;
  let nearestDistanceSq = Infinity;

  const consider = (candidate: THREE.Vector3) => {
    if (zRange && (candidate.z < zRange.min - epsilon || candidate.z > zRange.max + epsilon)) return;
    const distanceSq = candidate.distanceToSquared(reference);
    if (distanceSq < nearestDistanceSq) {
      nearestDistanceSq = distanceSq;
      nearest = candidate.clone();
    }
  };

  const considerSegment = (rawStart: THREE.Vector3, rawEnd: THREE.Vector3) => {
    const segment = zRange
      ? clipSegmentToZRange(rawStart, rawEnd, zRange.min, zRange.max)
      : [rawStart, rawEnd] as const;
    if (!segment) return;
    consider(closestPointOnSegment(reference, segment[0], segment[1]));
  };

  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += 1) {
    const indices = [0, 1, 2].map((corner) => {
      const offset = triangleIndex * 3 + corner;
      return index ? index.getX(offset) : offset;
    });
    const vertices = indices.map(vertex);
    const distances = vertices.map((point) => plane.distanceToPoint(point));
    const coplanar = distances.every((distance) => Math.abs(distance) <= epsilon);
    if (coplanar) {
      considerSegment(vertices[0], vertices[1]);
      considerSegment(vertices[1], vertices[2]);
      considerSegment(vertices[2], vertices[0]);
      continue;
    }

    const intersections: THREE.Vector3[] = [];
    const addUnique = (point: THREE.Vector3) => {
      if (!intersections.some((other) => other.distanceToSquared(point) < epsilon * epsilon)) {
        intersections.push(point);
      }
    };
    for (let edge = 0; edge < 3; edge += 1) {
      const next = (edge + 1) % 3;
      const start = vertices[edge];
      const end = vertices[next];
      const startDistance = distances[edge];
      const endDistance = distances[next];
      if (Math.abs(startDistance) <= epsilon) addUnique(start);
      if (startDistance * endDistance < -epsilon * epsilon) {
        const t = startDistance / (startDistance - endDistance);
        addUnique(start.clone().lerp(end, t));
      }
    }
    if (intersections.length === 1) consider(intersections[0]);
    else if (intersections.length >= 2) considerSegment(intersections[0], intersections[1]);
  }

  return nearest;
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
  return hit && hit.distance <= length + 0.001 ? hit : null;
}

/**
 * Return a unique world-space surface normal at an intersection. A hit inside
 * a triangle always has a unique tangent plane. At a triangle boundary we
 * inspect every incident triangle: coplanar STL seams are valid, while a real
 * model edge or vertex deliberately returns null.
 */
function uniqueSurfaceNormal(mesh: THREE.Mesh, hit: THREE.Intersection) {
  if (!hit.face) return null;
  const geometry = mesh.geometry;
  const positions = geometry.getAttribute("position");
  if (!positions) return null;
  const localPoint = mesh.worldToLocal(hit.point.clone());
  const vertex = (index: number) => new THREE.Vector3(
    positions.getX(index),
    positions.getY(index),
    positions.getZ(index),
  );
  const hitTriangle = new THREE.Triangle(
    vertex(hit.face.a),
    vertex(hit.face.b),
    vertex(hit.face.c),
  );
  const barycentric = hitTriangle.getBarycoord(localPoint, new THREE.Vector3());
  const normalMatrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
  const toWorldNormal = (normal: THREE.Vector3) => normal.applyNormalMatrix(normalMatrix).normalize();
  const hitNormal = toWorldNormal(hitTriangle.getNormal(new THREE.Vector3()));
  if (Math.min(barycentric.x, barycentric.y, barycentric.z) > 1e-4) return hitNormal;

  geometry.computeBoundingBox();
  const diagonal = geometry.boundingBox?.getSize(new THREE.Vector3()).length() ?? 1;
  const pointTolerance = Math.max(1e-5, diagonal * 1e-7);
  const index = geometry.index;
  const triangleCount = Math.floor((index?.count ?? positions.count) / 3);
  const incidentNormals: THREE.Vector3[] = [];
  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex += 1) {
    const indices = [0, 1, 2].map((corner) => {
      const offset = triangleIndex * 3 + corner;
      return index ? index.getX(offset) : offset;
    });
    const triangle = new THREE.Triangle(vertex(indices[0]), vertex(indices[1]), vertex(indices[2]));
    const closest = triangle.closestPointToPoint(localPoint, new THREE.Vector3());
    if (closest.distanceToSquared(localPoint) <= pointTolerance * pointTolerance) {
      const normal = triangle.getNormal(new THREE.Vector3());
      if (normal.lengthSq() > 1e-12) incidentNormals.push(normal.normalize());
    }
  }
  if (incidentNormals.length < 2) return null;
  const reference = incidentNormals[0];
  const coplanar = incidentNormals.every((normal) => Math.abs(reference.dot(normal)) > 0.9999);
  return coplanar ? toWorldNormal(reference.clone()) : null;
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

function Metric({
  label,
  value,
  delta,
  onInfo,
}: {
  label: string;
  value: string;
  delta?: string | null;
  onInfo?: () => void;
}) {
  return (
    <div className={`metric ${label === "IMPACT PATH" ? "metric-accent" : ""}`}>
      <div className="metric-label">
        <span>{label}</span>
        {onInfo && <button className="metric-info" onClick={onInfo} aria-label={`About ${label.toLowerCase()}`}>i</button>}
      </div>
      <div className="metric-value">
        <strong>{value}</strong>
        {delta && <em className={delta.startsWith("+") ? "delta-up" : "delta-down"}>{delta}</em>}
      </div>
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
  const [distanceOffset, setDistanceOffset] = useState(0);
  const [rightTab, setRightTab] = useState<"setup" | "history">("setup");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [metricDeltas, setMetricDeltas] = useState<MetricDeltas | null>(null);
  const [showImpactInfo, setShowImpactInfo] = useState(false);

  const viewportRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const threeRef = useRef<ThreeState | null>(null);
  const armorSourceRef = useRef<ArmorSource>({ kind: "demo" });
  const historyRef = useRef<HistoryEntry[]>([]);
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
  useEffect(() => { historyRef.current = history; }, [history]);

  useEffect(() => {
    if (!showImpactInfo) return;
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setShowImpactInfo(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [showImpactInfo]);

  useEffect(() => {
    let active = true;
    readSimulationHistory()
      .then((entries) => { if (active) setHistory(entries); })
      .catch(() => { if (active) setStatus("Simulation history is unavailable in this browser."); });
    return () => { active = false; };
  }, []);

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

  const metricDelta = (key: keyof MetricSnapshot, unit: string, digits: number) => {
    if (stage !== "result" || !metricDeltas) return null;
    const delta = metricDeltas[key];
    if (typeof delta !== "number" || !Number.isFinite(delta)) return null;
    const rounded = Number(delta.toFixed(digits));
    if (rounded === 0) return null;
    const sign = rounded > 0 ? "+" : "−";
    return `${sign}${Math.abs(rounded).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    })}${unit ? ` ${unit}` : ""}`;
  };

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
      armorSourceRef.current = { kind: "stl", buffer: buffer.slice(0) };
    } catch {
      setMeshStatus("Could not read this STL");
      setStatus("Try a binary or ASCII STL containing a closed solid.");
    } finally {
      event.target.value = "";
    }
  }, [replaceArmorGeometry]);

  const restoreDemo = useCallback(() => {
    armorSourceRef.current = { kind: "demo" };
    replaceArmorGeometry(makeDemoPlateGeometry(), "Demo wedge plate", "250 mm wide × 30 mm thick · 45° half-height ramp");
  }, [replaceArmorGeometry]);

  const frameScene = useCallback(() => {
    const state = threeRef.current;
    if (!state) return;
    frameObjects(state, [state.armorRoot, state.spinner, state.directionArrow]);
    setStatus("Camera framed around both the armor and spinner.");
  }, []);

  const runMaximumBiteAssist = useCallback((offset = distanceOffset) => {
    const state = threeRef.current;
    if (!state || stageRef.current !== "editor") return;
    const current = configRef.current;
    const center = new THREE.Vector3(current.startX, current.startY, current.startZ);
    state.armorRoot.updateMatrixWorld(true);
    state.armorMesh.updateMatrixWorld(true);

    let plane: THREE.Plane;
    let zRange: { min: number; max: number } | undefined;
    if (current.spinPlane === "horizontal") {
      plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -current.startZ);
    } else {
      const heading = current.moveDirection * DEG;
      const lateral = new THREE.Vector3(-Math.sin(heading), Math.cos(heading), 0);
      plane = new THREE.Plane().setFromNormalAndCoplanarPoint(lateral, center);
      zRange = { min: current.startZ - current.radius, max: current.startZ + current.radius };
    }

    const contact = nearestPointOnArmorSlice(state.armorMesh, plane, center, zRange);
    if (!contact) {
      setStatus("Maximum bite assist could not find armor on the spinner's middle plane. Adjust the spinner height or orientation.");
      return;
    }

    const towardArmor = new THREE.Vector2(contact.x - center.x, contact.y - center.y);
    if (towardArmor.lengthSq() < 1e-9) {
      setStatus("Maximum bite assist needs the spinner center to be laterally separated from the armor.");
      return;
    }
    towardArmor.normalize();
    const moveDirection = normalizeDegrees(Math.atan2(towardArmor.y, towardArmor.x) / DEG);
    let phase = moveDirection;
    let horizontalReach = current.radius;

    if (current.spinPlane === "vertical") {
      const verticalReach = contact.z - current.startZ;
      if (Math.abs(verticalReach) > current.radius + 1e-5) {
        setStatus("Maximum bite assist found armor outside the vertical spinner's radius. Adjust spinner Z and try again.");
        return;
      }
      horizontalReach = Math.sqrt(Math.max(0, current.radius ** 2 - verticalReach ** 2));
      phase = normalizeDegrees(Math.atan2(verticalReach, horizontalReach) / DEG);
    }

    // Positive offset backs the spinner away from the established contact;
    // negative values intentionally advance it into the armor.
    const approachDistance = horizontalReach + offset;
    const startX = contact.x - towardArmor.x * approachDistance;
    const startY = contact.y - towardArmor.y * approachDistance;
    setDistanceOffset(offset);
    setConfig((previous) => ({
      ...previous,
      phase: Number(phase.toFixed(3)),
      moveDirection: Number(moveDirection.toFixed(3)),
      startX: Number(startX.toFixed(3)),
      startY: Number(startY.toFixed(3)),
    }));
    setStatus(
      `Maximum bite aligned at (${contact.x.toFixed(1)}, ${contact.y.toFixed(1)}, ${contact.z.toFixed(1)}) mm · ${offset >= 0 ? offset.toFixed(1) : `−${Math.abs(offset).toFixed(1)}`} mm offset.`,
    );
  }, [distanceOffset]);

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
    let contactKinematics: ContactKinematics | null = null;

    for (let step = 0; step <= steps && !finished; step += 1) {
      const t = Math.min(endTime, step * dt);
      const center = start.clone().addScaledVector(direction, speed * t);
      const angle = phase + omega * t;

      if (trackedTooth === null) {
        let earliest: {
          tooth: number;
          sample: number;
          point: THREE.Vector3;
          current: THREE.Vector3;
          score: number;
          intersection: THREE.Intersection | null;
        } | null = null;
        for (let tooth = 0; tooth < config.toothCount; tooth += 1) {
          for (let sample = 0; sample < rodSamples.length; sample += 1) {
            const point = toothPoint(config, center, angle, tooth, rodSamples[sample]);
            const inside = pointInsideMesh(state.armorMesh, point, worldBox);
            if (step > 0 && inside && !previousInside[tooth][sample]) {
              const intersection = segmentSurfaceHit(state.armorMesh, previousPoints[tooth][sample], point);
              const hit = intersection?.point.clone() ?? point.clone();
              const segmentLength = Math.max(1e-9, previousPoints[tooth][sample].distanceTo(point));
              const score = previousPoints[tooth][sample].distanceTo(hit) / segmentLength;
              if (!earliest || score < earliest.score) {
                earliest = { tooth, sample, point: hit, current: point.clone(), score, intersection };
              }
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
          if (earliest.intersection) {
            const surfaceNormal = uniqueSurfaceNormal(state.armorMesh, earliest.intersection);
            if (surfaceNormal) {
              const previousT = Math.max(0, t - dt);
              const contactT = previousT + (t - previousT) * clamp(earliest.score, 0, 1);
              const contactCenter = start.clone().addScaledVector(direction, speed * contactT);
              const contactAngle = phase + omega * contactT;
              const contactTooth = toothPoint(
                config,
                contactCenter,
                contactAngle,
                earliest.tooth,
                rodSamples[earliest.sample],
              );
              const radiusVector = contactTooth.sub(contactCenter);
              const spinAxis = new THREE.Vector3(0, 0, 1)
                .applyQuaternion(baseQuaternion(config))
                .normalize();
              const rotationalVelocity = new THREE.Vector3()
                .crossVectors(spinAxis, radiusVector)
                .multiplyScalar(omega);
              const contactVelocity = direction.clone().multiplyScalar(speed).add(rotationalVelocity);
              const normalVelocityMm = Math.abs(contactVelocity.dot(surfaceNormal));
              const tangentialVelocityMm = Math.sqrt(Math.max(
                0,
                contactVelocity.lengthSq() - normalVelocityMm ** 2,
              ));
              contactKinematics = {
                normalVelocity: normalVelocityMm / 1000,
                tangentialVelocity: tangentialVelocityMm / 1000,
                impactAngle: Math.atan2(normalVelocityMm, tangentialVelocityMm) / DEG,
              };
            }
          }
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
          const surfaceExit = segmentSurfaceHit(state.armorMesh, lastTrackedPoint, current)?.point.clone() ?? current.clone();
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
      contactKinematics,
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
    const metrics: MetricSnapshot = {
      rpm: config.rpm,
      tipMps: derived.tipMps,
      closingSpeed: config.closingSpeed,
      energy: derived.energy,
      biteMm: derived.biteMm,
      pathLength: nextResult.pathLength,
      normalVelocity: nextResult.contactKinematics?.normalVelocity ?? null,
      tangentialVelocity: nextResult.contactKinematics?.tangentialVelocity ?? null,
      impactAngle: nextResult.contactKinematics?.impactAngle ?? null,
    };
    const previousRun = historyRef.current[0];
    const difference = (current: number | null, previous: number | null | undefined) =>
      typeof current === "number" && typeof previous === "number" ? current - previous : null;
    setMetricDeltas(previousRun ? {
      rpm: difference(metrics.rpm, previousRun.metrics.rpm),
      tipMps: difference(metrics.tipMps, previousRun.metrics.tipMps),
      closingSpeed: difference(metrics.closingSpeed, previousRun.metrics.closingSpeed),
      energy: difference(metrics.energy, previousRun.metrics.energy),
      biteMm: difference(metrics.biteMm, previousRun.metrics.biteMm),
      pathLength: difference(metrics.pathLength, previousRun.metrics.pathLength),
      normalVelocity: difference(metrics.normalVelocity, previousRun.metrics.normalVelocity),
      tangentialVelocity: difference(metrics.tangentialVelocity, previousRun.metrics.tangentialVelocity),
      impactAngle: difference(metrics.impactAngle, previousRun.metrics.impactAngle),
    } : null);
    const armorSource = armorSourceRef.current.kind === "stl"
      ? { kind: "stl" as const, buffer: armorSourceRef.current.buffer.slice(0) }
      : { kind: "demo" as const };
    const historyEntry: HistoryEntry = {
      id: typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      createdAt: Date.now(),
      config: { ...config },
      armorTransform: { ...armorTransform },
      armorName,
      meshStatus,
      armorSource,
      distanceOffset,
      metrics,
      hit: nextResult.hit,
    };
    const nextHistory = [historyEntry, ...historyRef.current];
    historyRef.current = nextHistory;
    setHistory(nextHistory);
    void writeHistoryEntry(historyEntry).catch(() => {
      setStatus("Simulation ran, but browser storage could not save this history entry.");
    });
    setResult(nextResult);
    resultRef.current = nextResult;
    setStage("running");
    const duration = clamp(3.6 + nextResult.physicsDuration * 5, 3.8, 7.2);
    playbackRef.current = { active: true, started: performance.now(), duration, lastIndex: -1 };
    setStatus(nextResult.hit ? "First tooth acquired. Tracking through armor…" : "Running geometric pass…");
  }, [armorName, armorTransform, calculateSimulation, config, derived, distanceOffset, meshStatus, setTracerGeometry]);

  const loadHistoryEntry = useCallback((entry: HistoryEntry) => {
    try {
      playbackRef.current.active = false;
      setStage("editor");
      setResult(null);
      setMetricDeltas(null);
      setLivePathLength(0);
      setTracerGeometry([]);
      const geometry = entry.armorSource.kind === "demo"
        ? makeDemoPlateGeometry()
        : new STLLoader().parse(entry.armorSource.buffer.slice(0));
      if (entry.armorSource.kind === "stl") normalizeStlUnitsToMillimeters(geometry);
      armorSourceRef.current = entry.armorSource.kind === "stl"
        ? { kind: "stl", buffer: entry.armorSource.buffer.slice(0) }
        : { kind: "demo" };
      replaceArmorGeometry(geometry, entry.armorName, entry.meshStatus);
      setArmorTransform({ ...entry.armorTransform });
      setConfig({ ...entry.config });
      setDistanceOffset(entry.distanceOffset);
      setRightTab("setup");
      setStatus(`Loaded simulation from ${new Date(entry.createdAt).toLocaleString()}. Adjust it or run again.`);
      requestAnimationFrame(() => {
        const state = threeRef.current;
        if (state) frameObjects(state, [state.armorRoot, state.spinner, state.directionArrow]);
      });
    } catch {
      setStatus("This history entry could not restore its saved STL.");
    }
  }, [replaceArmorGeometry, setTracerGeometry]);

  const deleteHistoryEntry = useCallback((id: string) => {
    const nextHistory = historyRef.current.filter((entry) => entry.id !== id);
    historyRef.current = nextHistory;
    setHistory(nextHistory);
    void removeHistoryEntry(id).catch(() => {
      setStatus("The history entry was removed from this view but could not be deleted from browser storage.");
    });
  }, []);

  const clearSimulationHistory = useCallback(() => {
    if (!window.confirm("Delete the entire simulation history stored in this browser?")) return;
    historyRef.current = [];
    setHistory([]);
    setMetricDeltas(null);
    void removeAllHistory().catch(() => {
      setStatus("History was cleared from this view but browser storage could not be emptied.");
    });
  }, []);

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
        <Metric label="RPM" value={Math.round(config.rpm).toLocaleString()} delta={metricDelta("rpm", "", 0)} />
        <Metric label="TIP SPEED" value={`${derived.tipMps.toFixed(1)} m/s`} delta={metricDelta("tipMps", "m/s", 1)} />
        <Metric label="CLOSING SPEED" value={`${config.closingSpeed.toFixed(1)} m/s`} delta={metricDelta("closingSpeed", "m/s", 1)} />
        <Metric label="STORED ENERGY" value={`${derived.energy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J`} delta={metricDelta("energy", "J", 0)} />
        <Metric label="THEORETICAL MAXIMUM BITE DEPTH" value={`${derived.biteMm.toFixed(1)} mm`} delta={metricDelta("biteMm", "mm", 1)} />
        <Metric label="IMPACT PATH" value={`${livePathLength.toFixed(1)} mm`} delta={metricDelta("pathLength", "mm", 1)} onInfo={() => setShowImpactInfo(true)} />
        <Metric
          label="NORMAL VELOCITY"
          value={result?.contactKinematics ? `${result.contactKinematics.normalVelocity.toFixed(1)} m/s` : "N/A"}
          delta={metricDelta("normalVelocity", "m/s", 1)}
        />
        <Metric
          label="IMPACT ANGLE"
          value={result?.contactKinematics ? `${result.contactKinematics.impactAngle.toFixed(1)}°` : "N/A"}
          delta={metricDelta("impactAngle", "°", 1)}
        />
        <Metric
          label="TANGENTIAL VELOCITY"
          value={result?.contactKinematics ? `${result.contactKinematics.tangentialVelocity.toFixed(1)} m/s` : "N/A"}
          delta={metricDelta("tangentialVelocity", "m/s", 1)}
        />
      </section>

      {showImpactInfo && (
        <div
          className="info-modal-backdrop"
          role="presentation"
          onMouseDown={(event) => { if (event.target === event.currentTarget) setShowImpactInfo(false); }}
        >
          <section className="info-modal" role="dialog" aria-modal="true" aria-labelledby="impact-info-title">
            <button className="info-modal-close" onClick={() => setShowImpactInfo(false)} aria-label="Close impact path information" autoFocus>×</button>
            <p className="info-modal-kicker">GEOMETRIC MODEL LIMITATION</p>
            <h2 id="impact-info-title">What impact path means in practice</h2>
            <p>This visualizer does not include a force model.</p>
            <p>In a real engagement with a sloped armor panel made from hardened steel or titanium, a longer engagement path can produce a larger accumulated reaction impulse. That impulse tends to deflect the weapon away from the wedge.</p>
            <p>Weapon setups that produce very long paths here are therefore more likely to disengage before completing the simulated path in real life. Minimizing path length increases the likelihood of effective penetration.</p>
            <p>However, a spinner with deep bite and a short impact path can still be deflected when the armor plate is sufficiently robust or the weapon has insufficient stored energy. A favorable geometric result is not a guarantee of penetration.</p>
            <button className="info-modal-action" onClick={() => setShowImpactInfo(false)}>Understood</button>
          </section>
        </div>
      )}

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

        <aside className="control-panel right-panel">
          <div className="panel-heading">
            <span>02</span>
            <div><p>WEAPON SYSTEM</p><h2>Spinner setup</h2></div>
          </div>

          <div className="panel-tabs" role="tablist" aria-label="Weapon panel view">
            <button role="tab" aria-selected={rightTab === "setup"} className={rightTab === "setup" ? "active" : ""} onClick={() => setRightTab("setup")}>Setup</button>
            <button role="tab" aria-selected={rightTab === "history"} className={rightTab === "history" ? "active" : ""} onClick={() => setRightTab("history")}>History <span>{history.length}</span></button>
          </div>

          {rightTab === "setup" ? <fieldset className="setup-panel-content" disabled={stage !== "editor"}>
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
            <p className="input-commit-note">↵ Press Enter to apply field changes</p>
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

          <section className="bite-assist" aria-labelledby="bite-assist-title">
            <div className="bite-assist-heading">
              <span>⌁</span>
              <div>
                <p>POSITIONING TOOL</p>
                <h3 id="bite-assist-title">Maximum bite assist</h3>
              </div>
            </div>
            <NumberField
              label="Distance offset"
              value={distanceOffset}
              onChange={(value) => runMaximumBiteAssist(value)}
              step={1}
              unit="mm"
            />
            <p className="bite-assist-tip">Use to prevent premature overlap. Positive values move the spinner backward along its path.</p>
            <button className="assist-action" onClick={() => runMaximumBiteAssist()} disabled={stage !== "editor"}>
              Align for maximum bite
            </button>
          </section>

          <div className="formula-card">
            <div><span>BITE</span><strong>{derived.biteMm.toFixed(2)} mm</strong></div>
            <p>Theoretical maximum assuming the armor intrudes into the weapon circle just as a tooth passes.</p>
          </div>

          <button className="primary-action" onClick={runSimulation} disabled={stage !== "editor"}>
            <span>▶</span> Run bite simulation
          </button>
          <p className="disclaimer">GEOMETRIC VISUALIZATION ONLY · NO FORCE OR PENETRATION MODEL</p>
          </fieldset> : (
            <section className="history-panel" role="tabpanel">
              <div className="history-toolbar">
                <div>
                  <strong>Simulation history</strong>
                  <span>Saved in this browser</span>
                </div>
                <button onClick={clearSimulationHistory} disabled={!history.length}>Delete all</button>
              </div>
              {!history.length ? (
                <div className="history-empty">
                  <span>◇</span>
                  <strong>No recorded simulations</strong>
                  <p>Completed runs will appear here for comparison and reloading.</p>
                </div>
              ) : (
                <div className="history-list">
                  {history.map((entry, index) => (
                    <article className="history-entry" key={entry.id}>
                      <div className="history-entry-top">
                        <span>RUN {history.length - index}</span>
                        <time dateTime={new Date(entry.createdAt).toISOString()}>{new Date(entry.createdAt).toLocaleString()}</time>
                      </div>
                      <strong title={entry.armorName}>{entry.armorName}</strong>
                      <div className="history-metrics">
                        <span>PATH <b>{entry.metrics.pathLength.toFixed(1)} mm</b></span>
                        <span>BITE <b>{entry.metrics.biteMm.toFixed(1)} mm</b></span>
                        <span>RPM <b>{Math.round(entry.metrics.rpm).toLocaleString()}</b></span>
                        <span>ENERGY <b>{entry.metrics.energy.toLocaleString(undefined, { maximumFractionDigits: 0 })} J</b></span>
                      </div>
                      <div className="history-actions">
                        <button onClick={() => loadHistoryEntry(entry)}>Reload configuration</button>
                        <button className="delete" onClick={() => deleteHistoryEntry(entry.id)} aria-label={`Delete simulation from ${new Date(entry.createdAt).toLocaleString()}`}>Delete</button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </section>
          )}
        </aside>
      </div>
    </main>
  );
}
