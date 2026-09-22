/**
 * Two-view geometry: fundamental/essential matrix estimation (normalized 8-point
 * + RANSAC), essential matrix decomposition, and DLT triangulation.
 *
 * OpenCV.js's WASM build does not expose findEssentialMat/recoverPose/
 * triangulatePoints, so this reimplements the classic Hartley & Zisserman
 * pipeline directly on top of lib/slam/linalg.ts. See recoverPoseAndPoints for
 * the entry point used by the rest of the SLAM pipeline.
 */
import {
  ataSymmetric,
  dot,
  fromRows,
  jacobiEigenSymmetric,
  matrixToRotationVector,
  multiply,
  norm,
  rotationVectorToMatrix,
  transpose,
} from "./linalg.ts";
import { levenbergMarquardt } from "./levenbergMarquardt.ts";
import type { CameraIntrinsics, Mat3, Vec3 } from "./types.ts";

export type Point2 = [number, number];

/** Below this parallax angle, DLT triangulation is numerically degenerate (see parallaxAngleDeg). */
const MIN_PARALLAX_DEG = 1.0;

interface Normalization {
  points: Point2[];
  T: number[]; // 3x3 row-major similarity transform
}

function normalizePoints(points: Point2[]): Normalization {
  const n = points.length;
  let cx = 0;
  let cy = 0;
  for (const [x, y] of points) {
    cx += x;
    cy += y;
  }
  cx /= n;
  cy /= n;

  let meanDist = 0;
  for (const [x, y] of points) {
    meanDist += Math.hypot(x - cx, y - cy);
  }
  meanDist /= n;
  const scale = meanDist > 1e-9 ? Math.SQRT2 / meanDist : 1;

  const T = [scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1];
  const normalized: Point2[] = points.map(([x, y]) => [scale * (x - cx), scale * (y - cy)]);
  return { points: normalized, T };
}

function matMul3x3(a: number[], b: number[]): number[] {
  const out = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[r * 3 + k] * b[k * 3 + c];
      out[r * 3 + c] = s;
    }
  }
  return out;
}

function transpose3x3(a: number[]): number[] {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}

/**
 * Normalized N-point algorithm (N >= 8) for the fundamental matrix: solves the
 * homogeneous linear system via the null space of A^T A (smallest eigenvector),
 * then enforces the rank-2 singularity constraint on the resulting 3x3 matrix.
 */
function estimateFundamentalFromNormalized(pts1: Point2[], pts2: Point2[]): number[] {
  const rows: number[][] = pts1.map(([x1, y1], i) => {
    const [x2, y2] = pts2[i];
    return [x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1];
  });
  const A = fromRows(rows);
  const AtA = ataSymmetric(A);
  const { eigenvectors } = jacobiEigenSymmetric(AtA);
  const f = eigenvectors[eigenvectors.length - 1]; // smallest eigenvalue -> null space

  // Enforce rank-2: F = sigma1*u1*v1^T + sigma2*u2*v2^T (drop the smallest singular value).
  const Fhat = fromRows([
    [f[0], f[1], f[2]],
    [f[3], f[4], f[5]],
    [f[6], f[7], f[8]],
  ]);
  const FtF = ataSymmetric(Fhat);
  const { eigenvalues: sv, eigenvectors: vvecs } = jacobiEigenSymmetric(FtF);
  const sigma1 = Math.sqrt(Math.max(sv[0], 0));
  const sigma2 = Math.sqrt(Math.max(sv[1], 0));
  const v1 = vvecs[0];
  const v2 = vvecs[1];

  const Fv1 = multiply(Fhat, fromRows([[v1[0]], [v1[1]], [v1[2]]])).data;
  const Fv2 = multiply(Fhat, fromRows([[v2[0]], [v2[1]], [v2[2]]])).data;
  const u1 = sigma1 > 1e-12 ? [Fv1[0] / sigma1, Fv1[1] / sigma1, Fv1[2] / sigma1] : [0, 0, 0];
  const u2 = sigma2 > 1e-12 ? [Fv2[0] / sigma2, Fv2[1] / sigma2, Fv2[2] / sigma2] : [0, 0, 0];

  const F = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      F[r * 3 + c] = sigma1 * u1[r] * v1[c] + sigma2 * u2[r] * v2[c];
    }
  }
  return F;
}

/**
 * Shared epipolar-constraint computation for a 3x3 matrix M relating two point
 * sets via x2^T M x1 = 0 (works for both a pixel-space fundamental matrix and
 * a normalized-coordinate essential matrix - the algebra is identical).
 */
function epipolarTerms(M: number[], p1: Point2, p2: Point2) {
  const x1 = [p1[0], p1[1], 1];
  const x2 = [p2[0], p2[1], 1];
  const Mx1 = [
    M[0] * x1[0] + M[1] * x1[1] + M[2] * x1[2],
    M[3] * x1[0] + M[4] * x1[1] + M[5] * x1[2],
    M[6] * x1[0] + M[7] * x1[1] + M[8] * x1[2],
  ];
  const Mt = transpose3x3(M);
  const Mtx2 = [
    Mt[0] * x2[0] + Mt[1] * x2[1] + Mt[2] * x2[2],
    Mt[3] * x2[0] + Mt[4] * x2[1] + Mt[5] * x2[2],
    Mt[6] * x2[0] + Mt[7] * x2[1] + Mt[8] * x2[2],
  ];
  const numerator = x2[0] * Mx1[0] + x2[1] * Mx1[1] + x2[2] * Mx1[2];
  const denom = Mx1[0] ** 2 + Mx1[1] ** 2 + Mtx2[0] ** 2 + Mtx2[1] ** 2;
  return { numerator, denom };
}

function sampsonDistance(F: number[], p1: Point2, p2: Point2): number {
  const { numerator, denom } = epipolarTerms(F, p1, p2);
  if (denom < 1e-12) return Infinity;
  return (numerator * numerator) / denom;
}

/** sqrt(Sampson distance) with the sign of the algebraic epipolar residual, for use as an LM residual. */
function signedEpipolarResidual(M: number[], p1: Point2, p2: Point2): number {
  const { numerator, denom } = epipolarTerms(M, p1, p2);
  if (denom < 1e-12) return 0;
  return numerator / Math.sqrt(denom);
}

export interface RansacFundamentalResult {
  F: number[];
  inlierIndices: number[];
}

/**
 * RANSAC-robust fundamental matrix estimation over pixel-space correspondences.
 * Mirrors what cv.findFundamentalMat(..., RANSAC, ...) would do, since that
 * function is not available in this OpenCV.js build.
 */
export function ransacFundamentalMatrix(
  pts1: Point2[],
  pts2: Point2[],
  options: { iterations?: number; pixelThreshold?: number; seed?: number } = {}
): RansacFundamentalResult | null {
  const n = pts1.length;
  if (n < 8) return null;
  const iterations = options.iterations ?? 400;
  const threshold = (options.pixelThreshold ?? 1.5) ** 2;

  const { points: norm1, T: T1 } = normalizePoints(pts1);
  const { points: norm2, T: T2 } = normalizePoints(pts2);
  const T2t = transpose3x3(T2);

  let rngState = options.seed ?? 0x9e3779b9;
  const rand = () => {
    // xorshift32 - deterministic and dependency-free, good enough for RANSAC sampling.
    rngState ^= rngState << 13;
    rngState ^= rngState >>> 17;
    rngState ^= rngState << 5;
    rngState >>>= 0;
    return rngState / 0xffffffff;
  };

  let bestInliers: number[] = [];
  let bestF: number[] | null = null;

  for (let iter = 0; iter < iterations; iter++) {
    const sample = new Set<number>();
    while (sample.size < 8) sample.add(Math.floor(rand() * n));
    const idxs = Array.from(sample);

    let Fnorm: number[];
    try {
      Fnorm = estimateFundamentalFromNormalized(
        idxs.map((i) => norm1[i]),
        idxs.map((i) => norm2[i])
      );
    } catch {
      continue;
    }
    const Fdenorm = matMul3x3(matMul3x3(T2t, Fnorm), T1);

    const inliers: number[] = [];
    for (let i = 0; i < n; i++) {
      if (sampsonDistance(Fdenorm, pts1[i], pts2[i]) < threshold) inliers.push(i);
    }
    if (inliers.length > bestInliers.length) {
      bestInliers = inliers;
      bestF = Fdenorm;
    }
  }

  if (!bestF || bestInliers.length < 8) return null;

  // Refit on all inliers for a less noisy final estimate.
  const inlierNorm1 = bestInliers.map((i) => norm1[i]);
  const inlierNorm2 = bestInliers.map((i) => norm2[i]);
  const refitted = matMul3x3(matMul3x3(T2t, estimateFundamentalFromNormalized(inlierNorm1, inlierNorm2)), T1);

  const finalInliers = pts1
    .map((_, i) => i)
    .filter((i) => sampsonDistance(refitted, pts1[i], pts2[i]) < threshold);

  return { F: finalInliers.length >= bestInliers.length ? refitted : bestF, inlierIndices: finalInliers.length >= 8 ? finalInliers : bestInliers };
}

function intrinsicsMatrix(K: CameraIntrinsics): number[] {
  return [K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1];
}

function essentialFromFundamental(F: number[], K: CameraIntrinsics): number[] {
  const Km = intrinsicsMatrix(K);
  const Kt = transpose3x3(Km);
  return matMul3x3(matMul3x3(Kt, F), Km);
}

interface Decomposition {
  R: Mat3;
  t: Vec3;
}

/**
 * Decomposes an essential matrix into the 4 candidate (R, t) solutions
 * (Hartley & Zisserman, "Multiple View Geometry", ch. 9). Caller disambiguates
 * via a cheirality (positive-depth) check.
 */
function decomposeEssential(E: number[]): Decomposition[] {
  const Emat = fromRows([
    [E[0], E[1], E[2]],
    [E[3], E[4], E[5]],
    [E[6], E[7], E[8]],
  ]);
  const EtE = ataSymmetric(Emat);
  const Et = transpose(Emat);
  const EEt = ataSymmetric(Et);

  const { eigenvectors: vAll } = jacobiEigenSymmetric(EtE);
  const { eigenvectors: uAll } = jacobiEigenSymmetric(EEt);
  const v1 = vAll[0];
  const v2 = vAll[1];
  const v3 = vAll[2];
  let u1 = uAll[0];
  let u2 = uAll[1];
  let u3 = uAll[2];

  const Ev = (v: number[]) => [
    E[0] * v[0] + E[1] * v[1] + E[2] * v[2],
    E[3] * v[0] + E[4] * v[1] + E[5] * v[2],
    E[6] * v[0] + E[7] * v[1] + E[8] * v[2],
  ];
  const dot3 = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  if (dot3(Ev(v1), u1) < 0) u1 = u1.map((x) => -x);
  if (dot3(Ev(v2), u2) < 0) u2 = u2.map((x) => -x);

  const det3 = (cols: number[][]) =>
    cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1]) -
    cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1]) +
    cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1]);

  if (det3([u1, u2, u3]) < 0) u3 = u3.map((x) => -x);
  if (det3([v1, v2, v3]) < 0) v3.forEach((_, i) => (v3[i] = -v3[i]));

  const U = [u1[0], u2[0], u3[0], u1[1], u2[1], u3[1], u1[2], u2[2], u3[2]]; // column-major cols u1,u2,u3 -> row-major 3x3
  const V = [v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2]];
  const Vt = transpose3x3(V);

  const W = [0, -1, 0, 1, 0, 0, 0, 0, 1];
  const Wt = [0, 1, 0, -1, 0, 0, 0, 0, 1];

  const Ra = matMul3x3(matMul3x3(U, W), Vt) as Mat3;
  const Rb = matMul3x3(matMul3x3(U, Wt), Vt) as Mat3;
  const t: Vec3 = [u3[0], u3[1], u3[2]];
  const tNeg: Vec3 = [-u3[0], -u3[1], -u3[2]];

  return [
    { R: Ra, t },
    { R: Ra, t: tNeg },
    { R: Rb, t },
    { R: Rb, t: tNeg },
  ];
}

/** DLT triangulation of a single point from two views given their 3x4 projection matrices. */
export function triangulatePoint(P1: number[], P2: number[], p1: Point2, p2: Point2): Vec3 {
  const rows = [
    [p1[0] * P1[8] - P1[0], p1[0] * P1[9] - P1[1], p1[0] * P1[10] - P1[2], p1[0] * P1[11] - P1[3]],
    [p1[1] * P1[8] - P1[4], p1[1] * P1[9] - P1[5], p1[1] * P1[10] - P1[6], p1[1] * P1[11] - P1[7]],
    [p2[0] * P2[8] - P2[0], p2[0] * P2[9] - P2[1], p2[0] * P2[10] - P2[2], p2[0] * P2[11] - P2[3]],
    [p2[1] * P2[8] - P2[4], p2[1] * P2[9] - P2[5], p2[1] * P2[10] - P2[6], p2[1] * P2[11] - P2[7]],
  ];
  const A = fromRows(rows);
  const AtA = ataSymmetric(A);
  const { eigenvectors } = jacobiEigenSymmetric(AtA);
  const X = eigenvectors[eigenvectors.length - 1];
  const w = X[3];
  return [X[0] / w, X[1] / w, X[2] / w];
}

export function cameraCenter(R: Mat3, t: Vec3): Vec3 {
  return [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
}

/**
 * Angle (degrees) between the two viewing rays from C1 and C2 to X. Near-zero
 * for points triangulated from two nearly-parallel rays (small baseline
 * relative to depth) - those configurations are numerically degenerate for
 * linear DLT triangulation and can place a point arbitrarily far away even
 * though it still reprojects with low error in both views. Rejecting points
 * below a minimum parallax angle is standard practice in VO/SLAM systems for
 * exactly this reason.
 */
export function parallaxAngleDeg(C1: Vec3, C2: Vec3, X: Vec3): number {
  const v1 = [X[0] - C1[0], X[1] - C1[1], X[2] - C1[2]];
  const v2 = [X[0] - C2[0], X[1] - C2[1], X[2] - C2[2]];
  const n1 = norm(v1);
  const n2 = norm(v2);
  if (n1 < 1e-9 || n2 < 1e-9) return 0;
  const cosAngle = dot(v1, v2) / (n1 * n2);
  return (Math.acos(Math.min(1, Math.max(-1, cosAngle))) * 180) / Math.PI;
}

export function projectPoint(K: CameraIntrinsics, R: Mat3, t: Vec3, X: Vec3): [number, number] {
  const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  return [(K.fx * xc) / zc + K.cx, (K.fy * yc) / zc + K.cy];
}

export function projectionMatrix(K: CameraIntrinsics, R: Mat3, t: Vec3): number[] {
  const Km = intrinsicsMatrix(K);
  const Rt = [R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2]];
  const out = new Array(12).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += Km[r * 3 + k] * Rt[k * 4 + c];
      out[r * 4 + c] = s;
    }
  }
  return out;
}

export interface RecoveredPose {
  R: Mat3;
  t: Vec3;
  /** index into the input inlier arrays (after RANSAC), with each point's triangulated 3D position */
  points: { index: number; position: Vec3 }[];
}

function normalizeToCamera(K: CameraIntrinsics, p: Point2): Point2 {
  return [(p[0] - K.cx) / K.fx, (p[1] - K.cy) / K.fy];
}

/**
 * The linear (normalized 8-point) estimate is exact in the noise-free case but
 * quite sensitive to pixel noise, particularly for the translation direction
 * (a well-known property of linear two-view solvers). This refines (R, t) with
 * a few Levenberg-Marquardt steps minimizing Sampson error in normalized
 * camera coordinates, which is standard practice after any linear essential
 * matrix estimate.
 */
function refineRelativePose(
  R: Mat3,
  t: Vec3,
  pts1: Point2[],
  pts2: Point2[],
  K: CameraIntrinsics
): { R: Mat3; t: Vec3 } {
  const norm1 = pts1.map((p) => normalizeToCamera(K, p));
  const norm2 = pts2.map((p) => normalizeToCamera(K, p));
  const rvec0 = matrixToRotationVector(R);
  const initialParams = [...rvec0, ...t];

  const residualFn = (params: number[]) => {
    const Rm = rotationVectorToMatrix(params.slice(0, 3));
    const tRaw = params.slice(3, 6);
    const tn = norm(tRaw) > 1e-9 ? tRaw.map((v) => v / norm(tRaw)) : [1, 0, 0];
    const tx = [0, -tn[2], tn[1], tn[2], 0, -tn[0], -tn[1], tn[0], 0];
    const E = matMul3x3(tx, Rm);
    const residuals = new Array(norm1.length);
    for (let i = 0; i < norm1.length; i++) {
      residuals[i] = signedEpipolarResidual(E, norm1[i], norm2[i]);
    }
    return residuals;
  };

  const result = levenbergMarquardt(initialParams, residualFn, { maxIterations: 30 });
  const Rrefined = rotationVectorToMatrix(result.params.slice(0, 3)) as Mat3;
  const tRawRefined = result.params.slice(3, 6);
  const tNormVal = norm(tRawRefined) > 1e-9 ? norm(tRawRefined) : 1;
  const tRefined = tRawRefined.map((v) => v / tNormVal) as Vec3;
  return { R: Rrefined, t: tRefined };
}

/**
 * Full two-view pipeline: RANSAC fundamental matrix -> essential matrix ->
 * decomposition -> cheirality check (via DLT triangulation) to pick the one
 * physically valid (R, t) among the 4 candidates.
 */
export function recoverPoseAndPoints(
  pts1: Point2[],
  pts2: Point2[],
  K: CameraIntrinsics,
  options?: { iterations?: number; pixelThreshold?: number }
): RecoveredPose | null {
  const ransac = ransacFundamentalMatrix(pts1, pts2, options);
  if (!ransac) return null;

  const E = essentialFromFundamental(ransac.F, K);
  const candidates = decomposeEssential(E);
  const P1 = projectionMatrix(K, [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);

  const isInFront = (R: Mat3, t: Vec3, X: Vec3) => {
    const depth1 = X[2];
    const depth2 = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    return depth1 > 0 && depth2 > 0;
  };

  let best: { candidate: Decomposition; points: { index: number; position: Vec3 }[]; score: number } | null = null;

  for (const candidate of candidates) {
    const P2 = projectionMatrix(K, candidate.R, candidate.t);
    const points: { index: number; position: Vec3 }[] = [];
    let positiveDepthCount = 0;

    for (const idx of ransac.inlierIndices) {
      const X = triangulatePoint(P1, P2, pts1[idx], pts2[idx]);
      if (isInFront(candidate.R, candidate.t, X)) positiveDepthCount++;
      points.push({ index: idx, position: X });
    }

    if (!best || positiveDepthCount > best.score) {
      best = { candidate, points, score: positiveDepthCount };
    }
  }

  if (!best) return null;

  // Keep only points that are in front of both cameras under the winning hypothesis,
  // then refine (R, t) nonlinearly - the linear estimate above is exact only in the
  // noise-free case and is otherwise noticeably biased, especially in translation.
  const linearInlierIndices = best.points
    .filter(({ position }) => isInFront(best!.candidate.R, best!.candidate.t, position))
    .map((p) => p.index);

  const refined = refineRelativePose(
    best.candidate.R,
    best.candidate.t,
    linearInlierIndices.map((i) => pts1[i]),
    linearInlierIndices.map((i) => pts2[i]),
    K
  );

  const P2refined = projectionMatrix(K, refined.R, refined.t);
  const C1: Vec3 = [0, 0, 0];
  const C2 = cameraCenter(refined.R, refined.t);
  const finalPoints = linearInlierIndices
    .map((idx) => ({ index: idx, position: triangulatePoint(P1, P2refined, pts1[idx], pts2[idx]) }))
    .filter(
      ({ position }) => isInFront(refined.R, refined.t, position) && parallaxAngleDeg(C1, C2, position) > MIN_PARALLAX_DEG
    );

  return { R: refined.R, t: refined.t, points: finalPoints };
}
