/**
 * Small dense linear algebra helpers, sized for SLAM's two-view geometry and
 * windowed bundle adjustment (matrices with at most a few thousand rows/cols).
 * OpenCV.js's WASM build does not expose findEssentialMat/recoverPose/
 * triangulatePoints (calib3d's two-view functions are excluded from the
 * standard build), so this module implements the underlying linear algebra
 * from scratch: a generic symmetric eigensolver (used for SVD-via-eigen of
 * A^T A) and a linear system solver (used for Levenberg-Marquardt normal
 * equations).
 */

/** Dense matrix stored row-major as a flat array, with explicit dimensions. */
export interface Matrix {
  rows: number;
  cols: number;
  data: Float64Array;
}

export function createMatrix(rows: number, cols: number, fill = 0): Matrix {
  return { rows, cols, data: new Float64Array(rows * cols).fill(fill) };
}

export function get(m: Matrix, r: number, c: number): number {
  return m.data[r * m.cols + c];
}

export function set(m: Matrix, r: number, c: number, v: number): void {
  m.data[r * m.cols + c] = v;
}

export function transpose(m: Matrix): Matrix {
  const out = createMatrix(m.cols, m.rows);
  for (let r = 0; r < m.rows; r++) {
    for (let c = 0; c < m.cols; c++) {
      set(out, c, r, get(m, r, c));
    }
  }
  return out;
}

export function multiply(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new Error(`Matrix dimension mismatch: ${a.rows}x${a.cols} * ${b.rows}x${b.cols}`);
  }
  const out = createMatrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const aik = a.data[i * a.cols + k];
      if (aik === 0) continue;
      for (let j = 0; j < b.cols; j++) {
        out.data[i * out.cols + j] += aik * b.data[k * b.cols + j];
      }
    }
  }
  return out;
}

/** A^T * A, exploiting symmetry of the result (only computes the upper triangle then mirrors). */
export function ataSymmetric(a: Matrix): Matrix {
  const n = a.cols;
  const out = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < a.rows; k++) {
        s += a.data[k * a.cols + i] * a.data[k * a.cols + j];
      }
      out.data[i * n + j] = s;
      out.data[j * n + i] = s;
    }
  }
  return out;
}

export function fromRows(rows: number[][]): Matrix {
  const r = rows.length;
  const c = rows[0]?.length ?? 0;
  const out = createMatrix(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) out.data[i * c + j] = rows[i][j];
  }
  return out;
}

export function toRows(m: Matrix): number[][] {
  const rows: number[][] = [];
  for (let i = 0; i < m.rows; i++) {
    rows.push(Array.from(m.data.slice(i * m.cols, (i + 1) * m.cols)));
  }
  return rows;
}

/**
 * Cyclic Jacobi eigenvalue algorithm for real symmetric matrices.
 * Returns eigenvalues descending and their corresponding eigenvectors (as columns of V,
 * i.e. eigenvectors[k] is the k-th eigenvector, matching eigenvalues[k]).
 * Numerically robust for the small sizes used here (n <= ~12); not intended for large matrices.
 */
export function jacobiEigenSymmetric(
  sym: Matrix,
  maxSweeps = 100,
  tolerance = 1e-12
): { eigenvalues: number[]; eigenvectors: number[][] } {
  const n = sym.rows;
  if (sym.cols !== n) throw new Error("jacobiEigenSymmetric requires a square matrix");
  const a = Array.from(sym.data);
  const idx = (r: number, c: number) => r * n + c;
  // V accumulates the rotation product; starts as identity.
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[idx(i, i)] = 1;

  const offDiagNorm = () => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) s += a[idx(i, j)] * a[idx(i, j)];
    }
    return Math.sqrt(s);
  };

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    if (offDiagNorm() < tolerance) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[idx(p, q)];
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[idx(p, p)];
        const aqq = a[idx(q, q)];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);

        for (let k = 0; k < n; k++) {
          const akp = a[idx(k, p)];
          const akq = a[idx(k, q)];
          a[idx(k, p)] = c * akp - s * akq;
          a[idx(k, q)] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[idx(p, k)];
          const aqk = a[idx(q, k)];
          a[idx(p, k)] = c * apk - s * aqk;
          a[idx(q, k)] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[idx(k, p)];
          const vkq = v[idx(k, q)];
          v[idx(k, p)] = c * vkp - s * vkq;
          v[idx(k, q)] = s * vkp + c * vkq;
        }
      }
    }
  }

  const eigenvalues: number[] = [];
  for (let i = 0; i < n; i++) eigenvalues.push(a[idx(i, i)]);

  const order = eigenvalues.map((_, i) => i).sort((i, j) => eigenvalues[j] - eigenvalues[i]);
  const sortedValues = order.map((i) => eigenvalues[i]);
  const sortedVectors = order.map((i) => {
    const vec: number[] = [];
    for (let k = 0; k < n; k++) vec.push(v[idx(k, i)]);
    return vec;
  });

  return { eigenvalues: sortedValues, eigenvectors: sortedVectors };
}

/**
 * Solves the linear system A x = b via Gaussian elimination with partial pivoting.
 * Used for Levenberg-Marquardt normal-equation solves, where A is small and dense
 * (bounded by the bundle-adjustment window size).
 */
export function solveLinearSystem(aIn: Matrix, bIn: number[]): number[] | null {
  const n = aIn.rows;
  if (aIn.cols !== n) throw new Error("solveLinearSystem requires a square matrix");
  const a = Array.from(aIn.data);
  const b = bIn.slice();
  const idx = (r: number, c: number) => r * n + c;

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let pivotVal = Math.abs(a[idx(col, col)]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[idx(r, col)]);
      if (v > pivotVal) {
        pivotVal = v;
        pivotRow = r;
      }
    }
    if (pivotVal < 1e-14) return null;

    if (pivotRow !== col) {
      for (let c = 0; c < n; c++) {
        const tmp = a[idx(col, c)];
        a[idx(col, c)] = a[idx(pivotRow, c)];
        a[idx(pivotRow, c)] = tmp;
      }
      const tmpB = b[col];
      b[col] = b[pivotRow];
      b[pivotRow] = tmpB;
    }

    const diag = a[idx(col, col)];
    for (let r = col + 1; r < n; r++) {
      const factor = a[idx(r, col)] / diag;
      if (factor === 0) continue;
      for (let c = col; c < n; c++) {
        a[idx(r, c)] -= factor * a[idx(col, c)];
      }
      b[r] -= factor * b[col];
    }
  }

  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= a[idx(r, c)] * x[c];
    x[r] = s / a[idx(r, r)];
  }
  return x;
}

export function cross3(a: number[], b: number[]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function scale(a: number[], s: number): number[] {
  return a.map((v) => v * s);
}

export function subtract(a: number[], b: number[]): number[] {
  return a.map((v, i) => v - b[i]);
}

/** Rodrigues' rotation formula: axis-angle vector -> 3x3 rotation matrix (row-major flat). */
export function rotationVectorToMatrix(rvec: number[]): number[] {
  const theta = norm(rvec);
  if (theta < 1e-12) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }
  const [kx, ky, kz] = scale(rvec, 1 / theta);
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  return [
    t * kx * kx + c,
    t * kx * ky - s * kz,
    t * kx * kz + s * ky,
    t * kx * ky + s * kz,
    t * ky * ky + c,
    t * ky * kz - s * kx,
    t * kx * kz - s * ky,
    t * ky * kz + s * kx,
    t * kz * kz + c,
  ];
}

/**
 * Inverse Rodrigues: 3x3 rotation matrix -> axis-angle vector. Only handles
 * angles away from Pi (not needed here - frame-to-frame camera rotations are
 * always small), which keeps this a simple, numerically direct formula.
 */
export function matrixToRotationVector(R: number[]): number[] {
  const trace = R[0] + R[4] + R[8];
  const cosTheta = Math.min(1, Math.max(-1, (trace - 1) / 2));
  const theta = Math.acos(cosTheta);
  if (theta < 1e-9) return [0, 0, 0];
  const rx = R[7] - R[5];
  const ry = R[2] - R[6];
  const rz = R[3] - R[1];
  const axisNorm = Math.hypot(rx, ry, rz);
  if (axisNorm < 1e-12) return [0, 0, 0];
  return [(theta * rx) / axisNorm, (theta * ry) / axisNorm, (theta * rz) / axisNorm];
}

/** Standard robust rotation-matrix -> quaternion conversion (xyzw), picking the largest denominator term to avoid divide-by-near-zero. */
export function rotationMatrixToQuaternion(R: number[]): [number, number, number, number] {
  const trace = R[0] + R[4] + R[8];
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    return [(R[7] - R[5]) * s, (R[2] - R[6]) * s, (R[3] - R[1]) * s, 0.25 / s];
  }
  if (R[0] > R[4] && R[0] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[0] - R[4] - R[8]);
    return [0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s, (R[7] - R[5]) / s];
  }
  if (R[4] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[4] - R[0] - R[8]);
    return [(R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s, (R[2] - R[6]) / s];
  }
  const s = 2 * Math.sqrt(1 + R[8] - R[0] - R[4]);
  return [(R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s, (R[3] - R[1]) / s];
}

export function determinant3x3(m: number[]): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}
