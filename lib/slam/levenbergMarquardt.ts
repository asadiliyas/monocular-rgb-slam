/**
 * Generic dense Levenberg-Marquardt least-squares solver, used both to refine
 * the linear two-view pose estimate (see twoView.ts) and for windowed bundle
 * adjustment (see bundleAdjust.ts). Problem sizes here are always small
 * (a handful to a few dozen parameters), so a dense normal-equations solve
 * with a numerically-differentiated Jacobian is simple, easy to verify, and
 * fast enough - no need for a sparse/Schur-complement solver at this scale.
 */
import { createMatrix, solveLinearSystem, type Matrix } from "./linalg.ts";

export interface LevenbergMarquardtOptions {
  maxIterations?: number;
  initialLambda?: number;
  lambdaUp?: number;
  lambdaDown?: number;
  costTolerance?: number;
  finiteDiffStep?: number;
}

export interface LevenbergMarquardtResult {
  params: number[];
  cost: number;
  iterations: number;
}

function sumSquares(r: number[]): number {
  let s = 0;
  for (const v of r) s += v * v;
  return s;
}

function numericJacobian(
  params: number[],
  residualFn: (p: number[]) => number[],
  r0: number[],
  step: number
): Matrix {
  const n = params.length;
  const m = r0.length;
  const J = createMatrix(m, n);
  for (let j = 0; j < n; j++) {
    const h = step * Math.max(1, Math.abs(params[j]));
    const perturbed = params.slice();
    perturbed[j] += h;
    const rPlus = residualFn(perturbed);
    for (let i = 0; i < m; i++) {
      J.data[i * n + j] = (rPlus[i] - r0[i]) / h;
    }
  }
  return J;
}

/**
 * Minimizes sum(residualFn(params)^2) starting from the given initial params.
 */
export function levenbergMarquardt(
  initialParams: number[],
  residualFn: (params: number[]) => number[],
  options: LevenbergMarquardtOptions = {}
): LevenbergMarquardtResult {
  const maxIterations = options.maxIterations ?? 50;
  let lambda = options.initialLambda ?? 1e-3;
  const lambdaUp = options.lambdaUp ?? 10;
  const lambdaDown = options.lambdaDown ?? 10;
  const costTolerance = options.costTolerance ?? 1e-10;
  const step = options.finiteDiffStep ?? 1e-6;

  let params = initialParams.slice();
  let residuals = residualFn(params);
  let cost = sumSquares(residuals);
  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;
    const J = numericJacobian(params, residualFn, residuals, step);
    const n = params.length;

    // Normal equations: (J^T J + lambda * diag(J^T J)) delta = -J^T r
    const JtJ = createMatrix(n, n);
    const Jtr = new Array(n).fill(0);
    for (let i = 0; i < J.rows; i++) {
      for (let a = 0; a < n; a++) {
        const jia = J.data[i * n + a];
        if (jia === 0) continue;
        Jtr[a] += jia * residuals[i];
        for (let b = 0; b < n; b++) {
          JtJ.data[a * n + b] += jia * J.data[i * n + b];
        }
      }
    }

    let accepted = false;
    for (let attempt = 0; attempt < 12 && !accepted; attempt++) {
      const damped = createMatrix(n, n);
      damped.data.set(JtJ.data);
      for (let d = 0; d < n; d++) {
        damped.data[d * n + d] += lambda * Math.max(JtJ.data[d * n + d], 1e-12);
      }
      const negJtr = Jtr.map((v) => -v);
      const delta = solveLinearSystem(damped, negJtr);
      if (!delta) {
        lambda *= lambdaUp;
        continue;
      }
      const candidate = params.map((p, i) => p + delta[i]);
      const candidateResiduals = residualFn(candidate);
      const candidateCost = sumSquares(candidateResiduals);

      if (candidateCost < cost) {
        const improvement = cost - candidateCost;
        params = candidate;
        residuals = candidateResiduals;
        cost = candidateCost;
        lambda /= lambdaDown;
        accepted = true;
        if (improvement < costTolerance) {
          return { params, cost, iterations };
        }
      } else {
        lambda *= lambdaUp;
      }
    }
    if (!accepted) break; // no improving step found even after backing off; converged or stuck
  }

  return { params, cost, iterations };
}
