import { matrixToRotationVector, rotationVectorToMatrix } from "./linalg.ts";
import { levenbergMarquardt } from "./levenbergMarquardt.ts";
import { projectPoint } from "./twoView.ts";
import type { CameraIntrinsics, Keyframe, Mat3, MapPoint, Vec3 } from "./types.ts";

/** Refines one keyframe's pose against its currently-triangulated map points, which stay fixed. */
function refineKeyframePose(kf: Keyframe, mapPoints: Map<number, MapPoint>, K: CameraIntrinsics): void {
  const observations: { point: Vec3; observed: [number, number] }[] = [];
  for (const [kpIdx, mpId] of kf.pointForKeypoint) {
    const mp = mapPoints.get(mpId);
    if (!mp) continue;
    const kp = kf.keypoints[kpIdx];
    observations.push({ point: mp.position, observed: [kp.x, kp.y] });
  }
  if (observations.length < 6) return; // underconstrained - leave the pose as tracking produced it

  const initialParams = [...matrixToRotationVector(kf.pose.R), ...kf.pose.t];
  const residualFn = (params: number[]) => {
    const R = rotationVectorToMatrix(params.slice(0, 3)) as Mat3;
    const t = params.slice(3, 6) as Vec3;
    const residuals: number[] = [];
    for (const obs of observations) {
      const [u, v] = projectPoint(K, R, t, obs.point);
      residuals.push(u - obs.observed[0], v - obs.observed[1]);
    }
    return residuals;
  };

  const result = levenbergMarquardt(initialParams, residualFn, { maxIterations: 20 });
  kf.pose = {
    R: rotationVectorToMatrix(result.params.slice(0, 3)) as Mat3,
    t: result.params.slice(3, 6) as Vec3,
  };
}

/** Refines one map point's position against all of its observing keyframes' current (fixed) poses. */
function refineMapPoint(mp: MapPoint, keyframesById: Map<number, Keyframe>, K: CameraIntrinsics): void {
  const observations: { pose: Keyframe["pose"]; observed: [number, number] }[] = [];
  for (const obs of mp.observations) {
    const kf = keyframesById.get(obs.keyframeId);
    if (!kf) continue;
    const kp = kf.keypoints[obs.keypointIndex];
    observations.push({ pose: kf.pose, observed: [kp.x, kp.y] });
  }
  if (observations.length < 2) return;

  const residualFn = (params: number[]) => {
    const X = params as Vec3;
    const residuals: number[] = [];
    for (const obs of observations) {
      const [u, v] = projectPoint(K, obs.pose.R, obs.pose.t, X);
      residuals.push(u - obs.observed[0], v - obs.observed[1]);
    }
    return residuals;
  };

  const result = levenbergMarquardt([...mp.position], residualFn, { maxIterations: 15 });
  mp.position = result.params as Vec3;
}

/**
 * Windowed bundle adjustment implemented as alternating pose-only and
 * point-only refinement (block coordinate descent) rather than one joint
 * sparse solve. A full joint solve would need a Schur-complement sparse
 * solver (what g2o/Ceres do internally); at this window size a couple of
 * alternating rounds of small, dense, independent LM problems converges to a
 * very similar result with far simpler and more verifiable code. The oldest
 * keyframe in the window is held fixed as a gauge anchor so the window can't
 * drift as a rigid body during refinement.
 */
export function runWindowedBundleAdjustment(
  allKeyframes: Keyframe[],
  mapPoints: Map<number, MapPoint>,
  K: CameraIntrinsics,
  options: { windowSize?: number; rounds?: number } = {}
): void {
  const windowSize = options.windowSize ?? 6;
  const rounds = options.rounds ?? 2;
  const window = allKeyframes.slice(-windowSize);
  if (window.length < 2) return;

  const anchorId = window[0].id;
  const keyframesById = new Map(allKeyframes.map((kf) => [kf.id, kf]));

  const touchedPointIds = new Set<number>();
  for (const kf of window) {
    for (const mpId of kf.pointForKeypoint.values()) touchedPointIds.add(mpId);
  }

  for (let round = 0; round < rounds; round++) {
    for (const kf of window) {
      if (kf.id === anchorId) continue;
      refineKeyframePose(kf, mapPoints, K);
    }
    for (const mpId of touchedPointIds) {
      const mp = mapPoints.get(mpId);
      if (mp) refineMapPoint(mp, keyframesById, K);
    }
  }
}
