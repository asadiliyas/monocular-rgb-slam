import { matchDescriptors } from "./matching.ts";
import { trackAgainstMap } from "./tracking.ts";
import { matrixToRotationVector, rotationVectorToMatrix } from "./linalg.ts";
import type { CameraIntrinsics, Keyframe, MapPoint, Pose } from "./types.ts";

const MIN_KEYFRAME_GAP = 12;
const MIN_LOOP_MATCHES = 30;

function poseRelativeTo(reference: Pose, pose: Pose): Pose {
  // R_rel = R * R_ref^T, t_rel = t - R_rel * t_ref  (transform from `reference`'s camera frame to `pose`'s)
  const Rref = reference.R;
  const RrefT = [Rref[0], Rref[3], Rref[6], Rref[1], Rref[4], Rref[7], Rref[2], Rref[5], Rref[8]];
  const R = pose.R;
  const Rrel = new Array(9) as Pose["R"];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += R[r * 3 + k] * RrefT[k * 3 + c];
      Rrel[r * 3 + c] = s;
    }
  }
  const tRef = reference.t;
  const tRel: Pose["t"] = [
    pose.t[0] - (Rrel[0] * tRef[0] + Rrel[1] * tRef[1] + Rrel[2] * tRef[2]),
    pose.t[1] - (Rrel[3] * tRef[0] + Rrel[4] * tRef[1] + Rrel[5] * tRef[2]),
    pose.t[2] - (Rrel[6] * tRef[0] + Rrel[7] * tRef[1] + Rrel[8] * tRef[2]),
  ];
  return { R: Rrel, t: tRel };
}

function composeFromRelative(reference: Pose, relative: Pose): Pose {
  const Rrel = relative.R;
  const Rref = reference.R;
  const R = new Array(9) as Pose["R"];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += Rrel[r * 3 + k] * Rref[k * 3 + c];
      R[r * 3 + c] = s;
    }
  }
  const tRef = reference.t;
  const t: Pose["t"] = [
    relative.t[0] + (Rrel[0] * tRef[0] + Rrel[1] * tRef[1] + Rrel[2] * tRef[2]),
    relative.t[1] + (Rrel[3] * tRef[0] + Rrel[4] * tRef[1] + Rrel[5] * tRef[2]),
    relative.t[2] + (Rrel[6] * tRef[0] + Rrel[7] * tRef[1] + Rrel[8] * tRef[2]),
  ];
  return { R, t };
}

export interface LoopClosureResult {
  matchedKeyframeId: number;
  correctedKeyframeCount: number;
}

/**
 * Looks for a revisit of an earlier part of the scene and, if found, corrects
 * accumulated drift across the keyframes since then.
 *
 * Verification reuses trackAgainstMap: matching the current keyframe's
 * descriptors against an older keyframe's own map points and running
 * solvePnPRansac gives a pose for the current keyframe expressed directly in
 * the map's existing (already-scaled) coordinate frame - this sidesteps the
 * scale ambiguity a fresh essential-matrix estimate would have, since it
 * reuses 3D points the map already agrees on.
 *
 * The discrepancy between that loop-implied pose and the current (drifted)
 * pose is then distributed across the intermediate keyframes with a linear
 * blend in (rotation-vector, translation) space relative to the matched
 * keyframe, which stays fixed as the anchor. This is a deliberate
 * simplification of full pose-graph optimization (no sparse solver, no
 * information-weighted smoothing) - appropriate for the short, single-loop
 * clips this system targets; seee the README for the tradeoff.
 */
export async function detectAndCorrectLoopClosure(
  currentKeyframe: Keyframe,
  allKeyframes: Keyframe[],
  mapPoints: Map<number, MapPoint>,
  K: CameraIntrinsics
): Promise<LoopClosureResult | null> {
  const currentIdx = allKeyframes.findIndex((kf) => kf.id === currentKeyframe.id);
  if (currentIdx < MIN_KEYFRAME_GAP) return null;

  const candidates = allKeyframes.slice(0, currentIdx - MIN_KEYFRAME_GAP + 1);
  if (candidates.length === 0) return null;

  let bestCandidate: Keyframe | null = null;
  let bestMatchCount = 0;
  for (const candidate of candidates) {
    const matches = await matchDescriptors(currentKeyframe.descriptors, candidate.descriptors, 0.75);
    if (matches.length > bestMatchCount) {
      bestMatchCount = matches.length;
      bestCandidate = candidate;
    }
  }

  if (!bestCandidate || bestMatchCount < MIN_LOOP_MATCHES) return null;

  const candidateMapPointIds = new Set(bestCandidate.pointForKeypoint.values());
  const candidateMapPoints = [...candidateMapPointIds]
    .map((id) => mapPoints.get(id))
    .filter((p): p is MapPoint => p !== undefined);
  if (candidateMapPoints.length < MIN_LOOP_MATCHES / 2) return null;

  const loopTracking = await trackAgainstMap(currentKeyframe.keypoints, currentKeyframe.descriptors, candidateMapPoints, K, {
    ratioThreshold: 0.75,
  });
  if (!loopTracking || loopTracking.matches.length < MIN_LOOP_MATCHES / 2) return null;

  const loopPose: Pose = { R: loopTracking.R, t: loopTracking.t };
  const candidateIdx = allKeyframes.findIndex((kf) => kf.id === bestCandidate!.id);

  const relativeDrifted = poseRelativeTo(bestCandidate.pose, currentKeyframe.pose);
  const relativeLoop = poseRelativeTo(bestCandidate.pose, loopPose);

  const rvDrifted = matrixToRotationVector(relativeDrifted.R);
  const rvLoop = matrixToRotationVector(relativeLoop.R);
  const deltaRv = rvLoop.map((v, i) => v - rvDrifted[i]);
  const deltaT = relativeLoop.t.map((v, i) => v - relativeDrifted.t[i]);

  const span = currentIdx - candidateIdx;
  let correctedCount = 0;
  for (let i = candidateIdx + 1; i <= currentIdx; i++) {
    const kf = allKeyframes[i];
    const alpha = (i - candidateIdx) / span;
    const relative = poseRelativeTo(bestCandidate.pose, kf.pose);
    const rv = matrixToRotationVector(relative.R).map((v, k) => v + alpha * deltaRv[k]);
    const t = relative.t.map((v, k) => v + alpha * deltaT[k]) as Pose["t"];
    const correctedRelative: Pose = { R: rotationVectorToMatrix(rv) as Pose["R"], t };
    kf.pose = composeFromRelative(bestCandidate.pose, correctedRelative);
    correctedCount++;
  }

  return { matchedKeyframeId: bestCandidate.id, correctedKeyframeCount: correctedCount };
}
