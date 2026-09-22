import { extractFrames } from "./frames.ts";
import { detectFeatures } from "./features.ts";
import { matchDescriptors } from "./matching.ts";
import {
  recoverPoseAndPoints,
  triangulatePoint,
  projectPoint,
  projectionMatrix,
  cameraCenter,
  parallaxAngleDeg,
  type Point2,
  type RecoveredPose,
} from "./twoView.ts";
import { trackAgainstMap } from "./tracking.ts";
import { runWindowedBundleAdjustment } from "./bundleAdjust.ts";
import { detectAndCorrectLoopClosure } from "./loopClosure.ts";
import { approximateIntrinsics } from "./intrinsics.ts";
import { rotationMatrixToQuaternion } from "./linalg.ts";
import type { CameraIntrinsics, Keyframe, MapPoint, Pose, PipelineResult, Vec3 } from "./types.ts";

export interface PipelineOptions {
  maxDurationSec?: number;
  fps?: number;
  width?: number;
  height?: number;
  maxFeatures?: number;
  bundleAdjustEveryNKeyframes?: number;
  loopClosureEveryNKeyframes?: number;
}

const BOOTSTRAP_OFFSETS = [3, 5, 8, 12, 16, 20, 25, 30];
const MIN_BOOTSTRAP_INLIERS = 40;
const MIN_NEW_POINT_MATCHES = 8;
const MAX_REPROJECTION_ERROR_PX = 6;
const MIN_PARALLAX_DEG = 1.0;

function identityPose(): Pose {
  return { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
}

function sampleColor(frame: Buffer, width: number, height: number, x: number, y: number): [number, number, number] {
  const xi = Math.min(width - 1, Math.max(0, Math.round(x)));
  const yi = Math.min(height - 1, Math.max(0, Math.round(y)));
  const idx = (yi * width + xi) * 3;
  return [frame[idx + 2], frame[idx + 1], frame[idx]]; // BGR -> RGB
}

/**
 * Final defensive cleanup: even with the parallax-angle guard on triangulation
 * (see extendMap and twoView.ts), a point with only two observations from a
 * near-parallel viewpoint pair can still drift to an extreme distance during
 * unconstrained bundle-adjustment refinement (its reprojection-error cost
 * surface is nearly flat along the depth direction, so LM has little to
 * anchor it). A last statistical-outlier pass on the exported cloud - reject
 * anything many times farther from the point-cloud centroid than the typical
 * point - catches stragglers regardless of which stage produced them. This is
 * the same idea as PCL's StatisticalOutlierRemoval, simplified to a single
 * median-distance threshold since we only need to protect the visualization
 * and export, not feed the result back into tracking.
 */
function removeStatisticalOutliers<T extends { position: Vec3 }>(points: T[], medianMultiplier = 8): T[] {
  if (points.length < 5) return points;
  const n = points.length;
  const centroid: Vec3 = [0, 0, 0];
  for (const p of points) {
    centroid[0] += p.position[0] / n;
    centroid[1] += p.position[1] / n;
    centroid[2] += p.position[2] / n;
  }
  const distances = points.map((p) => Math.hypot(p.position[0] - centroid[0], p.position[1] - centroid[1], p.position[2] - centroid[2]));
  const sorted = [...distances].sort((a, b) => a - b);
  const median = sorted[Math.floor(n / 2)];
  const threshold = median * medianMultiplier;
  return points.filter((_, i) => distances[i] <= threshold);
}

function cameraCenterAndOrientation(pose: Pose): { position: Vec3; quaternion: [number, number, number, number] } {
  const R = pose.R;
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]; // R^T = camera-to-world rotation
  const t = pose.t;
  const position: Vec3 = [
    -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]),
    -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]),
    -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]),
  ];
  return { position, quaternion: rotationMatrixToQuaternion(Rt) };
}

/**
 * Runs the full monocular SLAM pipeline on a video file: frame extraction,
 * two-view bootstrap, per-frame tracking against the growing 3D map, periodic
 * windowed bundle adjustment, and loop-closure correction. Returns the
 * trajectory and sparse point cloud in world coordinates (up to the
 * unobservable monocular scale factor - see the README).
 */
export async function runSlamPipeline(videoPath: string, options: PipelineOptions = {}): Promise<PipelineResult> {
  const totalStart = Date.now();
  const maxFeatures = options.maxFeatures ?? 500;
  const bundleAdjustEvery = options.bundleAdjustEveryNKeyframes ?? 3;
  const loopClosureEvery = options.loopClosureEveryNKeyframes ?? 5;

  const timings = { featureDetectionMs: 0, poseEstimationMs: 0, bundleAdjustmentMs: 0, loopClosureMs: 0 };
  const timeSync = <T>(bucket: keyof typeof timings, fn: () => T): T => {
    const t0 = Date.now();
    const result = fn();
    timings[bucket] += Date.now() - t0;
    return result;
  };
  const timeAsync = async <T>(bucket: keyof typeof timings, fn: () => Promise<T>): Promise<T> => {
    const t0 = Date.now();
    const result = await fn();
    timings[bucket] += Date.now() - t0;
    return result;
  };

  const extractStart = Date.now();
  const frameSet = await extractFrames(videoPath, {
    maxDurationSec: options.maxDurationSec ?? 12,
    fps: options.fps ?? 8,
    width: options.width ?? 640,
    height: options.height ?? 360,
  });
  const frameExtractionMs = Date.now() - extractStart;

  if (frameSet.frames.length < 4) {
    throw new Error("Video too short or unreadable: fewer than 4 usable frames were extracted.");
  }

  const K: CameraIntrinsics = approximateIntrinsics(frameSet.width, frameSet.height);

  const featuresByFrame = new Map<number, Awaited<ReturnType<typeof detectFeatures>>>();
  const getFeatures = async (frameIdx: number) => {
    let f = featuresByFrame.get(frameIdx);
    if (!f) {
      f = await timeAsync("featureDetectionMs", () =>
        detectFeatures(frameSet.frames[frameIdx].data, frameSet.width, frameSet.height, maxFeatures)
      );
      featuresByFrame.set(frameIdx, f);
    }
    return f;
  };

  const feat0 = await getFeatures(0);

  // --- Bootstrap: find the first later frame with enough parallax relative to frame 0 ---
  let bootstrapIdx = -1;
  let bootstrapMatches: { queryIdx: number; trainIdx: number }[] = [];
  let bootstrapResult: RecoveredPose | null = null;

  for (const offset of BOOTSTRAP_OFFSETS) {
    if (offset >= frameSet.frames.length) break;
    const featI = await getFeatures(offset);
    const found = await timeAsync("poseEstimationMs", async () => {
      const matches = await matchDescriptors(feat0.descriptors, featI.descriptors, 0.75);
      if (matches.length < MIN_BOOTSTRAP_INLIERS) return null;

      const pts1: Point2[] = matches.map((m) => [feat0.keypoints[m.queryIdx].x, feat0.keypoints[m.queryIdx].y]);
      const pts2: Point2[] = matches.map((m) => [featI.keypoints[m.trainIdx].x, featI.keypoints[m.trainIdx].y]);
      const result = recoverPoseAndPoints(pts1, pts2, K, { iterations: 400, pixelThreshold: 3 });
      if (result && result.points.length >= MIN_BOOTSTRAP_INLIERS) {
        return { matches, result };
      }
      return null;
    });
    if (found) {
      bootstrapIdx = offset;
      bootstrapMatches = found.matches;
      bootstrapResult = found.result;
      break;
    }
  }

  if (bootstrapIdx === -1 || !bootstrapResult) {
    throw new Error(
      "Could not initialize SLAM from this video: no pair of early frames had enough matched, well-triangulated points. The camera may not have moved enough, or the scene may lack texture."
    );
  }

  let nextKeyframeId = 0;
  let nextMapPointId = 0;
  const mapPoints = new Map<number, MapPoint>();
  const keyframes: Keyframe[] = [];

  const kf0: Keyframe = {
    id: nextKeyframeId++,
    frameIndex: 0,
    timestampSec: frameSet.frames[0].timestampSec,
    pose: identityPose(),
    keypoints: feat0.keypoints,
    descriptors: feat0.descriptors,
    pointForKeypoint: new Map(),
  };
  const feat1 = await getFeatures(bootstrapIdx);
  const kf1: Keyframe = {
    id: nextKeyframeId++,
    frameIndex: bootstrapIdx,
    timestampSec: frameSet.frames[bootstrapIdx].timestampSec,
    pose: { R: bootstrapResult.R, t: bootstrapResult.t },
    keypoints: feat1.keypoints,
    descriptors: feat1.descriptors,
    pointForKeypoint: new Map(),
  };

  for (const { index, position } of bootstrapResult.points) {
    const match = bootstrapMatches[index];
    const id = nextMapPointId++;
    const color = sampleColor(frameSet.frames[bootstrapIdx].data, frameSet.width, frameSet.height, feat1.keypoints[match.trainIdx].x, feat1.keypoints[match.trainIdx].y);
    const mapPoint: MapPoint = {
      id,
      position,
      descriptor: feat0.descriptors[match.queryIdx],
      color,
      observations: [
        { keyframeId: kf0.id, keypointIndex: match.queryIdx },
        { keyframeId: kf1.id, keypointIndex: match.trainIdx },
      ],
    };
    mapPoints.set(id, mapPoint);
    kf0.pointForKeypoint.set(match.queryIdx, id);
    kf1.pointForKeypoint.set(match.trainIdx, id);
  }

  keyframes.push(kf0, kf1);

  /** Triangulates new map points between the two most recent keyframes for keypoints neither has already matched to the map. */
  async function extendMap(prevKf: Keyframe, newKf: Keyframe, newFrameIdx: number) {
    const prevUnmatched: number[] = [];
    const prevUnmatchedDescriptors: Uint8Array[] = [];
    prevKf.keypoints.forEach((_, i) => {
      if (!prevKf.pointForKeypoint.has(i)) {
        prevUnmatched.push(i);
        prevUnmatchedDescriptors.push(prevKf.descriptors[i]);
      }
    });
    const newUnmatched: number[] = [];
    const newUnmatchedDescriptors: Uint8Array[] = [];
    newKf.keypoints.forEach((_, i) => {
      if (!newKf.pointForKeypoint.has(i)) {
        newUnmatched.push(i);
        newUnmatchedDescriptors.push(newKf.descriptors[i]);
      }
    });
    if (prevUnmatched.length < MIN_NEW_POINT_MATCHES || newUnmatched.length < MIN_NEW_POINT_MATCHES) return;

    const matches = await matchDescriptors(newUnmatchedDescriptors, prevUnmatchedDescriptors, 0.75);
    if (matches.length === 0) return;

    const P1 = projectionMatrix(K, prevKf.pose.R, prevKf.pose.t);
    const P2 = projectionMatrix(K, newKf.pose.R, newKf.pose.t);
    const C1 = cameraCenter(prevKf.pose.R, prevKf.pose.t);
    const C2 = cameraCenter(newKf.pose.R, newKf.pose.t);

    for (const m of matches) {
      const newKpIdx = newUnmatched[m.queryIdx];
      const prevKpIdx = prevUnmatched[m.trainIdx];
      if (newKf.pointForKeypoint.has(newKpIdx) || prevKf.pointForKeypoint.has(prevKpIdx)) continue; // claimed by an earlier match in this same loop

      const p1: Point2 = [prevKf.keypoints[prevKpIdx].x, prevKf.keypoints[prevKpIdx].y];
      const p2: Point2 = [newKf.keypoints[newKpIdx].x, newKf.keypoints[newKpIdx].y];
      const X = triangulatePoint(P1, P2, p1, p2);

      const depth1 = prevKf.pose.R[6] * X[0] + prevKf.pose.R[7] * X[1] + prevKf.pose.R[8] * X[2] + prevKf.pose.t[2];
      const depth2 = newKf.pose.R[6] * X[0] + newKf.pose.R[7] * X[1] + newKf.pose.R[8] * X[2] + newKf.pose.t[2];
      if (depth1 <= 0 || depth2 <= 0) continue;
      if (parallaxAngleDeg(C1, C2, X) < MIN_PARALLAX_DEG) continue; // degenerate near-parallel-ray triangulation

      const [u1, v1] = projectPoint(K, prevKf.pose.R, prevKf.pose.t, X);
      const [u2, v2] = projectPoint(K, newKf.pose.R, newKf.pose.t, X);
      const err1 = Math.hypot(u1 - p1[0], v1 - p1[1]);
      const err2 = Math.hypot(u2 - p2[0], v2 - p2[1]);
      if (err1 > MAX_REPROJECTION_ERROR_PX || err2 > MAX_REPROJECTION_ERROR_PX) continue;

      const id = nextMapPointId++;
      const color = sampleColor(frameSet.frames[newFrameIdx].data, frameSet.width, frameSet.height, p2[0], p2[1]);
      const mapPoint: MapPoint = {
        id,
        position: X,
        descriptor: newKf.descriptors[newKpIdx],
        color,
        observations: [
          { keyframeId: prevKf.id, keypointIndex: prevKpIdx },
          { keyframeId: newKf.id, keypointIndex: newKpIdx },
        ],
      };
      mapPoints.set(id, mapPoint);
      prevKf.pointForKeypoint.set(prevKpIdx, id);
      newKf.pointForKeypoint.set(newKpIdx, id);
    }
  }

  await timeAsync("poseEstimationMs", () => extendMap(kf0, kf1, bootstrapIdx));

  /** Local map used for per-frame tracking: points seen by recently-added keyframes only, so
   * matching cost stays bounded as the video (and total map size) grows, rather than matching
   * every frame against the entire historical map. Loop closure separately searches older
   * keyframes directly (see loopClosure.ts) - that's the deliberate exception to "recent only". */
  const LOCAL_MAP_WINDOW = 20;
  function getLocalMapPoints(): MapPoint[] {
    const recentKeyframes = keyframes.slice(-LOCAL_MAP_WINDOW);
    const ids = new Set<number>();
    for (const kf of recentKeyframes) {
      for (const id of kf.pointForKeypoint.values()) ids.add(id);
    }
    const points: MapPoint[] = [];
    for (const id of ids) {
      const p = mapPoints.get(id);
      if (p) points.push(p);
    }
    return points;
  }

  // --- Track every remaining sampled frame against the growing map ---
  let loopClosuresDetected = 0;
  for (let frameIdx = bootstrapIdx + 1; frameIdx < frameSet.frames.length; frameIdx++) {
    const feat = await getFeatures(frameIdx);
    const localMapPoints = getLocalMapPoints();
    const tracked = await timeAsync("poseEstimationMs", () =>
      trackAgainstMap(feat.keypoints, feat.descriptors, localMapPoints, K)
    );
    if (!tracked) continue; // tracking lost for this frame; skip and keep trying subsequent frames

    const newKf: Keyframe = {
      id: nextKeyframeId++,
      frameIndex: frameIdx,
      timestampSec: frameSet.frames[frameIdx].timestampSec,
      pose: { R: tracked.R, t: tracked.t },
      keypoints: feat.keypoints,
      descriptors: feat.descriptors,
      pointForKeypoint: new Map(),
    };
    for (const m of tracked.matches) {
      newKf.pointForKeypoint.set(m.keypointIndex, m.mapPointId);
      mapPoints.get(m.mapPointId)?.observations.push({ keyframeId: newKf.id, keypointIndex: m.keypointIndex });
    }

    const prevKf = keyframes[keyframes.length - 1];
    keyframes.push(newKf);
    await timeAsync("poseEstimationMs", () => extendMap(prevKf, newKf, frameIdx));

    if (keyframes.length % bundleAdjustEvery === 0) {
      timeSync("bundleAdjustmentMs", () =>
        runWindowedBundleAdjustment(keyframes, mapPoints, K, { windowSize: 6, rounds: 2 })
      );
    }

    if (keyframes.length % loopClosureEvery === 0) {
      const loop = await timeAsync("loopClosureMs", () => detectAndCorrectLoopClosure(newKf, keyframes, mapPoints, K));
      if (loop) {
        loopClosuresDetected++;
        timeSync("bundleAdjustmentMs", () =>
          runWindowedBundleAdjustment(keyframes, mapPoints, K, { windowSize: Math.min(keyframes.length, 12), rounds: 2 })
        );
      }
    }
  }

  // Final cleanup bundle adjustment pass over the tail of the trajectory.
  timeSync("bundleAdjustmentMs", () => runWindowedBundleAdjustment(keyframes, mapPoints, K, { windowSize: 8, rounds: 2 }));

  const trajectory = keyframes.map((kf) => {
    const { position, quaternion } = cameraCenterAndOrientation(kf.pose);
    return { keyframeId: kf.id, timestampSec: kf.timestampSec, position, quaternion };
  });

  const points = removeStatisticalOutliers(
    Array.from(mapPoints.values())
      .filter((p) => p.observations.length >= 2)
      .map((p) => ({ position: p.position, color: p.color }))
  );

  const totalMs = Date.now() - totalStart;

  return {
    trajectory,
    points,
    timings: {
      frameExtractionMs,
      ...timings,
      totalMs,
    },
    meta: {
      frameCount: frameSet.frames.length,
      keyframeCount: keyframes.length,
      videoWidth: frameSet.width,
      videoHeight: frameSet.height,
      loopClosuresDetected,
    },
  };
}
