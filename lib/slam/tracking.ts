import { getCv } from "./opencvLoader.ts";
import { matchDescriptors } from "./matching.ts";
import { rotationVectorToMatrix } from "./linalg.ts";
import type { CameraIntrinsics, Mat3, MapPoint, Vec3 } from "./types.ts";

export interface TrackingResult {
  R: Mat3;
  t: Vec3;
  /** Successful 2D (new frame keypoint) <-> 3D (existing map point) correspondences. */
  matches: { mapPointId: number; keypointIndex: number }[];
}

const MIN_MATCHES_FOR_PNP = 8;

/**
 * Localizes a new frame against the existing 3D map: matches its ORB
 * descriptors to map points' descriptors, then recovers absolute pose via
 * solvePnPRansac. This is the main tracking step (registering each frame
 * against the accumulated, bundle-adjusted map rather than only the previous
 * frame) which is what keeps drift from compounding as fast as naive
 * frame-to-frame chaining would.
 */
export async function trackAgainstMap(
  keypoints: { x: number; y: number }[],
  descriptors: Uint8Array[],
  mapPoints: MapPoint[],
  K: CameraIntrinsics,
  options: { ratioThreshold?: number; reprojectionErrorPx?: number; iterations?: number } = {}
): Promise<TrackingResult | null> {
  if (mapPoints.length < MIN_MATCHES_FOR_PNP || descriptors.length === 0) return null;

  const matches = await matchDescriptors(
    descriptors,
    mapPoints.map((p) => p.descriptor),
    options.ratioThreshold ?? 0.8
  );
  if (matches.length < MIN_MATCHES_FOR_PNP) return null;

  const cv = await getCv();
  const objectPointsFlat: number[] = [];
  const imagePointsFlat: number[] = [];
  for (const m of matches) {
    const pos = mapPoints[m.trainIdx].position;
    objectPointsFlat.push(pos[0], pos[1], pos[2]);
    const kp = keypoints[m.queryIdx];
    imagePointsFlat.push(kp.x, kp.y);
  }

  const objMat = cv.matFromArray(matches.length, 1, cv.CV_64FC3, objectPointsFlat);
  const imgMat = cv.matFromArray(matches.length, 1, cv.CV_64FC2, imagePointsFlat);
  const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1]);
  const distCoeffs = new cv.Mat();
  const rvec = new cv.Mat();
  const tvec = new cv.Mat();
  const inliers = new cv.Mat();

  try {
    const ok = cv.solvePnPRansac(
      objMat,
      imgMat,
      cameraMatrix,
      distCoeffs,
      rvec,
      tvec,
      false,
      options.iterations ?? 200,
      options.reprojectionErrorPx ?? 4.0,
      0.999,
      inliers
    );
    if (!ok || inliers.rows < MIN_MATCHES_FOR_PNP) return null;

    const R = rotationVectorToMatrix(Array.from(rvec.data64F)) as Mat3;
    const t = Array.from(tvec.data64F) as Vec3;
    const inlierIdx = Array.from(inliers.data32S as Int32Array);
    const resultMatches = inlierIdx.map((i) => ({
      mapPointId: mapPoints[matches[i].trainIdx].id,
      keypointIndex: matches[i].queryIdx,
    }));

    return { R, t, matches: resultMatches };
  } finally {
    objMat.delete();
    imgMat.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();
    inliers.delete();
  }
}
