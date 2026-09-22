import type { CameraIntrinsics } from "./types.ts";

/**
 * Approximates camera intrinsics from resolution alone, assuming a typical
 * smartphone-camera horizontal field of view. There is no calibration step
 * (no chessboard, no per-device metadata), so this is a known source of
 * systematic error - see the README's limitations section. A ~65 degree
 * horizontal FOV is a reasonable default for handheld phone video.
 */
export function approximateIntrinsics(width: number, height: number, horizontalFovDeg = 65): CameraIntrinsics {
  const fx = width / (2 * Math.tan((horizontalFovDeg * Math.PI) / 180 / 2));
  return {
    fx,
    fy: fx,
    cx: width / 2,
    cy: height / 2,
  };
}
