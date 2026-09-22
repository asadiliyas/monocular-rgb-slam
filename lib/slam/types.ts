export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

/** Row-major 3x3 rotation matrix, flattened. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];
export type Vec3 = [number, number, number];

export interface Pose {
  R: Mat3;
  t: Vec3;
}

export interface Observation {
  keyframeId: number;
  keypointIndex: number;
}

export interface MapPoint {
  id: number;
  position: Vec3;
  descriptor: Uint8Array;
  color: [number, number, number];
  observations: Observation[];
}

export interface Keyframe {
  id: number;
  frameIndex: number;
  timestampSec: number;
  pose: Pose;
  keypoints: { x: number; y: number }[];
  descriptors: Uint8Array[];
  /** keypointIndex -> mapPoint id, when triangulated/matched */
  pointForKeypoint: Map<number, number>;
}

export interface ExtractedFrame {
  index: number;
  timestampSec: number;
  data: Buffer;
}

export interface FrameSet {
  width: number;
  height: number;
  fps: number;
  frames: ExtractedFrame[];
}

export interface PipelineTimings {
  frameExtractionMs: number;
  featureDetectionMs: number;
  poseEstimationMs: number;
  bundleAdjustmentMs: number;
  loopClosureMs: number;
  totalMs: number;
}

export interface PipelineResult {
  trajectory: {
    keyframeId: number;
    timestampSec: number;
    position: Vec3;
    quaternion: [number, number, number, number];
  }[];
  points: {
    position: Vec3;
    color: [number, number, number];
  }[];
  timings: PipelineTimings;
  meta: {
    frameCount: number;
    keyframeCount: number;
    videoWidth: number;
    videoHeight: number;
    loopClosuresDetected: number;
  };
}
