import { getCv } from "./opencvLoader.ts";

export interface DetectedFeatures {
  keypoints: { x: number; y: number }[];
  descriptors: Uint8Array[];
}

/**
 * Runs ORB detection on a raw BGR24 frame. OpenCV.js Mats/vectors are backed
 * by manually-managed WASM heap memory (no GC), so every Mat/vector created
 * here is deleted before returning - descriptor rows are copied out into
 * plain Uint8Arrays first since they're views into memory that becomes
 * invalid once the source Mat is deleted.
 */
export async function detectFeatures(
  frame: Buffer,
  width: number,
  height: number,
  maxFeatures = 500
): Promise<DetectedFeatures> {
  const cv = await getCv();
  const bgr = cv.matFromArray(height, width, cv.CV_8UC3, frame);
  const gray = new cv.Mat();
  const mask = new cv.Mat();
  const kp = new cv.KeyPointVector();
  const desc = new cv.Mat();
  const orb = new cv.ORB(maxFeatures);

  try {
    cv.cvtColor(bgr, gray, cv.COLOR_BGR2GRAY);
    orb.detectAndCompute(gray, mask, kp, desc);

    const keypoints: { x: number; y: number }[] = [];
    const descriptors: Uint8Array[] = [];
    const cols = desc.cols;
    for (let i = 0; i < kp.size(); i++) {
      const pt = kp.get(i).pt;
      keypoints.push({ x: pt.x, y: pt.y });
      descriptors.push(Uint8Array.from(desc.data.subarray(i * cols, (i + 1) * cols)));
    }
    return { keypoints, descriptors };
  } finally {
    bgr.delete();
    gray.delete();
    mask.delete();
    kp.delete();
    desc.delete();
    orb.delete();
  }
}
