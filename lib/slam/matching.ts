import { getCv } from "./opencvLoader.ts";

export interface DescriptorMatch {
  queryIdx: number;
  trainIdx: number;
  distance: number;
}

function flattenDescriptors(rows: Uint8Array[]): { flat: Uint8Array; cols: number } {
  const cols = rows[0]?.length ?? 32;
  const flat = new Uint8Array(rows.length * cols);
  rows.forEach((row, i) => flat.set(row, i * cols));
  return { flat, cols };
}

/**
 * ORB descriptor matching (Hamming distance) with Lowe's ratio test, mirroring
 * the classic BFMatcher + knnMatch(k=2) pattern used throughout the pipeline
 * (frame-to-frame bootstrap, frame-to-map tracking, and loop-closure checks).
 */
export async function matchDescriptors(
  queryDescriptors: Uint8Array[],
  trainDescriptors: Uint8Array[],
  ratioThreshold = 0.75
): Promise<DescriptorMatch[]> {
  if (queryDescriptors.length === 0 || trainDescriptors.length < 2) return [];

  const cv = await getCv();
  const { flat: queryFlat, cols } = flattenDescriptors(queryDescriptors);
  const { flat: trainFlat } = flattenDescriptors(trainDescriptors);

  const queryMat = cv.matFromArray(queryDescriptors.length, cols, cv.CV_8U, queryFlat);
  const trainMat = cv.matFromArray(trainDescriptors.length, cols, cv.CV_8U, trainFlat);
  const bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
  const knn = new cv.DMatchVectorVector();

  try {
    bf.knnMatch(queryMat, trainMat, knn, 2);
    const matches: DescriptorMatch[] = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() < 2) continue;
      const best = pair.get(0);
      const second = pair.get(1);
      if (best.distance < ratioThreshold * second.distance) {
        matches.push({ queryIdx: best.queryIdx, trainIdx: best.trainIdx, distance: best.distance });
      }
    }
    return matches;
  } finally {
    queryMat.delete();
    trainMat.delete();
    bf.delete();
    knn.delete();
  }
}
