import { execa } from "execa";
import ffmpegPath from "ffmpeg-static";
import type { FrameSet } from "./types.ts";

export interface ExtractOptions {
  /** Hard cap on how much of the source video is read, protecting the performance budget against long uploads. */
  maxDurationSec?: number;
  fps?: number;
  /** Longer side of the output frame is capped at this; the shorter side is derived to preserve the source aspect ratio (see below for why this matters). */
  maxDimension?: number;
}

const BYTES_PER_PIXEL = 3; // bgr24

/**
 * Extracts a bounded, downsampled sequence of raw BGR24 frames from a video
 * file using a bundled static ffmpeg binary. Frames are read as one
 * concatenated raw stream on stdout (no intermediate image files, no JPEG
 * decode step needed) since OpenCV.js's WASM build has no image codec support
 * outside a browser canvas.
 *
 * Scaling preserves the source aspect ratio (capping the longer side, no
 * padding) rather than forcing a fixed landscape frame size. An earlier
 * version force-fit every video into a fixed 640x360 box with letterbox
 * padding, which is fine for landscape sources but silently wrecked portrait
 * ones: a 1080x1920 phone video would be scaled down to fit *within* 640x360,
 * which for that aspect ratio means shrinking to ~202x360 and padding the
 * remaining ~438px on each side with black - discarding most of the actual
 * frame content before ORB ever saw it, and starving two-view bootstrap of
 * the matches it needs. Since final dimensions now depend on the source's
 * own aspect ratio and aren't known until ffmpeg computes them, they're
 * parsed back out of ffmpeg's own stderr rather than assumed up front.
 */
export async function extractFrames(inputPath: string, options: ExtractOptions = {}): Promise<FrameSet> {
  if (!ffmpegPath) {
    throw new Error("ffmpeg-static did not resolve a bundled ffmpeg binary for this platform.");
  }

  const fps = options.fps ?? 6;
  const maxDimension = options.maxDimension ?? 640;
  const maxDurationSec = options.maxDurationSec ?? 10;

  // "if landscape-or-square: cap width, derive height; else: cap height, derive width" - -2 forces the derived side to be even, which raw/most pixel formats require.
  const filter = `fps=${fps},scale='if(gte(iw,ih),${maxDimension},-2)':'if(gte(iw,ih),-2,${maxDimension})'`;

  const result = await execa(
    ffmpegPath,
    ["-nostdin", "-t", String(maxDurationSec), "-i", inputPath, "-vf", filter, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
    { encoding: "buffer", maxBuffer: 300_000_000, reject: true }
  );

  const stderrText = Buffer.from(result.stderr).toString("utf8");
  const { width, height } = parseOutputDimensions(stderrText);

  const raw = Buffer.from(result.stdout);
  const frameBytes = width * height * BYTES_PER_PIXEL;
  const frameCount = Math.floor(raw.length / frameBytes);

  const frames = [];
  for (let i = 0; i < frameCount; i++) {
    frames.push({
      index: i,
      timestampSec: i / fps,
      data: raw.subarray(i * frameBytes, (i + 1) * frameBytes),
    });
  }

  return { width, height, fps, frames };
}

function parseOutputDimensions(stderrText: string): { width: number; height: number } {
  const outputSectionStart = stderrText.indexOf("Output #0");
  const outputSection = outputSectionStart >= 0 ? stderrText.slice(outputSectionStart) : stderrText;
  const match = outputSection.match(/Video:.*?(\d{2,5})x(\d{2,5})/);
  if (!match) {
    throw new Error("Could not determine the extracted frame size from ffmpeg's output.");
  }
  return { width: parseInt(match[1], 10), height: parseInt(match[2], 10) };
}
