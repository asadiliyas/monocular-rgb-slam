import { execa } from "execa";
import ffmpegPath from "ffmpeg-static";
import type { FrameSet } from "./types.ts";

export interface ExtractOptions {
  /** Hard cap on how much of the source video is read, protecting the performance budget against long uploads. */
  maxDurationSec?: number;
  fps?: number;
  width?: number;
  height?: number;
}

const BYTES_PER_PIXEL = 3; // bgr24

/**
 * Extracts a bounded, downsampled sequence of raw BGR24 frames from a video
 * file using a bundled static ffmpeg binary. Frames are read as one
 * concatenated raw stream on stdout (no intermediate image files, no JPEG
 * decode step needed) since OpenCV.js's WASM build has no image codec support
 * outside a browser canvas.
 */
export async function extractFrames(inputPath: string, options: ExtractOptions = {}): Promise<FrameSet> {
  if (!ffmpegPath) {
    throw new Error("ffmpeg-static did not resolve a bundled ffmpeg binary for this platform.");
  }

  const fps = options.fps ?? 8;
  const width = options.width ?? 640;
  const height = options.height ?? 360;
  const maxDurationSec = options.maxDurationSec ?? 12;

  const filter = `fps=${fps},scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2`;

  const result = await execa(
    ffmpegPath,
    [
      "-nostdin",
      "-t",
      String(maxDurationSec),
      "-i",
      inputPath,
      "-vf",
      filter,
      "-f",
      "rawvideo",
      "-pix_fmt",
      "bgr24",
      "pipe:1",
    ],
    { encoding: "buffer", maxBuffer: 300_000_000, reject: true }
  );

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
