import { NextRequest, NextResponse } from "next/server";
import { writeFile, rm, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { runSlamPipeline } from "@/lib/slam/pipeline";

// Needs the Node runtime (child_process for ffmpeg, WASM for OpenCV.js) - not Edge.
export const runtime = "nodejs";

const MAX_UPLOAD_BYTES = 150 * 1024 * 1024;

export async function POST(request: NextRequest) {
  let tempDir: string | null = null;
  try {
    const formData = await request.formData();
    const file = formData.get("video");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "No video file was provided." }, { status: 400 });
    }
    if (file.size === 0) {
      return NextResponse.json({ error: "The uploaded file is empty." }, { status: 400 });
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      return NextResponse.json(
        { error: `File too large - max ${Math.floor(MAX_UPLOAD_BYTES / 1024 / 1024)}MB.` },
        { status: 413 }
      );
    }

    tempDir = await mkdtemp(path.join(tmpdir(), "slam-"));
    const tempPath = path.join(tempDir, `input${path.extname(file.name) || ".mp4"}`);
    await writeFile(tempPath, Buffer.from(await file.arrayBuffer()));

    const result = await runSlamPipeline(tempPath);
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error while processing the video.";
    const isExpectedFailure = /could not initialize slam|too short or unreadable/i.test(message);
    return NextResponse.json({ error: message }, { status: isExpectedFailure ? 422 : 500 });
  } finally {
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true }).catch(() => {});
    }
  }
}
