"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { PipelineResult } from "@/lib/slam/types";

const Viewer3D = dynamic(() => import("@/components/Viewer3D"), { ssr: false });

type Status = "idle" | "processing" | "done" | "error";

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [resultKey, setResultKey] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (status === "processing") {
      const start = Date.now();
      timerRef.current = setInterval(() => setElapsedMs(Date.now() - start), 100);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [status]);

  const processFile = useCallback(async (file: File) => {
    setStatus("processing");
    setError(null);
    setResult(null);
    setFileName(file.name);
    setElapsedMs(0);

    try {
      const formData = new FormData();
      formData.append("video", file);
      const res = await fetch("/api/process", { method: "POST", body: formData });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.error ?? `Request failed with status ${res.status}`);
      }
      setResult(body as PipelineResult);
      setResultKey((k) => k + 1);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) void processFile(file);
    },
    [processFile]
  );

  return (
    <div className="flex min-h-screen flex-col bg-neutral-50">
      <header className="border-b border-neutral-200 bg-white px-6 py-4">
        <h1 className="text-lg font-semibold text-neutral-900">Monocular RGB Sparse SLAM</h1>
        <p className="text-sm text-neutral-500">
          Upload a short handheld video to estimate the camera trajectory and a sparse 3D point cloud.
        </p>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-8">
        <div
          className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
            dragActive ? "border-cyan-500 bg-cyan-50" : "border-neutral-300 bg-white"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragActive(false);
            handleFiles(e.dataTransfer.files);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <p className="text-neutral-600">Drag and drop a video here, or</p>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={status === "processing"}
            className="mt-3 rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 disabled:opacity-50"
          >
            Choose a video
          </button>
          <p className="mt-2 text-xs text-neutral-400">MP4/MOV, up to 150MB. Only the first ~12s are processed.</p>
        </div>

        {status === "processing" && (
          <div className="flex items-center gap-3 rounded-lg border border-neutral-200 bg-white p-4">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-900" />
            <span className="text-sm text-neutral-700">
              Processing {fileName}... {(elapsedMs / 1000).toFixed(1)}s elapsed
            </span>
          </div>
        )}

        {status === "error" && error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            <p className="font-medium">Processing failed</p>
            <p className="mt-1">{error}</p>
          </div>
        )}

        {status === "done" && result && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Total time" value={`${(result.timings.totalMs / 1000).toFixed(2)}s`} />
              <Stat label="Keyframes" value={String(result.meta.keyframeCount)} />
              <Stat label="Map points" value={String(result.points.length)} />
              <Stat label="Loop closures" value={String(result.meta.loopClosuresDetected)} />
            </div>

            <div className="h-[560px] w-full overflow-hidden rounded-xl border border-neutral-200">
              <Viewer3D key={resultKey} result={result} />
            </div>

            <details className="rounded-lg border border-neutral-200 bg-white p-4 text-sm text-neutral-600">
              <summary className="cursor-pointer font-medium text-neutral-800">Timing breakdown</summary>
              <ul className="mt-2 space-y-1">
                <li>Frame extraction: {result.timings.frameExtractionMs}ms</li>
                <li>Feature detection: {result.timings.featureDetectionMs}ms</li>
                <li>Pose estimation &amp; tracking: {result.timings.poseEstimationMs}ms</li>
                <li>Bundle adjustment: {result.timings.bundleAdjustmentMs}ms</li>
                <li>Loop closure: {result.timings.loopClosureMs}ms</li>
              </ul>
            </details>
          </>
        )}
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-3">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="text-lg font-semibold text-neutral-900">{value}</div>
    </div>
  );
}
