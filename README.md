# Monocular RGB Sparse SLAM

Upload a short handheld video and get back an estimated camera trajectory and
a sparse 3D point cloud, visualized in the browser.

**Live app:** _deploying — see [Deployment](#deployment) for status_

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Major technical decisions](#major-technical-decisions)
- [Libraries, frameworks, and models](#libraries-frameworks-and-models)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)
- [Measured performance](#measured-performance)

## Quick start

Requirements: Node.js 24+, npm. No native build tools needed — `ffmpeg-static`
and `@techstark/opencv-js` both ship prebuilt binaries/WASM.

```bash
npm install
npm run dev
# open http://localhost:3000, upload a short video
```

Production build:

```bash
npm run build
npm start
```

Or via Docker (this is what's actually deployed):

```bash
docker build -t monocular-rgb-slam .
docker run -p 3000:3000 monocular-rgb-slam
```

## Architecture

Single Next.js 16 (App Router) + TypeScript app, deployed as one container.
There is no separate backend service — the SLAM pipeline runs inside a Next.js
Route Handler on the Node.js runtime.

```
Browser (upload UI, react-three-fiber viewer)
        │  POST /api/process (multipart video)
        ▼
Next.js Route Handler (app/api/process/route.ts)
        │  writes upload to a temp file
        ▼
SLAM pipeline (lib/slam/pipeline.ts), synchronous, single request/response
  1. frames.ts        — ffmpeg-static extracts a downsampled raw BGR24 stream
  2. features.ts       — ORB keypoints + descriptors (OpenCV.js/WASM)
  3. twoView.ts         — two-view bootstrap: RANSAC fundamental matrix →
                          essential matrix → decomposition → DLT triangulation
                          (hand-implemented — see below for why)
  4. tracking.ts        — per-frame localization via solvePnPRansac against
                          the accumulated 3D map (OpenCV.js)
  5. bundleAdjust.ts    — windowed pose/point refinement (hand-rolled
                          Levenberg-Marquardt) to control local drift
  6. loopClosure.ts     — revisit detection + drift correction across the
                          affected keyframes
        │  JSON: trajectory + point cloud + timings
        ▼
Viewer3D.tsx (react-three-fiber / three.js), rendered client-side
```

Processing is synchronous (upload → JSON response in one request) because the
whole pipeline is designed to finish in single-digit seconds — there's no
value in a job queue at this scale, and it keeps the architecture simple.

## Major technical decisions

**Everything in TypeScript, including the CV math.** The team's preferred
stack is Next.js/TypeScript, and the alternative (a Python microservice using
OpenCV-Python) would have made pose estimation and triangulation trivial via
`cv2.findEssentialMat`/`recoverPose`/`triangulatePoints`. We tested
`@techstark/opencv-js` (the standard OpenCV.js WASM build) directly in Node
before committing to this path, and found that **calib3d's two-view geometry
functions are not compiled into any standard OpenCV.js build** — this is a
long-standing, deliberate upstream limitation
([opencv#14293](https://github.com/opencv/opencv/issues/14293)), not specific
to this package. `solvePnP`/`solvePnPRansac`/`findHomography`/`Rodrigues` are
present and used directly; the fundamental/essential matrix estimation and
triangulation in `lib/slam/twoView.ts` and the small linear-algebra kernel
(`lib/slam/linalg.ts` — a generic Jacobi eigenvalue solver, used for
SVD-via-eigendecomposition) had to be implemented from the classic Hartley &
Zisserman formulation. This was validated against synthetic ground-truth data
(known R/t, projected points, recovered pose compared to ground truth) before
being wired into the pipeline.

**Windowed bundle adjustment as alternating pose/point refinement, not a
joint sparse solve.** A textbook joint bundle adjustment couples every pose
and every point in one large sparse system, normally solved via Schur
complement (what g2o/Ceres do internally). Implementing that from scratch was
judged too much additional hand-rolled numerical code for the time available.
Instead, `bundleAdjust.ts` alternates two small, dense, well-conditioned
problems: refine each keyframe's pose (6 parameters) with points fixed, then
refine each point's position (3 parameters) with poses fixed, repeated for a
couple of rounds. This is a standard block-coordinate-descent approximation
of full BA — cheaper and much simpler to verify correct, at some cost to
optimality versus a joint solve.

**Tracking against the map, not frame-to-frame chaining.** Each frame is
localized via `solvePnPRansac` against the accumulated (and periodically
bundle-adjusted) 3D map, rather than only against the immediately preceding
frame. Errors in one frame don't directly propagate into the next frame's
pose estimate the way naive relative-pose chaining would — this is the
primary mechanism satisfying the "minimize accumulated pose error and drift"
requirement, with bundle adjustment and loop closure as secondary corrections.

**Loop closure via re-localization, not a second essential-matrix estimate.**
Detecting a revisit and estimating a fresh essential matrix between the two
keyframes would only recover a relative pose up to an independent, unrelated
scale factor — not directly comparable to the map's existing scale. Instead,
loop closure matches the current frame's descriptors against an older
keyframe's own map points and re-runs `solvePnPRansac`, which yields a pose
in the map's existing (already-scaled) coordinate frame. The discrepancy
between that and the current drifted pose is distributed across the
intermediate keyframes with a linear blend in (rotation-vector, translation)
space, anchored at the matched keyframe. This is a deliberate simplification
of full pose-graph optimization (no sparse solver, no uncertainty weighting)
appropriate for short, single-loop clips.

**No camera calibration step.** Intrinsics are approximated from the video's
resolution assuming a ~65° horizontal field of view (typical for a phone
camera) — see [Known limitations](#known-limitations).

**AWS deployment via a single Docker container on App Runner**, not the
`output: "standalone"` Next.js build mode. Standalone's file-tracing step is
known to sometimes miss non-JS assets bundled by native-binary npm packages
(exactly what `ffmpeg-static`'s downloaded binary and OpenCV.js's WASM file
are), and this is a container deployment with no serverless size budget to
optimize for — a full `npm ci` in the runtime image is simpler and more
reliably correct. App Runner was chosen over Lambda (its ~6MB synchronous
payload limit and cold starts poorly fit a video-upload workload) and over
ECS Fargate + ALB (meaningfully more infrastructure to wire up correctly for
no benefit at this scale).

## Libraries, frameworks, and models

| Purpose | Choice |
|---|---|
| App framework | Next.js 16 (App Router), TypeScript, Tailwind CSS |
| Video decoding | `ffmpeg-static` (bundled static ffmpeg binary) |
| Feature detection & matching | `@techstark/opencv-js` — ORB detector, brute-force Hamming matcher |
| Pose estimation (frame-to-map) | `@techstark/opencv-js` — `solvePnPRansac`, `Rodrigues` |
| Two-view geometry, triangulation, bundle adjustment, loop-closure math | Hand-implemented (`lib/slam/`) — see [Major technical decisions](#major-technical-decisions) |
| 3D visualization | `three.js` via `@react-three/fiber` and `@react-three/drei` |
| Deployment | Docker, AWS ECR, AWS App Runner |

No pretrained ML models are used — ORB is a classical, non-learned feature
detector, and all pose/geometry estimation is closed-form or least-squares
optimization, not learned.

## Deployment

Deployed as a single Docker container:

1. `docker build` produces the image from the `Dockerfile` in this repo.
2. Pushed to a private Amazon ECR repository.
3. Run on AWS App Runner (2 vCPU / 4 GB), which provides the public HTTPS
   URL, health checks, and autoscaling.

To redeploy after a code change:

```bash
docker build -t monocular-rgb-slam .
docker tag monocular-rgb-slam:local <account-id>.dkr.ecr.<region>.amazonaws.com/monocular-rgb-slam:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/monocular-rgb-slam:latest
aws apprunner start-deployment --service-arn <service-arn>
```

## Known limitations

- **Monocular scale is unobservable.** The trajectory and point cloud are
  correct up to an arbitrary, unrecoverable scale factor — there is no
  physical unit (meters, etc.) without an external reference. This is a
  fundamental property of monocular SLAM, not an implementation gap.
- **Camera intrinsics are approximated**, not calibrated per-device or
  per-video. This introduces a small, systematic geometric bias.
- **Windowed bundle adjustment**, not a full joint sparse solve. See
  [Major technical decisions](#major-technical-decisions).
- **Loop closure correction** is a simplified linear pose blend, not a
  proper pose-graph optimization. It handles a single revisit well; it isn't
  built for scenes with many overlapping loops.
- **No relocalization.** If tracking is lost for a stretch of frames (fast
  motion, motion blur, a textureless view), the pipeline simply skips those
  frames and keeps trying — it doesn't attempt to recover a lost track.
- **Sparse reconstruction only** — no dense surface/mesh output.
- **Degenerate-triangulation outliers.** Points triangulated from a
  near-zero-parallax frame pair are numerically unstable (can end up
  arbitrarily far away even with low reprojection error). A minimum-parallax
  filter at triangulation time and a final statistical-outlier pass on the
  exported point cloud both mitigate this, but a handful of such points can
  still slip through bundle adjustment's per-point refinement step.
- **First ~12 seconds only.** Longer uploads are accepted but only their
  first ~12 seconds are processed, both to respect the performance budget and
  because AWS App Runner enforces a hard 30-second request timeout.

With more time, the next improvements would be: a full sparse joint bundle
adjustment (Schur complement), a proper pose-graph solver for loop closure,
and an optional device-supplied calibration (or a quick built-in calibration
step) instead of an assumed FOV.

## Measured performance

Assignment target: a 10-second input video should process in ≤10 seconds.

| Test video | Resolution → processed at | Frames sampled | Total time |
|---|---|---|---|
| 10.0s synthetic (static textured scene, panning camera) | 1280×720 → 640×360 @ 8fps | 80 | **5.4s** |

Test environment: Docker container (`node:24-slim` base), running locally via
Docker Desktop on Windows 11 (WSL2 backend) — the same image deployed to AWS
App Runner (2 vCPU / 4 GB). Timing breakdown for the run above:

| Stage | Time |
|---|---|
| Frame extraction (ffmpeg) | 0.48s |
| Feature detection (ORB, all frames) | 1.23s |
| Pose estimation & tracking | 0.81s |
| Bundle adjustment | 0.98s |
| Loop closure | 1.88s |
| **Total** | **5.39s** |

This leaves meaningful headroom under the 10-second target. Processing time
scales primarily with the number of sampled frames (fixed at 8fps) and the
number of keyframes that trigger loop-closure checks; both are bounded
regardless of input video length, since only the first ~12 seconds are read.
