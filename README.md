# Monocular RGB Sparse SLAM

Upload a short handheld video and get back an estimated camera trajectory and
a sparse 3D point cloud, visualized in the browser.

**Live app:** http://18.214.188.37

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

**Aspect ratio is preserved, not forced into a fixed landscape frame.** An
early version scaled every video into a fixed 640×360 box with letterbox
padding. That's fine for landscape sources but silently broke portrait ones
(the overwhelming majority of real handheld phone video): a 1080×1920 clip
would shrink to fit *within* 640×360, which for that aspect ratio means
~202×360 of actual content padded with black bars on both sides - discarding
most of the frame before ORB ever saw it. Caught via a real failed upload
during testing. Fixed by capping the longer side at 640px and deriving the
other side to preserve the source's own aspect ratio, with the final
dimensions parsed back out of ffmpeg's own output (they're no longer fixed
in advance).

**Bootstrap tries multiple reference frames, not just frame 0.** Two-view
initialization needs a reference frame with genuine texture; the original
version always used frame 0 and only searched *later* frames as the second
view. If frame 0 itself has poor texture (motion blur while the phone is
still being raised, a moment pointed at the floor or sky at the very start
of a recording), no later frame can fix that, since frame 0 stays the fixed
query side of every match attempt. The bootstrap now also advances the
reference frame itself (trying frame 0, then a few candidates further in) if
the current one doesn't have enough keypoints to be viable at all.

**No camera calibration step.** Intrinsics are approximated from the video's
resolution assuming a ~65° horizontal field of view (typical for a phone
camera) — see [Known limitations](#known-limitations).

**No database.** Nothing in this assignment needs persistence — it's a
stateless upload → process → visualize flow, so there's no job history, no
user accounts, and no stored videos. An earlier pass added optional Supabase
logging to match the team's preferred stack, but it was removed: it wasn't
required, and it added an external dependency for zero functional benefit.

**Loop-closure candidates are capped and evenly subsampled, not exhaustive.**
The first working version checked every earlier keyframe as a loop-closure
candidate on every attempt — O(keyframes²) overall, and it showed up directly
in profiling on the deployed instance (loop closure was the single largest
timing bucket). It's capped at 20 evenly-spaced candidates per attempt now,
which is virtually as effective at catching a revisit in a short clip and
keeps cost roughly linear in video length instead of quadratic.

**AWS deployment: a single Docker container on a free-tier EC2 instance**,
not App Runner or Fargate. App Runner was the initial choice, but it doesn't
scale to zero — it bills for provisioned compute continuously even fully
idle — which conflicts with a hard $0 budget constraint; ECS Fargate has the
same problem for a comparably larger setup cost (VPC/ALB/target groups) with
no offsetting benefit at this scale. A `t3.micro` under the AWS free tier
runs the same image with no code changes and a measured ~300MB peak memory
footprint (comfortable headroom under the instance's 1GB). The build step
itself was moved off the instance: `t3.micro`'s burstable CPU could compile
the app (`next build`, tsc, Tailwind) but it was slow and unpredictable
alongside everything else fighting for burst credit, so the instance instead
pulls the already-built image from ECR — the same image validated locally in
Docker before pushing. It also sidesteps App Runner's ECR-access subscription
gate entirely, since plain EC2 needed no special account opt-in.
Not using Next's `output: "standalone"` build mode was a separate, earlier
decision: standalone's file-tracing step is known to sometimes miss non-JS
assets bundled by native-binary npm packages (exactly what `ffmpeg-static`'s
downloaded binary and OpenCV.js's WASM file are), and with no serverless size
budget to optimize for, a full `npm ci` in the image is simpler and more
reliably correct.

## Libraries, frameworks, and models

| Purpose | Choice |
|---|---|
| App framework | Next.js 16 (App Router), TypeScript, Tailwind CSS |
| Video decoding | `ffmpeg-static` (bundled static ffmpeg binary) |
| Feature detection & matching | `@techstark/opencv-js` — ORB detector, brute-force Hamming matcher |
| Pose estimation (frame-to-map) | `@techstark/opencv-js` — `solvePnPRansac`, `Rodrigues` |
| Two-view geometry, triangulation, bundle adjustment, loop-closure math | Hand-implemented (`lib/slam/`) — see [Major technical decisions](#major-technical-decisions) |
| 3D visualization | `three.js` via `@react-three/fiber` and `@react-three/drei` |
| Deployment | Docker, AWS ECR, AWS EC2 (free-tier `t3.micro`) |

No pretrained ML models are used — ORB is a classical, non-learned feature
detector, and all pose/geometry estimation is closed-form or least-squares
optimization, not learned.

## Deployment

Deployed as a single Docker container on a free-tier EC2 instance:

1. `docker build` produces the image from the `Dockerfile` in this repo
   (built and smoke-tested locally first).
2. Pushed to a private Amazon ECR repository.
3. A `t3.micro` EC2 instance (Amazon Linux 2023, free-tier eligible) pulls
   the image from ECR via an attached IAM instance role (no long-lived
   credentials on the box) and runs it with `--restart unless-stopped`.
4. An Elastic IP keeps the public address stable across instance restarts.
5. Security group: inbound 80 (app) and 22 (SSH, for redeploys) from
   anywhere — fine for a short-lived demo instance, but scope this down for
   anything longer-lived.

To redeploy after a code change:

```bash
docker build -t monocular-rgb-slam .
docker tag monocular-rgb-slam:local <account-id>.dkr.ecr.<region>.amazonaws.com/monocular-rgb-slam:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/monocular-rgb-slam:latest

ssh -i slam-app-key.pem ec2-user@<instance-ip> '
  aws ecr get-login-password --region <region> | sudo docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
  sudo docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/monocular-rgb-slam:latest
  sudo docker rm -f slam-app
  sudo docker run -d --name slam-app --restart unless-stopped -p 80:3000 <account-id>.dkr.ecr.<region>.amazonaws.com/monocular-rgb-slam:latest
'
```

**Cost:** designed to run at $0 — a single `t3.micro` instance plus one
Elastic IP (free while associated with a running instance) both fall under
the AWS free tier. This depends on the account's free-tier allowance still
being available (shared across any other `t2`/`t3.micro` usage on the same
account) — worth a quick check in Billing → Free Tier before leaving this
running long-term.

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
- **First ~10 seconds only.** Longer uploads are accepted but only their
  first ~10 seconds are processed, to keep processing time bounded regardless
  of input length.
- **Burstable-CPU variance.** The deployed `t3.micro` has burstable, not
  dedicated, CPU. Measured times below have a couple of seconds of headroom
  under the 10s target, but sustained heavy use (many uploads back to back)
  could exhaust CPU credit and slow individual runs.

With more time, the next improvements would be: a full sparse joint bundle
adjustment (Schur complement), a proper pose-graph solver for loop closure,
and an optional device-supplied calibration (or a quick built-in calibration
step) instead of an assumed FOV.

## Measured performance

Assignment target: a 10-second input video should process in ≤10 seconds.

**Test environment: the actual deployed instance** — AWS EC2 `t3.micro`
(1 vCPU burstable, 1GB RAM), Amazon Linux 2023, running the same Docker image
served at the live URL above. Measured via the deployed `/api/process`
endpoint (server-reported `totalMs`, excluding upload transfer time), using a
10.0s synthetic test video (1280×720 source, static textured scene with a
panning camera motion) processed at 640×360 @ 6fps → 60 frames sampled:

| Run | Frame extraction | Feature detection | Pose estimation | Bundle adjustment | Loop closure | **Total** |
|---|---|---|---|---|---|---|
| 1 | 1.95s | 1.96s | 1.31s | 1.22s | 1.35s | **7.81s** |
| 2 | 1.68s | 1.28s | 1.08s | 1.14s | 1.33s | **6.53s** |

Both runs land comfortably under the 10-second target, with 2-3.5s of
headroom. Run-to-run variance is expected on a burstable-CPU instance.

An earlier version of the pipeline (8fps/60-frame-cap, and an unbounded
loop-closure candidate search that turned out to be O(keyframes²) — see
[Major technical decisions](#major-technical-decisions)) measured **11.4-11.6s**
on this same instance, over budget. The fix was algorithmic (cap loop-closure
candidates) plus a small sampling-rate reduction, not a change in test
conditions — worth noting since it's a reminder that this pipeline's
performance is meaningfully hardware-dependent: the same image processes a
10s video in ~5-6s on a development machine with dedicated cores, vs.
~6.5-7.8s on the free-tier instance actually serving the live URL. The
timing breakdown is also reported live in the app's UI for any video you
upload, via the "Timing breakdown" panel under the results.
