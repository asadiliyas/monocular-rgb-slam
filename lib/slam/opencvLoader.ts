import { createRequire } from "node:module";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CvModule = any;

let cvPromise: Promise<CvModule> | null = null;

/**
 * Lazily initializes the OpenCV.js WASM runtime once per server process and
 * reuses it across requests (initialization takes real time and should not
 * happen per-video).
 *
 * This deliberately uses `require` (via `createRequire`) instead of a dynamic
 * `import()`. @techstark/opencv-js's CommonJS export is an Emscripten module
 * object that is itself thenable during WASM instantiation, and bundler ESM/
 * CJS interop for `await import(...)` on a thenable CJS export throws
 * "Promise.prototype.then called on incompatible receiver [object Module]"
 * under Turbopack/webpack (it does not happen under plain Node, which is why
 * this only surfaces once the app is bundled). `require` sidesteps that
 * interop path entirely.
 */
export async function getCv(): Promise<CvModule> {
  if (!cvPromise) {
    cvPromise = (async () => {
      const require = createRequire(import.meta.url);
      const cvModule = require("@techstark/opencv-js") as CvModule;
      if (cvModule instanceof Promise) return cvModule;
      if (cvModule.Mat) return cvModule;
      await new Promise<void>((resolve) => {
        cvModule.onRuntimeInitialized = () => resolve();
      });
      return cvModule;
    })();
  }
  return cvPromise;
}
