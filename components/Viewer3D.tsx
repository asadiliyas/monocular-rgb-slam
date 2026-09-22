"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import type { PipelineResult, Vec3 } from "@/lib/slam/types";

interface Viewer3DProps {
  result: PipelineResult;
}

function PointCloud({ points, size }: { points: PipelineResult["points"]; size: number }) {
  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      positions[i * 3] = p.position[0];
      positions[i * 3 + 1] = p.position[1];
      positions[i * 3 + 2] = p.position[2];
      colors[i * 3] = p.color[0] / 255;
      colors[i * 3 + 1] = p.color[1] / 255;
      colors[i * 3 + 2] = p.color[2] / 255;
    });
    return { positions, colors };
  }, [points]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={size} vertexColors sizeAttenuation />
    </points>
  );
}

function Trajectory({ trajectory }: { trajectory: PipelineResult["trajectory"] }) {
  const path = useMemo(() => trajectory.map((t) => t.position as [number, number, number]), [trajectory]);
  if (path.length < 2) return null;
  return <Line points={path} color="#22d3ee" lineWidth={2} />;
}

function CameraMarkers({ trajectory, size }: { trajectory: PipelineResult["trajectory"]; size: number }) {
  const step = Math.max(1, Math.floor(trajectory.length / 20));
  const markers = useMemo(() => trajectory.filter((_, i) => i % step === 0), [trajectory, step]);

  return (
    <>
      {markers.map((kf) => {
        const quat = new THREE.Quaternion(kf.quaternion[0], kf.quaternion[1], kf.quaternion[2], kf.quaternion[3]);
        return (
          <group key={kf.keyframeId} position={kf.position} quaternion={quat}>
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              <coneGeometry args={[size, size * 2.4, 4]} />
              <meshBasicMaterial color="#f97316" wireframe />
            </mesh>
          </group>
        );
      })}
    </>
  );
}

/** Canvas's `camera` prop only sets position, not orientation - without this the camera stays pointed along its default -Z axis regardless of where it was moved to, so the scene (anywhere else) never appears in frame. */
function CameraLookAt({ target }: { target: Vec3 }) {
  const { camera } = useThree();
  useEffect(() => {
    camera.lookAt(target[0], target[1], target[2]);
  }, [camera, target]);
  return null;
}

/**
 * Bounding sphere of the whole scene (points + trajectory), used to size the
 * camera, controls, and markers to whatever scale this video's monocular
 * reconstruction landed on.
 *
 * Uses the 85th-percentile distance from the centroid rather than the true
 * max: a sparse point cloud can legitimately keep a handful of points much
 * farther out than the main cluster (real background geometry, or map points
 * only barely surviving the pipeline's own outlier filter), and framing the
 * camera around the true max lets a few such points shrink everything else
 * to a speck. A percentile radius keeps the framing tied to where most of the
 * data actually is; the rare far point just extends past the frame edge,
 * which orbit/zoom can still reach.
 */
function useSceneScale(result: PipelineResult) {
  return useMemo(() => {
    const positions: Vec3[] = [...result.points.map((p) => p.position), ...result.trajectory.map((t) => t.position)];
    if (positions.length === 0) {
      return { center: [0, 0, 0] as Vec3, radius: 5 };
    }
    const center: Vec3 = [0, 0, 0];
    for (const p of positions) {
      center[0] += p[0] / positions.length;
      center[1] += p[1] / positions.length;
      center[2] += p[2] / positions.length;
    }
    const distances = positions
      .map((p) => Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]))
      .sort((a, b) => a - b);
    const radius = Math.max(0.5, distances[Math.floor(distances.length * 0.85)]);
    return { center, radius };
  }, [result]);
}

export default function Viewer3D({ result }: Viewer3DProps) {
  const { center, radius } = useSceneScale(result);
  const cameraDistance = radius * 2.5;

  return (
    <div className="h-full w-full rounded-lg bg-neutral-950">
      <Canvas
        camera={{
          fov: 50,
          near: radius / 100,
          far: radius * 50,
          position: [center[0] + cameraDistance * 0.5, center[1] + cameraDistance * 0.4, center[2] + cameraDistance],
        }}
      >
        <ambientLight intensity={1} />
        <CameraLookAt target={center} />
        <PointCloud points={result.points} size={radius * 0.006} />
        <Trajectory trajectory={result.trajectory} />
        <CameraMarkers trajectory={result.trajectory} size={radius * 0.025} />
        <axesHelper args={[radius * 0.15]} />
        <OrbitControls makeDefault target={center} enableDamping dampingFactor={0.1} />
      </Canvas>
    </div>
  );
}
