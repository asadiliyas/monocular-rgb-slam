import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["@techstark/opencv-js", "ffmpeg-static"],
};

export default nextConfig;
