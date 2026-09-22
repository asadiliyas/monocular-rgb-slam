# Multi-stage build. This app is not built with Next's `output: "standalone"` -
# ffmpeg-static's downloaded binary and OpenCV.js's WASM asset are exactly the
# kind of non-JS files that standalone's file-tracing step can miss, and this
# is a Docker deployment (no serverless size budget to optimize for), so the
# simpler, more reliable option is to ship a real node_modules install.

FROM node:24-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:24-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["npm", "start"]
