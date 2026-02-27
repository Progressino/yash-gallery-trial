# ── Stage 1: Build ────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 2: Serve with nginx ─────────────────────────────────
FROM nginx:alpine AS production

# Copy built React app
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config is mounted via docker-compose or copied separately
# For standalone use, copy the project nginx.conf:
# COPY ../nginx/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
