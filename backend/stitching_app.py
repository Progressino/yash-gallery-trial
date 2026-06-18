"""
Standalone stitching backend — zero-downtime isolated service.

Runs on port 8001 inside the stitching-backend container.
Routes only:
  /api/stitching/*  — production entry, karigar, challan, costing, etc.
  /api/auth/*       — login/logout (same JWT secret as main backend)

Starts in < 3 seconds. Never blocks on warm cache or GitHub downloads.
Karigar entries work 24/7 regardless of main-backend restarts/deployments.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stitching-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.warning("Stitching backend ready (fast startup — no warm cache)")
    yield
    log.warning("Stitching backend stopping")


app = FastAPI(title="Stitching Backend", version="1.0.0", lifespan=lifespan)

# CORS
_extra = os.environ.get("EXTRA_CORS_ORIGIN", "").strip()
_origins = [
    "http://localhost:5173",
    "https://progressino.com",
    "https://app.progressino.com",
]
if _extra:
    _origins.append(_extra)

app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Auth middleware — same JWT logic as main backend
from .routers.auth import decode_token
from .services.permissions import karigar_may_access_api, KARIGAR_ROLE
from .session import store as _session_store

_EXEMPT = {"/api/auth/login", "/api/auth/logout", "/api/auth/me", "/api/health"}


@app.middleware("http")
async def auth_mw(request: Request, call_next):
    path = request.url.path
    if path in _EXEMPT or not path.startswith("/api/"):
        return await call_next(request)
    token = request.cookies.get("auth_token")
    payload = decode_token(token) if token else None
    if not payload or not payload.get("sub"):
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
    request.state.auth = payload
    role = payload.get("role", "Admin")
    if role == KARIGAR_ROLE and not karigar_may_access_api(path, request.method):
        return JSONResponse(status_code=403, content={"detail": "Access denied"})
    return await call_next(request)


@app.middleware("http")
async def session_mw(request: Request, call_next):
    sid = request.cookies.get("session_id")
    sess = _session_store.get(sid) if sid else None
    request.state.session_id = sid
    request.state.session = sess
    return await call_next(request)


from .middleware.timing import register_timing_middleware

register_timing_middleware(app)


# Routes
from .routers.auth import router as auth_router
from .routers.stitching import router as stitching_router

app.include_router(auth_router,      prefix="/api/auth",      tags=["auth"])
app.include_router(stitching_router, prefix="/api/stitching", tags=["stitching"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "stitching-backend"}
