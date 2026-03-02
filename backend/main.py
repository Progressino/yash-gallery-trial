"""
Yash Gallery ERP — FastAPI backend
Serves all business logic as a REST API.
Session state is stored server-side keyed by a UUID cookie.
"""
from dotenv import load_dotenv
load_dotenv()   # loads .env from cwd (run from repo root or backend/)

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .session import store
from .routers import upload, data, cache, po

app = FastAPI(
    title="Yash Gallery ERP API",
    version="1.0.0",
    description="FastAPI backend for the Yash Gallery ERP system",
)

# ── CORS (allow Vite dev server + production domain) ─────────
import os as _os
_extra = _os.environ.get("EXTRA_CORS_ORIGIN", "").strip()
_origins = [
    "http://localhost:5173",        # Vite dev
    "http://localhost:3000",        # alternate dev
    "https://progressino.com",      # production root
    "https://www.progressino.com",  # www
    "https://app.progressino.com",  # app subdomain
    "https://yashgallery.com",      # yashgallery production
    "https://www.yashgallery.com",  # www
    "https://app.yashgallery.com",  # app subdomain
]
if _extra:
    _origins.append(_extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,       # needed for the session cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session cookie middleware ─────────────────────────────────
SESSION_COOKIE = "session_id"

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    sid = request.cookies.get(SESSION_COOKIE)
    sid, session = store.get_or_create(sid)
    request.state.session_id = sid
    request.state.session = session

    response: Response = await call_next(request)

    # Set / refresh cookie on every response
    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        max_age=4 * 3600,   # 4 hours
    )
    return response


# ── Routers ───────────────────────────────────────────────────
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(data.router,   prefix="/api/data",   tags=["data"])
app.include_router(cache.router,  prefix="/api/cache",  tags=["cache"])
app.include_router(po.router,     prefix="/api/po",     tags=["po"])


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": store.count}
