#!/usr/bin/env python3
"""Chunked daily-auto upload (same path as the Upload page) for a folder of CSV/XLSX files.

Usage:
  PYTHONPATH=. python3 scripts/upload_daily_chunked_http.py \\
    --base-url https://app.progressino.com \\
    --folder "/path/to/Sales 1 APR to 14" \\
    --username admin --password '***'

Or set FORECAST_BASE_URL, AUTH_USERNAME, AUTH_PASSWORD in .env (password plain or use login interactively).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 4 * 1024 * 1024


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _collect_files(folder: Path) -> list[Path]:
    allowed = {".csv", ".xlsx", ".xls", ".zip", ".rar"}
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed:
            out.append(p)
    return out


def _collect_rar_paths(paths: list[Path]) -> list[Path]:
    """Accept explicit .rar paths (e.g. Downloads/Sales 1-5-26.rar)."""
    out: list[Path] = []
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".rar":
            out.append(p)
        elif p.is_dir():
            out.extend(_collect_files(p))
    return sorted(out, key=lambda x: x.name.lower())


def login(
    sess: requests.Session,
    base: str,
    username: str,
    password: str,
    *,
    otp_code: str = "",
    device_id: str = "upload-script-cli",
) -> None:
    headers = {"X-Device-Id": device_id}
    r = sess.post(
        f"{base}/api/auth/login",
        json={"username": username, "password": password},
        timeout=60,
        headers=headers,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("otp_required") and data.get("challenge_id"):
        code = (otp_code or os.environ.get("AUTH_OTP", "")).strip()
        if not code:
            raise SystemExit(
                f"OTP required (sent to {data.get('masked_phone', 'mobile')}). "
                "Set AUTH_OTP=###### or pass --otp."
            )
        vr = sess.post(
            f"{base}/api/auth/otp/verify",
            json={
                "challenge_id": data["challenge_id"],
                "code": code,
                "trust_device": True,
            },
            timeout=60,
            headers=headers,
        )
        vr.raise_for_status()
        data = vr.json()
    if not data.get("ok"):
        raise SystemExit(f"Login failed: {data}")


def poll_coverage(sess: requests.Session, base: str, max_sec: int = 900) -> dict:
    start = time.time()
    saw_running = False
    while time.time() - start < max_sec:
        r = sess.get(f"{base}/api/data/coverage", params={"light": "true"}, timeout=90)
        r.raise_for_status()
        cov = r.json()
        ingest = cov.get("daily_auto_ingest_status") or "idle"
        sales = cov.get("sales_rebuild") or "idle"
        if ingest == "running" or sales == "running":
            saw_running = True
            msg = cov.get("daily_auto_ingest_message") or cov.get("sales_rebuild_message") or "…"
            print(f"  … {msg}", flush=True)
            time.sleep(2)
            continue
        if ingest == "error":
            raise SystemExit(cov.get("daily_auto_ingest_message") or "Ingest failed")
        if ingest == "done" or (saw_running and ingest == "idle"):
            return cov
        print("  … waiting for server to start", flush=True)
        time.sleep(2)
    raise SystemExit("Timed out waiting for ingest — refresh app in a minute.")


def upload_chunked(sess: requests.Session, base: str, paths: list[Path]) -> dict:
    files_meta = [{"name": p.name, "size": p.stat().st_size} for p in paths]
    r = sess.post(
        f"{base}/api/upload/chunk/init",
        json={"target": "daily-auto", "files": files_meta},
        timeout=60,
    )
    if r.status_code == 502:
        print("502 on chunk/init — server may still process; polling…", flush=True)
        poll_coverage(sess, base)
        return {"ok": True, "message": "Recovered after 502 on init"}
    r.raise_for_status()
    init = r.json()
    if not init.get("ok"):
        raise SystemExit(init.get("message") or "chunk init failed")
    upload_id = init["upload_id"]
    chunk_size = int(init["chunk_size"])

    for fi, path in enumerate(paths):
        size = path.stat().st_size
        total_chunks = max(1, (size + chunk_size - 1) // chunk_size)
        raw = path.read_bytes()
        for ci in range(total_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, size)
            blob = raw[start:end]
            fd = {
                "upload_id": (None, upload_id),
                "file_index": (None, str(fi)),
                "chunk_index": (None, str(ci)),
                "total_chunks": (None, str(total_chunks)),
                "chunk": (f"{path.name}.part{ci}", blob, "application/octet-stream"),
            }
            for attempt in range(5):
                try:
                    pr = sess.post(f"{base}/api/upload/chunk", files=fd, timeout=120)
                    if pr.status_code == 502:
                        time.sleep(2 * (attempt + 1))
                        continue
                    pr.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt >= 4:
                        raise
                    time.sleep(2 * (attempt + 1))
            else:
                pr.raise_for_status()
            print(f"  chunk {path.name} {ci + 1}/{total_chunks}", flush=True)

    for attempt in range(5):
        try:
            cr = sess.post(
                f"{base}/api/upload/chunk/complete",
                json={"upload_id": upload_id},
                timeout=45,
            )
            if cr.status_code == 502:
                print("502 on chunk/complete — polling ingest (chunks likely received)…", flush=True)
                poll_coverage(sess, base)
                return {"ok": True, "message": "Complete accepted via poll after 502"}
            cr.raise_for_status()
            done = cr.json()
            if not done.get("ok"):
                raise SystemExit(done.get("message") or "complete failed")
            return done
        except requests.Timeout:
            if attempt >= 4:
                print("Timeout on complete — polling…", flush=True)
                poll_coverage(sess, base)
                return {"ok": True, "message": "Complete via poll after timeout"}
            time.sleep(2 * (attempt + 1))
    raise SystemExit("chunk complete failed")


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("FORECAST_BASE_URL", "https://app.progressino.com"))
    ap.add_argument("--folder", type=Path, help="Folder of CSV/XLSX/ZIP/RAR")
    ap.add_argument("rars", nargs="*", type=Path, help="One or more .rar daily bundles")
    ap.add_argument("--username", default=os.environ.get("AUTH_USERNAME", ""))
    ap.add_argument("--password", default=os.environ.get("AUTH_PASSWORD", ""))
    ap.add_argument("--otp", default=os.environ.get("AUTH_OTP", ""), help="6-digit OTP if login requires it")
    args = ap.parse_args()
    if args.rars:
        paths = _collect_rar_paths(list(args.rars))
    elif args.folder:
        folder = args.folder.expanduser().resolve()
        if not folder.is_dir():
            raise SystemExit(f"Not a folder: {folder}")
        paths = _collect_files(folder)
    else:
        raise SystemExit("Provide --folder or list .rar paths")
    if not paths:
        raise SystemExit(f"No csv/xlsx in {folder}")
    if not args.username or not args.password:
        raise SystemExit("Set --username/--password or AUTH_USERNAME/AUTH_PASSWORD in .env")

    base = args.base_url.rstrip("/")
    sess = requests.Session()
    print(f"Login {base} as {args.username}…", flush=True)
    login(sess, base, args.username, args.password, otp_code=args.otp)
    print(f"Uploading {len(paths)} file(s) via chunked daily-auto…", flush=True)
    out = upload_chunked(sess, base, paths)
    print("Upload response:", json.dumps(out, indent=2)[:500], flush=True)
    if out.get("ingest_async") or out.get("sales_rebuild") == "pending":
        print("Polling ingest + sales rebuild…", flush=True)
        cov = poll_coverage(sess, base)
        print(
            f"Done. sales_rows={cov.get('sales_rows')} platforms: "
            f"mtr={cov.get('mtr')} myntra={cov.get('myntra')} meesho={cov.get('meesho')} flipkart={cov.get('flipkart')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
