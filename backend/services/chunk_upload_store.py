"""Disk-backed assembly for chunked multipart uploads (daily-auto / inventory-auto)."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_log = logging.getLogger(__name__)

ChunkTarget = Literal["daily-auto", "inventory-auto"]

CHUNK_SIZE_BYTES = int(os.environ.get("UPLOAD_CHUNK_SIZE_BYTES", str(4 * 1024 * 1024)))
MAX_FILES_PER_UPLOAD = int(os.environ.get("CHUNK_UPLOAD_MAX_FILES", "50"))
MAX_BYTES_PER_UPLOAD = int(os.environ.get("CHUNK_UPLOAD_MAX_BYTES", str(500 * 1024 * 1024)))
SESSION_TTL_SEC = int(os.environ.get("CHUNK_UPLOAD_TTL_SEC", str(2 * 3600)))

_BASE = Path(os.environ.get("CHUNK_UPLOAD_DIR", "/tmp/chunk_uploads"))


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "upload").strip() or "upload"
    return re.sub(r"[^\w.\-()+ ]", "_", base)[:200]


@dataclass
class _FileMeta:
    name: str
    size: int
    total_chunks: int = 0
    received_chunks: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "total_chunks": self.total_chunks,
            "received_chunks": sorted(self.received_chunks),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_FileMeta":
        fm = cls(name=d["name"], size=int(d["size"]), total_chunks=int(d.get("total_chunks") or 0))
        fm.received_chunks = set(int(x) for x in (d.get("received_chunks") or []))
        return fm


class ChunkUploadStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or _BASE
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_root(self, session_id: str) -> Path:
        safe_sid = re.sub(r"[^\w\-]", "_", session_id)[:80]
        return self.base_dir / safe_sid

    def _upload_dir(self, session_id: str, upload_id: str) -> Path:
        safe_uid = re.sub(r"[^\w\-]", "_", upload_id)[:80]
        return self._session_root(session_id) / safe_uid

    def purge_expired(self) -> int:
        """Remove upload dirs older than SESSION_TTL_SEC. Returns count removed."""
        cutoff = time.time() - SESSION_TTL_SEC
        removed = 0
        for sess_dir in self.base_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            for up_dir in sess_dir.iterdir():
                if not up_dir.is_dir():
                    continue
                try:
                    if up_dir.stat().st_mtime < cutoff:
                        shutil.rmtree(up_dir, ignore_errors=True)
                        removed += 1
                except OSError:
                    pass
        return removed

    def create(
        self,
        session_id: str,
        *,
        target: ChunkTarget,
        files: list[tuple[str, int]],
    ) -> tuple[str, int]:
        if not files:
            raise ValueError("No files in chunk upload session")
        if len(files) > MAX_FILES_PER_UPLOAD:
            raise ValueError(f"Too many files (max {MAX_FILES_PER_UPLOAD})")
        total = sum(s for _, s in files)
        if total > MAX_BYTES_PER_UPLOAD:
            raise ValueError(f"Total size exceeds {MAX_BYTES_PER_UPLOAD // (1024 * 1024)} MB limit")

        self.purge_expired()
        upload_id = str(uuid.uuid4())
        root = self._upload_dir(session_id, upload_id)
        root.mkdir(parents=True, exist_ok=False)
        meta = {
            "created": time.time(),
            "target": target,
            "chunk_size": CHUNK_SIZE_BYTES,
            "files": [_FileMeta(name=_safe_name(n), size=int(s)).to_dict() for n, s in files],
        }
        (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        for i in range(len(files)):
            (root / f"file_{i}").mkdir(exist_ok=True)
        return upload_id, CHUNK_SIZE_BYTES

    def _load_meta(self, session_id: str, upload_id: str) -> tuple[Path, dict, list[_FileMeta]]:
        root = self._upload_dir(session_id, upload_id)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError("Unknown or expired upload_id")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        files = [_FileMeta.from_dict(f) for f in meta.get("files") or []]
        return root, meta, files

    def _save_meta(self, root: Path, meta: dict, files: list[_FileMeta]) -> None:
        meta["files"] = [f.to_dict() for f in files]
        (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def write_chunk(
        self,
        session_id: str,
        upload_id: str,
        *,
        file_index: int,
        chunk_index: int,
        total_chunks: int,
        data: bytes,
    ) -> dict:
        if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
            raise ValueError("Invalid chunk indices")
        if not data:
            raise ValueError("Empty chunk")

        root, meta, files = self._load_meta(session_id, upload_id)
        if file_index < 0 or file_index >= len(files):
            raise ValueError("Invalid file_index")

        fm = files[file_index]
        if total_chunks != fm.total_chunks and fm.total_chunks == 0:
            fm.total_chunks = total_chunks
        elif fm.total_chunks and total_chunks != fm.total_chunks:
            raise ValueError("total_chunks mismatch")

        chunk_path = root / f"file_{file_index}" / f"chunk_{chunk_index:06d}"
        chunk_path.write_bytes(data)
        fm.received_chunks.add(chunk_index)
        self._save_meta(root, meta, files)

        total_expected = sum(
            (f.total_chunks or max(1, (f.size + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES))
            for f in files
        )
        total_received = sum(len(f.received_chunks) for f in files)
        return {
            "file_index": file_index,
            "chunk_index": chunk_index,
            "files_complete": sum(
                1 for f in files
                if f.total_chunks and len(f.received_chunks) >= f.total_chunks
            ),
            "file_count": len(files),
            "chunks_received": total_received,
            "chunks_expected": total_expected,
        }

    def assemble(self, session_id: str, upload_id: str) -> tuple[ChunkTarget, list[tuple[str, bytes]]]:
        root, meta, files = self._load_meta(session_id, upload_id)
        target: ChunkTarget = meta.get("target") or "daily-auto"
        out: list[tuple[str, bytes]] = []

        for i, fm in enumerate(files):
            fdir = root / f"file_{i}"
            if not fdir.is_dir():
                raise ValueError(f"Missing chunks for file {fm.name}")
            if not fm.total_chunks:
                raise ValueError(f"Incomplete upload for {fm.name}")
            if len(fm.received_chunks) < fm.total_chunks:
                missing = sorted(set(range(fm.total_chunks)) - fm.received_chunks)
                raise ValueError(f"Missing chunks for {fm.name}: {missing[:5]}")

            parts: list[bytes] = []
            for c in range(fm.total_chunks):
                cp = fdir / f"chunk_{c:06d}"
                if not cp.is_file():
                    raise ValueError(f"Missing chunk file {c} for {fm.name}")
                parts.append(cp.read_bytes())
            blob = b"".join(parts)
            if fm.size and len(blob) != fm.size:
                raise ValueError(
                    f"Size mismatch for {fm.name}: expected {fm.size}, got {len(blob)}"
                )
            out.append((fm.name, blob))

        shutil.rmtree(root, ignore_errors=True)
        return target, out

    def abort(self, session_id: str, upload_id: str) -> None:
        root = self._upload_dir(session_id, upload_id)
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)


chunk_store = ChunkUploadStore()
