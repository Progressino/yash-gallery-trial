"""Chunked upload disk store."""
import tempfile
from pathlib import Path

import pytest

from backend.services.chunk_upload_store import ChunkUploadStore, CHUNK_SIZE_BYTES


@pytest.fixture
def store(tmp_path):
    return ChunkUploadStore(base_dir=tmp_path)


def test_assemble_multi_chunk_file(store: ChunkUploadStore):
    sid = "sess-test"
    data = b"x" * (CHUNK_SIZE_BYTES + 1234)
    upload_id, chunk_size = store.create(
        sid, target="daily-auto", files=[("report.csv", len(data))]
    )
    assert chunk_size == CHUNK_SIZE_BYTES
    n_chunks = 2
    store.write_chunk(
        sid, upload_id, file_index=0, chunk_index=0, total_chunks=n_chunks,
        data=data[:chunk_size],
    )
    store.write_chunk(
        sid, upload_id, file_index=0, chunk_index=1, total_chunks=n_chunks,
        data=data[chunk_size:],
    )
    target, parts = store.assemble(sid, upload_id)
    assert target == "daily-auto"
    assert len(parts) == 1
    assert parts[0][0] == "report.csv"
    assert parts[0][1] == data


def test_missing_chunk_raises(store: ChunkUploadStore):
    sid = "sess-2"
    upload_id, _ = store.create(sid, target="inventory-auto", files=[("a.csv", 100)])
    store.write_chunk(sid, upload_id, file_index=0, chunk_index=0, total_chunks=2, data=b"a" * 50)
    with pytest.raises(ValueError, match="Missing chunks"):
        store.assemble(sid, upload_id)
