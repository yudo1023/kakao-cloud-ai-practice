# FastAPI 기초_요청 및 응답 처리
# p.205
# 1. 목표 : 멀티 파일 관리 API 구현
# 2. 요구사항
# 단일/다중 파일 업로드
# 파일 목록 조회 : 페이징, 필터링 지원
# 파일 다운로드 : 단일/다중 ZIP 다운로드
# 파일 삭제 : 단일/다중 삭제

import os
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="멀티 파일 관리 API", version="1.0.0")

class FileMeta(BaseModel):
    id: int
    filename: str
    stored_name: str
    content_type: Optional[str]
    size: int
    uploaded_at: datetime

class FilesPage(BaseModel):
    items: List[FileMeta]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int

class FilesZipIn(BaseModel):
    ids: List[int] = Field(..., min_items=1)
    zip_name: Optional[str] = None

class FilesDeleteIn(BaseModel):
    ids: List[int] = Field(..., min_items=1)

_files: List[FileMeta] = []
_id_lock = Lock()
_next_id = 1

def _gen_id() -> int:
    global _next_id
    with _id_lock:
        i = _next_id
        _next_id += 1
    return i

# 업로드 (단일/다중)
@app.post("/files", response_model=List[FileMeta], summary="단일/다중 파일 업로드")
async def upload_files(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
):
    if (not files) and file is not None:
        files = [file]

    if not files:
        raise HTTPException(status_code=400, detail="업로드할 파일을 한 개 이상 선택하세요.")

    results: List[FileMeta] = []

    for up in files:
        if not up or not up.filename:
            raise HTTPException(status_code=400, detail="파일 이름이 없습니다.")

        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        safe_orig = os.path.basename(up.filename)
        stored_name = f"{ts}__{safe_orig}"
        dest_path = UPLOAD_DIR / stored_name

        size = 0
        with open(dest_path, "wb") as out:
            while True:
                chunk = await up.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                out.write(chunk)

        meta = FileMeta(
            id=_gen_id(),
            filename=safe_orig,
            stored_name=stored_name,
            content_type=up.content_type or "application/octet-stream",
            size=size,
            uploaded_at=datetime.now(),
        )
        _files.append(meta)
        results.append(meta)

    return results

# 목록 조회
@app.get("/files", response_model=FilesPage, summary="파일 목록 조회(페이징/필터)")
async def list_files(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    q: Optional[str] = Query(None, description="파일명 부분검색"),
    content_type: Optional[str] = Query(None, description="Content-Type 완전일치"),
):
    items = _files

    if q:
        q_lower = q.lower()
        items = [f for f in items if q_lower in f.filename.lower()]
    if content_type:
        items = [f for f in items if (f.content_type or "") == content_type]

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    return FilesPage(items=page_items, page=page, page_size=page_size, total=total)

# 단일 파일 다운로드
@app.get("/files/{file_id}/download", summary="단일 파일 다운로드")
async def download_file(file_id: int):
    meta = next((f for f in _files if f.id == file_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="파일 메타를 찾을 수 없습니다.")

    path = UPLOAD_DIR / meta.stored_name
    if not path.exists():
        raise HTTPException(status_code=410, detail="스토리지에 파일이 없습니다.")

    return FileResponse(
        path,
        media_type=meta.content_type or "application/octet-stream",
        filename=meta.filename,
    )

# 다중 ZIP 다운로드
@app.post("/files/download", summary="다중 파일 ZIP 다운로드")
async def download_zip(payload: FilesZipIn):
    metas: List[FileMeta] = [m for m in _files if m.id in set(payload.ids)]
    if not metas:
        raise HTTPException(status_code=404, detail="다운로드할 파일이 없습니다.")

    mem = BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for m in metas:
            src = UPLOAD_DIR / m.stored_name
            if src.exists():
                zf.write(src, arcname=f"{m.id}_{m.filename}")
            else:
                continue

# 단일 파일 삭제
@app.delete("/files/{file_id}", summary="단일 파일 삭제")
async def delete_one(file_id: int):
    idx = next((i for i, f in enumerate(_files) if f.id == file_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    meta = _files.pop(idx)
    try:
        (UPLOAD_DIR / meta.stored_name).unlink(missing_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {e}")

    return {"deleted": 1}

# 다중 파일 삭제
@app.delete("/files", summary="다중 파일 삭제")
async def delete_many(payload: FilesDeleteIn):
    target_ids = set(payload.ids)
    kept: List[FileMeta] = []
    deleted = 0

    for m in list(_files):
        if m.id in target_ids:
            try:
                (UPLOAD_DIR / m.stored_name).unlink(missing_ok=True)
            except Exception:
                pass
            deleted += 1
        else:
            kept.append(m)

    _files.clear()
    _files.extend(kept)

    return {"deleted": deleted, "requested": len(payload.ids)}

@app.get("/")
async def root():
    return {
        "app": "멀티 파일 관리 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            {"POST": "/files"},
            {"GET": "/files"},
            {"GET": "/files/{id}/download"},
            {"POST": "/files/download"},
            {"DELETE": "/files/{id}"},
            {"DELETE": "/files"},
        ],
    }
