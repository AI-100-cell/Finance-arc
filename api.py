"""FastAPI REST interface for the Earnings Intelligence system.

Endpoints
---------
POST /query          Ask a question about earnings calls
POST /ingest         Upload a PDF/TXT transcript and ingest it
GET  /health         Liveness check

Run with:
    uvicorn api:app --reload
or via the venv directly:
    .\\venv\\Scripts\\uvicorn api:app --reload
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from pydantic import BaseModel

from src.embeddings import build_vector_store, load_vector_store
from src.graph import build_graph, run_query
from src.ingest import load_document
from src.retriever import HybridRetriever

load_dotenv()

# ── SHARED STATE loaded once at startup ─────────────────────────────────────

_graph = None


def _build_system():
    vs = load_vector_store()
    results = vs.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=d, metadata=m)
        for d, m in zip(results["documents"], results["metadatas"])
    ]
    retriever = HybridRetriever(vs, chunks)
    return build_graph(retriever)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = _build_system()
    yield


# ── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Earnings Intelligence API",
    description="Multi-agent RAG over earnings call transcripts",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SCHEMAS ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"


class Citation(BaseModel):
    ticker: Optional[str] = None
    quarter: Optional[str] = None
    source: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class IngestResponse(BaseModel):
    chunks_ingested: int
    ticker: str
    quarter: str


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _graph is None:
        raise HTTPException(status_code=503, detail="System not ready")
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    result = run_query(_graph, req.question, thread_id=req.thread_id)
    citations = [Citation(**c) for c in result.get("citations", []) if isinstance(c, dict)]
    return QueryResponse(answer=result["answer"], citations=citations)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    quarter: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=422, detail="No file provided")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=422, detail="Only PDF and TXT files are supported")

    # Parse year from quarter string like "Q3 2024"
    parts = quarter.strip().split()
    try:
        year = int(parts[-1])
    except (ValueError, IndexError):
        raise HTTPException(status_code=422, detail="quarter must end with a year, e.g. 'Q3 2024'")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = load_document(tmp_path, ticker.upper(), quarter, year)
        build_vector_store(chunks)
        # Rebuild the graph so the new chunks are picked up by BM25
        global _graph
        _graph = _build_system()
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        chunks_ingested=len(chunks),
        ticker=ticker.upper(),
        quarter=quarter,
    )
