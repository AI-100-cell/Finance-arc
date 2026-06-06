"""ChromaDB vector store (Concept #3).

Every chunk is embedded with OpenAI's text-embedding-3-small and stored
persistently. The store survives restarts, so ingestion only runs once.
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "text-embedding-3-small"


def _embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")
    return OpenAIEmbeddings(model=EMBED_MODEL)


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed `chunks` and persist them to PERSIST_DIR.

    Note: since Chroma 0.4 the store auto-persists when `persist_directory` is
    set, so the old explicit `vs.persist()` call is gone (it raises on
    chromadb >= 0.4 / the version pinned here).
    """
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=PERSIST_DIR,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {PERSIST_DIR}")
    return vs


def load_vector_store() -> Chroma:
    """Reopen the persisted store for querying."""
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings())
