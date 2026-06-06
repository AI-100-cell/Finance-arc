"""Data loading + chunking (Concept #8).

Entry point for all data: loads a file (PDF or TXT), splits it into
semantically meaningful chunks, and attaches metadata so retrieval can later
filter by company or time period.
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Canonical home of the splitter on langchain 0.2.x (the old
# `langchain.text_splitter` import path is deprecated).
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_document(filepath: str, ticker: str, quarter: str, year: int) -> list[Document]:
    """Load one file, chunk it with overlap, and tag every chunk with metadata."""
    # LangChain document loader (Concept #1 — LangChain fundamentals)
    loader = PyPDFLoader(filepath) if filepath.lower().endswith(".pdf") else TextLoader(
        filepath, encoding="utf-8"
    )
    docs = loader.load()

    # Chunk with overlap to preserve context across boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # ~200 words per chunk
        chunk_overlap=100,    # overlap prevents info loss at chunk boundaries
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    # Attach metadata to every chunk for filtered retrieval.
    for chunk in chunks:
        chunk.metadata.update(
            {
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "source": filepath,
            }
        )
    return chunks
