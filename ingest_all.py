"""One-time data ingestion — run once before launching the app.

Loads each transcript, chunks it, and embeds everything into ChromaDB.
Run with: python ingest_all.py
"""

import os

from src.ingest import load_document
from src.embeddings import build_vector_store

# Add your files here: (path, ticker, quarter_label, year)
documents = [
    ("data/transcripts/AAPL_Q3_2024.txt", "AAPL", "Q3 2024", 2024),
    ("data/transcripts/MSFT_Q4_2024.txt", "MSFT", "Q4 2024", 2024),
    ("data/transcripts/TSLA_Q2_2024.txt", "TSLA", "Q2 2024", 2024),
    ("data/transcripts/GOOGL_Q3_2024.txt", "GOOGL", "Q3 2024", 2024),
    ("data/transcripts/NVDA_Q2_2024.txt", "NVDA", "Q2 2024", 2024),
]

all_chunks = []
for path, ticker, quarter, year in documents:
    if not os.path.exists(path):
        print(f"SKIP {ticker} {quarter}: file not found ({path})")
        continue
    chunks = load_document(path, ticker, quarter, year)
    all_chunks.extend(chunks)
    print(f"{ticker} {quarter}: {len(chunks)} chunks ingested")

if not all_chunks:
    raise SystemExit("No transcripts found in data/transcripts/ — add .txt files first.")

build_vector_store(all_chunks)
print(f"Done. Total: {len(all_chunks)} chunks in ChromaDB.")
