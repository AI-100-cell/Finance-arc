# Earnings Call Intelligence

A multi-agent RAG system for querying earnings call transcripts. Ask natural-language questions about any company's financials, management tone, risks, or competitive positioning — powered by LangGraph, ChromaDB, and GPT-4o.

## Architecture

```
User question
     │
     ▼
Query Planner (GPT-4o)          ← decides which agents to run
     │
     ▼
Hybrid Retriever                 ← ChromaDB (semantic) + BM25 (keyword)
     │
     ├── Metrics Agent           ← revenue, margins, EPS, guidance
     ├── Tone Agent              ← management sentiment & confidence
     ├── Risk Agent              ← headwinds, uncertainties, warnings
     └── Comparison Agent        ← cross-company / cross-quarter diffs
     │
     ▼
Synthesizer (GPT-4o)            ← merges agent outputs into one answer
     │
     ▼
Response + Citations
```

Conversation memory is persisted across turns via LangGraph's `MemorySaver` checkpointer.

## Features

- Upload PDF or TXT earnings transcripts via the UI or REST API
- Hybrid retrieval (dense + sparse) for higher recall
- Four specialist agents — each focused on a different analysis dimension
- Multi-turn chat with conversation history
- FastAPI REST interface alongside the Streamlit UI
- Source citations on every answer

## Quick Start

**1. Clone and set up the environment**

```bash
git clone https://github.com/AI-100-cell/Finance-arc.git
cd Finance-arc
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required keys:

| Variable | Where to get it |
|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `FMP_API_KEY` | https://financialmodelingprep.com/developer/docs |

**3. Ingest transcripts**

```bash
python ingest_all.py
```

Or upload directly from the sidebar in the UI.

**4. Run the Streamlit app**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**5. Run the REST API (optional)**

```bash
uvicorn api:app --reload
```

API docs at `http://localhost:8000/docs`

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/query` | Ask a question |
| `POST` | `/ingest` | Upload a transcript |

**Example query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue growth for AAPL?", "thread_id": "session1"}'
```

## Project Structure

```
├── app.py                  # Streamlit UI entry point
├── api.py                  # FastAPI REST interface
├── ingest_all.py           # Batch ingestion script
├── requirements.txt
├── src/
│   ├── graph.py            # LangGraph workflow
│   ├── memory.py           # Conversation state & history
│   ├── retriever.py        # Hybrid retriever (ChromaDB + BM25)
│   ├── embeddings.py       # Vector store setup
│   ├── ingest.py           # Document loading & chunking
│   ├── agents/
│   │   ├── metrics.py
│   │   ├── tone.py
│   │   ├── risk.py
│   │   └── comparison.py
│   └── tools/
│       ├── api_tools.py    # FMP financial data tools
│       ├── calculator.py   # Numeric computation
│       ├── chart.py        # Chart generation
│       └── mcp_server.py   # MCP tool definitions
└── data/
    └── transcripts/        # Place PDF/TXT transcripts here
```

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o)
- Financial Modeling Prep API key (for live financial data tools)
