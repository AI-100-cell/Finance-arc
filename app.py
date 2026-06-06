"""Streamlit UI (Concept #1).

Entry point for the chat interface. Run with: streamlit run app.py
"""

# TODO: build the Streamlit chat UI wired to src/graph.py.
# app.py
import streamlit as st
from src.embeddings import load_vector_store
from src.retriever  import HybridRetriever
from src.graph      import build_graph, run_query
from src.ingest     import load_document
from src.embeddings import build_vector_store
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title='Earnings Intelligence', page_icon='$', layout='wide')
st.title('Earnings Call Intelligence')
st.caption('Ask anything about any company earnings — powered by multi-agent RAG + LangGraph')

# ── CACHE RESOURCES (load once per session) ──
@st.cache_resource
def load_system():
    vs      = load_vector_store()
    # Load all chunks into memory for BM25 (retrieve metadata from vs)
    results = vs.get(include=['documents', 'metadatas'])
    from langchain_core.documents import Document
    chunks  = [Document(page_content=d, metadata=m)
               for d, m in zip(results['documents'], results['metadatas'])]
    retriever = HybridRetriever(vs, chunks)
    graph   = build_graph(retriever)
    return graph

graph = load_system()

# ── SIDEBAR: add new transcripts ──
with st.sidebar:
    st.header('Add new transcript')
    uploaded = st.file_uploader('Upload PDF or TXT', type=['pdf', 'txt'])
    ticker   = st.text_input('Ticker (e.g. AAPL)')
    quarter  = st.text_input('Quarter (e.g. Q3 2024)')
    if st.button('Ingest') and uploaded and ticker and quarter:
        path = f'data/transcripts/{uploaded.name}'
        with open(path, 'wb') as f:
            f.write(uploaded.getbuffer())
        chunks = load_document(path, ticker, quarter, int(quarter[-4:]))
        build_vector_store(chunks)
        st.cache_resource.clear()
        st.success(f'Ingested {len(chunks)} chunks for {ticker}')

# ── CHAT INTERFACE ──
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input('Ask about any earnings call...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner('Agents thinking...'):
            result = run_query(graph, prompt)
        st.markdown(result['answer'])

        if result.get('citations'):
            with st.expander('View sources'):
                for c in result['citations'][:4]:
                    src = f"{c.get('ticker','')} {c.get('quarter','')} — {c.get('source','')}"
                    st.caption(src)

    st.session_state.messages.append({'role': 'assistant', 'content': result['answer']})
