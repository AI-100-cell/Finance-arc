"""LangGraph workflow (Concept #5).

Wires the agents, retriever, memory, and tools into a LangGraph state machine.
"""

# TODO: define the graph nodes/edges and compile the workflow.

# src/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json

from src.memory import AgentState, build_history_context, memory_update_node
from src.retriever import HybridRetriever
from src.agents.metrics    import metrics_agent
from src.agents.tone       import tone_agent
from src.agents.comparison import comparison_agent
from src.agents.risk       import risk_agent
from src.tools.mcp_server  import ALL_TOOLS

PLAN_PROMPT = ChatPromptTemplate.from_template('''
You are a financial query planner. Output ONLY valid JSON — no other text.
{{
  "tickers": ["AAPL"],
  "agents":  ["metrics", "tone"],
  "sub_questions": {{
    "metrics": "What were the gross margins?",
    "tone":    "How confident was the CFO?"
  }}
}}
Available agents: metrics, tone, comparison, risk
Conversation so far: {history}
Question: {question}
''')

# ── NODE FUNCTIONS ──────────────────────────────────

def query_planner_node(state: AgentState) -> dict:
    llm   = ChatOpenAI(model='gpt-4o', temperature=0)
    chain = PLAN_PROMPT | llm | StrOutputParser()
    history = build_history_context(state)
    raw  = chain.invoke({'question': state['user_query'], 'history': history})
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {'tickers': [], 'agents': [], 'sub_questions': {}}
    return {'plan': plan}

def run_agents_node(state: AgentState, retriever: HybridRetriever) -> dict:
    plan = state.get('plan') or {}
    if not isinstance(plan, dict):
        plan = {}

    history = build_history_context(state)
    ticker  = plan.get('tickers', [None])[0] if plan.get('tickers') else None
    results = {}

    for agent_name in plan.get('agents', []):
        sub_q   = plan.get('sub_questions', {}).get(agent_name, state['user_query'])
        chunks  = retriever.retrieve(sub_q, ticker=ticker)
        context = '\n\n'.join([c.page_content for c in chunks])
        cites   = [c.metadata for c in chunks]

        if   agent_name == 'metrics':    results['metrics']    = metrics_agent(sub_q, context, history)
        elif agent_name == 'tone':        results['tone']       = tone_agent(sub_q, context, history)
        elif agent_name == 'comparison':  results['comparison'] = comparison_agent(sub_q, context, history)
        elif agent_name == 'risk':        results['risk']       = risk_agent(sub_q, context, history)
        results['citations'] = cites

    return {'agent_outputs': results}

def synthesizer_node(state: AgentState) -> dict:
    llm     = ChatOpenAI(model='gpt-4o', temperature=0)
    outputs = state.get('agent_outputs', {})
    prompt  = f'''Combine the agent outputs below into one clear answer.
Preserve all citations. Keep it concise.
Agent outputs: {json.dumps(outputs, indent=2)}
Original question: {state['user_query']}'''
    answer = llm.invoke(prompt).content
    return {'final_answer': answer, 'citations': (outputs or {}).get('citations', [])}

# ── CONDITIONAL EDGE ─────────────────────────────────
def route_to_agents(state: AgentState):
    agents = (state.get('plan') or {}).get('agents', [])
    if not agents:
        return 'synthesizer'
    return 'run_agents'

# ── BUILD THE GRAPH ───────────────────────────────────
def build_graph(retriever: HybridRetriever):
    graph = StateGraph(AgentState)

    def _memory_inject(state: AgentState) -> dict:
        return {}

    def _run_agents(state: AgentState) -> dict:
        return run_agents_node(state, retriever)

    # Add nodes
    graph.add_node('memory_inject', _memory_inject)
    graph.add_node('query_planner', query_planner_node)
    graph.add_node('run_agents',    _run_agents)
    graph.add_node('synthesizer',   synthesizer_node)
    graph.add_node('memory_update', memory_update_node)

    # Add edges
    graph.set_entry_point('memory_inject')
    graph.add_edge('memory_inject', 'query_planner')
    graph.add_conditional_edges('query_planner', route_to_agents,
        {'run_agents': 'run_agents', 'synthesizer': 'synthesizer'})
    graph.add_edge('run_agents',  'synthesizer')
    graph.add_edge('synthesizer', 'memory_update')
    graph.add_edge('memory_update', END)

    # MemorySaver persists state across turns — Concept #2
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

def run_query(graph, query: str, thread_id: str = 'default') -> dict:
    config = {'configurable': {'thread_id': thread_id}}
    result = graph.invoke({'user_query': query, 'chat_history': []}, config)
    return {'answer': result.get('final_answer'), 'citations': result.get('citations', [])}

