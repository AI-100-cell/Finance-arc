"""Conversation memory (Concept #2).

Tracks dialogue history so multi-turn questions retain context.
"""

# TODO: implement a memory buffer / summary memory for the chat session.
# src/memory.py
from typing import TypedDict, List, Optional, Annotated
import operator

class Message(TypedDict):
    role: str   # 'user' or 'assistant'
    content: str

class AgentState(TypedDict):
    # Core inputs
    user_query:   str
    # Conversation memory — Annotated with operator.add for state merging
    chat_history: Annotated[List[Message], operator.add]
    # Query planning outputs
    plan:         Optional[dict]
    # Agent results
    agent_outputs: Optional[dict]
    # Final answer
    final_answer: Optional[str]
    citations:    Optional[list]
def build_history_context(state: AgentState) -> str:
    """Formats chat_history into a string for injection into prompts."""
    history = state.get('chat_history', [])
    if not history:
        return 'No prior conversation.'
    lines = []
    for msg in history[-6:]:   # Last 3 exchanges to avoid context overflow
        prefix = 'User' if msg['role'] == 'user' else 'Assistant'
        lines.append(f"{prefix}: {msg['content'][:300]}")
    return '\n'.join(lines)

def memory_injection_node(_state: AgentState) -> dict:
    """LangGraph node: inject memory into state before planning."""
    # State is already loaded by LangGraph checkpointer
    # Just return unchanged — agents read from state directly
    return {}

def memory_update_node(state: AgentState) -> dict:
    """LangGraph node: append this Q&A turn to history."""
    new_messages = [
        {'role': 'user',      'content': state['user_query']},
        {'role': 'assistant', 'content': state.get('final_answer', '')},
    ]
    return {'chat_history': new_messages}
