"""Risk agent (Concept #4).

Surfaces risks, guidance changes, and red flags from the transcript.
"""

# src/agents/risk.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT = ChatPromptTemplate.from_template('''
You are a risk-focused equity analyst.
Identify ALL risk signals: regulatory concerns, margin pressure,
customer concentration, macro headwinds, guidance reductions.

Conversation so far: {history}
Context: {context}
Question: {question}

Format: Risk category | Severity (High/Med/Low) | Quote | Implication
''')

def risk_agent(question: str, context: str, history: str = '') -> str:
    chain = PROMPT | ChatOpenAI(model='gpt-4o', temperature=0) | StrOutputParser()
    return chain.invoke({'question': question, 'context': context, 'history': history})
