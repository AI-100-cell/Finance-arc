"""Comparison agent (Concept #4).

Compares metrics/tone across periods or peer companies.
"""

# src/agents/comparison.py
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
PROMPT = ChatPromptTemplate.from_template('''
You are a comparative financial analyst.
Using the multi-period or multi-company context below,
produce a structured comparison and highlight the most significant changes.

Conversation so far: {history}
Context: {context}
Question: {question}

Format: 1) Comparison table  2) Key insight  3) Source quotes
''')

def comparison_agent(question: str, context: str, history: str = '') -> str:
    chain = PROMPT | ChatOpenAI(model='gpt-4o', temperature=0) | StrOutputParser()
    return chain.invoke({'question': question, 'context': context, 'history': history})
