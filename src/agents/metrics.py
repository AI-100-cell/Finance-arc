"""Metrics agent (Concept #4).

Extracts and reasons over quantitative financial metrics from transcripts.
"""

# src/agents/metrics.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT = ChatPromptTemplate.from_template('''
You are a senior financial analyst.
Use ONLY the context below from earnings transcripts.
Extract: metric name, value, period, YoY change, and the exact source sentence.

Conversation so far:
{history}

Context: {context}
Question: {question}

Format your answer as structured bullet points with citations.
''')

def metrics_agent(question: str, context: str, history: str = '') -> str:
    llm   = ChatOpenAI(model='gpt-4o', temperature=0)
    chain = PROMPT | llm | StrOutputParser()  # LCEL chain — Concept #1
    return chain.invoke({'question': question, 'context': context, 'history': history})

