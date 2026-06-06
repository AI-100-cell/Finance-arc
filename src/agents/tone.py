"""Tone agent (Concept #4).

Analyzes management sentiment / tone in the earnings call.
"""

# src/agents/tone.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT = ChatPromptTemplate.from_template('''
You are an expert in executive communication analysis.
Analyze the transcript below for: hedging words (believe/expect/may),
confidence signals (clear/strong/committed), topic avoidance, and tone shifts.

Conversation so far: {history}
Context: {context}
Question: {question}

Rate overall tone: Bullish / Cautious / Bearish.
Provide 2-3 specific quotes that support your rating.
''')

def tone_agent(question: str, context: str, history: str = '') -> str:
    chain = PROMPT | ChatOpenAI(model='gpt-4o', temperature=0) | StrOutputParser()
    return chain.invoke({'question': question, 'context': context, 'history': history})
