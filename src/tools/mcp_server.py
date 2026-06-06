"""MCP tool server (Concept #6).

Exposes the project's tools over the Model Context Protocol.
"""

# src/tools/mcp_server.py
from langchain_core.tools import tool
import yfinance as yf
import requests, json

@tool
def financial_calculator(expression: str) -> str:
    """Evaluate a financial math expression safely.
    Example: 'gross_margin = (revenue - cogs) / revenue * 100'"""
    import builtins
    allowed = {name: getattr(builtins, name)
               for name in ('abs', 'round', 'min', 'max', 'sum')}
    try:
        result = eval(expression, {'__builtins__': {}}, allowed)
        return f'Result: {result}'
    except Exception as e:
        return f'Calculation error: {e}'

@tool
def get_stock_price(ticker: str) -> str:
    """Fetch the current price and key metrics for a stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        return json.dumps({
            'ticker':        ticker.upper(),
            'price':         info.get('currentPrice', 'N/A'),
            'market_cap':    info.get('marketCap', 'N/A'),
            'pe_ratio':      info.get('trailingPE', 'N/A'),
            'revenue_growth':info.get('revenueGrowth', 'N/A'),
        }, indent=2)
    except Exception as e:
        return f'Error fetching {ticker}: {e}'

@tool
def fetch_sec_summary(ticker: str) -> str:
    """Fetch the latest SEC filing summary for a ticker from EDGAR."""
    url = (f'https://data.sec.gov/submissions/CIK'
           f'{ticker}.json')
    try:
        r = requests.get(url, headers={'User-Agent': 'earnings-ai@demo.com'})
        data = r.json()
        filings = data.get('filings', {}).get('recent', {})
        forms   = filings.get('form', [])[:5]
        dates   = filings.get('filingDate', [])[:5]
        return json.dumps({'recent_forms': forms, 'dates': dates})
    except Exception as e:
        return f'Error fetching SEC data: {e}'

# Export all tools as a list for LangGraph binding
ALL_TOOLS = [financial_calculator, get_stock_price, fetch_sec_summary]
