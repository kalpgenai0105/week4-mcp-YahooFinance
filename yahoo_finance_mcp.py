#!/usr/bin/env python3
"""
Yahoo Finance MCP Server
========================
An MCP (Model Context Protocol) server that exposes Yahoo Finance data as tools
that Claude (or any MCP-compatible LLM) can call.

This server uses:
  - FastMCP  : high-level wrapper around the official MCP Python SDK
  - yfinance : unofficial but widely-used Python wrapper for Yahoo Finance
  - Pydantic : runtime input validation and schema generation

Transport: stdio (run as a subprocess by the MCP client)

Usage (add to Claude Desktop / Claude Code MCP config):
    {
      "mcpServers": {
        "yahoo_finance": {
          "command": "python",
          "args": ["path/to/yahoo_finance_mcp.py"]
        }
      }
    }

Dependencies (install once):
    pip install mcp yfinance pydantic
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import json                          # Serialize Python objects → JSON strings
from typing import Optional, List    # Type-hint helpers
from enum import Enum                # Define finite choice sets (e.g. output format)
from datetime import datetime        # Date arithmetic for history ranges

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import yfinance as yf                # Yahoo Finance data fetcher (no API key needed)
from pydantic import BaseModel, Field, field_validator, ConfigDict  # Input validation
from mcp.server.fastmcp import FastMCP  # High-level MCP server framework

# ---------------------------------------------------------------------------
# MCP server initialisation
# Naming convention: {service}_mcp  (all lowercase, underscores)
# ---------------------------------------------------------------------------
mcp = FastMCP("yahoo_finance_mcp")


# ===========================================================================
# SHARED ENUMS
# ===========================================================================

class ResponseFormat(str, Enum):
    """
    Controls whether a tool returns human-readable Markdown or raw JSON.
    Agents that need to parse the output programmatically should use JSON.
    """
    MARKDOWN = "markdown"
    JSON     = "json"


class HistoryPeriod(str, Enum):
    """
    Named time-window shortcuts accepted by yfinance's Ticker.history().
    Using an Enum prevents the model from passing arbitrary invalid strings.
    """
    ONE_DAY      = "1d"
    FIVE_DAYS    = "5d"
    ONE_MONTH    = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS   = "6mo"
    ONE_YEAR     = "1y"
    TWO_YEARS    = "2y"
    FIVE_YEARS   = "5y"
    TEN_YEARS    = "10y"
    YTD          = "ytd"
    MAX          = "max"


class HistoryInterval(str, Enum):
    """
    OHLCV bar size / granularity.
    Note: intraday intervals (1m–90m) only work with short periods (≤60 days).
    """
    ONE_MIN      = "1m"
    FIVE_MIN     = "5m"
    FIFTEEN_MIN  = "15m"
    THIRTY_MIN   = "30m"
    ONE_HOUR     = "1h"
    ONE_DAY      = "1d"
    ONE_WEEK     = "1wk"
    ONE_MONTH    = "1mo"


# ===========================================================================
# SHARED UTILITIES
# Called by every tool; keeps error handling consistent and avoids repetition.
# ===========================================================================

def _get_ticker(symbol: str) -> yf.Ticker:
    """
    Create a yfinance Ticker object for *symbol*.

    yfinance is lazy – no network request is made here; the Ticker object
    just stores the symbol and builds URLs on demand.
    """
    return yf.Ticker(symbol.upper())


def _safe_value(value) -> str:
    """
    Convert a value that might be None / NaN / a large number into a printable
    string.  Handles pandas NA values without importing pandas directly.
    """
    if value is None:
        return "N/A"
    try:
        # pandas.NA, numpy.nan, float('nan') all evaluate truthy for isnan
        import math
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return str(value)


def _format_large_number(n) -> str:
    """
    Express large market-cap / revenue figures in a human-readable way.
    E.g.  1_500_000_000 → "1.50B",  250_000_000 → "250.00M"
    """
    if n is None:
        return "N/A"
    try:
        n = float(n)
        if abs(n) >= 1_000_000_000_000:
            return f"{n / 1_000_000_000_000:.2f}T"
        elif abs(n) >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        elif abs(n) >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif abs(n) >= 1_000:
            return f"{n / 1_000:.2f}K"
        return f"{n:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _handle_error(e: Exception) -> str:
    """
    Produce a concise, actionable error string so the LLM can diagnose the
    problem without reading a full stack trace.
    """
    msg = str(e)
    if "No data found" in msg or "symbol may be delisted" in msg:
        return "Error: No data found. The ticker symbol may be invalid or delisted."
    if "rate limit" in msg.lower() or "too many requests" in msg.lower():
        return "Error: Yahoo Finance rate limit reached. Wait a minute and retry."
    if "connection" in msg.lower() or "timeout" in msg.lower():
        return "Error: Network error connecting to Yahoo Finance. Check your internet connection."
    return f"Error: {type(e).__name__}: {msg}"


# ===========================================================================
# PYDANTIC INPUT MODELS
# Each tool has its own model so Pydantic can enforce constraints and generate
# a JSON Schema that the MCP client shows to the LLM.
# ===========================================================================

class StockQuoteInput(BaseModel):
    """Input for retrieving a real-time stock quote."""
    model_config = ConfigDict(
        str_strip_whitespace=True,   # Trim accidental whitespace from ticker
        validate_assignment=True,
        extra="forbid",              # Reject unknown fields (prevents typos)
    )

    symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'TSLA')",
        min_length=1,
        max_length=10,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable, 'json' for machine-readable",
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        """Normalise ticker to uppercase so 'aapl' and 'AAPL' both work."""
        return v.strip().upper()


class StockHistoryInput(BaseModel):
    """Input for fetching historical OHLCV price data."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    symbol: str = Field(..., description="Stock ticker symbol (e.g. 'GOOGL')", min_length=1, max_length=10)
    period: HistoryPeriod = Field(
        default=HistoryPeriod.ONE_MONTH,
        description="Time window: '1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'",
    )
    interval: HistoryInterval = Field(
        default=HistoryInterval.ONE_DAY,
        description="Bar size: '1m','5m','15m','30m','1h','1d','1wk','1mo'",
    )
    max_rows: Optional[int] = Field(
        default=30,
        description="Cap the number of rows returned to avoid context overflow (default 30)",
        ge=1,
        le=500,
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()


class CompanyInfoInput(BaseModel):
    """Input for fetching detailed company metadata."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    symbol: str = Field(..., description="Stock ticker symbol (e.g. 'NVDA')", min_length=1, max_length=10)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable, 'json' for machine-readable",
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()


class FinancialsInput(BaseModel):
    """Input for retrieving financial statements (income, balance sheet, cash flow)."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    symbol: str = Field(..., description="Stock ticker symbol (e.g. 'AMZN')", min_length=1, max_length=10)
    statement: str = Field(
        default="income",
        description="Which statement: 'income' | 'balance' | 'cashflow'",
    )
    quarterly: bool = Field(
        default=False,
        description="True → most recent 4 quarters; False (default) → annual data",
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, v: str) -> str:
        allowed = {"income", "balance", "cashflow"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"statement must be one of {allowed}, got '{v}'")
        return v


class NewsInput(BaseModel):
    """Input for fetching recent news articles about a stock."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    symbol: str = Field(..., description="Stock ticker symbol (e.g. 'META')", min_length=1, max_length=10)
    max_articles: Optional[int] = Field(
        default=5,
        description="Number of news articles to return (1–20)",
        ge=1,
        le=20,
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()


class OptionsInput(BaseModel):
    """Input for fetching options chain data (calls + puts)."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    symbol: str = Field(..., description="Stock ticker symbol (e.g. 'SPY')", min_length=1, max_length=10)
    expiry_date: Optional[str] = Field(
        default=None,
        description="Expiry date string 'YYYY-MM-DD'. If omitted the nearest expiry is used.",
    )
    max_strikes: Optional[int] = Field(
        default=10,
        description="Max number of strikes per side (calls/puts) to return",
        ge=1,
        le=50,
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()


# ===========================================================================
# TOOLS
# Each @mcp.tool function is exposed to the LLM as a callable tool.
# The docstring becomes the tool description the LLM reads when deciding
# which tool to call.
# ===========================================================================

@mcp.tool(
    name="yfinance_get_stock_quote",
    annotations={
        "title":          "Get Stock Quote",
        "readOnlyHint":   True,   # does not modify any data
        "destructiveHint":False,
        "idempotentHint": True,   # same inputs → same output
        "openWorldHint":  True,   # fetches live data from the internet
    },
)
async def yfinance_get_stock_quote(params: StockQuoteInput) -> str:
    """
    Fetch the latest real-time (15-min delayed) stock quote for a ticker.

    Returns price, change, volume, market cap, P/E ratio, 52-week range,
    and a handful of other key metrics from Yahoo Finance's fast-quote endpoint.

    Args:
        params (StockQuoteInput):
            - symbol (str): Ticker, e.g. 'AAPL'
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Formatted stock quote (Markdown or JSON).
             Returns "Error: ..." on failure.

    Examples:
        - "What is Apple's current stock price?" → symbol='AAPL'
        - "Give me a JSON quote for Tesla"       → symbol='TSLA', format='json'
    """
    try:
        ticker = _get_ticker(params.symbol)

        # .fast_info is a lightweight endpoint – much faster than .info
        fi = ticker.fast_info

        # .info is heavier but contains P/E, analyst target, etc.
        # We wrap it in a try/except because it can fail for some tickers
        try:
            info = ticker.info
        except Exception:
            info = {}

        # Collect the fields we care about into a plain dict
        quote = {
            "symbol":            params.symbol,
            "name":              info.get("longName") or info.get("shortName", "N/A"),
            "price":             _safe_value(fi.last_price),
            "previous_close":    _safe_value(fi.previous_close),
            "open":              _safe_value(fi.open),
            "day_high":          _safe_value(fi.day_high),
            "day_low":           _safe_value(fi.day_low),
            "volume":            _safe_value(fi.last_volume),
            "market_cap":        _format_large_number(fi.market_cap),
            "fifty_two_week_high": _safe_value(fi.year_high),
            "fifty_two_week_low":  _safe_value(fi.year_low),
            "currency":          _safe_value(fi.currency),
            "exchange":          _safe_value(fi.exchange),
            "pe_ratio":          _safe_value(info.get("trailingPE")),
            "eps":               _safe_value(info.get("trailingEps")),
            "dividend_yield":    _safe_value(info.get("dividendYield")),
            "analyst_target":    _safe_value(info.get("targetMeanPrice")),
        }

        # ------------------------------------------------------------------
        # Format output
        # ------------------------------------------------------------------
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(quote, indent=2)

        # Markdown path: human-friendly layout
        change_raw = None
        try:
            change_raw = float(fi.last_price) - float(fi.previous_close)
            change_pct = (change_raw / float(fi.previous_close)) * 100
            change_str = f"{change_raw:+.2f} ({change_pct:+.2f}%)"
        except (TypeError, ValueError):
            change_str = "N/A"

        lines = [
            f"# {quote['name']} ({quote['symbol']})",
            f"**Exchange**: {quote['exchange']}  |  **Currency**: {quote['currency']}",
            "",
            f"## Price",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Current Price | **{quote['price']}** |",
            f"| Change | {change_str} |",
            f"| Previous Close | {quote['previous_close']} |",
            f"| Open | {quote['open']} |",
            f"| Day High | {quote['day_high']} |",
            f"| Day Low  | {quote['day_low']} |",
            f"| 52-Week High | {quote['fifty_two_week_high']} |",
            f"| 52-Week Low  | {quote['fifty_two_week_low']} |",
            "",
            f"## Key Metrics",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Market Cap | {quote['market_cap']} |",
            f"| Volume | {quote['volume']} |",
            f"| P/E Ratio | {quote['pe_ratio']} |",
            f"| EPS | {quote['eps']} |",
            f"| Dividend Yield | {quote['dividend_yield']} |",
            f"| Analyst Target | {quote['analyst_target']} |",
        ]
        return "\n".join(lines)

    except Exception as e:
        return _handle_error(e)


# ---------------------------------------------------------------------------

@mcp.tool(
    name="yfinance_get_stock_history",
    annotations={
        "title":          "Get Historical Price Data",
        "readOnlyHint":   True,
        "destructiveHint":False,
        "idempotentHint": True,
        "openWorldHint":  True,
    },
)
async def yfinance_get_stock_history(params: StockHistoryInput) -> str:
    """
    Download historical OHLCV (Open/High/Low/Close/Volume) price data.

    Useful for charting, back-testing, or analysing price trends over time.
    Returns a table of bars capped at max_rows to avoid context overflow.

    Args:
        params (StockHistoryInput):
            - symbol (str): Ticker, e.g. 'MSFT'
            - period (HistoryPeriod): Window like '1mo', '1y', 'max'
            - interval (HistoryInterval): Bar size like '1d', '1wk'
            - max_rows (int): Cap on returned rows (default 30)

    Returns:
        str: JSON object with 'symbol', 'period', 'interval', 'rows' list.
             Each row: {date, open, high, low, close, volume, change_pct}.
             Returns "Error: ..." on failure.

    Examples:
        - "Show daily prices for the last month"  → symbol='AAPL', period='1mo', interval='1d'
        - "Weekly OHLCV for TSLA last year"        → symbol='TSLA', period='1y',  interval='1wk'
    """
    try:
        ticker = _get_ticker(params.symbol)

        # Download the price history dataframe from Yahoo Finance
        df = ticker.history(period=params.period.value, interval=params.interval.value)

        if df.empty:
            return f"No historical data found for '{params.symbol}' with period={params.period.value}."

        # Optionally cap the number of rows to prevent overwhelming the context
        df = df.tail(params.max_rows)  # keep the MOST RECENT rows

        # Build a plain list of dicts (JSON-serialisable)
        rows = []
        for ts, row in df.iterrows():
            # Compute daily % change relative to that bar's open
            try:
                chg_pct = ((row["Close"] - row["Open"]) / row["Open"]) * 100
            except (ZeroDivisionError, TypeError):
                chg_pct = None

            rows.append({
                "date":       ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
                "open":       round(float(row["Open"]),   4),
                "high":       round(float(row["High"]),   4),
                "low":        round(float(row["Low"]),    4),
                "close":      round(float(row["Close"]),  4),
                "volume":     int(row["Volume"]),
                "change_pct": round(chg_pct, 2) if chg_pct is not None else None,
            })

        result = {
            "symbol":        params.symbol,
            "period":        params.period.value,
            "interval":      params.interval.value,
            "total_rows":    len(rows),
            "rows":          rows,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


# ---------------------------------------------------------------------------

@mcp.tool(
    name="yfinance_get_company_info",
    annotations={
        "title":          "Get Company Information",
        "readOnlyHint":   True,
        "destructiveHint":False,
        "idempotentHint": True,
        "openWorldHint":  True,
    },
)
async def yfinance_get_company_info(params: CompanyInfoInput) -> str:
    """
    Retrieve detailed company profile: sector, industry, employees, description,
    website, headquarters, and valuation multiples.

    Args:
        params (CompanyInfoInput):
            - symbol (str): Ticker, e.g. 'NVDA'
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Company profile (Markdown or JSON).
             Returns "Error: ..." on failure.

    Examples:
        - "Tell me about Apple as a company"    → symbol='AAPL'
        - "What sector is NVIDIA in?"           → symbol='NVDA'
        - "Get JSON company data for Microsoft" → symbol='MSFT', format='json'
    """
    try:
        ticker = _get_ticker(params.symbol)
        info   = ticker.info   # full metadata dict from Yahoo Finance

        if not info or info.get("quoteType") is None:
            return f"Error: No company information found for '{params.symbol}'."

        # Extract a curated subset of the 100+ fields Yahoo provides
        profile = {
            "symbol":            params.symbol,
            "name":              info.get("longName", "N/A"),
            "summary":           info.get("longBusinessSummary", "N/A"),
            "sector":            info.get("sector", "N/A"),
            "industry":          info.get("industry", "N/A"),
            "website":           info.get("website", "N/A"),
            "country":           info.get("country", "N/A"),
            "city":              info.get("city", "N/A"),
            "employees":         info.get("fullTimeEmployees", "N/A"),
            "market_cap":        _format_large_number(info.get("marketCap")),
            "enterprise_value":  _format_large_number(info.get("enterpriseValue")),
            "revenue_ttm":       _format_large_number(info.get("totalRevenue")),
            "gross_margins":     _safe_value(info.get("grossMargins")),
            "operating_margins": _safe_value(info.get("operatingMargins")),
            "profit_margins":    _safe_value(info.get("profitMargins")),
            "pe_ratio":          _safe_value(info.get("trailingPE")),
            "peg_ratio":         _safe_value(info.get("pegRatio")),
            "price_to_book":     _safe_value(info.get("priceToBook")),
            "beta":              _safe_value(info.get("beta")),
            "52w_high":          _safe_value(info.get("fiftyTwoWeekHigh")),
            "52w_low":           _safe_value(info.get("fiftyTwoWeekLow")),
            "recommendation":    info.get("recommendationKey", "N/A"),
        }

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(profile, indent=2)

        # Markdown layout
        lines = [
            f"# {profile['name']} ({profile['symbol']})",
            f"**{profile['sector']}** › {profile['industry']}",
            f"📍 {profile['city']}, {profile['country']}  |  🌐 {profile['website']}",
            f"👥 Employees: {profile['employees']}",
            "",
            f"## Business Summary",
            profile["summary"],
            "",
            f"## Financials",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Market Cap | {profile['market_cap']} |",
            f"| Enterprise Value | {profile['enterprise_value']} |",
            f"| Revenue (TTM) | {profile['revenue_ttm']} |",
            f"| Gross Margin | {profile['gross_margins']} |",
            f"| Operating Margin | {profile['operating_margins']} |",
            f"| Net Profit Margin | {profile['profit_margins']} |",
            "",
            f"## Valuation",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| P/E Ratio | {profile['pe_ratio']} |",
            f"| PEG Ratio | {profile['peg_ratio']} |",
            f"| Price/Book | {profile['price_to_book']} |",
            f"| Beta | {profile['beta']} |",
            f"| Analyst Rating | {profile['recommendation'].upper()} |",
        ]
        return "\n".join(lines)

    except Exception as e:
        return _handle_error(e)


# ---------------------------------------------------------------------------

@mcp.tool(
    name="yfinance_get_financials",
    annotations={
        "title":          "Get Financial Statements",
        "readOnlyHint":   True,
        "destructiveHint":False,
        "idempotentHint": True,
        "openWorldHint":  True,
    },
)
async def yfinance_get_financials(params: FinancialsInput) -> str:
    """
    Retrieve financial statement data: income statement, balance sheet, or cash flow.

    Supports both annual (up to 4 years) and quarterly (most recent 4 quarters)
    views.  All figures are in the company's reporting currency.

    Args:
        params (FinancialsInput):
            - symbol (str): Ticker, e.g. 'AMZN'
            - statement (str): 'income' | 'balance' | 'cashflow'
            - quarterly (bool): True for quarterly, False for annual

    Returns:
        str: JSON object with {symbol, statement, frequency, columns, data}.
             'data' is a list of {row_label: value} dicts.
             Returns "Error: ..." on failure.

    Examples:
        - "Show Apple's annual income statement"   → symbol='AAPL', statement='income'
        - "Quarterly balance sheet for Amazon"      → symbol='AMZN', statement='balance', quarterly=True
        - "Tesla cash flow statement"               → symbol='TSLA', statement='cashflow'
    """
    try:
        ticker = _get_ticker(params.symbol)

        # Pick the correct dataframe attribute based on user input
        if params.statement == "income":
            df = ticker.quarterly_income_stmt if params.quarterly else ticker.income_stmt
        elif params.statement == "balance":
            df = ticker.quarterly_balance_sheet if params.quarterly else ticker.balance_sheet
        else:  # cashflow
            df = ticker.quarterly_cashflow if params.quarterly else ticker.cashflow

        if df is None or df.empty:
            return f"No {params.statement} statement data found for '{params.symbol}'."

        # Transpose so rows = metrics, columns = dates
        # yfinance returns rows=metrics, columns=dates already, but let's confirm
        # Column headers are Timestamps → convert to readable date strings
        columns = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c) for c in df.columns]

        # Build a list of {label, col1_value, col2_value, ...} for each metric
        rows_out = []
        for label, series in df.iterrows():
            row = {"label": str(label)}
            for col_ts, col_name in zip(df.columns, columns):
                val = series[col_ts]
                # Convert pandas NA / numpy floats to plain Python
                try:
                    import math
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        row[col_name] = None
                    else:
                        row[col_name] = float(val)
                except (TypeError, ValueError):
                    row[col_name] = None
            rows_out.append(row)

        result = {
            "symbol":    params.symbol,
            "statement": params.statement,
            "frequency": "quarterly" if params.quarterly else "annual",
            "columns":   columns,   # list of date strings (most recent first)
            "data":      rows_out,  # list of metric rows
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


# ---------------------------------------------------------------------------

@mcp.tool(
    name="yfinance_get_news",
    annotations={
        "title":          "Get Stock News",
        "readOnlyHint":   True,
        "destructiveHint":False,
        "idempotentHint": False,  # news changes over time
        "openWorldHint":  True,
    },
)
async def yfinance_get_news(params: NewsInput) -> str:
    """
    Fetch recent news articles related to a stock ticker from Yahoo Finance.

    Returns headlines, publishers, links, and publish times.  Useful for
    understanding sentiment and recent events around a company.

    Args:
        params (NewsInput):
            - symbol (str): Ticker, e.g. 'META'
            - max_articles (int): Number of articles to return (1–20, default 5)

    Returns:
        str: JSON list of {title, publisher, link, publish_time}.
             Returns "Error: ..." or "No news found" on failure.

    Examples:
        - "Any recent news about Google?"      → symbol='GOOGL'
        - "Latest 10 news items for NVIDIA"    → symbol='NVDA', max_articles=10
    """
    try:
        ticker = _get_ticker(params.symbol)
        news   = ticker.news   # returns a list of dicts

        if not news:
            return f"No recent news found for '{params.symbol}'."

        # Trim to requested count
        news = news[: params.max_articles]

        articles = []
        for item in news:
            # Yahoo Finance news schema: contentType, id, title, publisher, link,
            # providerPublishTime (Unix epoch), thumbnail, relatedTickers
            pub_time = item.get("providerPublishTime")
            if pub_time:
                pub_time = datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M UTC")
            else:
                pub_time = "N/A"

            articles.append({
                "title":        item.get("title", "N/A"),
                "publisher":    item.get("publisher", "N/A"),
                "link":         item.get("link", "N/A"),
                "publish_time": pub_time,
            })

        result = {
            "symbol":   params.symbol,
            "articles": articles,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


# ---------------------------------------------------------------------------

@mcp.tool(
    name="yfinance_get_options_chain",
    annotations={
        "title":          "Get Options Chain",
        "readOnlyHint":   True,
        "destructiveHint":False,
        "idempotentHint": True,
        "openWorldHint":  True,
    },
)
async def yfinance_get_options_chain(params: OptionsInput) -> str:
    """
    Retrieve the options chain (calls and puts) for a given expiry date.

    If no expiry_date is provided the nearest available expiry is used.
    Returns strike prices, bid/ask, implied volatility, open interest, and
    in-the-money status for both calls and puts.

    Args:
        params (OptionsInput):
            - symbol (str): Ticker, e.g. 'SPY'
            - expiry_date (Optional[str]): 'YYYY-MM-DD'. Nearest expiry used if omitted.
            - max_strikes (int): Max strikes per side (default 10)

    Returns:
        str: JSON with {symbol, expiry, calls:[...], puts:[...]}.
             Returns "Error: ..." on failure.

    Examples:
        - "Show me SPY options expiring nearest date"  → symbol='SPY'
        - "AAPL options expiring 2025-01-17"           → symbol='AAPL', expiry_date='2025-01-17'
    """
    try:
        ticker  = _get_ticker(params.symbol)
        expiries = ticker.options   # tuple of expiry date strings "YYYY-MM-DD"

        if not expiries:
            return f"No options data available for '{params.symbol}'."

        # Choose expiry: use the first (nearest) if not specified
        if params.expiry_date:
            # Validate the requested expiry exists
            if params.expiry_date not in expiries:
                return (
                    f"Expiry '{params.expiry_date}' not found for '{params.symbol}'. "
                    f"Available expiries: {', '.join(expiries[:10])}"
                )
            chosen_expiry = params.expiry_date
        else:
            chosen_expiry = expiries[0]

        # Fetch the option chain for the chosen expiry
        chain = ticker.option_chain(chosen_expiry)  # namedtuple: .calls, .puts

        def _parse_chain(df, side: str) -> List[dict]:
            """Helper: convert a calls/puts DataFrame into a list of dicts."""
            keep_cols = [
                "strike", "lastPrice", "bid", "ask", "impliedVolatility",
                "openInterest", "volume", "inTheMoney",
            ]
            # Only keep columns that actually exist in the dataframe
            cols = [c for c in keep_cols if c in df.columns]
            df   = df[cols].head(params.max_strikes)

            rows = []
            for _, row in df.iterrows():
                entry = {"side": side}
                for col in cols:
                    val = row[col]
                    try:
                        import math
                        if isinstance(val, float) and math.isnan(val):
                            entry[col] = None
                        elif col == "impliedVolatility":
                            entry[col] = round(float(val) * 100, 2)  # express as %
                        elif col in ("openInterest", "volume"):
                            entry[col] = int(val) if val is not None else None
                        else:
                            entry[col] = round(float(val), 4) if isinstance(val, float) else val
                    except (TypeError, ValueError):
                        entry[col] = None
                rows.append(entry)
            return rows

        result = {
            "symbol":          params.symbol,
            "expiry":          chosen_expiry,
            "available_expiries": list(expiries[:10]),   # preview of nearby expiries
            "calls":           _parse_chain(chain.calls, "call"),
            "puts":            _parse_chain(chain.puts,  "put"),
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return _handle_error(e)


# ===========================================================================
# ENTRY POINT
# When run directly (not imported), start the MCP server over stdio transport.
# The MCP client (e.g. Claude Desktop) launches this script as a subprocess
# and communicates through stdin/stdout using the MCP JSON-RPC protocol.
# ===========================================================================
if __name__ == "__main__":
    # mcp.run() starts the server event loop.
    # By default FastMCP uses stdio transport (reads from stdin, writes to stdout).
    # This is the standard pattern for local MCP servers invoked by a client.
    mcp.run()
