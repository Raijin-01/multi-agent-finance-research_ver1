"""
AI Finance Research Team - Gemini + LangGraph
Data-grounded, production-style multi-agent financial research system.

This module retrieves market/fundamental data from Yahoo Finance, news from RSS,
calculates technical indicators in Python, and uses Gemini only for interpretation.
"""

from __future__ import annotations

import functools
import logging
import operator
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict
from urllib.parse import quote_plus


def _ensure_packages(packages: dict) -> None:
    for pip_name, import_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            except Exception as exc:
                print(f"[setup] Could not auto-install '{pip_name}': {exc}")


_ensure_packages({"yfinance": "yfinance", "feedparser": "feedparser", "langgraph": "langgraph"})
_ensure_packages({"markdown2": "markdown2", "xhtml2pdf": "xhtml2pdf"})

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
from langgraph.graph import StateGraph, START, END


@dataclass(frozen=True)
class Config:
    gemini_model: str = "gemini-3.6-flash"
    max_retries: int = 3
    retry_base_delay: float = 1.5
    request_timeout: float = 20.0
    history_period: str = "1y"
    history_interval: str = "1d"
    cache_ttl_seconds: int = 900
    stale_market_hours: float = 24.0
    stale_news_days: int = 14
    news_lookback_days: int = 14
    max_news_items: int = 12
    price_consistency_tolerance: float = 0.02


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("finance_research_team")
RUN_LOG: list[dict] = []

NODE_LABELS = {
    "manager": "Manager", "market_data": "Market (data)", "market": "Market (analysis)",
    "news_data": "News (data)", "news": "News (analysis)", "fundamentals_data": "Fundamentals (data)",
    "fundamentals": "Fundamentals (analysis)", "technical_data": "Technical (data)",
    "technical": "Technical (analysis)", "risk": "Risk", "verification": "Verification",
    "critic": "Critic", "report": "Report",
}


def retry_with_backoff(max_retries: int = CFG.max_retries, base_delay: float = CFG.retry_base_delay, exceptions: tuple = (Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    wait = base_delay * (2 ** (attempt - 1))
                    logger.warning("%s failed (attempt %d/%d): %s - retrying in %.1fs", func.__name__, attempt, max_retries, exc, wait)
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


_CACHE: dict[str, tuple[float, Any]] = {}


def ttl_cache(seconds: int = CFG.cache_ttl_seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            now = time.time()
            cached = _CACHE.get(key)
            if cached and now - cached[0] < seconds:
                return cached[1]
            result = func(*args, **kwargs)
            _CACHE[key] = (now, result)
            return result
        return wrapper
    return decorator


def safe_node(node_name: str, critical: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: "State") -> dict:
            start = time.perf_counter()
            logger.info("[%s] started", node_name)
            try:
                result = func(state) or {}
                elapsed = time.perf_counter() - start
                RUN_LOG.append({"node": node_name, "status": "ok", "seconds": round(elapsed, 2)})
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("[%s] FAILED after %.2fs: %s", node_name, elapsed, exc)
                RUN_LOG.append({"node": node_name, "status": "failed", "seconds": round(elapsed, 2), "error": str(exc)})
                if critical:
                    raise
                return {"errors": [{"node": node_name, "error": str(exc), "time": datetime.now(timezone.utc).isoformat()}]}
        return wrapper
    return decorator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hours_since(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        then = datetime.fromisoformat(iso_ts)
        return (datetime.now(timezone.utc) - then).total_seconds() / 3600.0
    except Exception:
        return None


@retry_with_backoff(exceptions=(Exception,))
def _call_gemini(prompt: str) -> str:
    if "client" not in globals() and "client" not in dir(sys.modules["__main__"]):
        raise RuntimeError("Gemini `client` is not initialized. Expected a global client with client.interactions.create().")
    gemini_client = globals().get("client") or getattr(sys.modules["__main__"], "client")
    response = gemini_client.interactions.create(model=CFG.gemini_model, input=prompt)
    text = getattr(response, "output_text", None)
    if not text:
        raise ValueError("Gemini response had no `output_text`.")
    return text


def ask_gemini(prompt: str, fallback: str = "") -> str:
    try:
        return _call_gemini(prompt)
    except Exception as exc:
        logger.error("[gemini] call failed permanently: %s", exc)
        return fallback or f"DATA NOT AVAILABLE (Gemini call failed: {exc})"


def _fast_info_get(fast_info, keys: list[str], default=None):
    for key in keys:
        try:
            value = fast_info[key]
            if value is not None:
                return value
        except Exception:
            pass
    return default


@ttl_cache()
@retry_with_backoff(max_retries=2, exceptions=(Exception,))
def fetch_market_data(ticker: str) -> dict:
    retrieved_at = now_iso()
    tk = yf.Ticker(ticker)
    hist = tk.history(period=CFG.history_period, interval=CFG.history_interval, auto_adjust=False, timeout=CFG.request_timeout)
    if hist is None or hist.empty:
        raise ValueError(f"No historical price data returned for '{ticker}'.")
    fast_info = tk.fast_info
    current_price = _fast_info_get(fast_info, ["lastPrice", "last_price"], float(hist["Close"].iloc[-1]))
    market_cap = _fast_info_get(fast_info, ["marketCap", "market_cap"])
    year_high = _fast_info_get(fast_info, ["yearHigh", "year_high"])
    year_low = _fast_info_get(fast_info, ["yearLow", "year_low"])
    return {
        "status": "ok", "ticker": ticker, "current_price": round(float(current_price), 2), "market_cap": market_cap,
        "52w_high": round(float(year_high), 2) if year_high else None, "52w_low": round(float(year_low), 2) if year_low else None,
        "latest_volume": int(hist["Volume"].iloc[-1]), "avg_volume_20d": int(hist["Volume"].tail(20).mean()),
        "last_trading_date": hist.index[-1].strftime("%Y-%m-%d"), "history": hist, "retrieved_at": retrieved_at,
        "source": "Yahoo Finance (yfinance)",
    }


def _latest_from_statement(df: Optional[pd.DataFrame], row_names: list[str]):
    if df is None or df.empty:
        return None, None
    for name in row_names:
        if name in df.index:
            row = df.loc[name].dropna()
            if not row.empty:
                try:
                    return float(row.iloc[0]), str(row.index[0])[:10]
                except (TypeError, ValueError):
                    return None, None
    return None, None


@ttl_cache()
@retry_with_backoff(max_retries=2, exceptions=(Exception,))
def fetch_fundamentals(ticker: str) -> dict:
    retrieved_at = now_iso()
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    income, balance, cashflow = tk.income_stmt, tk.balance_sheet, tk.cashflow
    revenue, revenue_period = _latest_from_statement(income, ["Total Revenue", "TotalRevenue"])
    net_income, ni_period = _latest_from_statement(income, ["Net Income", "NetIncome"])
    op_income, oi_period = _latest_from_statement(income, ["Operating Income", "OperatingIncome"])
    gross_profit, gp_period = _latest_from_statement(income, ["Gross Profit", "GrossProfit"])
    fcf, fcf_period = _latest_from_statement(cashflow, ["Free Cash Flow", "FreeCashFlow"])
    op_cashflow, ocf_period = _latest_from_statement(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex, capex_period = _latest_from_statement(cashflow, ["Capital Expenditure", "CapitalExpenditure", "Purchase Of PP E", "Net PPE Purchase And Sale"])
    buybacks, buyback_period = _latest_from_statement(cashflow, ["Repurchase Of Capital Stock", "Common Stock Payments"])
    dividends_paid, div_period = _latest_from_statement(cashflow, ["Cash Dividends Paid", "Common Stock Dividend Paid"])
    total_cash, cash_period = _latest_from_statement(balance, ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash Cash Equivalents And Short Term Investments"])
    total_debt, debt_period = _latest_from_statement(balance, ["Total Debt", "TotalDebt"])
    total_equity, equity_period = _latest_from_statement(balance, ["Stockholders Equity", "Total Stockholders Equity"])
    return {
        "status": "ok", "ticker": ticker,
        "revenue": {"value": revenue, "period": revenue_period}, "gross_profit": {"value": gross_profit, "period": gp_period},
        "operating_income": {"value": op_income, "period": oi_period}, "net_income": {"value": net_income, "period": ni_period},
        "free_cash_flow": {"value": fcf, "period": fcf_period}, "operating_cash_flow": {"value": op_cashflow, "period": ocf_period},
        "capital_expenditure": {"value": capex, "period": capex_period}, "share_buybacks": {"value": buybacks, "period": buyback_period},
        "dividends_paid": {"value": dividends_paid, "period": div_period}, "total_cash": {"value": total_cash, "period": cash_period},
        "total_debt": {"value": total_debt, "period": debt_period}, "total_equity": {"value": total_equity, "period": equity_period},
        "shares_outstanding": info.get("sharesOutstanding"), "eps_ttm": info.get("trailingEps"), "eps_forward": info.get("forwardEps"),
        "pe_ttm": info.get("trailingPE"), "pe_forward": info.get("forwardPE"), "ps_ttm": info.get("priceToSalesTrailing12Months"),
        "peg_ratio": info.get("pegRatio") or info.get("trailingPegRatio"), "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "price_to_fcf": round(info["marketCap"] / fcf, 2) if info.get("marketCap") and fcf and fcf > 0 else None,
        "dividend_yield": info.get("dividendYield"), "debt_to_equity": info.get("debtToEquity"),
        "revenue_segments_note": "Segment-level revenue requires primary filings and is DATA NOT AVAILABLE unless surfaced from a primary source.",
        "retrieved_at": retrieved_at, "source": f"Yahoo Finance (yfinance) - {ticker} financial statements & key stats",
    }


@ttl_cache()
def fetch_news(ticker: str, company_name: str) -> list[dict]:
    retrieved_at = now_iso()
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={quote_plus(company_name)}+when:{CFG.news_lookback_days}d&hl=en-US&gl=US&ceid=US:en",
    ]
    items, seen_titles = [], set()
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            feed_title = parsed.feed.get("title", "Unknown feed") if hasattr(parsed, "feed") else "Unknown feed"
            for entry in parsed.entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                key = title.lower()[:90]
                if key in seen_titles:
                    continue
                pub_dt = None
                if getattr(entry, "published_parsed", None):
                    try:
                        pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pass
                if pub_dt and (datetime.now(timezone.utc) - pub_dt).days > CFG.news_lookback_days:
                    continue
                seen_titles.add(key)
                source_field = entry.get("source")
                source_name = source_field.get("title") if isinstance(source_field, dict) else feed_title
                items.append({"title": title, "source": source_name, "date": pub_dt.strftime("%Y-%m-%d") if pub_dt else "unknown", "url": entry.get("link", ""), "summary": (entry.get("summary") or "")[:400], "retrieved_at": retrieved_at})
        except Exception as exc:
            logger.warning("[news] feed failed (%s): %s", url, exc)
    items.sort(key=lambda x: x["date"], reverse=True)
    return items[:CFG.max_news_items]


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = _sma(series, window)
    std = series.rolling(window).std()
    return mid, mid + num_std * std, mid - num_std * std


def _safe_last(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return round(float(val), 3)


def _pivot_points(last_bar: pd.Series) -> dict:
    h, l, c = float(last_bar["High"]), float(last_bar["Low"]), float(last_bar["Close"])
    pivot = (h + l + c) / 3
    return {"pivot": round(pivot, 2), "r1": round(2 * pivot - l, 2), "s1": round(2 * pivot - h, 2), "r2": round(pivot + h - l, 2), "s2": round(pivot - h + l, 2)}


def compute_technical_indicators(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty:
        return {"status": "unavailable", "reason": "no price history available"}
    close, high, low = hist["Close"], hist["High"], hist["Low"]
    n = len(hist)
    macd_line, signal_line, macd_hist = _macd(close)
    _, boll_upper, boll_lower = _bollinger(close)
    daily = {
        "last_close": round(float(close.iloc[-1]), 2), "last_date": str(hist.index[-1].date()),
        "sma_20": _safe_last(_sma(close, 20)), "sma_50": _safe_last(_sma(close, 50)), "sma_200": _safe_last(_sma(close, 200)),
        "ema_20": _safe_last(_ema(close, 20)), "ema_50": _safe_last(_ema(close, 50)), "ema_200": _safe_last(_ema(close, 200)),
        "rsi_14": _safe_last(_rsi(close, 14)), "macd": _safe_last(macd_line), "macd_signal": _safe_last(signal_line),
        "macd_histogram": _safe_last(macd_hist), "atr_14": _safe_last(_atr(hist, 14)),
        "bollinger_upper": _safe_last(boll_upper), "bollinger_lower": _safe_last(boll_lower),
        "support_20d": round(float(low.tail(min(20, n)).min()), 2), "resistance_20d": round(float(high.tail(min(20, n)).max()), 2),
        "support_60d": round(float(low.tail(min(60, n)).min()), 2), "resistance_60d": round(float(high.tail(min(60, n)).max()), 2),
        "pivot_points_daily": _pivot_points(hist.iloc[-1]), "avg_volume_20d": int(hist["Volume"].tail(min(20, n)).mean()),
        "latest_volume": int(hist["Volume"].iloc[-1]),
    }
    sma50, sma200 = daily["sma_50"], daily["sma_200"]
    daily["ma_alignment"] = "golden_cross (SMA50 > SMA200)" if sma50 and sma200 and sma50 > sma200 else "death_cross (SMA50 < SMA200)" if sma50 and sma200 and sma50 < sma200 else "DATA NOT AVAILABLE (insufficient history for 50/200 SMA)"
    daily["price_vs_sma20"] = "above" if daily["sma_20"] and daily["last_close"] > daily["sma_20"] else "below" if daily["sma_20"] else "DATA NOT AVAILABLE"
    weekly_result = {"status": "unavailable", "reason": "insufficient history for weekly resample"}
    try:
        weekly = hist.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        if len(weekly) >= 10:
            w_close = weekly["Close"]
            weekly_result = {"status": "ok", "last_week_close": round(float(w_close.iloc[-1]), 2), "sma_10w": _safe_last(_sma(w_close, 10)), "sma_40w": _safe_last(_sma(w_close, 40)), "rsi_14w": _safe_last(_rsi(w_close, 14))}
    except Exception as exc:
        weekly_result = {"status": "unavailable", "reason": str(exc)}
    return {"status": "ok", "daily": daily, "weekly": weekly_result, "note": "Chart patterns are qualitative interpretations, not algorithmically detected facts.", "retrieved_at": now_iso()}


class State(TypedDict):
    company: str
    ticker: str
    research_question: str
    research_plan: str
    market_data: dict
    market_analysis: str
    news: list
    news_analysis: str
    fundamentals: dict
    fundamental_analysis: str
    technical_data: dict
    technical_analysis: str
    risk_analysis: str
    verification_results: list
    critique: str
    final_report: str
    sources: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]


def _fmt(data: Any, exclude: tuple = ("history",)) -> str:
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if k in exclude:
                continue
            if isinstance(v, dict):
                lines.append(f"{k}:")
                lines.extend(f"  {sk}: {sv}" for sk, sv in v.items())
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines) if lines else "DATA NOT AVAILABLE"
    if isinstance(data, list):
        if not data:
            return "DATA NOT AVAILABLE (no items retrieved)"
        return "\n".join(f"[{i}] {item.get('title', item)} | source: {item.get('source', '?')} | date: {item.get('date', '?')} | url: {item.get('url', '?')}\n    summary: {item.get('summary', '')}" if isinstance(item, dict) else f"[{i}] {item}" for i, item in enumerate(data, 1))
    return str(data)


GROUNDING_RULE = """
STRICT GROUNDING RULE: Only use the data explicitly provided above. If a figure is missing, null, or marked DATA NOT AVAILABLE, say exactly 'DATA NOT AVAILABLE' for that item. Do not estimate, infer, or invent a number. Clearly separate reported facts, qualitative interpretation, and explicitly-labeled assumptions. Do not give a final buy/sell/hold investment recommendation.
"""


@safe_node("manager")
def manager_agent(state: State) -> dict:
    prompt = f"""You are the Manager Agent of an AI Finance Research Team.\nCompany: {state['company']} ({state['ticker']})\nResearch Question: {state['research_question']}\nCreate a short structured plan for Market, News, Fundamentals, Technical, and Risk. Do not give an investment recommendation."""
    return {"research_plan": ask_gemini(prompt, "1. Market: industry and competition.\n2. News: recent catalysts.\n3. Fundamentals: financials and valuation.\n4. Technical: trend and momentum.\n5. Risk: major material risks.")}


@safe_node("market_data")
def market_data_node(state: State) -> dict:
    try:
        data = fetch_market_data(state["ticker"])
    except Exception as exc:
        data = {"status": "unavailable", "ticker": state["ticker"], "error": str(exc), "retrieved_at": now_iso(), "source": "Yahoo Finance (yfinance)"}
    return {"market_data": data, "sources": [{"type": "market_data", "source": data.get("source"), "retrieved_at": data.get("retrieved_at"), "status": data.get("status")}]}


@safe_node("market")
def market_agent(state: State) -> dict:
    prompt = f"""You are the Market Analysis Agent.\nCompany: {state['company']} ({state['ticker']})\nPlan:\n{state['research_plan']}\nRetrieved market data:\n{_fmt(state['market_data'])}\nAnalyze industry trends, competition, supply chain, and how the price/market cap/52-week range fit the context. {GROUNDING_RULE}"""
    return {"market_analysis": ask_gemini(prompt)}


@safe_node("news_data")
def news_data_node(state: State) -> dict:
    try:
        items = fetch_news(state["ticker"], state["company"])
    except Exception:
        items = []
    return {"news": items, "sources": [{"type": "news", "title": it["title"], "source": it["source"], "date": it["date"], "url": it["url"], "retrieved_at": it.get("retrieved_at")} for it in items]}


@safe_node("news")
def news_agent(state: State) -> dict:
    prompt = f"""You are the News and Catalyst Analysis Agent.\nCompany: {state['company']} ({state['ticker']})\nRetrieved news:\n{_fmt(state['news'])}\nSummarize important developments strictly from these articles, citing [n]. Distinguish confirmed information from speculation. {GROUNDING_RULE}"""
    return {"news_analysis": ask_gemini(prompt)}


@safe_node("fundamentals_data")
def fundamentals_data_node(state: State) -> dict:
    try:
        data = fetch_fundamentals(state["ticker"])
    except Exception as exc:
        data = {"status": "unavailable", "ticker": state["ticker"], "error": str(exc), "retrieved_at": now_iso(), "source": "Yahoo Finance (yfinance)"}
    return {"fundamentals": data, "sources": [{"type": "fundamentals", "source": data.get("source"), "retrieved_at": data.get("retrieved_at"), "status": data.get("status")}]}


@safe_node("fundamentals")
def fundamental_agent(state: State) -> dict:
    prompt = f"""You are the Fundamental Analysis Agent.\nCompany: {state['company']} ({state['ticker']})\nRetrieved fundamentals:\n{_fmt(state['fundamentals'])}\nAnalyze revenue, margins, cash generation, balance sheet, capital returns, and valuation. Note missing segment data. {GROUNDING_RULE}"""
    return {"fundamental_analysis": ask_gemini(prompt)}


@safe_node("technical_data")
def technical_data_node(state: State) -> dict:
    market_data = state.get("market_data", {})
    hist = market_data.get("history") if market_data.get("status") == "ok" else None
    try:
        indicators = compute_technical_indicators(hist)
    except Exception as exc:
        indicators = {"status": "unavailable", "reason": str(exc)}
    return {"technical_data": indicators, "sources": [{"type": "technical_data", "source": "Calculated in Python (pandas) from Yahoo Finance OHLCV", "retrieved_at": indicators.get("retrieved_at", now_iso()), "status": indicators.get("status")}]}


@safe_node("technical")
def technical_agent(state: State) -> dict:
    prompt = f"""You are the Technical Analysis Agent.\nCompany: {state['company']} ({state['ticker']})\nComputed indicators:\n{_fmt(state['technical_data'])}\nInterpret daily/weekly trend, RSI/MACD, volatility, volume, support/resistance and pivots. Chart structures are qualitative interpretations. {GROUNDING_RULE}"""
    return {"technical_analysis": ask_gemini(prompt)}


@safe_node("risk")
def risk_agent(state: State) -> dict:
    prompt = f"""You are the Risk Analysis Agent for {state['company']} ({state['ticker']}).\nMarket:\n{state['market_analysis']}\nNews:\n{state['news_analysis']}\nFundamentals:\n{state['fundamental_analysis']}\nTechnical:\n{state['technical_analysis']}\nRank major risks by potential materiality. Do not provide an investment recommendation."""
    return {"risk_analysis": ask_gemini(prompt)}


@safe_node("verification")
def verification_agent(state: State) -> dict:
    results = []
    def check_freshness(name: str, data: dict, stale_after_hours: float):
        if not data or data.get("status") != "ok":
            results.append({"field": name, "status": "UNVERIFIED", "reason": data.get("error", "DATA NOT AVAILABLE") if data else "no data"})
            return
        age = hours_since(data.get("retrieved_at"))
        if age is None:
            results.append({"field": name, "status": "UNVERIFIED", "reason": "no retrieval timestamp"})
        elif age > stale_after_hours:
            results.append({"field": name, "status": "STALE DATA", "reason": f"retrieved {age:.1f}h ago (>{stale_after_hours}h threshold)"})
        else:
            results.append({"field": name, "status": "VERIFIED", "reason": f"retrieved {age:.1f}h ago, source: {data.get('source')}"})
    check_freshness("market_data", state.get("market_data", {}), CFG.stale_market_hours)
    check_freshness("fundamentals", state.get("fundamentals", {}), CFG.stale_market_hours * 7)
    td = state.get("technical_data", {})
    results.append({"field": "technical_data", "status": "VERIFIED" if td.get("status") == "ok" else "UNVERIFIED", "reason": "calculated in Python from retrieved OHLCV" if td.get("status") == "ok" else td.get("reason", "unavailable")})
    news = state.get("news", [])
    results.append({"field": "news", "status": "VERIFIED" if news else "UNVERIFIED", "reason": f"{len(news)} sourced articles retrieved via RSS" if news else "no news articles retrieved"})
    md, daily = state.get("market_data", {}), td.get("daily", {}) if td.get("status") == "ok" else {}
    if md.get("current_price") and daily.get("last_close"):
        diff = abs(md["current_price"] - daily["last_close"]) / max(md["current_price"], 1e-9)
        results.append({"field": "price_consistency", "status": "CONTRADICTED" if diff > CFG.price_consistency_tolerance else "VERIFIED", "reason": f"prices differ by {diff:.2%}"})
    if state.get("errors"):
        results.append({"field": "pipeline_errors", "status": "PARTIALLY VERIFIED", "reason": f"{len(state['errors'])} node(s) reported errors"})
    return {"verification_results": results}


@safe_node("critic")
def critic_agent(state: State) -> dict:
    prompt = f"""You are the Critic Agent. Review the following for contradictions, unsupported claims, missing information, framing problems, missed risks, and verification flags.\nMARKET: {state['market_analysis']}\nNEWS: {state['news_analysis']}\nFUNDAMENTALS: {state['fundamental_analysis']}\nTECHNICAL: {state['technical_analysis']}\nRISK: {state['risk_analysis']}\nVERIFICATION: {_fmt(state['verification_results'])}\nProvide a concise QC assessment."""
    return {"critique": ask_gemini(prompt)}


@safe_node("report")
def report_agent(state: State) -> dict:
    prompt = f"""You are the Senior Financial Research Report Agent.\nCompany: {state['company']} ({state['ticker']})\nResearch Question: {state['research_question']}\nMarket: {state['market_analysis']}\nNews: {state['news_analysis']}\nFundamentals: {state['fundamental_analysis']}\nTechnical: {state['technical_analysis']}\nRisk: {state['risk_analysis']}\nCritic: {state['critique']}\nVerification: {_fmt(state['verification_results'])}\nSources: {_fmt(state['sources'])}\nWrite professional Markdown with exactly these sections: 1. Executive Summary; 2. Company Overview; 3. Market & Competitive Landscape; 4. Recent News & Catalysts; 5. Fundamental Analysis; 6. Valuation; 7. Technical Analysis; 8. Risk Analysis; 9. Scenario Analysis with higher-growth/base/growth-deceleration scenarios and every number labeled MODEL ASSUMPTION; 10. Key Bullish and Bearish Factors; 11. Key Uncertainties & Data-Quality Notes; 12. Sources. Numeric tables need source/date columns. {GROUNDING_RULE}"""
    return {"final_report": ask_gemini(prompt)}


def build_graph():
    workflow = StateGraph(State)
    for name, node in [("manager", manager_agent), ("market_data", market_data_node), ("market", market_agent), ("news_data", news_data_node), ("news", news_agent), ("fundamentals_data", fundamentals_data_node), ("fundamentals", fundamental_agent), ("technical_data", technical_data_node), ("technical", technical_agent), ("risk", risk_agent), ("verification", verification_agent), ("critic", critic_agent), ("report", report_agent)]:
        workflow.add_node(name, node)
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", "market_data")
    workflow.add_edge("manager", "news_data")
    workflow.add_edge("manager", "fundamentals_data")
    workflow.add_edge("market_data", "market")
    workflow.add_edge("news_data", "news")
    workflow.add_edge("fundamentals_data", "fundamentals")
    workflow.add_edge("market", "technical_data")
    workflow.add_edge("news", "technical_data")
    workflow.add_edge("fundamentals", "technical_data")
    workflow.add_edge("technical_data", "technical")
    workflow.add_edge("technical", "risk")
    workflow.add_edge("risk", "verification")
    workflow.add_edge("verification", "critic")
    workflow.add_edge("critic", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


app = build_graph()


def show_graph():
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception as exc:
        logger.info("Graph diagram skipped: %s", exc)


def export_report(markdown_text: str, filename_prefix: str = "Finance_Research_Report"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = f"{filename_prefix}_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    pdf_path = None
    try:
        import markdown2
        from xhtml2pdf import pisa
        html = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])
        pdf_path = f"{filename_prefix}_{ts}.pdf"
        with open(pdf_path, "wb") as f:
            result = pisa.CreatePDF(html, dest=f)
        if result.err:
            pdf_path = None
    except Exception:
        pass
    return md_path, pdf_path


def run_research(company_name: str, ticker: str, research_question: str) -> dict:
    initial_state: State = {"company": company_name, "ticker": ticker, "research_question": research_question, "research_plan": "", "market_data": {}, "market_analysis": "", "news": [], "news_analysis": "", "fundamentals": {}, "fundamental_analysis": "", "technical_data": {}, "technical_analysis": "", "risk_analysis": "", "verification_results": [], "critique": "", "final_report": "", "sources": [], "errors": []}
    final_state = dict(initial_state)
    for update in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            for key, value in node_output.items():
                final_state[key] = final_state.get(key, []) + list(value) if key in ("sources", "errors") else value
            print(f"{'[x]' if node_output.get('errors') else '[v]'} {NODE_LABELS.get(node_name, node_name)}")
    print("=" * 70)
    print("FINAL FINANCIAL RESEARCH REPORT")
    print("=" * 70)
    print(final_state["final_report"])
    md_path, pdf_path = export_report(final_state["final_report"], filename_prefix=f"{ticker}_Research_Report")
    print(f"Markdown report: {md_path}")
    print(f"PDF report: {pdf_path or '(skipped)'}")
    return final_state


RUN_EXAMPLE = True
if RUN_EXAMPLE:
    show_graph()
    result = run_research("NVIDIA Corporation", "NVDA", "Conduct a comprehensive financial analysis of NVIDIA. Evaluate its market position, news and catalysts, financial fundamentals, technical setup, and major risks.")
