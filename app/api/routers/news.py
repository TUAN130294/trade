from fastapi import APIRouter, HTTPException, WebSocket, Query
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
from app.core import state
from pathlib import Path
from datetime import datetime
import pandas as pd
from quantum_stock.services.interpretation_service import get_interpretation_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize interpretation service at module level
interp_service = get_interpretation_service()

class QueryRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    symbol: str = "MWG"

# NEWS INTELLIGENCE ENDPOINTS
# ============================================================

# In-memory news cache
_news_cache = {
    "alerts": [],
    "watchlist": ["MWG", "HPG", "FPT", "VNM", "VIC"],
    "last_scan": None
}

@router.get("/api/news/status")
async def get_news_status():
    """Get news scanner status"""
    # Check if orchestrator's news scanner is actually running
    scanner_running = bool(state.orchestrator and hasattr(state.orchestrator, 'news_scanner') and state.orchestrator.news_scanner)
    return {
        "is_running": scanner_running,
        "sources": ["VietStock", "CafeF", "VnExpress", "Fireant"],
        "last_scan": _news_cache.get("last_scan"),
        "total_alerts": len(_news_cache.get("alerts", [])),
        "scan_interval": "60s"
    }


@router.get("/api/news/alerts")
async def get_news_alerts(interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get news alerts list"""
    # Try to get real news from scanner
    alerts = _news_cache.get("alerts", [])

    # If no alerts, generate some based on RSS
    if not alerts:
        try:
            from quantum_stock.news.rss_news_fetcher import VNStockNewsFetcher
            fetcher = VNStockNewsFetcher()
            news_items = fetcher.fetch_all_feeds()

            for item in news_items[:10]:
                alerts.append({
                    "title": item.get("headline", "")[:100],
                    "summary": item.get("summary", "")[:200],
                    "source": item.get("source", "VietStock"),
                    "priority": item.get("priority", "MEDIUM"),
                    "symbols": item.get("related_symbols", []),
                    "published_at": item.get("timestamp", datetime.now().isoformat()),
                    "sentiment": item.get("news_sentiment", 0.5)
                })
            _news_cache["alerts"] = alerts
        except Exception as e:
            logger.warning(f"RSS fetch error: {e}")

    result = {
        "alerts": alerts[:20],
        "total": len(alerts),
        "timestamp": datetime.now().isoformat()
    }

    # Add interpretation if requested
    if interpret:
        # Get high priority alerts
        high_priority_count = sum(1 for a in alerts if a.get("priority") == "HIGH")
        result["interpretation"] = await interp_service.interpret(
            "news_alerts",
            {
                "count": len(alerts),
                "high_priority": high_priority_count
            }
        )

    return result


@router.get("/api/news/market-mood")
async def get_market_mood(interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get overall market sentiment/mood"""
    try:
        alerts = _news_cache.get("alerts", [])

        # Calculate mood based on alerts
        if alerts:
            positive = sum(1 for a in alerts if a.get("sentiment", 0.5) > 0.55)
            negative = sum(1 for a in alerts if a.get("sentiment", 0.5) < 0.45)

            if positive > negative * 1.5:
                mood = "bullish"
            elif positive > negative:
                mood = "slightly_bullish"
            elif negative > positive * 1.5:
                mood = "bearish"
            elif negative > positive:
                mood = "slightly_bearish"
            else:
                mood = "neutral"
        else:
            mood = "neutral"

        result = {
            "current_mood": mood,
            "confidence": 0.72,
            "positive_news": len([a for a in alerts if a.get("sentiment", 0.5) > 0.55]),
            "negative_news": len([a for a in alerts if a.get("sentiment", 0.5) < 0.45]),
            "neutral_news": len([a for a in alerts if 0.45 <= a.get("sentiment", 0.5) <= 0.55]),
            "timestamp": datetime.now().isoformat()
        }

        # Add interpretation if requested
        if interpret:
            result["interpretation"] = await interp_service.interpret(
                "news_mood",
                {
                    "mood": mood,
                    "positive_news": result["positive_news"],
                    "negative_news": result["negative_news"]
                }
            )

        return result
    except Exception as e:
        logger.error(f"Market mood error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/news/watchlist")
async def get_news_watchlist():
    """Get news watchlist"""
    return {
        "watchlist": _news_cache.get("watchlist", []),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/api/news/watchlist")
async def update_news_watchlist(watchlist: List[str]):
    """Update news watchlist"""
    _news_cache["watchlist"] = [s.upper() for s in watchlist]
    return {
        "success": True,
        "watchlist": _news_cache["watchlist"],
        "timestamp": datetime.now().isoformat()
    }


@router.post("/api/news/scan")
async def trigger_news_scan():
    """Trigger manual news scan"""
    try:
        alerts = []

        # Fetch from RSS
        try:
            from quantum_stock.news.rss_news_fetcher import VNStockNewsFetcher
            fetcher = VNStockNewsFetcher()
            news_items = fetcher.fetch_all_feeds()

            for item in news_items[:15]:
                # Use correct field names from VNStockNewsFetcher
                headline = item.get("headline", "")
                summary = item.get("summary", headline)
                source = item.get("source", "VietStock")
                priority = item.get("priority", "MEDIUM")
                symbols = item.get("related_symbols", [])
                timestamp = item.get("timestamp", datetime.now().isoformat())
                sentiment = item.get("news_sentiment", 0.5)

                alerts.append({
                    "title": headline[:100],
                    "summary": summary[:200],
                    "source": source,
                    "priority": priority,
                    "symbols": symbols,
                    "published_at": timestamp,
                    "sentiment": sentiment
                })
        except Exception as e:
            logger.warning(f"RSS scan error: {e}")

        _news_cache["alerts"] = alerts
        _news_cache["last_scan"] = datetime.now().isoformat()

        return {
            "success": True,
            "alerts": alerts[:20],
            "total_found": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"News scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/backtest/run")
async def run_backtest(request: Dict[str, Any], interpret: bool = Query(False, description="Add LLM interpretation")):
    """Run backtest for a given strategy"""
    try:
        strategy_name = request.get('strategy', 'momentum')
        symbol = request.get('symbol', 'MWG')
        days = request.get('days', 365)

        # Import backtest engine
        from quantum_stock.core.backtest_engine import (
            BacktestEngine, MACrossoverStrategy, RSIReversalStrategy,
            MACDStrategy, BollingerBreakoutStrategy
        )

        # Load historical data
        df = None
        data_source = "unknown"

        # Try CafeF first
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()
            historical = connector.get_stock_historical(symbol.upper(), days=days)
            if historical and len(historical) >= 50:
                df = pd.DataFrame(historical)
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
                df.set_index('date', inplace=True)
                data_source = "CafeF"
        except Exception as e:
            logger.warning(f"CafeF fetch for backtest: {e}")

        # Fallback to parquet
        if df is None:
            parquet_path = Path(f"data/historical/{symbol.upper()}.parquet")
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                data_source = "Parquet"

        if df is None or len(df) < 50:
            return {
                "success": False,
                "error": f"Not enough data for {symbol}",
                "data_source": data_source
            }

        # Select strategy
        strategy_map = {
            'momentum': MACrossoverStrategy(fast_period=10, slow_period=30),
            'mean_reversion': RSIReversalStrategy(period=14, oversold=30, overbought=70),
            'macd': MACDStrategy(fast=12, slow=26, signal=9),
            'bollinger': BollingerBreakoutStrategy(period=20, std_dev=2.0),
        }

        strategy = strategy_map.get(strategy_name.lower(), strategy_map['momentum'])

        # Run backtest
        engine = BacktestEngine(
            initial_capital=100_000_000,  # 100M VND
            commission_rate=0.0015,  # 0.15%
            slippage=0.001  # 0.1%
        )

        result = engine.run(
            df.tail(days),
            strategy,
            symbol=symbol.upper(),
            position_size=0.8,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )

        response = {
            "success": True,
            "strategy": strategy_name,
            "symbol": symbol.upper(),
            "data_source": data_source,
            "days_tested": len(df),
            "results": {
                "total_return_pct": round(result.total_return_pct, 2),
                "annualized_return": round(result.annualized_return * 100, 2),
                "max_drawdown_pct": round(result.max_drawdown_pct, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 2),
                "sortino_ratio": round(result.sortino_ratio, 2),
                "win_rate": round(result.win_rate, 1),  # Already percentage
                "profit_factor": round(result.profit_factor, 2),
                "total_trades": result.total_trades,
                "winning_trades": result.winning_trades,
                "losing_trades": result.losing_trades,
                "avg_win": round(result.avg_win, 0),
                "avg_loss": round(result.avg_loss, 0),
                "expectancy": round(result.expectancy, 2),
                "avg_holding_days": round(result.avg_holding_days, 1)
            },
            "timestamp": datetime.now().isoformat()
        }

        # Add interpretation if requested
        if interpret:
            response["interpretation"] = await interp_service.interpret(
                "backtest_result",
                {
                    "strategy": strategy_name,
                    "return_pct": round(result.total_return_pct, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 2),
                    "win_rate": round(result.win_rate, 1)
                }
            )

        return response

    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def is_port_in_use(port: int) -> bool:
    """Check if port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True


