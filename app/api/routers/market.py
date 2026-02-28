from fastapi import APIRouter, HTTPException, WebSocket, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from app.core import state
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from app.api.routers.trading import is_market_open
from quantum_stock.services.interpretation_service import get_interpretation_service

# Try to import ConversationalQuant (optional)
try:
    from quantum_stock.agents.conversational_quant import ConversationalQuant
except ImportError:
    ConversationalQuant = None

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize interpretation service at module level
interp_service = get_interpretation_service()

class QueryRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    symbol: str = "MWG"

# API Routes - Market Data & Analysis
# =====================

@router.get("/api/market/status")
async def get_market_status(interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get market status - VN-Index real-time data"""
    try:
        now = datetime.now()

        # Use unified market hours check
        is_open, session_info = is_market_open()

        # Default values — flagged as stale when using fallback
        vnindex = 1249.05
        change = 0.0
        change_pct = 0.0
        data_source = "hardcoded_fallback"

        # PRIORITY 1: Banggia API for intraday realtime price
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            banggia_url = "https://banggia.cafef.vn/stockhandler.ashx?index=true"
            resp = requests.get(banggia_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                items = resp.json()
                for item in items:
                    name = str(item.get('name', item.get('a', ''))).upper()
                    if name == 'VNINDEX':
                        # Index API: fields are index/change/percent (strings with commas)
                        idx_str = str(item.get('index', item.get('l', '0'))).replace(',', '')
                        vnindex = round(float(idx_str), 2)
                        chg_str = str(item.get('change', item.get('k', '0'))).replace(',', '')
                        change = round(float(chg_str), 2)
                        pct_str = str(item.get('percent', '0')).replace(',', '')
                        change_pct = round(float(pct_str), 2)
                        data_source = "CafeF Banggia (realtime)"
                        break
        except Exception as e:
            logger.warning(f"Banggia API Error: {e}")

        # PRIORITY 2: PriceHistory API (closing prices, 1 day delayed)
        if data_source == "hardcoded_fallback":
            try:
                import requests
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                url = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol=VNINDEX&StartDate=&EndDate=&PageIndex=1&PageSize=2"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('Success') and data.get('Data', {}).get('Data'):
                        items = data['Data']['Data']
                        if len(items) > 0:
                            latest = items[0]
                            vnindex = round(float(latest.get('GiaDongCua', 0)), 2)
                            data_source = "CafeF PriceHistory (closing)"
                            change_str = latest.get('ThayDoi', '0(0%)')
                            try:
                                parts = change_str.replace('(', ' ').replace('%)', '').split()
                                change = round(float(parts[0]), 2)
                                change_pct = round(float(parts[1]) if len(parts) > 1 else 0, 2)
                            except:
                                if len(items) > 1:
                                    prev_close = float(items[1].get('GiaDongCua', vnindex))
                                    change = round(vnindex - prev_close, 2)
                                    change_pct = round((change / prev_close) * 100, 2) if prev_close > 0 else 0
            except Exception as e:
                logger.warning(f"PriceHistory API Error: {e}")

        # PRIORITY 3: Parquet cache
        if data_source == "hardcoded_fallback":
            try:
                parquet_path = Path("data/historical/VNINDEX.parquet")
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    vnindex = round(float(df.iloc[-1]['close']), 2)
                    data_source = "Parquet (cached)"
                    if len(df) > 1:
                        prev_close = float(df.iloc[-2]['close'])
                        change = round(vnindex - prev_close, 2)
                        change_pct = round((change / prev_close) * 100, 2)
            except:
                pass

        result = {
            "is_open": is_open,
            "session_info": session_info,
            "current_time": now.isoformat(),
            "trading_hours": {
                "morning": "09:00 - 11:30",
                "afternoon": "13:00 - 14:45"
            },
            "vnindex": vnindex,
            "change": change,
            "change_pct": change_pct,
            "source": data_source
        }

        # Add interpretation if requested
        if interpret:
            result["interpretation"] = await interp_service.interpret(
                "market_status",
                {
                    "status": "mở cửa" if is_open else "đóng cửa",
                    "vnindex": vnindex,
                    "change": change,
                    "change_pct": change_pct
                }
            )

        return result
    except Exception as e:
        logger.error(f"Market status error: {e}")
        return {
            "error": str(e),
            "vnindex": 1249.05,
            "change": 0.0,
            "change_pct": 0.0
        }


@router.get("/api/market/regime")
async def get_market_regime(interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get current market regime detection"""
    try:
        # Try to load real VN-Index data
        vnindex_data = []
        try:
            parquet_path = Path("data/historical/VNINDEX.parquet")
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                vnindex_data = df.tail(200).copy()
        except:
            pass

        # No real data available — return unknown instead of fake data
        if len(vnindex_data) < 50:
            result = {
                "market_regime": "UNKNOWN",
                "volatility_regime": "UNKNOWN",
                "liquidity_regime": "UNKNOWN",
                "confidence": 0.0,
                "recommended_strategies": [],
                "risk_adjustment": 0.5,
                "summary": "Không có đủ dữ liệu VNINDEX để phân tích regime",
                "data_source": "unavailable",
                "error": "No VNINDEX data available (need 50+ days)"
            }
        else:
            # Real regime detection on real data
            close = vnindex_data['close']
            sma20 = close.rolling(20).mean()
            current = close.iloc[-1]

            if current > sma20.iloc[-1]:
                market_regime = "UPTREND"
                confidence = 0.8
            elif current < sma20.iloc[-1]:
                market_regime = "DOWNTREND"
                confidence = 0.75
            else:
                market_regime = "SIDEWAYS"
                confidence = 0.6

            result = {
                "market_regime": market_regime,
                "volatility_regime": "NORMAL",
                "liquidity_regime": "HIGH",
                "confidence": confidence,
                "recommended_strategies": ["TREND_FOLLOWING"] if market_regime == "UPTREND" else ["MEAN_REVERSION"] if market_regime == "DOWNTREND" else ["SCALPING"],
                "risk_adjustment": 1.0 if market_regime == "UPTREND" else 0.7 if market_regime == "DOWNTREND" else 0.85,
                "summary": f"Thị trường đang trong trạng thái {market_regime}",
                "data_source": "real"
            }

        # Add interpretation if requested (only when real data available)
        if interpret and result.get("data_source") != "unavailable":
            result["interpretation"] = await interp_service.interpret(
                "market_regime",
                {
                    "regime": result["market_regime"],
                    "confidence": result["confidence"],
                    "volatility": result["volatility_regime"],
                    "liquidity": result["liquidity_regime"]
                }
            )

        return result
    except Exception as e:
        logger.error(f"Market regime error: {e}")
        return {
            "market_regime": "NEUTRAL",
            "volatility_regime": "NORMAL",
            "confidence": 0.5,
            "error": str(e)
        }


@router.get("/api/market/smart-signals")
async def get_smart_signals(interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get comprehensive market signals: breadth, foreign flow, smart money, circuit breakers"""
    signals = []
    signal_summary = {}

    try:
        from quantum_stock.dataconnector.realtime_market import get_realtime_connector
        from quantum_stock.dataconnector.vps_market import get_vps_connector
        connector = get_realtime_connector()
        vps_connector = get_vps_connector()

        # 1️⃣ Market Breadth (Độ rộng thị trường)
        breadth = connector.get_market_breadth()
        if breadth:
            breadth_signal = breadth.get('breadth_signal', 'NEUTRAL')
            severity = "HIGH" if breadth_signal == "POSITIVE" else "WARNING" if breadth_signal == "NEGATIVE" else "INFO"

            signals.append({
                "type": "BREADTH",
                "emoji": "📊",
                "name": f"📊 Market Breadth: {breadth_signal}",
                "description": f"{breadth.get('advancing', 0)} mã tăng vs {breadth.get('declining', 0)} mã giảm | A/D ratio: {breadth.get('advance_decline_ratio', 0):.2f}",
                "severity": severity,
                "source": "CafeF Real-time",
                "details": {
                    "advancing": breadth.get('advancing', 0),
                    "declining": breadth.get('declining', 0),
                    "ceiling_hits": breadth.get('ceiling_hits', 0),
                    "floor_hits": breadth.get('floor_hits', 0),
                    "ratio": breadth.get('advance_decline_ratio', 0)
                },
                "timestamp": datetime.now().isoformat()
            })

            # Bull Trap Detection
            if breadth.get('is_bull_trap'):
                signals.append({
                    "type": "BULL_TRAP",
                    "emoji": "⚠️",
                    "name": "⚠️ CẢNH BÁO: Bull Trap Detection",
                    "description": breadth.get('bull_trap_reason', 'VN-Index tăng nhưng số mã giảm nhiều hơn'),
                    "severity": "HIGH",
                    "source": "CafeF Analysis",
                    "action": "CAUTION - Có thể là cảnh báo đảo chiều",
                    "timestamp": datetime.now().isoformat()
                })

            signal_summary['breadth'] = breadth_signal

        # 2️⃣ Foreign Capital Flow (Khối ngoại) - Use VPS API for real data
        # Get all stock symbols for VPS foreign flow query
        try:
            all_stocks_data = connector._get_cached_or_fetch()
            symbols = [s.get('a', '') for s in all_stocks_data if s.get('a')][:100]  # Limit to 100 symbols
        except Exception as e:
            logger.warning(f"Failed to get symbols for foreign flow: {e}")
            symbols = ['VNM', 'VCB', 'HPG', 'FPT', 'SSI', 'MWG', 'VHM', 'VIC', 'MSN', 'VPB']  # Default top stocks

        # Use VPS connector (async) instead of CafeF for accurate foreign flow data
        foreign = await vps_connector.get_foreign_flow(symbols)
        if foreign:
            flow_type = foreign.get('flow_type', 'NEUTRAL')
            severity = "HIGH" if flow_type == "BUY" else "WARNING" if flow_type == "SELL" else "INFO"
            emoji = "💰" if flow_type == "BUY" else "📉" if flow_type == "SELL" else "➡️"

            net_value_bn = foreign.get('net_value_billion', 0)
            stocks_flow = foreign.get('stocks', [])

            # Extract top buyers and sellers from VPS data
            top_buyers = [s for s in stocks_flow if s.get('net_value', 0) > 0][:3]
            top_sellers = [s for s in stocks_flow if s.get('net_value', 0) < 0][:3]

            # For buyers: show buy volume, for sellers: show sell volume
            top_buy_str = ", ".join([f"{s['symbol']}({int(s.get('foreign_buy_volume', 0))//1000}k)" for s in top_buyers])
            top_sell_str = ", ".join([f"{s['symbol']}({int(s.get('foreign_sell_volume', 0))//1000}k)" for s in top_sellers])

            description = f"Khối ngoại {'mua ròng' if flow_type == 'BUY' else 'bán ròng'} {abs(net_value_bn):.1f} tỷ"
            if top_buy_str:
                description += f"\nMua: {top_buy_str}"
            if top_sell_str:
                description += f"\nBán: {top_sell_str}"

            signals.append({
                "type": "FOREIGN_FLOW",
                "emoji": emoji,
                "name": f"{emoji} Khối Ngoại: {flow_type}",
                "description": description,
                "severity": severity,
                "source": "VPS Real-time",
                "details": {
                    "flow_type": flow_type,
                    "net_value_billion": net_value_bn,
                    "top_buy_symbols": [s['symbol'] for s in top_buyers],
                    "top_sell_symbols": [s['symbol'] for s in top_sellers]
                },
                "action": f"{'FOLLOW - Khối ngoại accumulating' if flow_type == 'BUY' else 'CAUTION - Khối ngoại distributing' if flow_type == 'SELL' else 'NEUTRAL'}",
                "timestamp": datetime.now().isoformat()
            })

            signal_summary['foreign_flow'] = flow_type

        # 3️⃣ Smart Money / Volume Anomalies (Lái gom / Phân phối)
        volume = connector.get_volume_anomalies()
        if volume:
            smart_signal = volume.get('smart_money_signal', 'Không phát hiện')
            smart_stocks = volume.get('smart_money_stocks', [])

            # Determine severity
            if "CLIMAX BUYING" in smart_signal:
                severity = "HIGH"
                emoji = "🚀"
                signal_type = "CLIMAX_BUY"
            elif "CLIMAX SELLING" in smart_signal:
                severity = "WARNING"
                emoji = "📉"
                signal_type = "CLIMAX_SELL"
            elif "CHURNING" in smart_signal:
                severity = "INFO"
                emoji = "⚙️"
                signal_type = "CHURNING"
            else:
                severity = "INFO"
                emoji = "➡️"
                signal_type = "NEUTRAL"

            if smart_signal != "Không phát hiện":
                description = smart_signal
                if smart_stocks:
                    description += f"\nSymbols: {', '.join(smart_stocks)}"

                signals.append({
                    "type": "SMART_MONEY",
                    "emoji": emoji,
                    "name": f"{emoji} Smart Money: {signal_type}",
                    "description": description,
                    "severity": severity,
                    "source": "CafeF Volume Analysis",
                    "details": {
                        "high_volume_gainers_count": len(volume.get('high_volume_gainers', [])),
                        "high_volume_losers_count": len(volume.get('high_volume_losers', [])),
                        "churning_count": len(volume.get('churning', []))
                    },
                    "action": "MONITOR" if severity == "INFO" else "FOLLOW" if severity == "HIGH" else "CAUTION",
                    "timestamp": datetime.now().isoformat()
                })

                signal_summary['smart_money'] = signal_type

        # 4️⃣ Circuit Breaker Alerts (Cảnh báo trần/sàn)
        if breadth and (breadth.get('ceiling_hits', 0) > 10 or breadth.get('floor_hits', 0) > 10):
            if breadth.get('ceiling_hits', 0) > 10:
                signals.append({
                    "type": "CIRCUIT_BREAKER",
                    "emoji": "🔴",
                    "name": "🔴 Circuit Breaker: Nhiều mã tăng trần",
                    "description": f"{breadth.get('ceiling_hits', 0)} mã tăng trần - Có thể bị circuit breaker",
                    "severity": "WARNING",
                    "source": "System Monitor",
                    "action": "CAUTION - Market may hit upper circuit breaker",
                    "timestamp": datetime.now().isoformat()
                })

            if breadth.get('floor_hits', 0) > 10:
                signals.append({
                    "type": "CIRCUIT_BREAKER",
                    "emoji": "🔵",
                    "name": "🔵 Circuit Breaker: Nhiều mã giảm sàn",
                    "description": f"{breadth.get('floor_hits', 0)} mã giảm sàn - Có thể bị circuit breaker",
                    "severity": "WARNING",
                    "source": "System Monitor",
                    "action": "CAUTION - Market may hit lower circuit breaker",
                    "timestamp": datetime.now().isoformat()
                })

    except Exception as e:
        logger.warning(f"Real-time connector error: {e}")

    # Fallback: return generic signals if no data
    if not signals:
        signals = [
            {
                "type": "INFO",
                "emoji": "📊",
                "name": "📊 Đang tải tín hiệu thị trường",
                "description": "Kết nối dữ liệu real-time từ CafeF",
                "severity": "INFO",
                "source": "System",
                "timestamp": datetime.now().isoformat()
            }
        ]

    # Generate quick summary for user
    quick_summary = generate_quick_summary(signal_summary)

    result = {
        "signals": signals,
        "count": len(signals),
        "signal_summary": signal_summary,
        "quick_summary": quick_summary,  # New: Simple 1-line summary
        "interpretation": generate_signal_interpretation(signal_summary),
        "source": "Real-time CafeF API",
        "timestamp": datetime.now().isoformat()
    }

    # Add LLM interpretation if requested
    if interpret:
        result["llm_interpretation"] = await interp_service.interpret(
            "smart_signals",
            {
                "count": len(signals),
                "breadth": signal_summary.get('breadth', 'NEUTRAL'),
                "foreign_flow": signal_summary.get('foreign_flow', 'NEUTRAL'),
                "smart_money": signal_summary.get('smart_money', 'NEUTRAL')
            }
        )

    return result


def generate_quick_summary(summary: Dict[str, str]) -> Dict[str, Any]:
    """Generate a simple, user-friendly market summary"""

    # Determine overall market sentiment
    breadth = summary.get('breadth', 'NEUTRAL')
    foreign = summary.get('foreign_flow', 'NEUTRAL')
    smart = summary.get('smart_money', 'NEUTRAL')

    # Score: +1 for bullish, -1 for bearish
    score = 0
    if breadth == 'POSITIVE': score += 1
    elif breadth == 'NEGATIVE': score -= 1

    if foreign == 'BUY': score += 1
    elif foreign == 'SELL': score -= 1

    if smart in ['CLIMAX_BUY']: score += 1
    elif smart in ['CLIMAX_SELL']: score -= 1

    # Determine verdict
    if score >= 2:
        verdict = "🟢 THỊ TRƯỜNG TÍCH CỰC"
        action = "CÓ THỂ MUA"
        color = "green"
    elif score <= -2:
        verdict = "🔴 THỊ TRƯỜNG TIÊU CỰC"
        action = "NÊN BÁN / CHỜ"
        color = "red"
    elif score == 1:
        verdict = "🟡 THỊ TRƯỜNG KHẢ QUAN"
        action = "CÂN NHẮC MUA"
        color = "yellow"
    elif score == -1:
        verdict = "🟠 THỊ TRƯỜNG THẬN TRỌNG"
        action = "CÂN NHẮC BÁN"
        color = "orange"
    else:
        verdict = "⚪ THỊ TRƯỜNG SIDEWAY"
        action = "CHỜ TÍN HIỆU RÕ"
        color = "gray"

    # Build simple explanation
    explanation_parts = []
    if breadth == 'POSITIVE':
        explanation_parts.append("Nhiều mã tăng")
    elif breadth == 'NEGATIVE':
        explanation_parts.append("Nhiều mã giảm")

    if foreign == 'BUY':
        explanation_parts.append("Khối ngoại mua")
    elif foreign == 'SELL':
        explanation_parts.append("Khối ngoại bán")

    if smart == 'CLIMAX_BUY':
        explanation_parts.append("Tiền vào mạnh")
    elif smart == 'CLIMAX_SELL':
        explanation_parts.append("Tiền thoát mạnh")

    explanation = " + ".join(explanation_parts) if explanation_parts else "Chưa có tín hiệu"

    return {
        "verdict": verdict,
        "action": action,
        "color": color,
        "score": score,
        "explanation": explanation,
        "one_liner": f"{verdict} → {action} ({explanation})"
    }


def generate_signal_interpretation(summary: Dict[str, str]) -> str:
    """Generate human-readable interpretation of market signals"""
    interpretations = []

    if summary.get('breadth') == 'POSITIVE':
        interpretations.append("📈 Thị trường tích cực với nhiều cổ phiếu tăng")
    elif summary.get('breadth') == 'NEGATIVE':
        interpretations.append("📉 Thị trường yếu với nhiều cổ phiếu giảm")
    else:
        interpretations.append("➡️ Thị trường sideway, không rõ hướng")

    if summary.get('foreign_flow') == 'BUY':
        interpretations.append("💰 Khối ngoại đang accumulate, tín hiệu tích cực")
    elif summary.get('foreign_flow') == 'SELL':
        interpretations.append("📉 Khối ngoại đang distribute, cần cảnh báo")

    if summary.get('smart_money') == 'CLIMAX_BUY':
        interpretations.append("🚀 Smart money đang mua mạnh, có cơ hội breakout")
    elif summary.get('smart_money') == 'CLIMAX_SELL':
        interpretations.append("⚠️ Smart money đang bán mạnh, cảnh báo đảo chiều")
    elif summary.get('smart_money') == 'CHURNING':
        interpretations.append("⚙️ Volume cao nhưng giá sideway, có thể phân phối")

    return " | ".join(interpretations) if interpretations else "Chưa có tín hiệu rõ ràng"


@router.get("/api/agents/status")
async def get_agents_status():
    """Get status of all agents for Radar display"""
    try:
        from quantum_stock.core.realtime_signals import get_radar_agent_status
        agents = get_radar_agent_status()
    except:
        agents = None

    # Fallback to default agents if not available
    if not agents:
        agents = [
            {
                "name": "Scout",
                "emoji": "🔭",
                "role": "Market Scanner",
                "description": "Quét thị trường 24/7, phát hiện cơ hội đầu tư",
                "status": "online",
                "accuracy": 0.85,
                "signals_today": 0,
                "last_signal": "Đang quét thị trường..." if datetime.now().hour >= 9 and datetime.now().hour < 15 else "Chờ phiên giao dịch",
                "specialty": "Pattern Recognition"
            },
            {
                "name": "Alex",
                "emoji": "📊",
                "role": "Technical Analyst",
                "description": "Phân tích kỹ thuật: RSI, MACD, Bollinger, Support/Resistance",
                "status": "online",
                "accuracy": 0.82,
                "signals_today": 0,
                "last_signal": "Sẵn sàng phân tích",
                "specialty": "Technical Indicators"
            },
            {
                "name": "Bull",
                "emoji": "🐂",
                "role": "Growth Hunter",
                "description": "Tìm kiếm cổ phiếu tăng trưởng, breakout",
                "status": "online",
                "accuracy": 0.78,
                "signals_today": 0,
                "last_signal": "Đang tìm cơ hội breakout",
                "specialty": "Momentum Trading"
            },
            {
                "name": "Bear",
                "emoji": "🐻",
                "role": "Risk Sentinel",
                "description": "Phát hiện rủi ro, cảnh báo đảo chiều",
                "status": "online",
                "accuracy": 0.91,
                "signals_today": 0,
                "last_signal": "Đang giám sát rủi ro",
                "specialty": "Risk Detection"
            },
            {
                "name": "Risk Doctor",
                "emoji": "🏥",
                "role": "Position Sizer",
                "description": "Tính toán khối lượng giao dịch an toàn, quản lý vốn",
                "status": "online",
                "accuracy": 0.95,
                "signals_today": 0,
                "last_signal": "Sẵn sàng tính position size",
                "specialty": "Money Management"
            },
            {
                "name": "Chief",
                "emoji": "⚖️",
                "role": "Decision Maker",
                "description": "Tổng hợp ý kiến, ra quyết định cuối cùng",
                "status": "online",
                "accuracy": 0.85,
                "signals_today": 0,
                "last_signal": "Chờ tín hiệu từ team",
                "specialty": "Consensus Building"
            }
        ]

    return {
        "agents": agents,
        "total_agents": len(agents),
        "online_count": sum(1 for a in agents if a.get('status') == 'online'),
        "avg_accuracy": sum(a.get('accuracy', 0.8) for a in agents) / len(agents) if agents else 0,
        "timestamp": datetime.now().isoformat(),
        "data_source": "static_config"
    }


@router.get("/api/analysis/technical/{symbol}")
async def get_technical_analysis(symbol: str, interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get technical analysis for a symbol"""
    try:
        symbol = symbol.upper()
        df = None
        data_source = "unknown"

        # PRIORITY 1: Fetch REAL data from CafeF API
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()
            historical = connector.get_stock_historical(symbol, days=100)

            if historical and len(historical) >= 20:
                df = pd.DataFrame(historical)
                data_source = "CafeF Real-time"
                logger.info(f"✅ Technical analysis using REAL CafeF data for {symbol}")
        except Exception as e:
            logger.warning(f"CafeF fetch error for {symbol}: {e}")

        # FALLBACK 2: Try parquet file
        if df is None:
            try:
                parquet_path = Path(f"data/historical/{symbol}.parquet")
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    data_source = "Parquet (cached)"
            except Exception as e:
                logger.warning(f"Parquet read error: {e}")

        # FALLBACK 3: Return error instead of synthetic data
        if df is None:
            return {
                "symbol": symbol,
                "error": f"No historical data available for {symbol}. Download data first.",
                "data_source": "unavailable",
                "hint": "Run: python download_all_stocks.py"
            }

        # Calculate support/resistance
        recent = df.tail(50)
        support_levels = [float(recent['low'].min()), float(recent['low'].quantile(0.25)), float(recent['low'].quantile(0.5))]
        resistance_levels = [float(recent['high'].max()), float(recent['high'].quantile(0.75)), float(recent['high'].quantile(0.95))]
        current_price = float(df['close'].iloc[-1])

        # RSI calculation
        close = df['close'].astype(float)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        # Bottom evaluation
        is_potential_bottom = current_rsi < 35 and current_price < sorted(support_levels)[1]
        bottom_score = 30
        bottom_reasons = []
        if current_rsi < 35:
            bottom_score += 25
            bottom_reasons.append(f"RSI thấp ({current_rsi:.0f}) - Vùng quá bán")
        if current_price < sorted(support_levels)[1]:
            bottom_score += 20
            bottom_reasons.append("Giá gần vùng hỗ trợ mạnh")
        if float(df['volume'].iloc[-1]) > float(df['volume'].mean()) * 1.5:
            bottom_score += 15
            bottom_reasons.append("Volume tăng đột biến")

        # Historical highs/lows
        all_highs = df['high'].astype(float)
        all_lows = df['low'].astype(float)
        historical_high = round(float(all_highs.max()), 0)
        historical_low = round(float(all_lows.min()), 0)

        result = {
            "symbol": symbol,
            "current_price": round(current_price, 0),
            "support_levels": [round(x, 0) for x in sorted(support_levels)],
            "resistance_levels": [round(x, 0) for x in sorted(resistance_levels)],
            "rsi": round(current_rsi, 1),
            "patterns": ["Sideway accumulation"] if 40 < current_rsi < 60 else ["Oversold bounce"] if current_rsi < 30 else ["Overbought warning"] if current_rsi > 70 else [],
            "bottom_evaluation": {
                "score": min(95, bottom_score),
                "is_potential_bottom": is_potential_bottom,
                "reasons": bottom_reasons,
                "dist_to_nearest_support_pct": round((current_price - min(support_levels)) / current_price * 100, 1)
            },
            "resistance_evaluation": {
                "nearest_resistance": round(max(resistance_levels), 0),
                "dist_to_resistance_pct": round((max(resistance_levels) - current_price) / current_price * 100, 1)
            },
            "historical": {
                "high_52w": historical_high,
                "low_52w": historical_low
            },
            "data_source": data_source
        }

        # Add interpretation if requested
        if interpret:
            # Determine signal (MUA/BÁN/CHỜ)
            if current_rsi < 35 and is_potential_bottom:
                signal = "MUA"
            elif current_rsi > 70:
                signal = "BÁN"
            else:
                signal = "CHỜ"

            result["interpretation"] = await interp_service.interpret(
                "technical_analysis",
                {
                    "symbol": symbol,
                    "signal": signal,
                    "rsi": round(current_rsi, 1),
                    "current_price": round(current_price, 0),
                    "bottom_score": min(95, bottom_score)
                }
            )

        return result
    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/agents/chat")
async def agent_chat(request: QueryRequest):
    """Process natural language query from agent chat using LLM"""
    try:
        query = request.query.strip()

        if not query:
            return {
                "response": "Please ask a question",
                "agent": "Chief",
                "timestamp": datetime.now().isoformat()
            }

        # Try to use ConversationalQuant if available
        if ConversationalQuant:
            try:
                conv = ConversationalQuant()
                result = conv.process_query(query)
                if hasattr(result, 'to_dict'):
                    return result.to_dict()
                else:
                    return result
            except Exception as e:
                logger.warning(f"ConversationalQuant error: {e}")

        # Use LLM interpretation service for intelligent chat
        try:
            # Get basic market context
            market_context = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_open": is_market_open()[0],
            }

            # Call LLM for chat response
            llm_response = await interp_service.interpret(
                template_name="agent_chat",
                data={
                    "query": query,
                    "market_context": market_context
                }
            )

            return {
                "response": llm_response,
                "agent": "Chief",
                "timestamp": datetime.now().isoformat(),
                "llm_powered": True
            }
        except Exception as llm_error:
            logger.warning(f"LLM interpretation failed: {llm_error}, using fallback")

        # Fallback: Simple response from Chief agent
        response_map = {
            "mwg": "MWG is a growth stock with strong momentum. Scout detected opportunities at current levels.",
            "hpg": "HPG shows support at resistance. Technical analysis confirms potential entry points.",
            "fpt": "FPT has positive sentiment. Bull trap detection shows healthy breadth.",
            "help": "Ask me about any stock symbol, market conditions, or trading strategies.",
            "status": "All 6 agents are online and ready. System is monitoring 102 stocks for opportunities.",
            "agent": "Chief is coordinating analysis. Bear is watching risk, Bull is seeking growth, Scout found opportunities."
        }

        # Match keywords
        query_lower = query.lower()
        for keyword, response in response_map.items():
            if keyword in query_lower:
                return {
                    "response": response,
                    "agent": "Chief",
                    "timestamp": datetime.now().isoformat()
                }

        # Generic response
        return {
            "response": f"Analyzing '{query}'. Scout is checking market opportunities, Alex reviewing technical levels, Bull watching momentum. More data needed for detailed analysis. Try asking about specific stocks (MWG, HPG, FPT) or market conditions.",
            "agent": "Chief",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        return {
            "response": f"Error processing query: {str(e)[:100]}",
            "agent": "Chief",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/api/agents/analyze")
async def analyze_with_agents(request: Dict[str, Any]):
    """Run multi-agent analysis on a symbol"""
    try:
        symbol = request.get('symbol', 'MWG').upper()
        data_source = "unknown"
        live_price = None

        # STEP 0: Get LIVE price first (most important for display)
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            live_connector = get_realtime_connector()
            live_price = live_connector.get_stock_price(symbol)
            if live_price and live_price > 0:
                logger.info(f"📈 Got LIVE price for {symbol}: {live_price:,.0f} VND")
        except Exception as e:
            logger.warning(f"Could not get live price for {symbol}: {e}")

        # PRIORITY 1: Fetch REAL data from CafeF API
        df = None
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()
            historical = connector.get_stock_historical(symbol, days=100)

            if historical and len(historical) >= 20:
                df = pd.DataFrame(historical)
                data_source = "CafeF Real-time"
                logger.info(f"✅ Using REAL CafeF data for {symbol} ({len(historical)} days)")
        except Exception as e:
            logger.warning(f"CafeF historical fetch error for {symbol}: {e}")

        # FALLBACK 2: Try parquet file (may have old data)
        if df is None:
            try:
                parquet_path = Path(f"data/historical/{symbol}.parquet")
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    data_source = "Parquet (cached)"
                    logger.info(f"📂 Using parquet data for {symbol} ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Parquet read error for {symbol}: {e}")

        # FALLBACK 3: Return error instead of synthetic data
        if df is None:
            return {
                "symbol": symbol,
                "error": f"No historical data available for {symbol}. Download data first.",
                "data_source": "unavailable",
                "messages": [],
                "hint": "Run: python download_all_stocks.py"
            }

        # Calculate technical indicators
        close = df['close'].astype(float)
        volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series([1] * len(df))

        # Get current price - PREFER LIVE PRICE over historical close
        current_price = float(close.iloc[-1])
        if live_price and live_price > 0:
            current_price = live_price
            data_source = f"{data_source} + Live"
            logger.info(f"✅ Final price for {symbol}: {current_price:,.0f} VND (Live)")

        prev_price = float(close.iloc[-2]) if len(close) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

        # RSI Calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else 50

        # MACD Calculation
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_value = round(float(macd.iloc[-1]), 2)
        macd_signal = "BULLISH 📈" if macd.iloc[-1] > signal_line.iloc[-1] else "BEARISH 📉"

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper_bb = sma20 + 2 * std20
        lower_bb = sma20 - 2 * std20

        # Volume Analysis
        avg_vol = volume.rolling(20).mean().iloc[-1]
        current_vol = int(volume.iloc[-1])
        vol_ratio = round(current_vol / avg_vol, 2) if avg_vol > 0 else 1

        # Support/Resistance (50-day)
        recent_lows = df['low'].astype(float).tail(50)
        recent_highs = df['high'].astype(float).tail(50)
        support = round(float(recent_lows.min()), 0)
        resistance = round(float(recent_highs.max()), 0)

        # Historical Highs/Lows (all available data ~ 52 weeks)
        all_lows = df['low'].astype(float)
        all_highs = df['high'].astype(float)
        historical_low = round(float(all_lows.min()), 0)
        historical_high = round(float(all_highs.max()), 0)

        # Key levels (multiple support/resistance)
        key_levels = {
            "support_levels": [
                round(float(recent_lows.quantile(0.1)), 0),  # Strong support
                round(float(recent_lows.quantile(0.25)), 0),  # Medium support
                support  # Immediate support
            ],
            "resistance_levels": [
                resistance,  # Immediate resistance
                round(float(recent_highs.quantile(0.75)), 0),  # Medium resistance
                historical_high  # All-time high
            ],
            "historical_high": historical_high,
            "historical_low": historical_low,
            "52w_range_position": round((current_price - historical_low) / (historical_high - historical_low) * 100, 1) if historical_high != historical_low else 50
        }

        # Bollinger Bands position
        bb_position = "TRUNG BÌNH"
        if current_price > float(upper_bb.iloc[-1]):
            bb_position = "TRÊN BB (Quá mua)"
        elif current_price < float(lower_bb.iloc[-1]):
            bb_position = "DƯỚI BB (Quá bán)"

        # Trend analysis
        sma5 = close.rolling(5).mean().iloc[-1]
        sma20_val = float(sma20.iloc[-1])
        trend = "TĂNG" if sma5 > sma20_val else "GIẢM" if sma5 < sma20_val else "SIDEWAY"

        # Foreign flow - REAL DATA from VPS
        foreign_net = 0.0  # Default neutral
        foreign_status = "KHÔNG RÕ"
        try:
            from quantum_stock.dataconnector.vps_market import get_vps_connector
            vps = get_vps_connector()
            flow_data = await vps.get_foreign_flow([symbol])
            foreign_net = flow_data.get('net_value_billion', 0.0)
            foreign_status = "MUA RÒNG" if foreign_net > 0 else "BÁN RÒNG" if foreign_net < 0 else "TRUNG LẬP"
            logger.info(f"✅ Real foreign flow for {symbol}: {foreign_net:.2f}B VND ({foreign_status})")
        except Exception as e:
            logger.warning(f"Failed to get real foreign flow for {symbol}: {e}")
            foreign_status = "KHÔNG CÓ DỮ LIỆU"

        # Risk/Reward calculation
        risk = current_price - support
        reward = resistance - current_price
        rr_ratio = reward / risk if risk > 0 else 0

        # Confidence score
        confidence_score = 50
        if current_rsi < 35:
            confidence_score += 15
        if macd_value > 0:
            confidence_score += 10
        if vol_ratio > 1.5:
            confidence_score += 10
        if foreign_net > 0:
            confidence_score += 10
        confidence_score = min(95, max(30, confidence_score))

        # Prepare technical data for LLM agents
        technical_data = {
            "symbol": symbol,
            "price": current_price,
            "change_pct": change_pct,
            "rsi": current_rsi,
            "macd": macd_value,
            "macd_signal": macd_signal,
            "volume_ratio": vol_ratio,
            "support": support,
            "resistance": resistance,
            "trend": trend,
            "bollinger_position": bb_position,
            "sma5": float(sma5),
            "sma20": float(sma20_val),
            "foreign_net": foreign_net,
            "foreign_status": foreign_status,
            "risk": risk,
            "reward": reward,
            "rr_ratio": rr_ratio,
            "confidence": confidence_score
        }

        # Agent role definitions for LLM
        agent_roles = {
            "Scout": {
                "emoji": "🔭",
                "role": "Scout - Người trinh sát thị trường",
                "description": "Thu thập và báo cáo dữ liệu thị trường thô. Tập trung vào giá, khối lượng, xu hướng."
            },
            "Alex": {
                "emoji": "📊",
                "role": "Alex - Chuyên gia phân tích kỹ thuật",
                "description": "Phân tích các chỉ báo kỹ thuật (RSI, MACD, Bollinger). Đánh giá hỗ trợ, kháng cự."
            },
            "Bull": {
                "emoji": "🐂",
                "role": "Bull - Nhà đầu tư lạc quan",
                "description": "Tìm kiếm cơ hội mua, điểm tăng trưởng. Nhấn mạnh tiềm năng tăng giá."
            },
            "Bear": {
                "emoji": "🐻",
                "role": "Bear - Nhà quản lý rủi ro",
                "description": "Cảnh báo rủi ro, điểm yếu. Tập trung vào downside risk và stop-loss."
            },
            "Risk Doctor": {
                "emoji": "🏥",
                "role": "Risk Doctor - Bác sĩ quản lý vốn",
                "description": "Tính toán quản lý vốn, position size, stop-loss, take-profit."
            },
            "Chief": {
                "emoji": "⚖️",
                "role": "Chief - Người đưa ra quyết định cuối cùng",
                "description": "Tổng hợp ý kiến các agent, đưa ra verdict: MUA/BÁN/CHỜ với confidence."
            }
        }

        # Try to get LLM-powered agent analysis
        llm_messages = []
        use_llm = False
        try:
            import asyncio

            # Call all 6 agents in parallel for speed
            async def get_agent_message(agent_name, agent_info):
                try:
                    analysis = await interp_service.interpret(
                        template_name="agent_analysis",
                        data={
                            "role": agent_info["role"],
                            "role_description": agent_info["description"],
                            "symbol": symbol,
                            **technical_data
                        }
                    )
                    return {
                        "sender": agent_name,
                        "emoji": agent_info["emoji"],
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "SUCCESS" if "MUA" in analysis.upper() else "WARNING" if "BÁN" in analysis.upper() else "INFO",
                        "content": analysis
                    }
                except Exception as e:
                    logger.warning(f"LLM failed for {agent_name}: {e}")
                    return None

            # Call all agents in parallel
            agent_tasks = [
                get_agent_message(name, info)
                for name, info in agent_roles.items()
            ]
            llm_messages = await asyncio.gather(*agent_tasks)
            llm_messages = [msg for msg in llm_messages if msg is not None]

            if len(llm_messages) >= 4:  # If we got at least 4 agents, use LLM results
                logger.info(f"✅ Using LLM-powered agent analysis ({len(llm_messages)}/6 agents)")
                use_llm = True
            else:
                logger.warning(f"⚠️ Only {len(llm_messages)}/6 agents responded, using fallback")

        except Exception as llm_error:
            logger.warning(f"LLM agent analysis failed: {llm_error}, using hardcoded fallback")

        # Messages from agents (Vietnamese, detailed)
        if use_llm and len(llm_messages) > 0:
            messages = llm_messages
        else:
            messages = [
            {
                "sender": "Scout",
                "emoji": "🔭",
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "INFO",
                "content": f"📊 **TỔNG QUAN {symbol}**\n\n"
                           f"💰 Giá hiện tại: {current_price:,.0f} VND\n"
                           f"{'📈' if change_pct >= 0 else '📉'} Thay đổi: {change_pct:+.2f}%\n"
                           f"📦 Khối lượng: {current_vol:,.0f} cp\n"
                           f"📊 KL/TB 20 phiên: {vol_ratio:.2f}x\n"
                           f"🔄 Xu hướng ngắn hạn: {trend}"
            },
            {
                "sender": "Alex",
                "emoji": "📊",
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "ANALYSIS",
                "content": f"📈 **PHÂN TÍCH KỸ THUẬT**\n\n"
                           f"• RSI(14): {current_rsi:.1f}/100 {'⚠️ Quá mua' if current_rsi > 70 else '⚠️ Quá bán' if current_rsi < 30 else '✅ Bình thường'}\n"
                           f"• MACD: {macd_value:,.0f} {'📈 Bullish' if macd_value > 0 else '📉 Bearish'}\n"
                           f"• Bollinger: {bb_position}\n"
                           f"• Hỗ trợ: {support:,.0f} VND\n"
                           f"• Kháng cự: {resistance:,.0f} VND\n"
                           f"• SMA(5): {sma5:,.0f} | SMA(20): {sma20_val:,.0f}"
            },
            {
                "sender": "Bull",
                "emoji": "🐂",
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "SUCCESS" if vol_ratio > 2 or (current_rsi < 35 and macd_value > 0) else "INFO",
                "content": f"🎯 **PHÂN TÍCH TĂNG TRƯỞNG**\n\n"
                           f"{'🔥 VOLUME BÙG NỔ! Dòng tiền đang đổ vào mạnh' if vol_ratio > 2 else '📊 Volume bình thường, chờ tín hiệu breakout' if vol_ratio > 0.8 else '😴 Volume thấp, thị trường đang tích lũy'}\n\n"
                           f"• Khoảng cách đến kháng cự: {((resistance - current_price) / current_price * 100):.1f}%\n"
                           f"• Tiềm năng tăng: {reward:,.0f} VND/cp\n"
                           f"• Điểm mua lý tưởng: {support:,.0f} - {support * 1.02:,.0f} VND"
            },
            {
                "sender": "Bear",
                "emoji": "🐻",
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "WARNING" if current_rsi > 70 or current_rsi < 30 or vol_ratio > 2.5 else "INFO",
                "content": f"⚠️ **CẢNH BÁO RỦI RO**\n\n"
                           f"{'🚨 QUÁ MUA! RSI > 70, khả năng điều chỉnh cao' if current_rsi > 70 else '📉 QUÁ BÁN! RSI < 30, có thể hồi phục' if current_rsi < 30 else '✅ RSI ổn định, không có tín hiệu cực đoan'}\n\n"
                           f"• Khoảng cách đến hỗ trợ: {((current_price - support) / current_price * 100):.1f}%\n"
                           f"• Rủi ro giảm: {risk:,.0f} VND/cp\n"
                           f"• Cắt lỗ tại: {support * 0.97:,.0f} VND (-3% dưới hỗ trợ)"
            },
            {
                "sender": "Risk Doctor",
                "emoji": "🏥",
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "INFO",
                "content": f"💊 **QUẢN LÝ VỐN & VỊ THẾ**\n\n"
                           f"📍 Điểm vào lệnh: {current_price:,.0f} VND\n"
                           f"🛑 Cắt lỗ (SL): {support:,.0f} VND ({((support - current_price) / current_price * 100):.1f}%)\n"
                           f"🎯 Chốt lời (TP): {resistance:,.0f} VND (+{((resistance - current_price) / current_price * 100):.1f}%)\n"
                           f"📊 Risk/Reward: 1:{rr_ratio:.1f}\n\n"
                           f"💰 Với vốn 100 triệu:\n"
                           f"• Rủi ro 2%: Position size = {int(2000000 / risk):,} cp\n"
                           f"• Giá trị: {int(2000000 / risk) * current_price:,.0f} VND"
            },
            {
                "sender": "Chief",
                "emoji": "⚖️",
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "SUCCESS" if confidence_score >= 70 else "WARNING" if confidence_score < 50 else "INFO",
                "content": f"🎖️ **QUYẾT ĐỊNH CUỐI CÙNG**\n\n"
                           f"{'🟢 KHUYẾN NGHỊ: MUA' if current_rsi < 40 and macd_value > 0 else '🔴 KHUYẾN NGHỊ: BÁN' if current_rsi > 70 else '🟡 KHUYẾN NGHỊ: CHỜ'}\n\n"
                           f"📊 Độ tin cậy: {confidence_score}%\n"
                           f"📈 Xu hướng: {trend}\n"
                           f"💹 MACD: {'Tích cực' if macd_value > 0 else 'Tiêu cực'}\n"
                           f"📦 Volume: {'Cao' if vol_ratio > 1.5 else 'Bình thường' if vol_ratio > 0.8 else 'Thấp'}\n\n"
                           f"💡 Lưu ý: {'Thời điểm tốt để tích lũy' if current_rsi < 40 else 'Cân nhắc chốt lời một phần' if current_rsi > 60 else 'Chờ tín hiệu rõ ràng hơn'}"
            }
        ]

        return {
            "success": True,
            "symbol": symbol,
            "messages": messages,
            "technical": {
                "rsi": current_rsi,
                "macd": macd_value,
                "macd_signal": macd_signal,
                "volume_ratio": vol_ratio,
                "support": support,
                "resistance": resistance,
                "price": current_price,
                "change_pct": round(change_pct, 2),
                "data_source": data_source
            },
            "key_levels": key_levels,
            "historical": {
                "high_52w": historical_high,
                "low_52w": historical_low,
                "range_position": key_levels["52w_range_position"],
                "days_of_data": len(df)
            }
        }

    except Exception as e:
        logger.error(f"Agent analysis error: {e}")
        return {
            "success": False,
            "error": str(e)[:100],
            "messages": [{
                "sender": "System",
                "emoji": "❌",
                "type": "ERROR",
                "content": f"Analysis failed: {str(e)[:100]}"
            }]
        }


# ============================================================
