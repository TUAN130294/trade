from fastapi import APIRouter, HTTPException, WebSocket, Query
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
from app.core import state
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from quantum_stock.services.interpretation_service import get_interpretation_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize interpretation service at module level
interp_service = get_interpretation_service()

class QueryRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    symbol: str = "MWG"

# MISSING ENDPOINTS - Added for 100% frontend compatibility
# ============================================================

@router.get("/api/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get OHLCV stock data for charts - REAL DATA ONLY"""
    try:
        symbol = symbol.upper()

        # PRIORITY 1: Try CafeF real-time API first (freshest data)
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()
            historical = connector.get_stock_historical(symbol, days=200)

            if historical and len(historical) >= 20:
                data = []
                for item in historical:
                    date_str = str(item.get('date', ''))
                    # Convert DD/MM/YYYY to YYYY-MM-DD if needed
                    if '/' in date_str:
                        try:
                            parts = date_str.split('/')
                            date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        except:
                            pass
                    data.append({
                        "date": date_str,
                        "open": float(item.get('open', 0)),
                        "high": float(item.get('high', 0)),
                        "low": float(item.get('low', 0)),
                        "close": float(item.get('close', 0)),
                        "volume": int(item.get('volume', 0))
                    })
                # Append today's live candle if not already in data
                today_str = datetime.now().strftime('%Y-%m-%d')
                if data and data[-1].get('date') != today_str:
                    try:
                        stocks = connector._get_cached_or_fetch()
                        for s in stocks:
                            if s.get('a', '').upper() == symbol:
                                live_price = float(s.get('l', 0)) * 1000
                                live_open = float(s.get('o', s.get('l', 0))) * 1000
                                live_high = float(s.get('v', s.get('l', 0))) * 1000
                                live_low = float(s.get('w', s.get('l', 0))) * 1000
                                live_vol = int(s.get('n', 0))
                                if live_price > 0:
                                    data.append({
                                        "date": today_str,
                                        "open": live_open,
                                        "high": live_high,
                                        "low": live_low,
                                        "close": live_price,
                                        "volume": live_vol
                                    })
                                break
                    except Exception as e:
                        logger.warning(f"Could not append today's candle: {e}")

                logger.info(f"✅ /api/stock/{symbol}: Using CafeF real-time ({len(data)} days)")
                return data
        except Exception as e:
            logger.warning(f"CafeF fetch failed for {symbol}: {e}")

        # PRIORITY 2: Try parquet cache (may be 1 day old)
        parquet_path = Path(f"data/historical/{symbol}.parquet")
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df = df.tail(200)  # Last 200 days

                # Convert date format for chart (DD/MM/YYYY -> YYYY-MM-DD)
                data = []
                for _, row in df.iterrows():
                    date_str = str(row.get('date', ''))
                    if '/' in date_str:
                        try:
                            parts = date_str.split('/')
                            date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                        except:
                            pass
                    data.append({
                        "date": date_str,
                        "open": float(row.get('open', 0)),
                        "high": float(row.get('high', 0)),
                        "low": float(row.get('low', 0)),
                        "close": float(row.get('close', 0)),
                        "volume": int(row.get('volume', 0))
                    })
                logger.info(f"✅ /api/stock/{symbol}: Using parquet cache ({len(data)} days)")
                return data
            except Exception as e:
                logger.error(f"Parquet read error: {e}")

        # NO SYNTHETIC DATA - Return error if no real data available
        logger.error(f"❌ /api/stock/{symbol}: No real data available!")
        raise HTTPException(
            status_code=404,
            detail=f"No data available for {symbol}. Please download data first."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predict/{symbol}")
async def get_prediction(symbol: str):
    """AI Prediction for stock price - REAL STOCKFORMER MODEL"""
    try:
        symbol = symbol.upper()

        # Try to use real model scanner prediction
        try:
            from quantum_stock.scanners.model_prediction_scanner import ModelPredictionScanner
            scanner = ModelPredictionScanner()
            # Use absolute path relative to project root (4 levels up from this file)
            base_dir = Path(__file__).parent.parent.parent.parent.resolve()
            model_path = base_dir / "models" / f"{symbol}_stockformer_simple_best.pt"

            if model_path.exists():
                # Run real prediction using Stockformer model
                prediction = await scanner._predict_single(model_path)

                if prediction:
                    logger.info(f"✅ /api/predict/{symbol}: Using REAL Stockformer model")
                    return {
                        "symbol": symbol,
                        "current_price": prediction.current_price,
                        "direction": prediction.direction,
                        "expected_return": round(prediction.expected_return_5d * 100, 2),  # Convert to %
                        "confidence": round(prediction.confidence, 2),
                        "volatility_forecast": round(abs(prediction.expected_return_5d) * 0.5, 4),
                        "model": "Stockformer v2.5 (REAL)",
                        "predictions": [round(p, 0) for p in prediction.predicted_prices],
                        "has_opportunity": prediction.has_opportunity,
                        "signal_strength": round(prediction.signal_strength, 4),
                        "timestamp": datetime.now().isoformat(),
                        "source": "model"
                    }
        except Exception as e:
            logger.warning(f"Model prediction failed for {symbol}: {e}")

        # FALLBACK: If no model, return error - don't generate fake data
        logger.error(f"❌ /api/predict/{symbol}: No trained model available!")
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {symbol}. Train model first or add to watchlist. Hint: Run 'python train_stockformer.py --symbol {symbol}'"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/data/stats")
async def get_data_stats(interpret: bool = Query(False, description="Add LLM interpretation")):
    """Get data hub statistics"""
    try:
        data_dir = Path("data/historical")
        parquet_files = list(data_dir.glob("*.parquet")) if data_dir.exists() else []

        total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)  # MB
        sample_symbols = [f.stem for f in parquet_files[:20]]

        result = {
            "total_files": len(parquet_files),
            "total_available": 1730,  # Approximate total stocks on Vietnam exchanges
            "coverage_pct": round(len(parquet_files) / 1730 * 100, 1),
            "total_size_mb": round(total_size, 2),
            "sample_symbols": sample_symbols,
            "last_update": datetime.fromtimestamp(max(f.stat().st_mtime for f in parquet_files)).strftime("%Y-%m-%d %H:%M") if parquet_files else "N/A",
            "status": "OK"
        }

        # Add interpretation if requested
        if interpret:
            result["interpretation"] = await interp_service.interpret(
                "data_stats",
                {
                    "total_files": len(parquet_files),
                    "coverage_pct": round(len(parquet_files) / 1730 * 100, 1)
                }
            )

        return result
    except Exception as e:
        logger.error(f"Data stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/analyze/deep_flow")
async def analyze_deep_flow(request: Dict[str, Any]):
    """Deep flow analysis for a symbol - uses REAL foreign flow data"""
    try:
        symbol = request.get("symbol", "MWG").upper()
        days = request.get("days", 60)

        # Get real foreign flow data from VPS
        flow_score = 50.0  # Default neutral
        recommendation = "WATCH"
        insights = []

        try:
            from quantum_stock.dataconnector.vps_market import get_vps_connector
            vps = get_vps_connector()
            flow_data = await vps.get_foreign_flow([symbol])

            net_value_bn = flow_data.get('net_value_billion', 0.0)
            flow_type = flow_data.get('flow_type', 'NEUTRAL')

            # Calculate flow score based on real data (0-100 scale)
            # Strong buy: 80-100, Moderate buy: 60-80, Neutral: 40-60, Sell: 0-40
            if net_value_bn > 5:  # Strong buy > 5B VND
                flow_score = min(85 + (net_value_bn / 2), 100)
                recommendation = "ACCUMULATE"
                insights.append({
                    "type": "FOREIGN_BUY",
                    "description": f"Khối ngoại mua ròng mạnh {net_value_bn:.1f} tỷ",
                    "confidence": 0.85
                })
            elif net_value_bn > 1:  # Moderate buy 1-5B
                flow_score = 60 + (net_value_bn * 4)
                recommendation = "WATCH"
                insights.append({
                    "type": "FOREIGN_BUY",
                    "description": f"Khối ngoại mua ròng {net_value_bn:.1f} tỷ",
                    "confidence": 0.70
                })
            elif net_value_bn < -5:  # Strong sell
                flow_score = max(15 - (abs(net_value_bn) / 2), 0)
                recommendation = "AVOID"
                insights.append({
                    "type": "FOREIGN_SELL",
                    "description": f"Khối ngoại bán ròng mạnh {abs(net_value_bn):.1f} tỷ",
                    "confidence": 0.85
                })
            elif net_value_bn < -1:  # Moderate sell
                flow_score = 40 - (abs(net_value_bn) * 4)
                recommendation = "WATCH"
                insights.append({
                    "type": "FOREIGN_SELL",
                    "description": f"Khối ngoại bán ròng {abs(net_value_bn):.1f} tỷ",
                    "confidence": 0.70
                })
            else:  # Neutral
                flow_score = 50
                recommendation = "WATCH"
                insights.append({
                    "type": "NEUTRAL",
                    "description": f"Khối ngoại giao dịch ít ({net_value_bn:.1f} tỷ)",
                    "confidence": 0.60
                })

            # Add volume insight if available
            stocks_flow = flow_data.get('stocks', [])
            if stocks_flow:
                stock_data = stocks_flow[0]
                buy_vol = stock_data.get('foreign_buy_volume', 0)
                sell_vol = stock_data.get('foreign_sell_volume', 0)
                if buy_vol > 0 or sell_vol > 0:
                    insights.append({
                        "type": "VOLUME_DATA",
                        "description": f"KL mua: {buy_vol:,.0f} | KL bán: {sell_vol:,.0f}",
                        "confidence": 0.75
                    })

        except Exception as e:
            logger.warning(f"Failed to get real flow data for {symbol}: {e}")
            insights.append({
                "type": "DATA_UNAVAILABLE",
                "description": "Không thể lấy dữ liệu khối ngoại thực tế",
                "confidence": 0.0
            })

        return {
            "symbol": symbol,
            "days_analyzed": days,
            "insights": insights,
            "flow_score": round(flow_score, 1),
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Deep flow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
