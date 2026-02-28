from fastapi import APIRouter, HTTPException, WebSocket, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import logging
from app.core import state
from app.core.auth import verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    symbol: str = "MWG"

@router.get("/api/status")
async def get_status():
    """Get system status"""
    if state.orchestrator:
        return state.orchestrator.get_status()
    raise HTTPException(status_code=503, detail="Orchestrator not initialized")


@router.get("/api/orders")
async def get_orders():
    """Get all orders history"""
    if state.orchestrator and state.orchestrator.broker:
        orders = []
        for order in state.orchestrator.broker.orders.values():
            orders.append(order.to_dict())
        # Sort by created_at descending
        orders.sort(key=lambda x: x['created_at'], reverse=True)
        return {"orders": orders}
    return {"orders": []}


@router.get("/api/positions")
async def get_positions():
    """Get current positions"""
    if state.orchestrator and state.orchestrator.broker:
        positions = []
        for pos in state.orchestrator.broker.positions.values():
            positions.append(pos.to_dict())
        return {"positions": positions}
    return {"positions": []}


@router.get("/api/trades")
async def get_trades():
    """Get trade history"""
    if state.orchestrator and state.orchestrator.broker:
        trades = state.orchestrator.broker.trade_history.copy()
        # Sort by timestamp descending
        trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return {"trades": trades}
    return {"trades": []}


@router.get("/api/discussions")
async def get_discussions():
    """Get all discussion history"""
    if state.orchestrator:
        discussions = list(state.orchestrator.discussion_history.values())
        # Sort by timestamp descending
        discussions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return {"discussions": discussions}
    return {"discussions": []}


@router.get("/api/discussion/{discussion_id}")
async def get_discussion(discussion_id: str):
    """Get discussion by ID"""
    if state.orchestrator and discussion_id in state.orchestrator.discussion_history:
        return state.orchestrator.discussion_history[discussion_id]
    raise HTTPException(status_code=404, detail="Discussion not found")


@router.get("/api/order/{order_id}/discussion")
async def get_order_discussion(order_id: str):
    """Get discussion associated with an order"""
    if state.orchestrator:
        discussion_id = state.orchestrator.order_to_discussion.get(order_id)
        if discussion_id and discussion_id in state.orchestrator.discussion_history:
            return state.orchestrator.discussion_history[discussion_id]
        raise HTTPException(status_code=404, detail=f"No discussion found for order {order_id}")
    raise HTTPException(status_code=503, detail="Orchestrator not initialized")


@router.post("/api/test/opportunity", dependencies=[Depends(verify_api_key)])
async def trigger_test_opportunity(symbol: str = "ACB"):
    """
    Trigger a test opportunity for testing
    This simulates the model scanner finding a strong opportunity
    """
    
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        from quantum_stock.scanners.model_prediction_scanner import ModelPrediction
        from datetime import datetime

        # Create mock prediction
        mock_prediction = ModelPrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=26500.0,
            predicted_prices=[26800, 27200, 27500, 27800, 28100],
            expected_return_5d=0.0604,  # +6.04%
            confidence=0.82,
            direction='UP',
            has_opportunity=True,
            signal_strength=0.0496,
            model_path=f'models/{symbol}_stockformer_simple_best.pt',
            features_used=15
        )

        # Trigger the opportunity callback
        await state.orchestrator._on_model_opportunity(mock_prediction)

        return {
            "status": "triggered",
            "symbol": symbol,
            "expected_return": f"{mock_prediction.expected_return_5d*100:.2f}%",
            "confidence": mock_prediction.confidence
        }

    except Exception as e:
        logger.error(f"Error triggering test opportunity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def is_market_open() -> tuple[bool, str]:
    """Check if Vietnam stock market is open"""
    from datetime import datetime, time
    now = datetime.now()
    current_time = now.time()
    weekday = now.weekday()  # 0=Monday, 6=Sunday

    # Weekend check
    if weekday >= 5:  # Saturday or Sunday
        return False, "Thị trường đóng cửa (Cuối tuần)"

    # Morning session: 09:00 - 11:30
    morning_open = time(9, 0)
    morning_close = time(11, 30)

    # Afternoon session: 13:00 - 14:45
    afternoon_open = time(13, 0)
    afternoon_close = time(14, 45)

    if morning_open <= current_time <= morning_close:
        return True, "Phiên sáng (09:00-11:30)"
    elif afternoon_open <= current_time <= afternoon_close:
        return True, "Phiên chiều (13:00-14:45)"
    else:
        next_session = "09:00 ngày mai" if current_time > afternoon_close else "09:00" if current_time < morning_open else "13:00"
        return False, f"Ngoài giờ giao dịch. Phiên tiếp theo: {next_session}"


@router.post("/api/test/trade", dependencies=[Depends(verify_api_key)])
async def trigger_test_trade(symbol: str = "MWG", action: str = "BUY", force: bool = False):
    """
    Direct test trade - bypasses agents for instant demo
    Set force=true to bypass market hours check (for testing only)
    """
    
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Check market hours (unless force=true)
    if not force:
        market_open, reason = is_market_open()
        if not market_open:
            return {
                "error": "Market closed",
                "reason": reason,
                "hint": "Add ?force=true to bypass (testing only)"
            }

    try:
        from quantum_stock.autonomous.position_exit_scheduler import Position
        from quantum_stock.core.broker_api import OrderSide, OrderType
        from quantum_stock.dataconnector.realtime_market import get_realtime_connector
        from datetime import datetime

        # Get REAL current price from CafeF
        # CafeF 'l' field returns price in thousands (86 = 86,000 VND)
        connector = get_realtime_connector()
        real_price = connector.get_stock_price(symbol)
        if real_price:
            current_price = real_price  # Already in thousands format from CafeF
        else:
            # Fallback prices in thousands format
            # Prices in full VND (matching get_stock_price() return format)
            fallback_prices = {"MWG": 86000, "HPG": 26200, "SSI": 30350, "ACB": 26500, "VNM": 70000, "FPT": 155000}
            current_price = fallback_prices.get(symbol, 50.0)

        # Update broker's market price for this symbol
        state.orchestrator.broker.market_prices[symbol] = {
            'last': current_price,
            'bid': current_price * 0.999,
            'ask': current_price * 1.001,
            'reference': current_price
        }

        # Calculate position size (12.5% of portfolio)
        account_info = await state.orchestrator.broker.get_account_info()
        portfolio_value = account_info.nav
        position_value = portfolio_value * 0.125
        
        # Buffer check
        if position_value > state.orchestrator.broker.cash_balance:
            position_value = state.orchestrator.broker.cash_balance * 0.95
            
        quantity = int(position_value / current_price / 100) * 100

        if action == "BUY" and quantity > 0:
            # Place buy order using broker directly
            order = await state.orchestrator.broker.place_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=current_price
            )

            # Add to position monitor
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=current_price,
                entry_date=datetime.now(),
                take_profit_pct=0.15,
                trailing_stop_pct=0.05,
                stop_loss_pct=-0.05,
                entry_reason="TEST - Direct trade for demo"
            )
            state.orchestrator.exit_scheduler.add_position(position)

            # Broadcast order
            await state.orchestrator.agent_message_queue.put({
                'type': 'order_executed',
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            })

            logger.info(f"TEST TRADE: {action} {quantity} {symbol} @ {current_price:,.0f}")

            return {
                "status": "success",
                "action": action,
                "symbol": symbol,
                "quantity": quantity,
                "price": current_price,
                "value": quantity * current_price
            }

        raise HTTPException(status_code=400, detail="Invalid trade parameters")

    except Exception as e:
        logger.error(f"Error in test trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/reset", dependencies=[Depends(verify_api_key)])
async def reset_trading():
    """Reset all trading history, positions, and restore initial balance"""
    
    if state.orchestrator and state.orchestrator.broker:
        # Clear broker data
        state.orchestrator.broker.reset()
        
        # Clear exit scheduler positions
        state.orchestrator.exit_scheduler.positions.clear()
        
        # Reset stats
        state.orchestrator.stats = {
            'opportunities_detected': 0,
            'agent_discussions': 0,
            'orders_executed': 0,
            'positions_exited': 0
        }
        
        # Broadcast reset message
        await state.orchestrator.agent_message_queue.put({
            'type': 'system_reset',
            'message': 'Trading history has been reset',
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info("🔄 Trading system reset - All history cleared")
        
        return {
            "status": "success",
            "message": "Trading history reset. Balance restored to 500,000,000 VND",
            "cash_balance": state.orchestrator.broker.cash_balance
        }
    raise HTTPException(status_code=503, detail="Orchestrator not initialized")


@router.post("/api/stop", dependencies=[Depends(verify_api_key)])
async def stop_system():
    """Stop autonomous system"""
    
    if state.orchestrator:
        state.orchestrator.is_running = False
        return {"status": "stopping"}
    raise HTTPException(status_code=400, detail="System not running")


# =====================
