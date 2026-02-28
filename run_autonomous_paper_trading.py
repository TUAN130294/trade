#!/usr/bin/env python3
"""
Autonomous Paper Trading Runner
================================
Chạy hệ thống autonomous trading với paper trading mode

User chỉ cần:
1. Chạy script này
2. Mở http://localhost:8000/autonomous
3. Xem agents thảo luận và trade real-time

KHÔNG CẦN xác nhận trades - hệ thống tự động hoàn toàn
"""

import asyncio
import sys
from pathlib import Path

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_stock.autonomous.orchestrator import AutonomousOrchestrator
try:
    from quantum_stock.agents.conversational_quant import ConversationalQuant
except:
    ConversationalQuant = None
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/autonomous_trading.log')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# =====================
# Pydantic Models
# =====================

class QueryRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    symbol: str = "MWG"

# Create FastAPI app
app = FastAPI(
    title="Autonomous Paper Trading",
    description="Real-time autonomous trading system with agent conversations",
    version="1.0.0"
)

from app.core import state
from app.api.routers import trading, market, data, news

app.include_router(trading.router)
app.include_router(market.router)
app.include_router(data.router)
app.include_router(news.router)

# CORS - Configurable via environment variables
# Read from env with localhost fallback
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else [
    "http://localhost:5176",  # Frontend (Nginx)
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8100",
    "http://localhost:8003",
    "http://127.0.0.1:5176",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8100",
    "http://127.0.0.1:8003",
]

# Add production origin if in production environment
if os.getenv("ENVIRONMENT") == "production" and os.getenv("PRODUCTION_ORIGIN"):
    ALLOWED_ORIGINS.append(os.getenv("PRODUCTION_ORIGIN"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator


@app.on_event("startup")
async def startup_event():
    """Start autonomous system on app startup"""

    logger.info("=" * 70)
    logger.info("🚀 STARTING AUTONOMOUS PAPER TRADING SYSTEM")
    logger.info("=" * 70)

    # Create orchestrator
    state.orchestrator = AutonomousOrchestrator(
        paper_trading=True,
        initial_balance=500_000_000  # 500M VND
    )

    # Start orchestrator in background
    asyncio.create_task(state.orchestrator.start())
    asyncio.create_task(broadcast_messages())

    logger.info("✅ System ready!")
    logger.info("📊 Open http://localhost:8001/autonomous to view dashboard")
    logger.info("")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop autonomous system on shutdown"""

    if state.orchestrator:
        logger.info("Stopping autonomous system...")
        await state.orchestrator.stop()


async def broadcast_messages():
    """Broadcast orchestrator messages to all WebSocket clients"""

    while True:
        try:
            if state.orchestrator and state.orchestrator.is_running:
                # Get message from orchestrator queue
                message = await asyncio.wait_for(
                    state.orchestrator.agent_message_queue.get(),
                    timeout=1.0
                )

                logger.info(f"📤 Broadcasting message type: {message.get('type')} to {len(state.active_websockets)} clients")

                # Broadcast to all connected clients
                for ws in state.active_websockets[:]:  # Copy list to avoid modification during iteration
                    try:
                        await ws.send_json(message)
                        logger.debug(f"✅ Sent to WebSocket client")
                    except Exception as e:
                        logger.warning(f"Failed to send to WebSocket: {e}")
                        state.active_websockets.remove(ws)

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            await asyncio.sleep(1)


@app.get("/")
async def homepage():
    """Healthcheck endpoint"""
    return {"status": "ok", "service": "Autonomous Trading Model API"}

@app.websocket("/ws/autonomous")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    state.active_websockets.append(websocket)

    logger.info(f"WebSocket client connected (total: {len(state.active_websockets)})")

    try:
        # Send initial status
        if state.orchestrator:
            await websocket.send_json({
                'type': 'status_update',
                'portfolio_value': state.orchestrator.broker.cash_balance,
                'active_positions': len(state.orchestrator.exit_scheduler.get_all_positions()),
                'today_pnl': 0
            })

        # Keep connection alive
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)
        logger.info(f"WebSocket client disconnected (total: {len(state.active_websockets)})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)


def is_port_in_use(port: int) -> bool:
    """Check if port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True

if __name__ == "__main__":
    PORT = int(os.environ.get("TRADING_PORT", 8100))

    # Check if port is already in use - prevent duplicate servers
    if is_port_in_use(PORT):
        logger.error(f"❌ Port {PORT} is already in use!")
        logger.error(f"Another server is already running on port {PORT}")
        logger.error(f"Please stop the existing server first, or use a different port")
        logger.error(f"To find the process: netstat -ano | findstr :{PORT}")
        sys.exit(1)

    logger.info("Starting Autonomous Paper Trading Server...")
    logger.info(f"Dashboard will be available at: http://localhost:{PORT}/autonomous")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
