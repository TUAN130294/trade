# Global state for Autonomous Trading System
from typing import List
from fastapi import WebSocket

orchestrator = None
active_websockets: List[WebSocket] = []

ALLOWED_ORIGINS = [
    "http://localhost:5176",  # Frontend (Nginx)
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8003",
    "http://127.0.0.1:5176",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8003",
]
