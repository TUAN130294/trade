import os
import re

src_file = "d:/testpapertr/run_autonomous_paper_trading.py"
with open(src_file, 'r', encoding='utf-8') as f:
    orig_content = f.read()

# Define the sections by index or string markers
# Section 1: core/state.py
state_py_content = """# Global state for Autonomous Trading System
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
"""
os.makedirs("d:/testpapertr/app/core", exist_ok=True)
with open("d:/testpapertr/app/core/state.py", 'w', encoding='utf-8') as f:
    f.write(state_py_content)

# We want to extract the different sections.
# Let's break the file into parts based on specific headers:
# 1. Base API routes (orders, positions, trades, status, reset_trading, stop_system, trigger_test_*) -> trading.py
# 2. API Routes - Market Data & Analysis -> market.py
# 3. MISSING ENDPOINTS -> agents.py or data.py
# 4. NEWS INTELLIGENCE ENDPOINTS -> news.py

# In new main file, we will replace the body with router imports.
# We will just write a python script that will generate all these routers with necessary imports.
# Actually, since the file contains heavy references to Orchestrator, it might be safer and easier to just use `app.state` or global `app` if it's imported.
# Let's write the whole refactoring script next.
print("Created core directories and state.py")
