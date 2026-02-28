import re
import os

with open("d:/testpapertr/run_autonomous_paper_trading.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Common Router Header
router_header = """from fastapi import APIRouter, HTTPException, WebSocket
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
from app.core import state

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    symbol: str = "MWG"

"""

# Prepare files
trading_lines = [router_header]
market_lines = [router_header]
data_lines = [router_header]
news_lines = [router_header]
main_lines = []

current_section = "main_top"
skip = False

for i, line in enumerate(lines):
    # Match section headers
    if "# API Routes - Market Data & Analysis" in line:
        current_section = "market"
    elif "MISSING ENDPOINTS" in line:
        current_section = "data"
    elif "NEWS INTELLIGENCE ENDPOINTS" in line:
        current_section = "news"
    elif "if __name__ == \"__main__\":" in line:
        current_section = "main_bottom"
    
    # Check if we should move initial endpoints to trading
    if current_section == "main_top" and "@app.get(\"/api/status\")" in line:
        current_section = "trading"

    # Process line replacements
    processed_line = line
    if current_section in ["trading", "market", "data", "news"]:
        processed_line = processed_line.replace("@app.get", "@router.get")
        processed_line = processed_line.replace("@app.post", "@router.post")
        processed_line = processed_line.replace("@app.websocket", "@router.websocket")
        
        # Fixing orchestrator and active_websockets references
        processed_line = processed_line.replace("global orchestrator", "")
        # But `if orchestrator:` needs to be `if state.orchestrator:`
        # We'll do a simple regex for orchestrator matching word boundary
        processed_line = re.sub(r'\borchestrator\b', 'state.orchestrator', processed_line)
        processed_line = re.sub(r'\bactive_websockets\b', 'state.active_websockets', processed_line)
        
        # Append to respective list
        if current_section == "trading":
            trading_lines.append(processed_line)
        elif current_section == "market":
            market_lines.append(processed_line)
        elif current_section == "data":
            data_lines.append(processed_line)
        elif current_section == "news":
            news_lines.append(processed_line)
            
    else:
        # For main_top and main_bottom
        # Add the state imports and router inclusions
        if current_section == "main_top":
            # We replace `global orchestrator` in main as well
            if "orchestrator: AutonomousOrchestrator = None" in processed_line or "active_websockets: List[WebSocket] = []" in processed_line:
                processed_line = "" # removed from main
            elif "global orchestrator" in processed_line or "global active_websockets" in processed_line:
                processed_line = "" 
            
            # Change orchestrator references in main too!
            if "orchestrator." in line or "orchestrator =" in line or "if orchestrator" in line:
                processed_line = re.sub(r'\borchestrator\b', 'state.orchestrator', processed_line)
                
            main_lines.append(processed_line)
            
        elif current_section == "main_bottom":
            main_lines.append(processed_line)


# Write files
os.makedirs("d:/testpapertr/app/api/routers", exist_ok=True)
with open("d:/testpapertr/app/api/routers/trading.py", "w", encoding="utf-8") as f:
    f.writelines(trading_lines)
with open("d:/testpapertr/app/api/routers/market.py", "w", encoding="utf-8") as f:
    f.writelines(market_lines)
with open("d:/testpapertr/app/api/routers/data.py", "w", encoding="utf-8") as f:
    f.writelines(data_lines)
with open("d:/testpapertr/app/api/routers/news.py", "w", encoding="utf-8") as f:
    f.writelines(news_lines)

# Now, we need to inject the imports and include_router into main_lines
# Right after creating FastAPI app: `app = FastAPI(...)`
# We'll search for `app = FastAPI` and after it finishes (around version="1.0.0"\n))
router_inclusion = """
from app.core import state
from app.api.routers import trading, market, data, news

app.include_router(trading.router)
app.include_router(market.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(news.router, prefix="/api")
"""

app_creation_idx = -1
for i, line in enumerate(main_lines):
    if "version=\"1.0.0\"" in line:
        app_creation_idx = i + 2
        break

if app_creation_idx != -1:
    main_lines.insert(app_creation_idx, router_inclusion)

with open("d:/testpapertr/run_autonomous_paper_trading.py", "w", encoding="utf-8") as f:
    f.writelines(main_lines)

print("Refactoring complete! Modules extracted into app/api/routers/")
