"""
Start VN-Quant API Backend on Port 8003
"""
import sys
from pathlib import Path

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "quantum_stock.web.vn_quant_api:app",
        host="0.0.0.0",
        port=8003,
        reload=True,  # Enable reload for development
        log_level="info"
    )

