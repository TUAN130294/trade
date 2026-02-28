# WebSocket & React Code Patterns

**Version:** 1.0
**Updated:** 2026-02-27
**Scope:** WebSocket communication and React components

---

## WebSocket Patterns (FastAPI + Client)

### FastAPI WebSocket Handler

```python
@app.websocket("/ws/autonomous")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time events"""
    await websocket.accept()
    connected_websockets.add(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            # Process incoming message
            response = await process_client_message(data)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        connected_websockets.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def broadcast_message(message: dict) -> None:
    """Broadcast to all connected WebSocket clients"""
    disconnected = set()

    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.warning(f"Broadcast to client failed: {e}")
            disconnected.add(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        connected_websockets.discard(ws)
```

### Message Broadcasting

```python
# In AutonomousOrchestrator
async def broadcast_agent_message(agent_name: str, message: str, confidence: float):
    """Broadcast agent discussion to dashboard"""
    ws_message = {
        "type": "agent_message",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "agent_name": agent_name,
            "agent_emoji": AGENT_EMOJIS[agent_name],
            "message_type": "ANALYSIS",
            "content": message,
            "confidence": confidence
        }
    }
    await broadcast_message(ws_message)

async def broadcast_order_executed(order: Order):
    """Broadcast order execution to dashboard"""
    ws_message = {
        "type": "order_executed",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status
        }
    }
    await broadcast_message(ws_message)
```

### Client-Side WebSocket Hook (React)

```javascript
// hooks/use-websocket.js
import { useEffect, useState, useCallback } from 'react';

const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export const useWebSocket = (url) => {
    const [data, setData] = useState(null);
    const [status, setStatus] = useState("disconnected");
    const [reconnectCount, setReconnectCount] = useState(0);

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(url);

            ws.onopen = () => {
                console.log("WebSocket connected");
                setStatus("connected");
                setReconnectCount(0);
            };

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    setData(message);
                } catch (error) {
                    console.error("Failed to parse message:", error);
                }
            };

            ws.onerror = (error) => {
                console.error("WebSocket error:", error);
                setStatus("error");
            };

            ws.onclose = () => {
                console.log("WebSocket closed, attempting reconnect...");
                setStatus("disconnected");

                if (reconnectCount < MAX_RECONNECT_ATTEMPTS) {
                    setTimeout(() => {
                        setReconnectCount(prev => prev + 1);
                        connect();
                    }, RECONNECT_DELAY_MS);
                } else {
                    setStatus("failed");
                }
            };

            return ws;
        } catch (error) {
            console.error("WebSocket connection failed:", error);
            setStatus("error");
        }
    }, [url, reconnectCount]);

    useEffect(() => {
        const ws = connect();
        return () => ws?.close();
    }, [connect]);

    return { data, status };
};
```

### Usage in React Component

```jsx
// components/AgentFeed.jsx
import { useWebSocket } from '../hooks/use-websocket';

export const AgentFeed = () => {
    const { data, status } = useWebSocket('ws://localhost:8100/ws/autonomous');
    const [messages, setMessages] = useState([]);

    useEffect(() => {
        if (data?.type === 'agent_message') {
            setMessages(prev => [...prev, data.data]);
        }
    }, [data]);

    return (
        <div className="p-4">
            <div className="mb-2">
                Status: <span className={status === 'connected' ? 'text-green-400' : 'text-red-400'}>
                    {status}
                </span>
            </div>
            <div className="space-y-2">
                {messages.map((msg, i) => (
                    <div key={i} className="p-3 bg-white/10 rounded">
                        <div className="flex items-center gap-2">
                            <span>{msg.agent_emoji}</span>
                            <strong>{msg.agent_name}</strong>
                            <span className="text-sm text-gray-300">{msg.confidence}%</span>
                        </div>
                        <p className="mt-1 text-sm">{msg.content}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};
```

---

## React Component Patterns

### Functional Component with Hooks

```jsx
import React, { useState, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';

export const StockChart = ({ symbol, initialData }) => {
    const [data, setData] = useState(initialData);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchStockData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`/api/stock/${symbol}`);
            if (!response.ok) throw new Error('Failed to fetch');

            const stockData = await response.json();
            setData(stockData);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [symbol]);

    useEffect(() => {
        fetchStockData();
        const interval = setInterval(fetchStockData, 5000); // Refresh every 5s
        return () => clearInterval(interval);
    }, [symbol, fetchStockData]);

    if (loading) return <div className="p-4 text-center">Loading...</div>;
    if (error) return <div className="p-4 text-red-400">{error}</div>;

    return (
        <div className="p-4 bg-glass rounded-lg">
            <h3 className="font-bold mb-2">{symbol}</h3>
            <div className="text-sm">
                <div>Price: {data.close}</div>
                <div className={data.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {data.pnl_pct:+.2f}%
                </div>
            </div>
        </div>
    );
};

StockChart.propTypes = {
    symbol: PropTypes.string.isRequired,
    initialData: PropTypes.object
};

StockChart.defaultProps = {
    initialData: null
};
```

### Tailwind CSS Classes (VN-Quant Theme)

**Glass-Morphism:**
```jsx
<div className="backdrop-blur-md bg-white/10 border border-white/20 rounded-xl p-4">
    {/* Content */}
</div>
```

**Button Styles:**
```jsx
// Primary button
<button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
    Action
</button>

// Danger button
<button className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
    Delete
</button>

// Subtle button
<button className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded">
    More
</button>
```

**Text Colors:**
```jsx
// Positive (gains, up move)
<span className="text-green-400">+5.2%</span>

// Negative (losses, down move)
<span className="text-red-400">-2.1%</span>

// Neutral
<span className="text-gray-300">Neutral</span>

// Emphasis
<span className="text-white font-bold">Important</span>
```

**Layout:**
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    {/* Responsive grid */}
</div>

<div className="flex items-center justify-between p-4 bg-black/50 rounded">
    {/* Flexbox container */}
</div>
```

---

## Best Practices

### WebSocket
- Always handle disconnection and auto-reconnect with exponential backoff
- Limit message size to prevent memory issues
- Use message queues for high-frequency updates
- Clean up stale connections periodically

### React
- Use functional components with hooks, avoid class components
- Memoize expensive computations with useMemo
- Keep components small and focused (single responsibility)
- Use PropTypes for type checking
- Cleanup subscriptions in useEffect return
- Avoid prop drilling with Context API or Redux

---

*These patterns ensure robust WebSocket communication and clean React components throughout VN-Quant frontend.*
