# Phase 02: Implement WebSocket Client

**Priority:** P0 (CRITICAL)
**Effort:** 2h
**Status:** Pending

## Context Links
- Backend: `run_autonomous_paper_trading.py` (line 176-235: WebSocket endpoint /ws/autonomous)
- Backend: `quantum_stock/autonomous/orchestrator.py` (broadcasts 5 event types)
- Frontend: `vn-quant-web/src/App.jsx` (line 1131: uses iframe, zero WebSocket code)
- Audit: "WebSocket not implemented in FE"

## Overview
Backend broadcasts real-time events via `/ws/autonomous`:
1. `opportunity_detected` - New trading signal found
2. `agent_discussion` - 6 agents discussing trade
3. `order_executed` - Order placed
4. `position_exited` - Position closed
5. `system_reset` - System state cleared

**Goal:** Build React WebSocket client to receive these events, display in dashboard real-time.

## Key Insights
- Backend already working, just need FE client
- 5 event types, all JSON payloads
- Use native WebSocket API (no external deps like socket.io)
- Auto-reconnect on disconnect
- Display events in chronological feed (newest first)

## Requirements

### Functional
- Connect to ws://localhost:8100/ws/autonomous on mount
- Parse JSON messages, identify event type
- Store events in React state (max 100 recent)
- Display in real-time feed with timestamps
- Show connection status indicator

### Non-Functional
- Auto-reconnect with exponential backoff (1s, 2s, 4s, max 30s)
- Handle connection errors gracefully
- Clear events on system_reset event
- Max 100 events in memory (prevent memory leak)

## Architecture

```
┌──────────────────────────────────────┐
│ React Component: WebSocketFeed      │
│ - useEffect: connect on mount       │
│ - State: events[], isConnected      │
│ - Handler: onMessage(event)         │
└─────────────┬────────────────────────┘
              │ WebSocket
              ↓
┌──────────────────────────────────────┐
│ FastAPI: /ws/autonomous              │
│ - Broadcasts 5 event types           │
│ - JSON payloads                      │
└──────────────────────────────────────┘

Event Types:
1. opportunity_detected → {symbol, source, confidence, ...}
2. agent_discussion → {discussion_id, agents[], messages[], ...}
3. order_executed → {order_id, symbol, side, price, qty, ...}
4. position_exited → {symbol, entry_price, exit_price, pnl, ...}
5. system_reset → {timestamp, message}
```

## Related Code Files

**Frontend (Create):**
- `vn-quant-web/src/components/websocket-feed.jsx` - WebSocket client component (~150 lines)
- `vn-quant-web/src/hooks/use-websocket.js` - Reusable WebSocket hook (~80 lines)

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Import and use WebSocketFeed in DashboardView

**Backend (No changes needed):**
- `run_autonomous_paper_trading.py` - Already has /ws/autonomous endpoint

## Implementation Steps

### Step 1: Create WebSocket Hook
Create `vn-quant-web/src/hooks/use-websocket.js`:

```javascript
import { useState, useEffect, useRef } from 'react'

export function useWebSocket(url, options = {}) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectDelayRef = useRef(1000) // Start with 1s

  useEffect(() => {
    let isMounted = true

    const connect = () => {
      if (!isMounted) return

      try {
        const ws = new WebSocket(url)
        wsRef.current = ws

        ws.onopen = () => {
          console.log('WebSocket connected:', url)
          setIsConnected(true)
          reconnectDelayRef.current = 1000 // Reset delay
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            setLastMessage(data)
            options.onMessage?.(data)
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err)
          }
        }

        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
        }

        ws.onclose = () => {
          console.log('WebSocket disconnected')
          setIsConnected(false)

          if (isMounted && options.autoReconnect !== false) {
            // Exponential backoff
            reconnectTimeoutRef.current = setTimeout(() => {
              console.log(`Reconnecting in ${reconnectDelayRef.current}ms...`)
              connect()
            }, reconnectDelayRef.current)

            reconnectDelayRef.current = Math.min(
              reconnectDelayRef.current * 2,
              30000 // Max 30s
            )
          }
        }
      } catch (err) {
        console.error('WebSocket connection failed:', err)
      }
    }

    connect()

    return () => {
      isMounted = false
      clearTimeout(reconnectTimeoutRef.current)
      wsRef.current?.close()
    }
  }, [url])

  return { isConnected, lastMessage }
}
```

### Step 2: Create WebSocket Feed Component
Create `vn-quant-web/src/components/websocket-feed.jsx`:

```javascript
import { useState } from 'react'
import { useWebSocket } from '../hooks/use-websocket'

const WS_URL = 'ws://localhost:8100/ws/autonomous'
const MAX_EVENTS = 100

export function WebSocketFeed() {
  const [events, setEvents] = useState([])

  const { isConnected } = useWebSocket(WS_URL, {
    onMessage: (data) => {
      // Handle system_reset: clear all events
      if (data.type === 'system_reset') {
        setEvents([])
        return
      }

      // Add new event to top, limit to MAX_EVENTS
      setEvents(prev => [
        { ...data, timestamp: data.timestamp || new Date().toISOString() },
        ...prev.slice(0, MAX_EVENTS - 1)
      ])
    }
  })

  const getEventIcon = (type) => {
    switch (type) {
      case 'opportunity_detected': return '🔍'
      case 'agent_discussion': return '💬'
      case 'order_executed': return '✅'
      case 'position_exited': return '🚪'
      case 'system_reset': return '🔄'
      default: return '📡'
    }
  }

  const getEventColor = (type) => {
    switch (type) {
      case 'opportunity_detected': return 'text-amber-400'
      case 'agent_discussion': return 'text-blue-400'
      case 'order_executed': return 'text-emerald-400'
      case 'position_exited': return 'text-purple-400'
      case 'system_reset': return 'text-red-400'
      default: return 'text-slate-400'
    }
  }

  return (
    <div className="glass-panel p-6 rounded-xl">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-white">Live Feed</h2>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
          <span className="text-xs text-slate-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="space-y-3 max-h-[600px] overflow-y-auto">
        {events.length === 0 ? (
          <p className="text-slate-500 text-sm text-center py-8">
            Waiting for events...
          </p>
        ) : (
          events.map((event, idx) => (
            <div
              key={`${event.type}-${event.timestamp}-${idx}`}
              className="p-4 bg-black/20 rounded-lg border border-white/5 hover:border-white/10 transition-colors"
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl">{getEventIcon(event.type)}</span>
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <h3 className={`font-semibold ${getEventColor(event.type)}`}>
                      {event.type.replace(/_/g, ' ').toUpperCase()}
                    </h3>
                    <span className="text-xs text-slate-500">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>

                  {event.type === 'opportunity_detected' && (
                    <div className="mt-2 text-sm">
                      <p className="text-white">Symbol: <span className="font-bold">{event.symbol}</span></p>
                      <p className="text-slate-400">Source: {event.source}</p>
                      <p className="text-slate-400">Confidence: {(event.confidence * 100).toFixed(0)}%</p>
                    </div>
                  )}

                  {event.type === 'order_executed' && (
                    <div className="mt-2 text-sm">
                      <p className="text-white">{event.symbol} - {event.side.toUpperCase()}</p>
                      <p className="text-slate-400">Price: {event.price.toLocaleString()} VND</p>
                      <p className="text-slate-400">Qty: {event.quantity}</p>
                    </div>
                  )}

                  {event.type === 'position_exited' && (
                    <div className="mt-2 text-sm">
                      <p className="text-white">{event.symbol}</p>
                      <p className={event.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                        P&L: {event.pnl >= 0 ? '+' : ''}{event.pnl.toFixed(2)}%
                      </p>
                    </div>
                  )}

                  {event.type === 'agent_discussion' && (
                    <div className="mt-2 text-sm">
                      <p className="text-slate-400">{event.agents?.length || 0} agents participated</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
```

### Step 3: Integrate into Dashboard
Modify `vn-quant-web/src/App.jsx`:

```javascript
// Add import
import { WebSocketFeed } from './components/websocket-feed'

// In DashboardView component, add WebSocketFeed:
function DashboardView({ marketStatus, regime }) {
  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      {/* Existing stat cards */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* ... existing cards ... */}
      </section>

      {/* Add WebSocket feed */}
      <section>
        <WebSocketFeed />
      </section>
    </div>
  )
}
```

### Step 4: Test WebSocket Connection
1. Start backend: `python run_autonomous_paper_trading.py`
2. Verify WebSocket endpoint: Check console for "WebSocket endpoint: /ws/autonomous"
3. Start frontend: `cd vn-quant-web && npm run dev`
4. Open http://localhost:5176
5. Check Live Feed shows "Connected" indicator
6. Trigger test event from backend:
   ```bash
   curl -X POST http://localhost:8100/api/test/opportunity?symbol=ACB
   ```
7. Verify event appears in Live Feed

## Todo List
- [ ] Create use-websocket.js hook with auto-reconnect
- [ ] Create websocket-feed.jsx component
- [ ] Add event rendering for 5 event types
- [ ] Add connection status indicator
- [ ] Import WebSocketFeed into App.jsx
- [ ] Add to DashboardView
- [ ] Test: Connection on page load
- [ ] Test: Auto-reconnect on disconnect
- [ ] Test: All 5 event types display correctly
- [ ] Test: system_reset clears event list
- [ ] Test: Max 100 events enforced

## Success Criteria
- [ ] WebSocket connects to /ws/autonomous on mount
- [ ] Connection status indicator shows green when connected
- [ ] All 5 event types render with correct icons/colors
- [ ] Events appear in real-time (< 500ms delay)
- [ ] Auto-reconnect works after backend restart
- [ ] system_reset event clears all previous events
- [ ] Max 100 events limit prevents memory leak
- [ ] No console errors on connection/disconnection

## Risk Assessment
- **Risk:** Backend restart breaks connection → Mitigated: Auto-reconnect with backoff
- **Risk:** Large event payloads slow UI → Mitigated: Max 100 events, virtualization if needed
- **Risk:** WebSocket blocked by firewall → Acceptable: Local development only

## Security Considerations
- WebSocket on localhost only (no external access)
- No auth on WebSocket (read-only events)
- Consider adding API key to WS query params if Phase 01 requires it

## Next Steps
After completing this phase:
1. Update plan.md with completion status
2. Proceed to Phase 03 (Remove iframe, integrate trading view)
3. Consider adding WebSocket events to other views (Analysis, Radar)
