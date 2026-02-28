# Phase 03: Remove iframe, Integrate Trading View

**Priority:** P0 (CRITICAL)
**Effort:** 2h
**Status:** Pending

## Context Links
- Frontend: `vn-quant-web/src/App.jsx` (line 1131: iframe to /autonomous)
- Backend: `/autonomous` serves HTML dashboard (legacy)
- Backend: `app/api/routers/trading.py` - Core endpoints: /api/orders, /api/positions, /api/trades
- Audit: "iframe /autonomous broken - integrate trading view directly"

## Overview
Current implementation uses iframe pointing to `/autonomous` HTML page. Problems:
- Iframe breaks CORS, styling conflicts
- Separate state management (React vs vanilla HTML)
- Can't share WebSocket connection
- Duplicates API calls

**Goal:** Replace iframe with native React components using existing backend endpoints.

## Key Insights
- Backend already has all needed endpoints: /api/orders, /api/positions, /api/trades, /api/status
- Can reuse WebSocket feed from Phase 02
- Build 3 panels: Portfolio Stats, Positions Table, Orders Table
- Leverage TailwindCSS glass-morphism from existing components

## Requirements

### Functional
- Display portfolio stats: cash, total_value, total_pnl
- Show current positions table: symbol, qty, entry_price, current_price, pnl%, days_held
- Show orders table: symbol, side, price, qty, status, timestamp
- Auto-refresh every 5 seconds
- Manual refresh button

### Non-Functional
- Load time <500ms for all 3 endpoints
- Responsive layout (mobile, tablet, desktop)
- Consistent glass-morphism styling
- No iframe

## Architecture

```
┌────────────────────────────────────────┐
│ TradingView Component                  │
├────────────────────────────────────────┤
│ ┌────────────────────────────────────┐ │
│ │ Portfolio Stats Panel              │ │
│ │ /api/status → cash, value, pnl     │ │
│ └────────────────────────────────────┘ │
│ ┌────────────────────────────────────┐ │
│ │ Positions Table                    │ │
│ │ /api/positions → symbol, pnl, etc  │ │
│ └────────────────────────────────────┘ │
│ ┌────────────────────────────────────┐ │
│ │ Orders History Table               │ │
│ │ /api/orders → side, price, status  │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘
```

## Related Code Files

**Frontend (Create):**
- `vn-quant-web/src/components/trading-view.jsx` - Main trading view (~180 lines)
- `vn-quant-web/src/components/portfolio-stats.jsx` - Stats panel (~60 lines)
- `vn-quant-web/src/components/positions-table.jsx` - Positions (~70 lines)
- `vn-quant-web/src/components/orders-table.jsx` - Orders (~70 lines)

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Replace iframe with TradingView component

**Backend (No changes):**
- Endpoints already exist in `app/api/routers/trading.py`

## Implementation Steps

### Step 1: Create Portfolio Stats Component
Create `vn-quant-web/src/components/portfolio-stats.jsx`:

```javascript
import { useState, useEffect } from 'react'

const API_URL = '/api'

export function PortfolioStats() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_URL}/status`)
      const data = await res.json()
      setStats(data)
      setLoading(false)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 5000) // Refresh every 5s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return <div className="text-slate-400 text-center py-4">Loading...</div>
  }

  const pnl = stats?.total_pnl || 0
  const pnlPct = stats?.total_pnl_pct || 0

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="glass-panel p-5 rounded-xl">
        <p className="text-slate-400 text-sm mb-1">Cash Available</p>
        <h3 className="text-2xl font-bold text-white">
          {(stats?.cash || 0).toLocaleString()} VND
        </h3>
      </div>

      <div className="glass-panel p-5 rounded-xl">
        <p className="text-slate-400 text-sm mb-1">Total Portfolio Value</p>
        <h3 className="text-2xl font-bold text-white">
          {(stats?.total_value || 0).toLocaleString()} VND
        </h3>
      </div>

      <div className="glass-panel p-5 rounded-xl">
        <p className="text-slate-400 text-sm mb-1">Total P&L</p>
        <h3 className={`text-2xl font-bold ${pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {pnl >= 0 ? '+' : ''}{pnl.toLocaleString()} VND
        </h3>
        <p className={`text-sm ${pnlPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
        </p>
      </div>
    </div>
  )
}
```

### Step 2: Create Positions Table Component
Create `vn-quant-web/src/components/positions-table.jsx`:

```javascript
import { useState, useEffect } from 'react'

const API_URL = '/api'

export function PositionsTable() {
  const [positions, setPositions] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchPositions = async () => {
    try {
      const res = await fetch(`${API_URL}/positions`)
      const data = await res.json()
      setPositions(data.positions || [])
      setLoading(false)
    } catch (err) {
      console.error('Failed to fetch positions:', err)
    }
  }

  useEffect(() => {
    fetchPositions()
    const interval = setInterval(fetchPositions, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6 rounded-xl">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-white">Current Positions</h2>
        <button
          onClick={fetchPositions}
          className="text-slate-400 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined">refresh</span>
        </button>
      </div>

      {loading ? (
        <div className="text-slate-400 text-center py-8">Loading positions...</div>
      ) : positions.length === 0 ? (
        <div className="text-slate-500 text-center py-8">No open positions</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-slate-400 text-sm border-b border-white/10">
                <th className="pb-3">Symbol</th>
                <th className="pb-3">Qty</th>
                <th className="pb-3">Entry Price</th>
                <th className="pb-3">Current Price</th>
                <th className="pb-3">P&L</th>
                <th className="pb-3">Days Held</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, idx) => {
                const pnlPct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                return (
                  <tr key={idx} className="border-b border-white/5">
                    <td className="py-3 font-semibold text-white">{pos.symbol}</td>
                    <td className="py-3 text-slate-300">{pos.quantity}</td>
                    <td className="py-3 text-slate-300">{pos.entry_price?.toLocaleString()}</td>
                    <td className="py-3 text-slate-300">{pos.current_price?.toLocaleString()}</td>
                    <td className={`py-3 font-semibold ${pnlPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
                    </td>
                    <td className="py-3 text-slate-400">{pos.days_held || 0}d</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
```

### Step 3: Create Orders Table Component
Create `vn-quant-web/src/components/orders-table.jsx`:

```javascript
import { useState, useEffect } from 'react'

const API_URL = '/api'

export function OrdersTable() {
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchOrders = async () => {
    try {
      const res = await fetch(`${API_URL}/orders`)
      const data = await res.json()
      setOrders(data.orders || [])
      setLoading(false)
    } catch (err) {
      console.error('Failed to fetch orders:', err)
    }
  }

  useEffect(() => {
    fetchOrders()
    const interval = setInterval(fetchOrders, 10000) // Refresh every 10s
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="glass-panel p-6 rounded-xl">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-white">Order History</h2>
        <button
          onClick={fetchOrders}
          className="text-slate-400 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined">refresh</span>
        </button>
      </div>

      {loading ? (
        <div className="text-slate-400 text-center py-8">Loading orders...</div>
      ) : orders.length === 0 ? (
        <div className="text-slate-500 text-center py-8">No orders yet</div>
      ) : (
        <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0 bg-[#0a0e17]">
              <tr className="text-left text-slate-400 text-sm border-b border-white/10">
                <th className="pb-3">Symbol</th>
                <th className="pb-3">Side</th>
                <th className="pb-3">Price</th>
                <th className="pb-3">Qty</th>
                <th className="pb-3">Status</th>
                <th className="pb-3">Time</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order, idx) => (
                <tr key={idx} className="border-b border-white/5">
                  <td className="py-3 font-semibold text-white">{order.symbol}</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      order.side === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                    }`}>
                      {order.side}
                    </span>
                  </td>
                  <td className="py-3 text-slate-300">{order.price?.toLocaleString()}</td>
                  <td className="py-3 text-slate-300">{order.quantity}</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs ${
                      order.status === 'FILLED' ? 'bg-emerald-500/20 text-emerald-400' :
                      order.status === 'PENDING' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-slate-500/20 text-slate-400'
                    }`}>
                      {order.status}
                    </span>
                  </td>
                  <td className="py-3 text-slate-400 text-sm">
                    {new Date(order.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
```

### Step 4: Create Main Trading View
Create `vn-quant-web/src/components/trading-view.jsx`:

```javascript
import { PortfolioStats } from './portfolio-stats'
import { PositionsTable } from './positions-table'
import { OrdersTable } from './orders-table'
import { WebSocketFeed } from './websocket-feed'

export function TradingView() {
  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Autonomous Trading</h1>
        <p className="text-slate-400">Real-time portfolio monitoring and order tracking</p>
      </header>

      {/* Portfolio Stats */}
      <section>
        <PortfolioStats />
      </section>

      {/* Positions and Orders Grid */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PositionsTable />
        <OrdersTable />
      </section>

      {/* Live Feed (from Phase 02) */}
      <section>
        <WebSocketFeed />
      </section>
    </div>
  )
}
```

### Step 5: Replace iframe in App.jsx
Modify `vn-quant-web/src/App.jsx`:

```javascript
// Add import
import { TradingView } from './components/trading-view'

// In renderView() function, replace iframe line:
case 'trading': return <TradingView />
// Remove: case 'trading': return <iframe src="/autonomous" ... />
```

### Step 6: Test Trading View
1. Start backend: `python run_autonomous_paper_trading.py`
2. Start frontend: `cd vn-quant-web && npm run dev`
3. Navigate to Trading view in sidebar
4. Verify:
   - Portfolio stats load
   - Positions table shows current positions
   - Orders table shows order history
   - WebSocket feed shows live events
   - Auto-refresh works every 5s
   - Manual refresh buttons work

## Todo List
- [ ] Create portfolio-stats.jsx component
- [ ] Create positions-table.jsx component
- [ ] Create orders-table.jsx component
- [ ] Create trading-view.jsx main component
- [ ] Import all components into App.jsx
- [ ] Replace iframe with <TradingView />
- [ ] Test: Portfolio stats load and refresh
- [ ] Test: Positions table displays correctly
- [ ] Test: Orders table displays correctly
- [ ] Test: WebSocket feed integrates seamlessly
- [ ] Test: Responsive layout on mobile/tablet
- [ ] Remove old /autonomous HTML endpoint (optional, backend cleanup)

## Success Criteria
- [ ] No iframe in App.jsx
- [ ] Trading view uses native React components
- [ ] All 3 backend endpoints integrated: /api/status, /api/positions, /api/orders
- [ ] Auto-refresh works every 5s
- [ ] Manual refresh buttons work
- [ ] WebSocket feed integrated in same view
- [ ] Consistent glass-morphism styling
- [ ] No CORS errors
- [ ] Mobile responsive layout works

## Risk Assessment
- **Risk:** Data refresh conflicts with WebSocket updates → Mitigated: Different update intervals
- **Risk:** Large order history slows render → Mitigated: Max height with scroll, limit to 100 recent
- **Risk:** Missing data fields from backend → Mitigated: Optional chaining, fallbacks

## Security Considerations
- Read-only endpoints (no auth needed)
- If Phase 01 adds auth, update fetch calls to include X-API-Key header

## Next Steps
After completing this phase:
1. Update plan.md with completion status
2. Test all P0 phases together (auth + WebSocket + trading view)
3. Proceed to Phase 04 (Schema mismatches)
4. Consider removing /autonomous HTML endpoint from backend (cleanup)
