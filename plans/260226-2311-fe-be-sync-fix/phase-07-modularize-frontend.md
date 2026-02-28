# Phase 07: Modularize Frontend

**Priority:** P2 (MEDIUM)
**Effort:** 1.5h
**Status:** Pending

## Context Links
- Frontend: `vn-quant-web/src/App.jsx` (1156 lines, 14 components in single file)
- Audit: "Split App.jsx monolith - Extract 14 components into separate files"
- Development Rules: Keep files under 200 lines

## Overview
App.jsx is a 1156-line monolith containing:
- 14 view components (Dashboard, Analysis, Radar, Command, etc.)
- Sidebar navigation
- Multiple utility functions
- All in one file

**Goal:** Extract components into separate files following React best practices.

## Key Insights
- Split into views/ and components/ directories
- Each view component in own file (~50-150 lines)
- Shared components (Sidebar) in components/
- Keep App.jsx as orchestrator (~80 lines)
- Use kebab-case for file names (React components still PascalCase)
- Maintain existing functionality, zero breaking changes

## Requirements

### Functional
- All existing features work identically
- No prop drilling issues
- Clean import structure
- Consistent naming conventions

### Non-Functional
- Each file under 200 lines
- Clear separation of concerns
- Easy to navigate for new developers
- Fast HMR (Hot Module Replacement) in dev

## Architecture

### Current Structure
```
vn-quant-web/src/
├── App.jsx (1156 lines) ❌
├── main.jsx
└── assets/
```

### Target Structure
```
vn-quant-web/src/
├── App.jsx (~80 lines) ✅
├── main.jsx
├── components/
│   ├── sidebar.jsx
│   ├── websocket-feed.jsx (from Phase 02)
│   ├── trading-view.jsx (from Phase 03)
│   ├── portfolio-stats.jsx (from Phase 03)
│   ├── positions-table.jsx (from Phase 03)
│   ├── orders-table.jsx (from Phase 03)
│   ├── discussions-view.jsx (from Phase 05)
│   ├── discussion-detail-modal.jsx (from Phase 05)
│   ├── agent-votes-table.jsx (from Phase 05)
│   └── stock-chart.jsx
├── views/
│   ├── dashboard-view.jsx
│   ├── analysis-view.jsx
│   ├── radar-view.jsx
│   ├── command-view.jsx
│   ├── backtest-view.jsx
│   ├── predict-view.jsx
│   ├── data-hub-view.jsx
│   └── news-intel-view.jsx
├── hooks/
│   └── use-websocket.js (from Phase 02)
└── utils/
    └── formatters.js
```

## Related Code Files

**Frontend (Create):**
- `vn-quant-web/src/components/sidebar.jsx`
- `vn-quant-web/src/components/stock-chart.jsx`
- `vn-quant-web/src/views/dashboard-view.jsx`
- `vn-quant-web/src/views/analysis-view.jsx`
- `vn-quant-web/src/views/radar-view.jsx`
- `vn-quant-web/src/views/command-view.jsx`
- `vn-quant-web/src/views/backtest-view.jsx`
- `vn-quant-web/src/views/predict-view.jsx`
- `vn-quant-web/src/views/data-hub-view.jsx`
- `vn-quant-web/src/views/news-intel-view.jsx`
- `vn-quant-web/src/utils/formatters.js`

**Frontend (Modify):**
- `vn-quant-web/src/App.jsx` - Slim down to ~80 lines, import views

## Implementation Steps

### Step 1: Create Utility Functions
Create `vn-quant-web/src/utils/formatters.js`:

```javascript
// Currency formatter
export const fmtMoney = (n) => new Intl.NumberFormat('vi-VN').format(n)

// Date formatter
export const fmtDate = (dateStr) => {
  return new Date(dateStr).toLocaleDateString('vi-VN')
}

// Time formatter
export const fmtTime = (dateStr) => {
  return new Date(dateStr).toLocaleTimeString('vi-VN')
}

// Percentage formatter
export const fmtPercent = (n, decimals = 2) => {
  return `${n >= 0 ? '+' : ''}${n.toFixed(decimals)}%`
}
```

### Step 2: Extract Sidebar Component
Create `vn-quant-web/src/components/sidebar.jsx`:

```javascript
export function Sidebar({ activeView, setView }) {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'dashboard' },
    { id: 'analysis', label: 'Analysis', icon: 'analytics' },
    { id: 'radar', label: 'Radar', icon: 'radar' },
    { id: 'command', label: 'Command', icon: 'terminal' },
    { id: 'trading', label: 'Trading', icon: 'currency_exchange' },
    { id: 'discussions', label: 'Discussions', icon: 'forum' },
    { id: 'backtest', label: 'Backtest', icon: 'history' },
    { id: 'predict', label: 'Predict', icon: 'psychology' },
    { id: 'data', label: 'Data Hub', icon: 'database' },
    { id: 'news', label: 'News Intel', icon: 'feed' },
  ]

  return (
    <aside className="hidden md:flex flex-col w-64 border-r border-white/5 bg-[#0a0e17]/50 backdrop-blur-sm">
      <div className="p-6 border-b border-white/5">
        <h1 className="text-2xl font-bold text-white">VN-QUANT</h1>
        <p className="text-xs text-slate-500 mt-1">Autonomous Trading</p>
      </div>

      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setView(item.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
              activeView === item.id
                ? 'bg-primary/20 text-primary border border-primary/30'
                : 'text-slate-400 hover:bg-white/5 hover:text-white'
            }`}
          >
            <span className="material-symbols-outlined text-[20px]">{item.icon}</span>
            <span className="font-medium">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-white/5">
        <div className="flex items-center gap-3 px-4 py-3 bg-white/5 rounded-lg">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-accent-cyan flex items-center justify-center text-white font-bold text-sm">
            VQ
          </div>
          <div>
            <p className="text-sm font-semibold text-white">VN-Quant</p>
            <p className="text-xs text-slate-500">v4.0.0</p>
          </div>
        </div>
      </div>
    </aside>
  )
}
```

### Step 3: Extract StockChart Component
Create `vn-quant-web/src/components/stock-chart.jsx`:

```javascript
import { useEffect, useRef } from 'react'
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts'

export function StockChart({ data }) {
  const chartContainerRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!data || data.length === 0 || !chartContainerRef.current) return

    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    try {
      const chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: '#94a3b8'
        },
        grid: {
          vertLines: { color: 'rgba(255,255,255,0.05)' },
          horzLines: { color: 'rgba(255,255,255,0.05)' }
        },
        width: chartContainerRef.current.clientWidth || 600,
        height: 400,
      })
      chartRef.current = chart

      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#0bda5e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#0bda5e',
        wickDownColor: '#ef4444',
      })

      const chartData = data
        .filter(d => d.date && d.open && d.high && d.low && d.close)
        .map(d => ({
          time: String(d.date).split('T')[0],
          open: Number(d.open),
          high: Number(d.high),
          low: Number(d.low),
          close: Number(d.close)
        }))
        .sort((a, b) => a.time.localeCompare(b.time))

      if (chartData.length > 0) {
        candlestickSeries.setData(chartData)
        chart.timeScale().fitContent()
      }

      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth })
        }
      }
      window.addEventListener('resize', handleResize)

      return () => {
        window.removeEventListener('resize', handleResize)
        if (chartRef.current) {
          chartRef.current.remove()
          chartRef.current = null
        }
      }
    } catch (err) {
      console.error('Chart error:', err)
    }
  }, [data])

  return <div ref={chartContainerRef} className="w-full h-[400px] min-h-[400px]" />
}
```

### Step 4: Extract View Components
Create each view in `vn-quant-web/src/views/`:

**Example: dashboard-view.jsx**
```javascript
import { useEffect, useState } from 'react'
import { WebSocketFeed } from '../components/websocket-feed'

const API_URL = '/api'

export function DashboardView() {
  const [marketStatus, setMarketStatus] = useState(null)
  const [regime, setRegime] = useState(null)
  const [smartSignals, setSmartSignals] = useState([])
  const [agentInfo, setAgentInfo] = useState(null)

  useEffect(() => {
    fetch(`${API_URL}/market/status`).then(r => r.json()).then(setMarketStatus)
    fetch(`${API_URL}/market/regime`).then(r => r.json()).then(setRegime)
    fetch(`${API_URL}/market/smart-signals`).then(r => r.json()).then(d => setSmartSignals(d.signals || []))
    fetch(`${API_URL}/agents/status`).then(r => r.json()).then(setAgentInfo)
  }, [])

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      {/* Market Stats Cards */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* VN-Index Card */}
        <div className="glass-panel p-5 rounded-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">VN-Index</p>
              <h3 className="text-2xl font-bold text-white">{marketStatus?.vnindex || '---'}</h3>
            </div>
            <div className="p-2 bg-primary/10 rounded-lg text-primary">
              <span className="material-symbols-outlined text-[20px]">trending_up</span>
            </div>
          </div>
          <div className="mt-4 flex items-center gap-2">
            <span className={`${marketStatus?.change >= 0 ? 'text-emerald-400' : 'text-red-400'} text-sm font-bold`}>
              {marketStatus?.change > 0 ? '+' : ''}{marketStatus?.change} ({marketStatus?.change_pct}%)
            </span>
          </div>
        </div>

        {/* Market Regime Card */}
        <div className="glass-panel p-5 rounded-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Market Regime</p>
              <h3 className="text-2xl font-bold text-white">{regime?.market_regime || '---'}</h3>
            </div>
            <div className="p-2 bg-amber-500/10 rounded-lg text-amber-500">
              <span className="material-symbols-outlined text-[20px]">psychology</span>
            </div>
          </div>
          <div className="mt-4">
            <span className="text-slate-400 text-xs">Conf: {((regime?.confidence || 0) * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Circuit Breaker Card */}
        <div className="glass-panel p-5 rounded-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Circuit Breaker</p>
              <h3 className="text-2xl font-bold text-white">
                {smartSignals.find(s => s.type === 'CIRCUIT_BREAKER')?.action || 'NORMAL'}
              </h3>
            </div>
            <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500">
              <span className="material-symbols-outlined text-[20px]">verified_user</span>
            </div>
          </div>
        </div>

        {/* Active Agents Card */}
        <div className="glass-panel p-5 rounded-xl">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Active Agents</p>
              <h3 className="text-2xl font-bold text-white">
                {agentInfo?.online_count ?? '...'} <span className="text-lg text-slate-500">/ {agentInfo?.total_agents ?? '...'}</span>
              </h3>
            </div>
            <div className="p-2 bg-accent-cyan/10 rounded-lg text-accent-cyan">
              <span className="material-symbols-outlined text-[20px]">smart_toy</span>
            </div>
          </div>
        </div>
      </section>

      {/* WebSocket Feed */}
      <section>
        <WebSocketFeed />
      </section>
    </div>
  )
}
```

Repeat for other views (analysis-view.jsx, radar-view.jsx, etc.) - extract from App.jsx.

### Step 5: Slim Down App.jsx
Modify `vn-quant-web/src/App.jsx`:

```javascript
import { useState } from 'react'
import { Sidebar } from './components/sidebar'
import { DashboardView } from './views/dashboard-view'
import { AnalysisView } from './views/analysis-view'
import { RadarView } from './views/radar-view'
import { CommandView } from './views/command-view'
import { TradingView } from './components/trading-view'
import { DiscussionsView } from './components/discussions-view'
import { BacktestView } from './views/backtest-view'
import { PredictView } from './views/predict-view'
import { DataHubView } from './views/data-hub-view'
import { NewsIntelView } from './views/news-intel-view'

function App() {
  const [activeView, setActiveView] = useState('dashboard')

  const renderView = () => {
    switch (activeView) {
      case 'dashboard': return <DashboardView />
      case 'analysis': return <AnalysisView />
      case 'radar': return <RadarView />
      case 'command': return <CommandView />
      case 'trading': return <TradingView />
      case 'discussions': return <DiscussionsView />
      case 'backtest': return <BacktestView />
      case 'predict': return <PredictView />
      case 'data': return <DataHubView />
      case 'news': return <NewsIntelView />
      default: return <DashboardView />
    }
  }

  return (
    <div className="flex w-full h-screen bg-[#0a0e17] text-slate-200 overflow-hidden font-sans selection:bg-primary/30">
      <div className="fixed inset-0 pointer-events-none bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 z-0"></div>
      <Sidebar activeView={activeView} setView={setActiveView} />
      <main className="flex-1 flex flex-col relative z-10 w-full max-w-full overflow-hidden">
        {renderView()}
      </main>
    </div>
  )
}

export default App
```

### Step 6: Verify All Imports Work
```bash
cd vn-quant-web
npm run dev
```

Check each view:
1. Dashboard → verify stats load
2. Analysis → verify chart renders
3. Radar → verify scanner works
4. Command → verify chat works
5. Trading → verify orders/positions load
6. Discussions → verify discussion list loads
7. Backtest → verify backtest form works
8. Predict → verify predictions load
9. Data Hub → verify data management works
10. News Intel → verify news feed works

## Todo List
- [ ] Create utils/formatters.js
- [ ] Create components/sidebar.jsx
- [ ] Create components/stock-chart.jsx
- [ ] Create views/dashboard-view.jsx
- [ ] Create views/analysis-view.jsx
- [ ] Create views/radar-view.jsx
- [ ] Create views/command-view.jsx
- [ ] Create views/backtest-view.jsx
- [ ] Create views/predict-view.jsx
- [ ] Create views/data-hub-view.jsx
- [ ] Create views/news-intel-view.jsx
- [ ] Update App.jsx with imports
- [ ] Test all views load correctly
- [ ] Verify no console errors
- [ ] Check HMR works for each file

## Success Criteria
- [ ] App.jsx under 100 lines
- [ ] Each component file under 200 lines
- [ ] All 10 views work identically to before
- [ ] No broken imports
- [ ] HMR works fast in dev mode
- [ ] Clear folder structure (components/, views/, hooks/, utils/)
- [ ] Zero functional regressions

## Risk Assessment
- **Risk:** Circular dependencies break imports → Mitigated: Clear component hierarchy
- **Risk:** Shared state breaks after split → Mitigated: Keep state in App.jsx, pass as props
- **Risk:** Missing imports cause runtime errors → Mitigated: Test each view after extraction

## Security Considerations
- No security impact (refactor only)

## Next Steps
After completing this phase:
1. Update plan.md with completion status
2. Proceed to Phase 08 (State persistence)
3. Consider adding PropTypes or TypeScript for type safety (future)
4. Update docs with new file structure
