# Phase 05: Integrate Core Endpoints

**Priority:** P1 (HIGH)
**Effort:** 1.5h
**Status:** Pending

## Context Links
- Backend: `app/api/routers/trading.py` - /api/orders, /api/positions, /api/trades, /api/discussions already exist
- Frontend: `vn-quant-web/src/App.jsx` - Phase 03 integrated orders/positions, but trades/discussions unused
- Audit: "Integrate unused core endpoints into React SPA"

## Overview
Backend has 4 core endpoints fully implemented but not wired to FE:
- `/api/trades` - Trade history with P&L calculations ✅ Partially used in Phase 03
- `/api/discussions` - Agent discussion history (6-agent consensus records)
- `/api/discussion/{id}` - Single discussion details
- `/api/order/{order_id}/discussion` - Discussion that led to specific order

**Goal:** Wire these endpoints into UI to show full audit trail of autonomous decisions.

## Key Insights
- Trades endpoint already used in Phase 03 (OrdersTable) → Skip if already done
- Discussions endpoint provides transparency: see WHY agents made each trade
- Link orders to discussions for full traceability
- Display agent votes, reasoning, confidence scores

## Requirements

### Functional
- Display recent discussions (last 20)
- Show discussion details: symbol, timestamp, agents, verdict, confidence
- Link orders to discussions (click order → see discussion)
- Display agent breakdown: Bull, Bear, Alex, Scout, RiskDoctor, Chief
- Show final consensus vote and reasoning

### Non-Functional
- Load time <500ms for discussions list
- Click-through latency <200ms
- Paginate if >100 discussions
- Responsive layout

## Architecture

```
┌─────────────────────────────────────────┐
│ DiscussionsView Component               │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ Discussions List                    │ │
│ │ /api/discussions → recent 20        │ │
│ │ Click → show detail modal           │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Discussion Detail Modal             │ │
│ │ /api/discussion/{id}                │ │
│ │ - Agents table with votes           │ │
│ │ - Chief verdict                     │ │
│ │ - Confidence score                  │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ OrdersTable Enhancement                 │
├─────────────────────────────────────────┤
│ - Add "View Discussion" button         │
│ - Click → fetch /api/order/{id}/discussion │
│ - Show modal with agent reasoning      │
└─────────────────────────────────────────┘
```

## Related Code Files

**Frontend (Create):**
- `vn-quant-web/src/components/discussions-view.jsx` - Main discussions view (~120 lines)
- `vn-quant-web/src/components/discussion-detail-modal.jsx` - Detail modal (~100 lines)
- `vn-quant-web/src/components/agent-votes-table.jsx` - Agent breakdown table (~60 lines)

**Frontend (Modify):**
- `vn-quant-web/src/components/orders-table.jsx` - Add "View Discussion" button
- `vn-quant-web/src/App.jsx` - Add discussions view to sidebar/navigation

**Backend (No changes):**
- Endpoints already exist in `app/api/routers/trading.py`

## Implementation Steps

### Step 1: Create Agent Votes Table Component
Create `vn-quant-web/src/components/agent-votes-table.jsx`:

```javascript
export function AgentVotesTable({ agents }) {
  const getAgentIcon = (name) => {
    const icons = {
      Bull: '🐂',
      Bear: '🐻',
      Alex: '📊',
      Scout: '🔍',
      RiskDoctor: '💊',
      Chief: '🎖'
    }
    return icons[name] || '🤖'
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="text-left text-slate-400 text-sm border-b border-white/10">
            <th className="pb-3">Agent</th>
            <th className="pb-3">Vote</th>
            <th className="pb-3">Confidence</th>
            <th className="pb-3">Reasoning</th>
          </tr>
        </thead>
        <tbody>
          {agents?.map((agent, idx) => (
            <tr key={idx} className="border-b border-white/5">
              <td className="py-3 flex items-center gap-2">
                <span className="text-xl">{getAgentIcon(agent.name)}</span>
                <span className="font-semibold text-white">{agent.name}</span>
              </td>
              <td className="py-3">
                <span className={`px-2 py-1 rounded text-xs font-semibold ${
                  agent.vote === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' :
                  agent.vote === 'SELL' ? 'bg-red-500/20 text-red-400' :
                  'bg-slate-500/20 text-slate-400'
                }`}>
                  {agent.vote}
                </span>
              </td>
              <td className="py-3 text-slate-300">
                {(agent.confidence * 100).toFixed(0)}%
              </td>
              <td className="py-3 text-slate-400 text-sm max-w-md truncate">
                {agent.reasoning}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

### Step 2: Create Discussion Detail Modal
Create `vn-quant-web/src/components/discussion-detail-modal.jsx`:

```javascript
import { AgentVotesTable } from './agent-votes-table'

export function DiscussionDetailModal({ discussionId, onClose }) {
  const [discussion, setDiscussion] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!discussionId) return

    fetch(`/api/discussion/${discussionId}`)
      .then(r => r.json())
      .then(data => {
        setDiscussion(data)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to fetch discussion:', err)
        setLoading(false)
      })
  }, [discussionId])

  if (!discussionId) return null

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="glass-panel rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-[#0a0e17]/90 backdrop-blur-md p-6 border-b border-white/10">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Agent Discussion</h2>
              <p className="text-slate-400">
                {discussion?.symbol} • {new Date(discussion?.timestamp).toLocaleString()}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white transition-colors"
            >
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {loading ? (
            <div className="text-center text-slate-400 py-8">Loading discussion...</div>
          ) : discussion ? (
            <>
              {/* Chief Verdict */}
              <section className="glass-panel p-5 rounded-lg border-l-4 border-l-emerald-500">
                <h3 className="font-bold text-white mb-2 flex items-center gap-2">
                  <span className="text-xl">🎖</span> Chief's Verdict
                </h3>
                <p className="text-slate-300 mb-3">{discussion.verdict?.decision}</p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-slate-400">Action: </span>
                    <span className={`font-semibold ${
                      discussion.verdict?.action === 'BUY' ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {discussion.verdict?.action}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400">Confidence: </span>
                    <span className="font-semibold text-white">
                      {(discussion.verdict?.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </section>

              {/* Agent Votes */}
              <section>
                <h3 className="font-bold text-white mb-4">Agent Breakdown</h3>
                <AgentVotesTable agents={discussion.agents} />
              </section>

              {/* Context */}
              {discussion.context && (
                <section className="glass-panel p-5 rounded-lg">
                  <h3 className="font-bold text-white mb-2">Context</h3>
                  <pre className="text-sm text-slate-400 whitespace-pre-wrap">
                    {JSON.stringify(discussion.context, null, 2)}
                  </pre>
                </section>
              )}
            </>
          ) : (
            <div className="text-center text-red-400 py-8">Discussion not found</div>
          )}
        </div>
      </div>
    </div>
  )
}
```

### Step 3: Create Discussions View Component
Create `vn-quant-web/src/components/discussions-view.jsx`:

```javascript
import { useState, useEffect } from 'react'
import { DiscussionDetailModal } from './discussion-detail-modal'

const API_URL = '/api'

export function DiscussionsView() {
  const [discussions, setDiscussions] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState(null)

  const fetchDiscussions = async () => {
    try {
      const res = await fetch(`${API_URL}/discussions`)
      const data = await res.json()
      setDiscussions(data.discussions || [])
      setLoading(false)
    } catch (err) {
      console.error('Failed to fetch discussions:', err)
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDiscussions()
  }, [])

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Agent Discussions</h1>
        <p className="text-slate-400">
          Full audit trail of AI agent consensus decisions
        </p>
      </header>

      <div className="glass-panel p-6 rounded-xl">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold text-white">Recent Discussions</h2>
          <button
            onClick={fetchDiscussions}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined">refresh</span>
          </button>
        </div>

        {loading ? (
          <div className="text-center text-slate-400 py-8">Loading...</div>
        ) : discussions.length === 0 ? (
          <div className="text-center text-slate-500 py-8">No discussions yet</div>
        ) : (
          <div className="space-y-3">
            {discussions.slice(0, 20).map((disc, idx) => (
              <div
                key={disc.discussion_id || idx}
                className="p-4 bg-black/20 rounded-lg border border-white/5 hover:border-white/20 transition-all cursor-pointer"
                onClick={() => setSelectedId(disc.discussion_id)}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-bold text-white text-lg">{disc.symbol}</h3>
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        disc.verdict?.action === 'BUY'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {disc.verdict?.action}
                      </span>
                      <span className="text-slate-400 text-sm">
                        Confidence: {(disc.verdict?.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-slate-400 text-sm line-clamp-2">
                      {disc.verdict?.decision}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-500">
                      {new Date(disc.timestamp).toLocaleDateString()}
                    </p>
                    <p className="text-xs text-slate-500">
                      {new Date(disc.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Discussion Detail Modal */}
      {selectedId && (
        <DiscussionDetailModal
          discussionId={selectedId}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  )
}
```

### Step 4: Add Discussion Link to Orders Table
Modify `vn-quant-web/src/components/orders-table.jsx`:

```javascript
import { useState } from 'react'
import { DiscussionDetailModal } from './discussion-detail-modal'

export function OrdersTable() {
  // ... existing state ...
  const [selectedDiscussionId, setSelectedDiscussionId] = useState(null)

  const handleViewDiscussion = async (orderId) => {
    try {
      const res = await fetch(`/api/order/${orderId}/discussion`)
      const data = await res.json()
      if (data.discussion_id) {
        setSelectedDiscussionId(data.discussion_id)
      } else {
        alert('No discussion found for this order')
      }
    } catch (err) {
      console.error('Failed to fetch order discussion:', err)
    }
  }

  return (
    <>
      <div className="glass-panel p-6 rounded-xl">
        {/* ... existing code ... */}
        <tbody>
          {orders.map((order, idx) => (
            <tr key={idx} className="border-b border-white/5">
              {/* ... existing columns ... */}
              <td className="py-3">
                <button
                  onClick={() => handleViewDiscussion(order.order_id)}
                  className="text-xs text-blue-400 hover:text-blue-300 underline"
                >
                  View Discussion
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </div>

      {/* Discussion Modal */}
      {selectedDiscussionId && (
        <DiscussionDetailModal
          discussionId={selectedDiscussionId}
          onClose={() => setSelectedDiscussionId(null)}
        />
      )}
    </>
  )
}
```

### Step 5: Add to Navigation
Modify `vn-quant-web/src/App.jsx`:

```javascript
// Add import
import { DiscussionsView } from './components/discussions-view'

// In Sidebar component, add new menu item:
const menuItems = [
  // ... existing items ...
  { id: 'discussions', label: 'Discussions', icon: 'forum' },
]

// In renderView function:
case 'discussions': return <DiscussionsView />
```

### Step 6: Test Integration
1. Start backend with some trade history
2. Navigate to Discussions view
3. Verify discussions list loads
4. Click a discussion → verify modal shows agent votes
5. Navigate to Trading view → Orders table
6. Click "View Discussion" on an order
7. Verify modal shows correct discussion

## Todo List
- [ ] Create agent-votes-table.jsx component
- [ ] Create discussion-detail-modal.jsx component
- [ ] Create discussions-view.jsx component
- [ ] Add "View Discussion" button to OrdersTable
- [ ] Wire order click to /api/order/{id}/discussion
- [ ] Add discussions menu item to sidebar
- [ ] Import DiscussionsView in App.jsx
- [ ] Test discussions list loads
- [ ] Test discussion detail modal
- [ ] Test order → discussion linking
- [ ] Test modal close/open UX

## Success Criteria
- [ ] Discussions view displays recent 20 discussions
- [ ] Click discussion opens modal with full details
- [ ] Agent votes table shows all 6 agents with reasoning
- [ ] Chief verdict displayed prominently
- [ ] Order "View Discussion" button works
- [ ] Modal closes properly on X button or background click
- [ ] No console errors
- [ ] Responsive layout works on mobile

## Risk Assessment
- **Risk:** Large discussion payloads slow modal → Mitigated: Fetch on-demand, not upfront
- **Risk:** Order has no linked discussion → Mitigated: Show "No discussion found" message
- **Risk:** Modal UX conflicts with existing modals → Mitigated: Use z-index layers properly

## Security Considerations
- Read-only endpoints (no auth needed)
- If Phase 01 adds auth, update fetch calls accordingly

## Next Steps
After completing this phase:
1. Update plan.md with completion status
2. Proceed to Phase 06 (Production deployment config)
3. Consider adding discussion search/filter (future enhancement)
4. Consider exporting discussions to CSV for analysis
