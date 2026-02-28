import { useState, useEffect } from 'react'
import { API_URL } from '../utils/constants'

export function CommandView({ initialSymbol }) {
  const [symbol, setSymbol] = useState(initialSymbol || 'MWG')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [hasRunInitial, setHasRunInitial] = useState(false)

  // Use a ref to track if we should auto-run
  useEffect(() => {
    if (initialSymbol && !hasRunInitial) {
      setSymbol(initialSymbol);
      runAgentAnalysis(initialSymbol);
      setHasRunInitial(true);
    }
  }, [initialSymbol, hasRunInitial])

  const runAgentAnalysis = async (sym = symbol) => {
    setLoading(true)
    setMessages([])

    try {
      const resp = await fetch(`${API_URL}/agents/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: sym })
      })
      const data = await resp.json()
      setMessages(data.messages || [])
    } catch (err) {
      setMessages([{ sender: 'System', emoji: '❌', content: 'Error connecting to AI agents', type: 'ERROR' }])
    }

    setLoading(false)
  }

  const getTypeColor = (type) => {
    switch (type) {
      case 'SUCCESS': return 'bg-emerald-500/20 text-emerald-400'
      case 'WARNING': return 'bg-amber-500/20 text-amber-400'
      case 'ERROR': return 'bg-red-500/20 text-red-400'
      default: return 'bg-blue-500/20 text-blue-400'
    }
  }

  const getSenderColor = (sender) => {
    switch (sender) {
      case 'Scout': return 'border-l-cyan-400'
      case 'Alex': return 'border-l-blue-400'
      case 'Bull': return 'border-l-emerald-400'
      case 'Bear': return 'border-l-red-400'
      case 'Chief': return 'border-l-purple-400 bg-purple-500/5'
      default: return 'border-l-slate-400'
    }
  }

  return (
    <div className="flex-1 flex flex-col p-6 overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <span className="material-symbols-outlined text-accent-pink">forum</span>
          Agent Communication Feed
          <span className="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded-full animate-pulse">● LIVE AI</span>
        </h2>
      </div>

      {/* Control Panel */}
      <div className="glass-panel p-4 rounded-xl mb-4 flex gap-4 items-center">
        <input
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          className="bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white w-32 focus:outline-none focus:border-primary"
          placeholder="Symbol"
        />
        <button
          onClick={() => runAgentAnalysis(symbol)}
          disabled={loading}
          className={`flex-1 py-3 rounded-xl font-bold transition-all flex items-center justify-center gap-2
            ${loading ? 'bg-white/10 text-slate-400 cursor-not-allowed' : 'bg-gradient-to-r from-primary to-purple-600 text-white hover:shadow-lg'}`}
        >
          {loading ? (
            <>
              <span className="animate-spin">⟳</span> Đang phân tích với AI...
            </>
          ) : (
            <>
              <span className="material-symbols-outlined">rocket_launch</span>
              PHÂN TÍCH VỚI AI (5 Agents)
            </>
          )}
        </button>
      </div>

      {/* Agent Stats */}
      <div className="grid grid-cols-5 gap-2 mb-4">
        {[
          { name: 'Scout', emoji: '🔭', color: 'text-cyan-400' },
          { name: 'Alex', emoji: '📊', color: 'text-blue-400' },
          { name: 'Bull', emoji: '🐂', color: 'text-emerald-400' },
          { name: 'Bear', emoji: '🐻', color: 'text-red-400' },
          { name: 'Chief', emoji: '👔', color: 'text-purple-400' },
        ].map(agent => (
          <div key={agent.name} className="glass-panel p-3 rounded-lg text-center">
            <span className="text-2xl">{agent.emoji}</span>
            <p className={`text-sm font-bold ${agent.color}`}>{agent.name}</p>
          </div>
        ))}
      </div>

      {/* Message Feed */}
      <div className="flex-1 glass-panel rounded-xl p-4 overflow-y-auto space-y-3">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-500">
            <div className="text-center">
              <span className="material-symbols-outlined text-6xl mb-2">smart_toy</span>
              <p>Nhập mã cổ phiếu và click <strong>PHÂN TÍCH VỚI AI</strong></p>
              <p className="text-sm">5 AI Agents sẽ thảo luận và đưa ra quyết định</p>
            </div>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`p-4 rounded-xl border-l-4 ${getSenderColor(msg.sender)} bg-white/5 animate-fade-in`}>
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xl">{msg.emoji}</span>
                  <span className="font-bold text-white">{msg.sender}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">{msg.time}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${getTypeColor(msg.type)}`}>{msg.type}</span>
                </div>
              </div>
              <div className="text-slate-200 leading-relaxed whitespace-pre-wrap">{msg.content}</div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
