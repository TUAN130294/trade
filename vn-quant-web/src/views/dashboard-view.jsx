import { useState, useEffect } from 'react'
import { API_URL } from '../utils/constants'
import { WebSocketFeed } from '../components/websocket-feed'

export function DashboardView({ marketStatus, regime }) {
  const [smartSignals, setSmartSignals] = useState([])
  const [agentInfo, setAgentInfo] = useState(null)

  useEffect(() => {
    // Fetch smart signals on load
    fetch(`${API_URL}/market/smart-signals`)
      .then(r => r.ok ? r.json() : Promise.reject('API error'))
      .then(d => setSmartSignals(d.signals || []))
      .catch(e => console.error('Smart signals error:', e))
    // Fetch agent status
    fetch(`${API_URL}/agents/status`)
      .then(r => r.ok ? r.json() : Promise.reject('API error'))
      .then(d => setAgentInfo(d))
      .catch(e => console.error('Agent status error:', e))
  }, [])

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'HIGH': return 'border-red-500 bg-red-500/10 text-red-400'
      case 'MEDIUM': return 'border-amber-500 bg-amber-500/10 text-amber-400'
      case 'WARNING': return 'border-amber-500 bg-amber-500/10 text-amber-400'
      case 'INFO': return 'border-blue-500 bg-blue-500/10 text-blue-400'
      default: return 'border-slate-500 bg-slate-500/10 text-slate-400'
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-xl flex flex-col justify-between group hover:border-primary/40 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">VN-INDEX</p>
              <h3 className="text-2xl font-bold text-white tracking-tight">{marketStatus?.vnindex || 'Loading...'}</h3>
            </div>
            <div className="p-2 bg-primary/10 rounded-lg text-primary"><span className="material-symbols-outlined text-[20px]">show_chart</span></div>
          </div>
          <div className="mt-4 flex items-center gap-2">
            <span className={`${marketStatus?.change >= 0 ? 'text-emerald-400' : 'text-red-400'} text-sm font-bold flex items-center`}>
              <span className="material-symbols-outlined text-[16px]">{marketStatus?.change >= 0 ? 'trending_up' : 'trending_down'}</span>
              {marketStatus?.change > 0 ? '+' : ''}{marketStatus?.change} ({marketStatus?.change_pct}%)
            </span>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-xl flex flex-col justify-between group hover:border-primary/40 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Market Regime</p>
              <h3 className="text-2xl font-bold text-white tracking-tight">{regime?.market_regime || '---'}</h3>
            </div>
            <div className="p-2 bg-amber-500/10 rounded-lg text-amber-500"><span className="material-symbols-outlined text-[20px]">psychology</span></div>
          </div>
          <div className="mt-4">
            <span className="text-slate-400 text-xs">Confidence: {((regime?.confidence || 0) * 100).toFixed(0)}%</span>
          </div>
        </div>

        <div className="glass-panel p-5 rounded-xl flex flex-col justify-between group hover:border-primary/40 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Circuit Breaker</p>
              <h3 className="text-2xl font-bold text-white tracking-tight">{smartSignals.find(s => s.type === 'CIRCUIT_BREAKER')?.action || 'NORMAL'}</h3>
            </div>
            <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-500"><span className="material-symbols-outlined text-[20px]">verified_user</span></div>
          </div>
          <div className="mt-4"><span className="text-emerald-400 text-xs font-bold">{smartSignals.find(s => s.type === 'CIRCUIT_BREAKER')?.description || 'Checking...'}</span></div>
        </div>

        <div className="glass-panel p-5 rounded-xl flex flex-col justify-between group hover:border-primary/40 transition-all">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Active Agents</p>
              <h3 className="text-2xl font-bold text-white tracking-tight">{agentInfo?.online_count ?? '...'} <span className="text-lg text-slate-500 font-normal">/ {agentInfo?.total_agents ?? '...'}</span></h3>
            </div>
            <div className="p-2 bg-accent-cyan/10 rounded-lg text-accent-cyan"><span className="material-symbols-outlined text-[20px]">smart_toy</span></div>
          </div>
          <div className="mt-4"><span className={`text-xs font-bold ${agentInfo?.online_count === agentInfo?.total_agents ? 'text-emerald-400' : 'text-amber-400'}`}>{agentInfo ? (agentInfo.online_count === agentInfo.total_agents ? 'ALL ONLINE' : 'PARTIAL') : 'LOADING...'}</span></div>
        </div>
      </section>

      {/* Smart Signals Panel */}
      <section className="glass-panel p-6 rounded-xl border-l-4 border-l-amber-500">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-amber-400">notifications_active</span>
          Tín Hiệu Thị Trường
          <span className="text-xs px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full">LIVE</span>
        </h2>

        {smartSignals.length === 0 ? (
          <p className="text-slate-400 text-sm">Đang tải tín hiệu...</p>
        ) : (
          <div className="space-y-3">
            {smartSignals.map((signal, idx) => (
              <div key={idx} className={`p-4 rounded-lg border-l-4 ${getSeverityColor(signal.severity)}`}>
                <div className="flex justify-between items-start mb-2">
                  <span className="font-bold text-white">{signal.name}</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${getSeverityColor(signal.severity)}`}>
                    {signal.severity}
                  </span>
                </div>
                <p className="text-slate-300 text-sm mb-2">{signal.description}</p>
                <p className="text-xs text-slate-400">
                  <span className="font-medium text-primary">→ {signal.action}</span>
                </p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="glass-panel p-6 rounded-xl">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-primary">recommend</span> Recommended Strategies
        </h2>
        <div className="flex flex-wrap gap-2">
          {(regime?.recommended_strategies || ['Loading...']).map((s, i) => (
            <span key={i} className="px-4 py-2 bg-primary/10 text-primary rounded-full text-sm font-medium border border-primary/20">{s}</span>
          ))}
        </div>
      </section>

      <section>
        <WebSocketFeed />
      </section>
    </div>
  )
}
