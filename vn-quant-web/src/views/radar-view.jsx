import { useState, useEffect } from 'react'
import { API_URL } from '../utils/constants'

export function RadarView({ onAgentChat }) {
  const [agents, setAgents] = useState([])

  useEffect(() => {
    fetch(`${API_URL}/agents/status`).then(r => r.json()).then(d => setAgents(d.agents || []))
  }, [])

  // Function to extract symbol from signal message (e.g. "Phát hiện cơ hội MWG, HPG" -> "MWG")
  const getSymbolFromSignal = (signal) => {
    if (!signal) return null;
    const match = signal.match(/[A-Z]{3}/);
    return match ? match[0] : null;
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-white flex items-center gap-2"><span className="material-symbols-outlined text-accent-cyan">radar</span> Agent Radar (MADDPG)</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {agents.map((agent, i) => {
          const symbol = getSymbolFromSignal(agent.last_signal);
          return (
            <div key={i}
              className="glass-panel p-5 rounded-xl border-l-4 border-l-primary hover:border-l-accent-cyan transition-all cursor-pointer group relative"
              onClick={() => symbol && onAgentChat(symbol)}
            >
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-3">
                  <div className="size-10 rounded-full bg-primary/20 flex items-center justify-center group-hover:bg-primary transition-colors"><span className="material-symbols-outlined text-primary group-hover:text-white">smart_toy</span></div>
                  <div>
                    <h3 className="font-bold text-white">{agent.name || agent.agent}</h3>
                    <span className={`text-xs flex items-center gap-1 ${agent.status === 'online' ? 'text-emerald-400' : 'text-slate-500'}`}>
                      <span className={`size-1.5 rounded-full ${agent.status === 'online' ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`}></span>
                      {agent.status || agent.role}
                    </span>
                  </div>
                </div>
                <span className="text-2xl font-bold text-white">{((agent.accuracy || agent.win_rate || 0) * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden mb-2"><div className="h-full bg-primary rounded-full" style={{ width: `${(agent.accuracy || agent.win_rate || 0.5) * 100}%` }}></div></div>
              <div className="text-sm text-slate-400 group-hover:text-white transition-colors">
                {agent.last_signal || `Role: ${agent.role}`}
                {symbol && <span className="block text-xs text-accent-cyan mt-1 font-bold">CLICK TO ANALYZE {symbol}</span>}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
