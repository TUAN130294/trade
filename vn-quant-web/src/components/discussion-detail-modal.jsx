import { useState, useEffect } from 'react'
import { AgentVotesTable } from './agent-votes-table'

export function DiscussionDetailModal({ discussionId, apiUrl, onClose }) {
  const [discussion, setDiscussion] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!discussionId) return

    fetch(`${apiUrl}/discussion/${discussionId}`)
      .then(res => res.json())
      .then(data => {
        setDiscussion(data)
        setLoading(false)
      })
      .catch(err => {
        console.error('Discussion fetch error:', err)
        setLoading(false)
      })
  }, [discussionId, apiUrl])

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'BUY': return 'text-green-400 bg-green-500/10 border-green-500'
      case 'SELL': return 'text-red-400 bg-red-500/10 border-red-500'
      case 'HOLD': return 'text-amber-400 bg-amber-500/10 border-amber-500'
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500'
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="glass-panel p-6 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-start mb-6">
          <h2 className="text-2xl font-bold text-white">Discussion Details</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
          >
            <span className="material-symbols-outlined text-slate-400">close</span>
          </button>
        </div>

        {loading ? (
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-slate-700/50 rounded w-1/3"></div>
            <div className="h-32 bg-slate-700/50 rounded"></div>
          </div>
        ) : discussion ? (
          <div className="space-y-6">
            {/* Header Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass-card p-4 rounded-lg">
                <p className="text-slate-400 text-xs mb-1">Symbol</p>
                <p className="text-white font-bold text-lg">{discussion.symbol}</p>
              </div>
              <div className="glass-card p-4 rounded-lg">
                <p className="text-slate-400 text-xs mb-1">Final Verdict</p>
                <span className={`px-3 py-1 rounded font-bold ${getVerdictColor(discussion.verdict)}`}>
                  {discussion.verdict}
                </span>
              </div>
              <div className="glass-card p-4 rounded-lg">
                <p className="text-slate-400 text-xs mb-1">Timestamp</p>
                <p className="text-slate-300 text-sm">
                  {new Date(discussion.timestamp).toLocaleString()}
                </p>
              </div>
            </div>

            {/* Reasoning */}
            {discussion.reasoning && (
              <div className="glass-card p-4 rounded-lg">
                <h3 className="text-white font-semibold mb-2">Consensus Reasoning</h3>
                <p className="text-slate-300 text-sm leading-relaxed">{discussion.reasoning}</p>
              </div>
            )}

            {/* Agent Votes */}
            <div className="glass-card p-4 rounded-lg">
              <h3 className="text-white font-semibold mb-4">Agent Votes</h3>
              <AgentVotesTable votes={discussion.agent_votes} />
            </div>

            {/* Market Context */}
            {discussion.market_context && (
              <div className="glass-card p-4 rounded-lg">
                <h3 className="text-white font-semibold mb-2">Market Context</h3>
                <pre className="text-slate-300 text-xs bg-slate-900/50 p-3 rounded overflow-x-auto">
                  {JSON.stringify(discussion.market_context, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ) : (
          <p className="text-red-400 text-center py-8">Failed to load discussion</p>
        )}
      </div>
    </div>
  )
}
