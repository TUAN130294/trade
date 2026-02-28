import { useState, useEffect } from 'react'
import { DiscussionDetailModal } from './discussion-detail-modal'

export function DiscussionsView({ apiUrl }) {
  const [discussions, setDiscussions] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState(null)

  useEffect(() => {
    const fetchDiscussions = () => {
      fetch(`${apiUrl}/discussions`)
        .then(res => res.json())
        .then(data => {
          setDiscussions(Array.isArray(data) ? data : data.discussions || [])
          setLoading(false)
        })
        .catch(err => {
          console.error('Discussions fetch error:', err)
          setLoading(false)
        })
    }

    fetchDiscussions()
    const interval = setInterval(fetchDiscussions, 10000) // Auto-refresh every 10s

    return () => clearInterval(interval)
  }, [apiUrl])

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'BUY': return 'text-green-400 bg-green-500/10 border-green-500'
      case 'SELL': return 'text-red-400 bg-red-500/10 border-red-500'
      case 'HOLD': return 'text-amber-400 bg-amber-500/10 border-amber-500'
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500'
    }
  }

  if (loading) {
    return (
      <div className="p-6 space-y-4">
        <div className="glass-panel p-6 rounded-xl animate-pulse h-32"></div>
        <div className="glass-panel p-6 rounded-xl animate-pulse h-32"></div>
      </div>
    )
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-3xl font-bold text-white mb-2">Agent Discussions</h2>
        <p className="text-slate-400">Multi-agent consensus decisions and reasoning</p>
      </div>

      {discussions.length === 0 ? (
        <div className="glass-panel p-12 rounded-xl text-center">
          <span className="material-symbols-outlined text-slate-600 text-5xl mb-4">forum</span>
          <p className="text-slate-500">No discussions yet</p>
        </div>
      ) : (
        <div className="space-y-4">
          {discussions.map((discussion) => (
            <div
              key={discussion.id}
              onClick={() => setSelectedId(discussion.id)}
              className="glass-panel p-6 rounded-xl hover:bg-slate-800/30 cursor-pointer transition-all"
            >
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-3">
                  <span className="material-symbols-outlined text-blue-400">forum</span>
                  <h3 className="text-xl font-bold text-white">{discussion.symbol}</h3>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 rounded border font-bold text-sm ${getVerdictColor(discussion.verdict)}`}>
                    {discussion.verdict}
                  </span>
                  <span className="text-slate-400 text-sm">
                    {new Date(discussion.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>

              {discussion.reasoning && (
                <p className="text-slate-300 text-sm line-clamp-2 mb-3">
                  {discussion.reasoning}
                </p>
              )}

              <div className="flex items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-1">
                  <span className="material-symbols-outlined text-[16px]">groups</span>
                  {discussion.agent_count || 0} agents
                </span>
                <span className="flex items-center gap-1">
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                  Click for details
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedId && (
        <DiscussionDetailModal
          discussionId={selectedId}
          apiUrl={apiUrl}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  )
}
