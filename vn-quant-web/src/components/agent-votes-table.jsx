export function AgentVotesTable({ votes }) {
  if (!votes || votes.length === 0) {
    return <p className="text-slate-500 text-sm text-center py-4">No votes recorded</p>
  }

  const getVoteColor = (vote) => {
    switch (vote) {
      case 'BUY': return 'text-green-400 bg-green-500/10'
      case 'SELL': return 'text-red-400 bg-red-500/10'
      case 'HOLD': return 'text-amber-400 bg-amber-500/10'
      default: return 'text-slate-400 bg-slate-500/10'
    }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-3 px-2 text-slate-400 font-semibold">Agent</th>
            <th className="text-left py-3 px-2 text-slate-400 font-semibold">Vote</th>
            <th className="text-left py-3 px-2 text-slate-400 font-semibold">Reasoning</th>
            <th className="text-right py-3 px-2 text-slate-400 font-semibold">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {votes.map((vote, idx) => (
            <tr key={idx} className="border-b border-slate-800/50">
              <td className="py-3 px-2 font-semibold text-white">{vote.agent_name}</td>
              <td className="py-3 px-2">
                <span className={`px-2 py-1 rounded text-xs font-bold ${getVoteColor(vote.vote)}`}>
                  {vote.vote}
                </span>
              </td>
              <td className="py-3 px-2 text-slate-300 max-w-md truncate">
                {vote.reasoning || '—'}
              </td>
              <td className="py-3 px-2 text-right text-slate-300">
                {((vote.confidence || 0) * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
