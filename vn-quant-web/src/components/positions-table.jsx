import { useState, useEffect } from 'react'

export function PositionsTable({ apiUrl }) {
  const [positions, setPositions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchPositions = () => {
      fetch(`${apiUrl}/positions`)
        .then(res => res.json())
        .then(data => {
          setPositions(Array.isArray(data) ? data : data.positions || [])
          setLoading(false)
        })
        .catch(err => {
          console.error('Positions fetch error:', err)
          setLoading(false)
        })
    }

    fetchPositions()
    const interval = setInterval(fetchPositions, 5000) // Auto-refresh every 5s

    return () => clearInterval(interval)
  }, [apiUrl])

  const fmtMoney = (n) => new Intl.NumberFormat('vi-VN').format(n)

  if (loading) {
    return <div className="glass-panel p-6 rounded-xl animate-pulse h-64"></div>
  }

  return (
    <div className="glass-panel p-6 rounded-xl">
      <h3 className="text-white font-bold text-lg mb-4">Active Positions</h3>

      {positions.length === 0 ? (
        <p className="text-slate-500 text-sm text-center py-8">No open positions</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-2 text-slate-400 font-semibold">Symbol</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Qty</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Entry</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Current</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">P&L%</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Days</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((pos, idx) => {
                const pnlPercent = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                const isProfitable = pnlPercent >= 0

                return (
                  <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="py-3 px-2 font-bold text-white">{pos.symbol}</td>
                    <td className="py-3 px-2 text-right text-slate-300">{pos.quantity}</td>
                    <td className="py-3 px-2 text-right text-slate-300">{fmtMoney(pos.entry_price)}</td>
                    <td className="py-3 px-2 text-right text-slate-300">{fmtMoney(pos.current_price)}</td>
                    <td className={`py-3 px-2 text-right font-semibold ${isProfitable ? 'text-green-400' : 'text-red-400'}`}>
                      {isProfitable ? '+' : ''}{pnlPercent.toFixed(2)}%
                    </td>
                    <td className="py-3 px-2 text-right text-slate-400">{pos.days_held || 0}</td>
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
