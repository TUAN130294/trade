import { useState, useEffect } from 'react'

export function PortfolioStats({ apiUrl }) {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStatus = () => {
      fetch(`${apiUrl}/status`)
        .then(res => res.json())
        .then(data => {
          setStatus(data)
          setLoading(false)
        })
        .catch(err => {
          console.error('Status fetch error:', err)
          setLoading(false)
        })
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 5000) // Auto-refresh every 5s

    return () => clearInterval(interval)
  }, [apiUrl])

  if (loading) {
    return <div className="glass-panel p-6 rounded-xl animate-pulse h-32"></div>
  }

  const fmtMoney = (n) => new Intl.NumberFormat('vi-VN').format(n)
  const pnl = status?.total_value - status?.initial_capital || 0
  const pnlPercent = ((pnl / status?.initial_capital) * 100) || 0

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="glass-panel p-6 rounded-xl">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-slate-400 text-sm font-medium mb-1">Available Cash</p>
            <h3 className="text-2xl font-bold text-white">{fmtMoney(status?.cash || 0)} VND</h3>
          </div>
          <div className="p-2 bg-blue-500/10 rounded-lg text-blue-500">
            <span className="material-symbols-outlined text-[20px]">account_balance_wallet</span>
          </div>
        </div>
      </div>

      <div className="glass-panel p-6 rounded-xl">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-slate-400 text-sm font-medium mb-1">Portfolio Value</p>
            <h3 className="text-2xl font-bold text-white">{fmtMoney(status?.total_value || 0)} VND</h3>
          </div>
          <div className="p-2 bg-green-500/10 rounded-lg text-green-500">
            <span className="material-symbols-outlined text-[20px]">trending_up</span>
          </div>
        </div>
      </div>

      <div className="glass-panel p-6 rounded-xl">
        <div className="flex justify-between items-start">
          <div>
            <p className="text-slate-400 text-sm font-medium mb-1">Total P&L</p>
            <h3 className={`text-2xl font-bold ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {pnl >= 0 ? '+' : ''}{fmtMoney(pnl)} VND
            </h3>
            <p className={`text-sm mt-1 ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {pnl >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
            </p>
          </div>
          <div className={`p-2 rounded-lg ${pnl >= 0 ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
            <span className="material-symbols-outlined text-[20px]">
              {pnl >= 0 ? 'arrow_upward' : 'arrow_downward'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
