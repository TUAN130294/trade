import { useState } from 'react'
import { API_URL } from '../utils/constants'

export function BacktestView() {
  const [results, setResults] = useState({})
  const [loading, setLoading] = useState({})
  const [symbol, setSymbol] = useState('MWG')

  const strategies = [
    { id: 'momentum', name: 'Momentum (MA Crossover)', winRate: null, profitFactor: null, maxDrawdown: null, sharpe: null },
    { id: 'mean_reversion', name: 'Mean Reversion (RSI)', winRate: null, profitFactor: null, maxDrawdown: null, sharpe: null },
    { id: 'macd', name: 'MACD Signal', winRate: null, profitFactor: null, maxDrawdown: null, sharpe: null },
  ]

  const runBacktest = async (strategyId) => {
    setLoading(prev => ({ ...prev, [strategyId]: true }))
    try {
      const resp = await fetch(`${API_URL}/backtest/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy: strategyId, symbol, days: 365 })
      })
      const data = await resp.json()
      if (data.success) {
        setResults(prev => ({ ...prev, [strategyId]: data.results }))
      } else {
        alert(`Backtest failed: ${data.error}`)
      }
    } catch (e) {
      alert(`Error: ${e.message}`)
    } finally {
      setLoading(prev => ({ ...prev, [strategyId]: false }))
    }
  }

  const getMetric = (strategyId, metric, defaultVal) => {
    if (results[strategyId]) {
      const map = { winRate: 'win_rate', sharpe: 'sharpe_ratio', maxDrawdown: 'max_drawdown_pct', profitFactor: 'profit_factor' }
      return results[strategyId][map[metric]] || defaultVal
    }
    return defaultVal
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <span className="material-symbols-outlined text-amber-400">history</span> Walk-Forward Backtest
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-sm">Symbol:</span>
          <input type="text" value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
            className="bg-white/10 border border-white/20 rounded px-3 py-1 text-white w-24 text-center" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {strategies.map((s) => (
          <div key={s.id} className="glass-panel p-6 rounded-xl">
            <h3 className="text-lg font-bold text-white mb-4">{s.name}</h3>
            <div className="grid grid-cols-2 gap-4">
              <div><p className="text-slate-500 text-xs mb-1">Win Rate</p><p className="text-2xl font-bold text-emerald-400">{getMetric(s.id, 'winRate', '--')}%</p></div>
              <div><p className="text-slate-500 text-xs mb-1">Sharpe Ratio</p><p className="text-2xl font-bold text-white">{getMetric(s.id, 'sharpe', '--')}</p></div>
              <div><p className="text-slate-500 text-xs mb-1">Max Drawdown</p><p className="text-2xl font-bold text-red-400">{getMetric(s.id, 'maxDrawdown', '--')}%</p></div>
              <div><p className="text-slate-500 text-xs mb-1">Profit Factor</p><p className="text-2xl font-bold text-emerald-400">{getMetric(s.id, 'profitFactor', '--')}</p></div>
            </div>
            {results[s.id] && (
              <div className="mt-3 text-xs text-slate-400">
                <span>Trades: {results[s.id].total_trades}</span>
                <span className="mx-2">|</span>
                <span className="text-emerald-400">W:{results[s.id].winning_trades}</span>
                <span className="mx-1">/</span>
                <span className="text-red-400">L:{results[s.id].losing_trades}</span>
              </div>
            )}
            <button
              onClick={() => runBacktest(s.id)}
              disabled={loading[s.id]}
              className={`w-full mt-4 py-2 rounded-lg transition-colors border border-white/10 ${loading[s.id] ? 'bg-amber-500/30 text-amber-300' : 'bg-white/10 hover:bg-white/20 text-white'}`}>
              {loading[s.id] ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
