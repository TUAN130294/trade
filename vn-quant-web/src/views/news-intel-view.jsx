import { useState, useEffect } from 'react'
import { API_URL } from '../utils/constants'

export function NewsIntelView() {
  const [status, setStatus] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [mood, setMood] = useState(null)
  const [scanning, setScanning] = useState(false)
  const [watchlist, setWatchlist] = useState([])
  const [newSymbol, setNewSymbol] = useState('')

  useEffect(() => {
    loadData()
    // DISABLED: Auto-refresh was consuming too many API tokens
    // const interval = setInterval(loadData, 60000) // Refresh every minute
    // return () => clearInterval(interval)
  }, [])

  const loadData = async () => {
    try {
      const [statusRes, alertsRes, moodRes, watchlistRes] = await Promise.all([
        fetch(`${API_URL}/news/status`).then(r => r.json()),
        fetch(`${API_URL}/news/alerts`).then(r => r.json()),
        fetch(`${API_URL}/news/market-mood`).then(r => r.json()),
        fetch(`${API_URL}/news/watchlist`).then(r => r.json())
      ])
      setStatus(statusRes)
      setAlerts(alertsRes.alerts || [])
      setMood(moodRes)
      setWatchlist(watchlistRes.watchlist || [])
    } catch (e) {
      console.error('Load error:', e)
    }
  }

  const runScan = async () => {
    setScanning(true)
    try {
      const res = await fetch(`${API_URL}/news/scan`, { method: 'POST' })
      const data = await res.json()
      if (data.alerts) {
        setAlerts(data.alerts)
      }
    } catch (e) {
      console.error('Scan error:', e)
    }
    setScanning(false)
    loadData()
  }

  const addToWatchlist = async () => {
    if (!newSymbol.trim()) return
    const updated = [...watchlist, newSymbol.toUpperCase()]
    try {
      await fetch(`${API_URL}/news/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated)
      })
      setWatchlist(updated)
      setNewSymbol('')
    } catch (e) { }
  }

  const removeFromWatchlist = async (symbol) => {
    const updated = watchlist.filter(s => s !== symbol)
    try {
      await fetch(`${API_URL}/news/watchlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updated)
      })
      setWatchlist(updated)
    } catch (e) { }
  }

  const getMoodColor = (m) => {
    switch (m) {
      case 'bullish': return 'text-emerald-400 bg-emerald-400/20'
      case 'slightly_bullish': return 'text-green-400 bg-green-400/20'
      case 'bearish': return 'text-red-400 bg-red-400/20'
      case 'slightly_bearish': return 'text-orange-400 bg-orange-400/20'
      default: return 'text-slate-400 bg-slate-400/20'
    }
  }

  const getPriorityColor = (p) => {
    switch (p) {
      case 'CRITICAL': return 'bg-red-500/20 text-red-400 border-red-500'
      case 'HIGH': return 'bg-orange-500/20 text-orange-400 border-orange-500'
      case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500'
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <span className="material-symbols-outlined text-accent-cyan">newspaper</span>
          News Intelligence
          <span className="text-xs px-2 py-1 bg-purple-500/20 text-purple-400 rounded-full ml-2">
            Tin tức TRƯỚC - Giá SAU
          </span>
        </h2>
        <button
          onClick={runScan}
          disabled={scanning}
          className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-primary to-accent-cyan text-white rounded-lg hover:opacity-90 disabled:opacity-50"
        >
          <span className={`material-symbols-outlined ${scanning ? 'animate-spin' : ''}`}>
            {scanning ? 'sync' : 'radar'}
          </span>
          {scanning ? 'Đang quét...' : 'Scan Now'}
        </button>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-xl">
          <p className="text-slate-400 text-xs font-bold uppercase">News Sentiment</p>
          <div className={`mt-2 text-xl font-bold px-3 py-1 rounded inline-block ${getMoodColor(mood?.current_mood)}`}>
            {mood?.current_mood?.replace('_', ' ').toUpperCase() || 'NEUTRAL'}
          </div>
        </div>
        <div className="glass-panel p-4 rounded-xl">
          <p className="text-slate-400 text-xs font-bold uppercase">Active Sources</p>
          <p className="text-white font-bold text-2xl mt-1">{status?.sources?.length || 0}</p>
        </div>
        <div className="glass-panel p-4 rounded-xl">
          <p className="text-slate-400 text-xs font-bold uppercase">High Priority Alerts</p>
          <p className="text-red-400 font-bold text-2xl mt-1">{alerts.filter(a => a.priority === 'HIGH' || a.priority === 'CRITICAL').length}</p>
        </div>
        <div className="glass-panel p-4 rounded-xl">
          <p className="text-slate-400 text-xs font-bold uppercase">Watchlist Stocks</p>
          <p className="text-white font-bold text-2xl mt-1">{watchlist.length}</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Watchlist Column */}
        <div className="glass-panel p-4 rounded-xl h-[500px] flex flex-col">
          <h3 className="font-bold text-white mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-yellow-400">star</span> Watchlist
          </h3>
          <div className="flex gap-2 mb-4">
            <input
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
              placeholder="Add Symbol..."
              className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-primary"
            />
            <button onClick={addToWatchlist} className="bg-white/10 hover:bg-white/20 px-3 rounded-lg text-white">
              +
            </button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2">
            {watchlist.map(sym => (
              <div key={sym} className="flex justify-between items-center bg-white/5 p-3 rounded-lg group">
                <span className="font-bold text-white">{sym}</span>
                <button onClick={() => removeFromWatchlist(sym)} className="text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="material-symbols-outlined text-sm">close</span>
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Alerts Feed */}
        <div className="lg:col-span-2 glass-panel p-4 rounded-xl h-[500px] flex flex-col">
          <h3 className="font-bold text-white mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-accent-pink">notifications</span> Live News Alerts
          </h3>
          <div className="flex-1 overflow-y-auto space-y-3 pr-2">
            {alerts.length === 0 ? (
              <div className="text-slate-500 text-center py-10">Waiting for news events...</div>
            ) : (
              alerts.map((alert, i) => (
                <div key={i} className={`p-4 rounded-xl border-l-4 ${getPriorityColor(alert.priority)} bg-white/5 hover:bg-white/10 transition-colors`}>
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-bold text-white text-lg">{alert.title}</span>
                    <span className="text-xs text-slate-400 whitespace-nowrap ml-2">{new Date(alert.published_at).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                  <p className="text-slate-300 text-sm mb-2 line-clamp-2">{alert.summary}</p>
                  <div className="flex justify-between items-center">
                    <div className="flex gap-2">
                      {alert.symbols?.map(s => (
                        <span key={s} className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded text-xs font-bold">{s}</span>
                      ))}
                    </div>
                    <span className="text-xs text-slate-500">{alert.source}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
