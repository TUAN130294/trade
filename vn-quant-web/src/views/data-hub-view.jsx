import { useState, useEffect } from 'react'
import { API_URL } from '../utils/constants'

export function DataHubView() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API_URL}/data/stats`).then(r => r.json()).then(d => { setStats(d); setLoading(false) })
  }, [])

  if (loading) return <div className="flex-1 flex items-center justify-center text-white">Loading data stats...</div>

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-white flex items-center gap-2">
        <span className="material-symbols-outlined text-accent-cyan">database</span> Data Hub
        <span className="text-xs px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-full">Auto Update 17:30</span>
      </h2>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-6 rounded-xl text-center">
          <div className="text-4xl font-bold text-primary">{stats?.total_files || 0}</div>
          <div className="text-slate-400 text-sm">Mã cổ phiếu đã tải</div>
        </div>
        <div className="glass-panel p-6 rounded-xl text-center">
          <div className="text-4xl font-bold text-emerald-400">{stats?.total_available || 1730}</div>
          <div className="text-slate-400 text-sm">Tổng mã toàn sàn</div>
        </div>
        <div className="glass-panel p-6 rounded-xl text-center">
          <div className="text-4xl font-bold text-amber-400">{stats?.coverage_pct || 0}%</div>
          <div className="text-slate-400 text-sm">Coverage</div>
        </div>
        <div className="glass-panel p-6 rounded-xl text-center">
          <div className="text-4xl font-bold text-white">{stats?.total_size_mb || 0} MB</div>
          <div className="text-slate-400 text-sm">Dung lượng</div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="glass-panel p-6 rounded-xl">
        <div className="flex justify-between mb-2">
          <span className="text-white font-bold">Data Coverage</span>
          <span className="text-primary">{stats?.total_files}/{stats?.total_available}</span>
        </div>
        <div className="h-4 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary to-emerald-400 rounded-full transition-all"
            style={{ width: `${stats?.coverage_pct || 0}%` }}
          ></div>
        </div>
        <p className="text-slate-400 text-sm mt-2">
          {stats?.coverage_pct < 100
            ? `Đang tải thêm... Chạy "python download_all_stocks.py" để tải đầy đủ.`
            : '✅ Đã tải đầy đủ toàn sàn!'}
        </p>
      </div>

      {/* Auto Update Info */}
      <div className="glass-panel p-6 rounded-xl border-l-4 border-l-emerald-400">
        <h3 className="text-white font-bold mb-2 flex items-center gap-2">
          <span className="material-symbols-outlined text-emerald-400">schedule</span>
          Tự động cập nhật
        </h3>
        <p className="text-slate-300">Data được cập nhật tự động lúc <strong>17:30</strong> hàng ngày sau khi thị trường đóng cửa.</p>
        <p className="text-slate-400 text-sm mt-1">Task Scheduler: VN-QUANT-DataUpdate</p>
      </div>

      {/* Sample Symbols */}
      <div className="glass-panel p-6 rounded-xl">
        <h3 className="text-white font-bold mb-4">Mã cổ phiếu đã tải (mẫu)</h3>
        <div className="flex flex-wrap gap-2">
          {(stats?.sample_symbols || []).map(s => (
            <span key={s} className="px-3 py-1 bg-white/5 text-white rounded-full text-sm border border-white/10">{s}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
