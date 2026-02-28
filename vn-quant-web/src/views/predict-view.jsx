import { useState, useEffect } from 'react'
import { API_URL, fmtMoney } from '../utils/constants'

export function PredictView({ symbol }) {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runPrediction = async () => {
    setLoading(true)
    setError(null)
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 30000) // 30s timeout

      const resp = await fetch(`${API_URL}/predict/${symbol}`, {
        signal: controller.signal
      })
      clearTimeout(timeoutId)

      if (!resp.ok) throw new Error(`API Error: ${resp.status}`)

      const data = await resp.json()
      setPrediction(data)
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.message || 'Lỗi kết nối API')
      // Don't set fake prediction data — show error state only
      setPrediction(null)
    }
    setLoading(false)
  }

  useEffect(() => { runPrediction() }, [symbol])

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-white flex items-center gap-2"><span className="material-symbols-outlined text-accent-pink">auto_graph</span> Stockformer AI Prediction</h2>

      {error && (
        <div className="bg-amber-500/20 border border-amber-500/50 text-amber-400 px-4 py-2 rounded-lg text-sm">
          ⚠️ {error} - Hiển thị dữ liệu dự phòng
        </div>
      )}

      <div className="glass-panel p-6 rounded-xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-3xl font-bold text-white">{symbol}</h3>
            <p className="text-slate-400">Current: {fmtMoney(prediction?.current_price || 0)} VND</p>
          </div>
          <div className={`px-6 py-3 rounded-xl font-bold text-xl ${prediction?.direction === 'UP' ? 'bg-emerald-500/20 text-emerald-400' : prediction?.direction === 'DOWN' ? 'bg-red-500/20 text-red-400' : 'bg-slate-500/20 text-slate-400'}`}>
            {prediction?.direction || '---'}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white/5 p-4 rounded-lg"><p className="text-slate-500 text-xs">Expected Return</p><p className="text-2xl font-bold text-white">{prediction?.expected_return?.toFixed(2) || '---'}%</p></div>
          <div className="bg-white/5 p-4 rounded-lg"><p className="text-slate-500 text-xs">Confidence</p><p className="text-2xl font-bold text-white">{((prediction?.confidence || 0) * 100).toFixed(0)}%</p></div>
          <div className="bg-white/5 p-4 rounded-lg"><p className="text-slate-500 text-xs">Volatility</p><p className="text-2xl font-bold text-white">{prediction?.volatility_forecast?.toFixed(4) || '---'}</p></div>
          <div className="bg-white/5 p-4 rounded-lg"><p className="text-slate-500 text-xs">Model</p><p className="text-lg font-bold text-primary">{loading ? 'Đang dự đoán...' : prediction?.model || 'N/A'}</p></div>
        </div>

        <div>
          <h4 className="text-white font-bold mb-2">5-Day Price Forecast</h4>
          <div className="flex gap-2">
            {(prediction?.predictions || []).map((p, i) => (
              <div key={i} className="flex-1 bg-white/5 p-3 rounded-lg text-center">
                <p className="text-slate-500 text-xs">Day {i + 1}</p>
                <p className="text-white font-bold">{fmtMoney(p)}</p>
              </div>
            ))}
            {(!prediction?.predictions || prediction.predictions.length === 0) && !loading && (
              <p className="text-slate-400">Chưa có dự đoán</p>
            )}
          </div>
        </div>

        <button onClick={runPrediction} disabled={loading} className="mt-6 bg-primary hover:bg-blue-600 text-white px-6 py-3 rounded-xl font-bold transition-all w-full disabled:opacity-50">
          {loading ? '⏳ Đang dự đoán...' : '🔄 Làm mới dự đoán'}
        </button>
      </div>
    </div>
  )
}
