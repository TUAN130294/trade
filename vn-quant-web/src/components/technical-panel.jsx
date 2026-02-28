import { useState, useEffect } from 'react'
import { API_URL, fmtMoney } from '../utils/constants'

export function TechnicalPanel({ symbol }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    setError(null);
    fetch(`${API_URL}/analysis/technical/${symbol}`)
      .then(res => {
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        return res.json();
      })
      .then(d => {
        if (d.detail) throw new Error(d.detail); // FastAPI error format
        setData(d);
      })
      .catch(err => {
        console.error("Tech Analysis Error:", err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [symbol]);

  if (loading) return <div className="glass-panel p-4 animate-pulse">Loading technicals...</div>;
  if (error) return (
    <div className="glass-panel p-4 rounded-xl border-l-4 border-l-amber-500">
      <p className="text-amber-400 text-sm">⚠️ Technical analysis unavailable</p>
      <p className="text-slate-500 text-xs mt-1">{error}</p>
    </div>
  );
  if (!data) return null;

  return (
    <div className="glass-panel p-5 rounded-xl border-l-4 border-l-accent-cyan">
      <h3 className="text-white font-bold mb-4 flex items-center gap-2">
        <span className="material-symbols-outlined text-accent-cyan">engineering</span>
        Technical Deep Dive
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Support & Resistance */}
        <div>
          <h4 className="text-slate-400 text-xs font-bold uppercase mb-2">Key Levels</h4>
          <div className="space-y-2">
            <div>
              <span className="text-emerald-400 text-sm font-medium">Support:</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {data.support_levels.slice(0, 3).map((l, i) => (
                  <span key={i} className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded text-xs border border-emerald-500/20">{fmtMoney(l)}</span>
                ))}
              </div>
            </div>
            <div>
              <span className="text-red-400 text-sm font-medium">Resistance:</span>
              <div className="flex flex-wrap gap-2 mt-1">
                {data.resistance_levels.slice(0, 3).map((l, i) => (
                  <span key={i} className="px-2 py-0.5 bg-red-500/10 text-red-400 rounded text-xs border border-red-500/20">{fmtMoney(l)}</span>
                ))}
              </div>
            </div>
          </div>
          {/* 52-Week Range */}
          {data.historical && (
            <div className="mt-3 pt-3 border-t border-slate-700/50">
              <h4 className="text-slate-400 text-xs font-bold uppercase mb-2">52-Week Range</h4>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-red-400">{fmtMoney(data.historical.low_52w)}</span>
                <div className="flex-1 h-2 bg-slate-700 rounded-full relative">
                  <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-red-500 via-yellow-500 to-emerald-500 rounded-full" style={{ width: '100%' }}></div>
                  <div className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow border-2 border-accent-cyan" style={{ left: `${Math.min(100, Math.max(0, ((data.current_price - data.historical.low_52w) / (data.historical.high_52w - data.historical.low_52w)) * 100))}%` }}></div>
                </div>
                <span className="text-emerald-400">{fmtMoney(data.historical.high_52w)}</span>
              </div>
              <p className="text-center text-slate-500 text-xs mt-1">Current: {fmtMoney(data.current_price)}</p>
            </div>
          )}
        </div>

        {/* Bottom Evaluation */}
        <div className={`p-3 rounded-lg border ${data.bottom_evaluation.is_potential_bottom ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-slate-700/30 border-slate-600'}`}>
          <h4 className="text-white text-sm font-bold flex justify-between">
            Bottom Probability
            <span className={data.bottom_evaluation.is_potential_bottom ? 'text-emerald-400' : 'text-slate-400'}>{data.bottom_evaluation.score}/100</span>
          </h4>
          <div className="h-1.5 bg-slate-700 rounded-full mt-2 mb-2 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400" style={{ width: `${data.bottom_evaluation.score}%` }}></div>
          </div>
          <ul className="text-xs space-y-1">
            {data.bottom_evaluation.reasons.map((r, i) => (
              <li key={i} className="flex items-center gap-1 text-slate-300">
                <span className="material-symbols-outlined text-[10px] text-emerald-400">check</span> {r}
              </li>
            ))}
            {data.bottom_evaluation.reasons.length === 0 && <li className="text-slate-500">No strong bottom signals</li>}
          </ul>
        </div>
      </div>

      {/* Patterns */}
      <div className="mt-4">
        <h4 className="text-slate-400 text-xs font-bold uppercase mb-2">Detected Patterns (Recently)</h4>
        <div className="flex flex-wrap gap-2">
          {data.patterns.length > 0 ? (
            data.patterns.map((p, i) => (
              <span key={i} className="px-3 py-1 bg-white/5 text-white rounded-lg text-sm border border-white/10 flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-400 text-[14px]">lightbulb</span> {p}
              </span>
            ))
          ) : (
            <span className="text-slate-500 text-sm italic">No significant patterns detected in last 5 days</span>
          )}
        </div>
      </div>
    </div>
  );
}
