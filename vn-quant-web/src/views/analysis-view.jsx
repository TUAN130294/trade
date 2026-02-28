import { useState } from 'react'
import { API_URL, fmtMoney } from '../utils/constants'
import { StockChart } from '../components/stock-chart'
import { TechnicalPanel } from '../components/technical-panel'

export function AnalysisView({ symbol, setSymbol, stockData, onAgentChat }) {
  const [inputSym, setInputSym] = useState(symbol)
  const handleSearch = (e) => { if (e.key === 'Enter') setSymbol(inputSym) }
  const latest = stockData && stockData.length > 0 ? stockData[stockData.length - 1] : null

  return (
    <div className="flex-1 overflow-y-auto p-4 lg:p-6 pb-20">
      <div className="max-w-[1600px] mx-auto space-y-6">
        <div className="flex flex-col md:flex-row gap-6 md:items-end justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <input value={inputSym} onChange={(e) => setInputSym(e.target.value.toUpperCase())} onKeyDown={handleSearch}
                className="bg-transparent text-3xl md:text-4xl font-bold text-white tracking-tight border-b border-white/10 focus:border-primary focus:outline-none w-[200px]" />
              <span className="bg-white/5 text-slate-400 text-xs px-2 py-1 rounded border border-white/10">HOSE</span>
            </div>
            {latest ? (
              <div className="flex items-baseline gap-4">
                <span className="text-4xl font-bold text-white tracking-tight">{fmtMoney(latest.close)} <span className="text-lg text-slate-400 font-normal">VND</span></span>
              </div>
            ) : (<div className="text-slate-500">Loading data for {symbol}...</div>)}
          </div>

          {/* Quick Action */}
          <div className="flex gap-2">
            <button onClick={() => onAgentChat(symbol)} className="px-4 py-2 bg-primary text-white rounded-lg flex items-center gap-2 hover:bg-blue-600 transition-colors">
              <span className="material-symbols-outlined text-[18px]">chat</span>
              Ask Agents
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          <div className="xl:col-span-8 glass-panel rounded-xl overflow-hidden shadow-2xl shadow-black/40 p-2">
            {stockData && stockData.length > 0 ? <StockChart data={stockData} /> : <div className="flex items-center justify-center h-[400px] text-slate-500">Waiting for Data...</div>}
          </div>

          <div className="xl:col-span-4 flex flex-col gap-4">
            {/* New Technical Panel */}
            <TechnicalPanel symbol={symbol} />

            {/* Deep Flow Agent */}
            <div className="glass-panel p-5 rounded-xl border-l-4 border-l-primary/50">
              <div className="flex items-center gap-3 mb-3">
                <div className="size-10 rounded-full bg-gradient-to-br from-primary to-blue-700 flex items-center justify-center shadow-lg"><span className="material-symbols-outlined text-white text-[20px]">manage_search</span></div>
                <div><h3 className="font-bold text-white text-base">Deep Flow AI</h3><p className="text-xs text-primary font-medium">Scanning: {symbol}</p></div>
              </div>
              <button className="w-full bg-white/10 hover:bg-white/20 text-white font-medium py-2 rounded-lg transition-colors border border-white/10"
                onClick={() => { fetch(`${API_URL}/analyze/deep_flow`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol, days: 60 }) }).then(res => res.json()).then(d => alert(`Deep Flow: ${d.insights?.length || 0} signals found!`)) }}>
                Start Deep Scan
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
