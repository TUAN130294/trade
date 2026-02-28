import { useState, useEffect } from 'react'
import { API_URL } from './utils/constants'
import { Sidebar } from './components/sidebar'
import { TradingView } from './components/trading-view'
import { DiscussionsView } from './components/discussions-view'
import { DashboardView } from './views/dashboard-view'
import { AnalysisView } from './views/analysis-view'
import { RadarView } from './views/radar-view'
import { CommandView } from './views/command-view'
import { BacktestView } from './views/backtest-view'
import { PredictView } from './views/predict-view'
import { DataHubView } from './views/data-hub-view'
import { NewsIntelView } from './views/news-intel-view'

function App() {
  // Phase 08: State persistence with localStorage
  const [activeView, setActiveView] = useState(
    localStorage.getItem('vn-quant-activeView') || 'dashboard'
  )
  const [analysisSymbol, setAnalysisSymbol] = useState(
    localStorage.getItem('vn-quant-analysisSymbol') || 'MWG'
  )

  const [marketStatus, setMarketStatus] = useState(null)
  const [stockData, setStockData] = useState([])
  const [regime, setRegime] = useState(null)
  const [chatSymbol, setChatSymbol] = useState(null)

  // Phase 08: Persist activeView to localStorage
  useEffect(() => {
    localStorage.setItem('vn-quant-activeView', activeView)
  }, [activeView])

  // Phase 08: Persist analysisSymbol to localStorage
  useEffect(() => {
    localStorage.setItem('vn-quant-analysisSymbol', analysisSymbol)
  }, [analysisSymbol])

  // Initial data load
  useEffect(() => {
    fetch(`${API_URL}/market/status`).then(r => r.json()).then(setMarketStatus)
    fetch(`${API_URL}/market/regime`).then(r => r.json()).then(setRegime)
  }, [])

  // Load stock data when analysis view is active
  useEffect(() => {
    if (activeView === 'analysis') {
      fetch(`${API_URL}/stock/${analysisSymbol}`)
        .then(r => r.json())
        .then(setStockData)
        .catch(e => console.error(e))
    }
  }, [activeView, analysisSymbol])

  // Handler for "Chat with Agents"
  const handleAgentChat = (symbol) => {
    setChatSymbol(symbol);
    setActiveView('command');
  }

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <DashboardView marketStatus={marketStatus} regime={regime} />
      case 'analysis':
        return <AnalysisView symbol={analysisSymbol} setSymbol={setAnalysisSymbol} stockData={stockData} onAgentChat={handleAgentChat} />
      case 'radar':
        return <RadarView onAgentChat={handleAgentChat} />
      case 'command':
        return <CommandView initialSymbol={chatSymbol} />
      case 'trading':
        return <TradingView apiUrl={API_URL} />
      case 'discussions':
        return <DiscussionsView apiUrl={API_URL} />
      case 'backtest':
        return <BacktestView />
      case 'predict':
        return <PredictView symbol={analysisSymbol} />
      case 'data':
        return <DataHubView />
      case 'news':
        return <NewsIntelView />
      default:
        return <DashboardView marketStatus={marketStatus} regime={regime} />
    }
  }

  return (
    <div className="flex w-full h-screen bg-[#0a0e17] text-slate-200 overflow-hidden font-sans selection:bg-primary/30">
      <div className="fixed inset-0 pointer-events-none bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 z-0"></div>
      <Sidebar
        activeView={activeView}
        setView={setView => {
          setActiveView(setView);
          if (setView !== 'command') setChatSymbol(null);
        }}
      />
      <main className="flex-1 flex flex-col relative z-10 w-full max-w-full overflow-hidden">
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-6 bg-[#0a0e17]/80 backdrop-blur-md md:hidden">
          <span className="font-bold text-white">VN-QUANT APP</span>
          <button className="text-slate-400"><span className="material-symbols-outlined">menu</span></button>
        </header>
        {renderView()}
      </main>
    </div>
  )
}

export default App
