export function Sidebar({ activeView, setView }) {
  const menuItems = [
    { id: 'dashboard', icon: 'dashboard', label: 'Overview' },
    { id: 'analysis', icon: 'show_chart', label: 'Analysis' },
    { id: 'news', icon: 'newspaper', label: 'News Intel' },
    { id: 'radar', icon: 'radar', label: 'Agent Radar' },
    { id: 'command', icon: 'forum', label: 'Agent Chat' },
    { id: 'trading', icon: 'account_balance', label: 'Auto Trading' },
    { id: 'discussions', icon: 'groups', label: 'Discussions' },
    { id: 'backtest', icon: 'history', label: 'Backtest' },
    { id: 'predict', icon: 'auto_graph', label: 'AI Predict' },
    { id: 'data', icon: 'database', label: 'Data Hub' },
  ]

  return (
    <aside className="glass-sidebar w-72 h-full flex-shrink-0 flex flex-col justify-between py-6 px-4 hidden md:flex z-50">
      <div className="flex flex-col gap-8">
        <div className="flex items-center gap-3 px-2">
          <div className="relative flex items-center justify-center size-10 rounded-lg bg-gradient-to-br from-primary to-accent-cyan shadow-lg shadow-primary/20">
            <span className="material-symbols-outlined text-white text-[24px]">token</span>
          </div>
          <div className="flex flex-col">
            <h1 className="text-white text-xl font-bold tracking-tight">VN-QUANT</h1>
            <p className="text-slate-400 text-xs font-medium tracking-wide">AGENTIC LEVEL 4</p>
          </div>
        </div>

        <nav className="flex flex-col gap-2">
          <div className="px-3 py-1 text-xs font-bold text-slate-500 uppercase tracking-widest">Platform</div>
          {menuItems.map(item => (
            <button key={item.id} onClick={() => setView(item.id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all group w-full text-left
                 ${activeView === item.id ? 'bg-primary/15 border border-primary/20 text-white shadow-sm' : 'hover:bg-white/5 text-slate-400 hover:text-white'}`}>
              <span className={`material-symbols-outlined transition-colors ${activeView === item.id ? 'text-primary' : 'group-hover:text-accent-cyan'}`}>{item.icon}</span>
              <span className="text-sm font-semibold">{item.label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="flex flex-col gap-2">
        <div className="mt-4 flex items-center gap-3 px-4 py-3 rounded-xl bg-white/5 border border-white/5">
          <div className="size-8 rounded-full bg-gray-600 flex items-center justify-center text-xs font-bold">AD</div>
          <div className="flex flex-col">
            <span className="text-white text-sm font-bold">Admin</span>
            <span className="text-emerald-400 text-xs flex items-center gap-1">
              <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse"></span>Online
            </span>
          </div>
        </div>
      </div>
    </aside>
  )
}
