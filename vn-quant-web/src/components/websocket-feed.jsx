import { useState, useEffect } from 'react'
import { useWebSocket } from '../hooks/use-websocket'

const EVENT_CONFIG = {
  opportunity_detected: { icon: 'search', color: 'amber', label: 'Opportunity' },
  agent_discussion: { icon: 'forum', color: 'blue', label: 'Discussion' },
  order_executed: { icon: 'shopping_cart', color: 'green', label: 'Order' },
  position_exited: { icon: 'logout', color: 'purple', label: 'Exit' },
  system_reset: { icon: 'refresh', color: 'red', label: 'Reset' }
}

export function WebSocketFeed() {
  const wsUrl = import.meta.env.VITE_WS_URL ||
    (import.meta.env.PROD ? 'wss://' + window.location.host + '/ws/autonomous' : 'ws://localhost:8101/ws/autonomous')

  const { isConnected, lastMessage } = useWebSocket(wsUrl)
  const [events, setEvents] = useState([])

  useEffect(() => {
    if (!lastMessage) return

    // Handle system_reset: clear events
    if (lastMessage.event_type === 'system_reset') {
      setEvents([lastMessage])
      return
    }

    // Add new event, keep max 100 (newest first)
    setEvents(prev => [lastMessage, ...prev].slice(0, 100))
  }, [lastMessage])

  const getEventStyle = (eventType) => {
    const config = EVENT_CONFIG[eventType] || { icon: 'info', color: 'slate', label: 'Event' }
    return config
  }

  return (
    <div className="glass-panel p-6 rounded-xl">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-white font-bold text-lg">Live Trading Feed</h3>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-xs text-slate-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {events.length === 0 ? (
          <p className="text-slate-500 text-sm text-center py-8">No events yet...</p>
        ) : (
          events.map((event, idx) => {
            const style = getEventStyle(event.event_type)
            const colorClasses = {
              amber: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
              blue: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
              green: 'bg-green-500/10 border-green-500/30 text-green-400',
              purple: 'bg-purple-500/10 border-purple-500/30 text-purple-400',
              red: 'bg-red-500/10 border-red-500/30 text-red-400',
              slate: 'bg-slate-500/10 border-slate-500/30 text-slate-400'
            }

            return (
              <div
                key={idx}
                className={`p-3 rounded-lg border-l-4 ${colorClasses[style.color] || colorClasses.slate}`}
              >
                <div className="flex items-start gap-3">
                  <span className="material-symbols-outlined text-[18px] mt-0.5">{style.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-semibold text-sm">{style.label}</span>
                      <span className="text-xs text-slate-500">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 break-words">
                      {event.data?.symbol && <strong>{event.data.symbol}</strong>}
                      {event.data?.message || event.data?.action || JSON.stringify(event.data)}
                    </p>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
