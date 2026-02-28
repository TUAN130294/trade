import { PortfolioStats } from './portfolio-stats'
import { PositionsTable } from './positions-table'
import { OrdersTable } from './orders-table'
import { WebSocketFeed } from './websocket-feed'

export function TradingView({ apiUrl }) {
  return (
    <div className="space-y-6 p-6">
      <PortfolioStats apiUrl={apiUrl} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PositionsTable apiUrl={apiUrl} />
        <OrdersTable apiUrl={apiUrl} />
      </div>

      <WebSocketFeed />
    </div>
  )
}
