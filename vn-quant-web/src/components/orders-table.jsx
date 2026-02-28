import { useState, useEffect } from 'react'

export function OrdersTable({ apiUrl }) {
  const [orders, setOrders] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchOrders = () => {
      fetch(`${apiUrl}/orders`)
        .then(res => res.json())
        .then(data => {
          setOrders(Array.isArray(data) ? data : data.orders || [])
          setLoading(false)
        })
        .catch(err => {
          console.error('Orders fetch error:', err)
          setLoading(false)
        })
    }

    fetchOrders()
    const interval = setInterval(fetchOrders, 10000) // Auto-refresh every 10s

    return () => clearInterval(interval)
  }, [apiUrl])

  const fmtMoney = (n) => new Intl.NumberFormat('vi-VN').format(n)

  const getStatusColor = (status) => {
    switch (status) {
      case 'FILLED': return 'text-green-400'
      case 'PENDING': return 'text-amber-400'
      case 'CANCELLED': return 'text-red-400'
      default: return 'text-slate-400'
    }
  }

  if (loading) {
    return <div className="glass-panel p-6 rounded-xl animate-pulse h-64"></div>
  }

  return (
    <div className="glass-panel p-6 rounded-xl">
      <h3 className="text-white font-bold text-lg mb-4">Recent Orders</h3>

      {orders.length === 0 ? (
        <p className="text-slate-500 text-sm text-center py-8">No orders yet</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left py-3 px-2 text-slate-400 font-semibold">Symbol</th>
                <th className="text-left py-3 px-2 text-slate-400 font-semibold">Side</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Price</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Qty</th>
                <th className="text-left py-3 px-2 text-slate-400 font-semibold">Status</th>
                <th className="text-right py-3 px-2 text-slate-400 font-semibold">Time</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order, idx) => {
                const isBuy = order.side === 'BUY'

                return (
                  <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-800/20">
                    <td className="py-3 px-2 font-bold text-white">{order.symbol}</td>
                    <td className={`py-3 px-2 font-semibold ${isBuy ? 'text-green-400' : 'text-red-400'}`}>
                      {order.side}
                    </td>
                    <td className="py-3 px-2 text-right text-slate-300">{fmtMoney(order.price)}</td>
                    <td className="py-3 px-2 text-right text-slate-300">{order.quantity}</td>
                    <td className={`py-3 px-2 font-semibold ${getStatusColor(order.status)}`}>
                      {order.status}
                    </td>
                    <td className="py-3 px-2 text-right text-slate-400">
                      {new Date(order.timestamp).toLocaleTimeString()}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
