import { useState, useEffect, useRef } from 'react'

export function useWebSocket(url) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectDelayRef = useRef(1000) // Start with 1s

  useEffect(() => {
    if (!url) return

    const connect = () => {
      try {
        const ws = new WebSocket(url)
        wsRef.current = ws

        ws.onopen = () => {
          console.log('[WS] Connected to', url)
          setIsConnected(true)
          reconnectDelayRef.current = 1000 // Reset delay on success
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            setLastMessage(data)
          } catch (e) {
            console.error('[WS] Parse error:', e)
          }
        }

        ws.onerror = (error) => {
          console.error('[WS] Error:', error)
        }

        ws.onclose = () => {
          console.log('[WS] Disconnected')
          setIsConnected(false)
          wsRef.current = null

          // Exponential backoff: 1s → 2s → 4s → 8s → 16s → 30s (max)
          const delay = Math.min(reconnectDelayRef.current, 30000)
          console.log(`[WS] Reconnecting in ${delay}ms...`)

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectDelayRef.current = Math.min(delay * 2, 30000)
            connect()
          }, delay)
        }
      } catch (e) {
        console.error('[WS] Connection error:', e)
      }
    }

    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [url])

  return { isConnected, lastMessage }
}
