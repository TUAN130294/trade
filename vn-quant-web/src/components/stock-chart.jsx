import { useEffect, useRef } from 'react'
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts'

export function StockChart({ data }) {
  const chartContainerRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!data || data.length === 0 || !chartContainerRef.current) return

    // Cleanup previous chart
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    try {
      const chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: '#94a3b8'
        },
        grid: {
          vertLines: { color: 'rgba(255,255,255,0.05)' },
          horzLines: { color: 'rgba(255,255,255,0.05)' }
        },
        width: chartContainerRef.current.clientWidth || 600,
        height: 400,
      })
      chartRef.current = chart

      // v5 API: Use addSeries with CandlestickSeries type
      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#0bda5e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#0bda5e',
        wickDownColor: '#ef4444',
      })

      // Parse, filter, sort, then color by prev close (VN market style)
      const parsed = data
        .filter(d => d.date && d.open && d.high && d.low && d.close)
        .map(d => ({
          time: String(d.date).split('T')[0],
          open: Number(d.open),
          high: Number(d.high),
          low: Number(d.low),
          close: Number(d.close)
        }))
        .sort((a, b) => a.time.localeCompare(b.time))

      // Color each candle: green if close >= prev close, red otherwise
      const chartData = parsed.map((d, i) => {
        const prevClose = i > 0 ? parsed[i - 1].close : d.open
        const isUp = d.close >= prevClose
        return {
          ...d,
          color: isUp ? '#0bda5e' : '#ef4444',
          wickColor: isUp ? '#0bda5e' : '#ef4444',
          borderColor: isUp ? '#0bda5e' : '#ef4444',
        }
      })

      if (chartData.length > 0) {
        candlestickSeries.setData(chartData)
        chart.timeScale().fitContent()
      }

      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth })
        }
      }
      window.addEventListener('resize', handleResize)

      return () => {
        window.removeEventListener('resize', handleResize)
        if (chartRef.current) {
          chartRef.current.remove()
          chartRef.current = null
        }
      }
    } catch (err) {
      console.error('Chart error:', err)
    }
  }, [data])

  return <div ref={chartContainerRef} className="w-full h-[400px] min-h-[400px]" />
}
