"""
VPS Market Data Connector
Fetches real-time data from VPS Securities API with CafeF fallback
API: https://bgapidatafeed.vps.com.vn/getliststockdata/{symbols}
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VPSMarketConnector:
    """
    Connector for VPS Securities real-time market data
    Primary source with CafeF fallback for reliability
    """

    VPS_API_URL = "https://bgapidatafeed.vps.com.vn/getliststockdata"

    def __init__(self, cache_ttl: int = 60, enable_cafef_fallback: bool = False):
        """
        Initialize VPS Market Connector

        Args:
            cache_ttl: Cache time-to-live in seconds (default 60s)
            enable_cafef_fallback: Enable CafeF fallback (default False to avoid circular import)
        """
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = cache_ttl
        self.connected = True
        self._cafef_fallback = None
        self._enable_cafef_fallback = enable_cafef_fallback

        logger.info("VPSMarketConnector initialized (CafeF fallback disabled to avoid circular import)")

    async def connect(self):
        """Connect to VPS API (no-op for HTTP API)"""
        self.connected = True
        logger.info("VPS connector ready")

    def _get_cached_or_fetch(self, symbols: List[str]) -> List[Dict]:
        """
        Get cached data or fetch fresh from VPS API

        Args:
            symbols: List of stock symbols

        Returns:
            List of stock data dicts
        """
        now = datetime.now()
        cache_key = ",".join(sorted(symbols))

        # Check cache
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (now - cached_time).total_seconds() < self._cache_ttl:
                logger.debug(f"Cache hit for {len(symbols)} symbols")
                return cached_data

        # Fetch from VPS
        try:
            symbols_str = ",".join(symbols)
            url = f"{self.VPS_API_URL}/{symbols_str}"

            resp = requests.get(url, headers=self.headers, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    # Cache successful response
                    self._cache[cache_key] = (data, now)
                    logger.info(f"✅ VPS API: Fetched {len(data)} stocks")
                    return data
                else:
                    logger.warning("VPS API returned empty data")
            else:
                logger.warning(f"VPS API error: {resp.status_code}")

        except Exception as e:
            logger.error(f"VPS API failed: {e}")

        # Fallback to CafeF if VPS fails (lazy load to avoid circular import)
        if self._enable_cafef_fallback:
            if self._cafef_fallback is None:
                try:
                    from quantum_stock.dataconnector.realtime_market import RealTimeMarketConnector
                    self._cafef_fallback = RealTimeMarketConnector(prefer_vps=False)
                    logger.info("CafeF fallback loaded")
                except Exception as e:
                    logger.warning(f"Failed to load CafeF fallback: {e}")

            if self._cafef_fallback:
                logger.info("Falling back to CafeF...")
                try:
                    cafef_data = self._cafef_fallback._get_cached_or_fetch()
                    # Filter for requested symbols
                    filtered = [
                        stock for stock in cafef_data
                        if stock.get('a', '').upper() in [s.upper() for s in symbols]
                    ]
                    if filtered:
                        logger.info(f"✅ CafeF fallback: {len(filtered)} stocks")
                        return self._convert_cafef_to_vps_format(filtered)
                except Exception as e:
                    logger.error(f"CafeF fallback failed: {e}")

        return []

    def _convert_cafef_to_vps_format(self, cafef_data: List[Dict]) -> List[Dict]:
        """
        Convert CafeF format to VPS format for compatibility

        CafeF fields: a=symbol, l=current price (x1000), k=change
        VPS fields: sym, lastPrice (x1000), r=reference, c=ceiling, f=floor
        """
        converted = []
        for stock in cafef_data:
            try:
                converted.append({
                    'sym': stock.get('a', ''),
                    'lastPrice': stock.get('l', 0),  # Already in x1000 format
                    'r': stock.get('re', 0),  # Reference price
                    'c': stock.get('b', 0),  # Ceiling
                    'f': stock.get('d', 0),  # Floor
                    'closePrice': stock.get('pc', 0),  # Previous close
                    'lot': stock.get('n', 0) or stock.get('totalvolume', 0),  # Volume
                    'changePc': stock.get('k', 0),  # % change
                    'fBVol': stock.get('tb', 0) or 0,  # Foreign buy volume
                    'fSVolume': stock.get('ts', 0) or 0,  # Foreign sell volume
                    'source': 'cafef_fallback'
                })
            except Exception as e:
                logger.warning(f"Failed to convert stock data: {e}")
                continue

        return converted

    async def get_stock_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch multiple stocks data from VPS API

        Args:
            symbols: List of stock symbols (e.g., ['SSI', 'VNM', 'FPT'])

        Returns:
            Dict with stocks data and metadata
        """
        if not symbols:
            return {"stocks": [], "count": 0, "source": "none"}

        data = self._get_cached_or_fetch(symbols)

        return {
            "stocks": data,
            "count": len(data),
            "symbols": symbols,
            "source": data[0].get('source', 'vps') if data else 'vps',
            "timestamp": datetime.now().isoformat()
        }

    async def get_single_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch single stock with VND conversion

        Args:
            symbol: Stock symbol (e.g., 'SSI')

        Returns:
            Dict with stock data in VND or None if not found
        """
        result = await self.get_stock_data([symbol])

        if result['count'] == 0:
            return None

        stock = result['stocks'][0]

        # Convert to VND (x1000 to actual VND)
        # Handle changePc which might be string or number
        change_pc = stock.get('changePc', 0)
        try:
            change_pc_float = float(change_pc) if change_pc else 0.0
        except (ValueError, TypeError):
            change_pc_float = 0.0

        return {
            'symbol': stock.get('sym', symbol),
            'price_vnd': stock.get('lastPrice', 0) * 1000,
            'reference_vnd': stock.get('r', 0) * 1000,
            'ceiling_vnd': stock.get('c', 0) * 1000,
            'floor_vnd': stock.get('f', 0) * 1000,
            'close_price_vnd': stock.get('closePrice', 0) * 1000,
            'volume': stock.get('lot', 0),
            'change_percent': change_pc_float,
            'price_display': f"{stock.get('lastPrice', 0) * 1000:,.0f} VND",
            'change_display': f"{change_pc_float:+.2f}%",
            'source': stock.get('source', 'vps'),
            'timestamp': datetime.now().isoformat()
        }

    async def get_foreign_flow(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Extract foreign trading flow (khối ngoại) data

        Args:
            symbols: List of stock symbols

        Returns:
            Dict with foreign buy/sell volumes and net flow
        """
        result = await self.get_stock_data(symbols)

        if result['count'] == 0:
            return {
                "total_buy_volume": 0,
                "total_sell_volume": 0,
                "net_volume": 0,
                "stocks": [],
                "flow_type": "NEUTRAL"
            }

        total_buy = 0
        total_sell = 0
        stocks_flow = []

        for stock in result['stocks']:
            symbol = stock.get('sym', '')

            # Safely convert to float
            try:
                fb_vol = float(stock.get('fBVol', 0) or 0)
                fs_vol = float(stock.get('fSVolume', 0) or 0)
                price = float(stock.get('lastPrice', 0) or 0)
            except (ValueError, TypeError):
                fb_vol = 0.0
                fs_vol = 0.0
                price = 0.0

            # Calculate values in VND
            fb_value = fb_vol * price * 1000
            fs_value = fs_vol * price * 1000

            total_buy += fb_value
            total_sell += fs_value

            if fb_vol > 0 or fs_vol > 0:
                stocks_flow.append({
                    'symbol': symbol,
                    'foreign_buy_volume': fb_vol,
                    'foreign_sell_volume': fs_vol,
                    'foreign_buy_value': fb_value,
                    'foreign_sell_value': fs_value,
                    'net_volume': fb_vol - fs_vol,
                    'net_value': fb_value - fs_value
                })

        # Sort by net value (descending)
        stocks_flow.sort(key=lambda x: x['net_value'], reverse=True)

        net_value = total_buy - total_sell
        net_value_bn = round(net_value / 1_000_000_000, 2)

        flow_type = "BUY" if net_value > 0 else "SELL" if net_value < 0 else "NEUTRAL"

        # Calculate total volumes with safe type conversion
        total_buy_vol = 0
        total_sell_vol = 0
        for s in result['stocks']:
            try:
                total_buy_vol += float(s.get('fBVol', 0) or 0)
                total_sell_vol += float(s.get('fSVolume', 0) or 0)
            except (ValueError, TypeError):
                pass

        return {
            "total_buy_volume": round(total_buy_vol),
            "total_sell_volume": round(total_sell_vol),
            "total_buy_value": total_buy,
            "total_sell_value": total_sell,
            "net_value": net_value,
            "net_value_billion": net_value_bn,
            "flow_type": flow_type,
            "stocks": stocks_flow,
            "summary": f"Khối ngoại {'mua ròng' if net_value > 0 else 'bán ròng' if net_value < 0 else 'cân bằng'} {abs(net_value_bn):.1f} tỷ",
            "timestamp": datetime.now().isoformat()
        }

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a specific stock (synchronous)

        Args:
            symbol: Stock symbol

        Returns:
            Current price in VND or None if not found
        """
        data = self._get_cached_or_fetch([symbol])

        if not data:
            return None

        stock = data[0]
        # Return price in VND (multiply by 1000)
        return stock.get('lastPrice', 0) * 1000

    async def get_market_depth(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market depth (order book) for a symbol
        Note: VPS basic API doesn't include order book, returns simplified data
        """
        result = await self.get_single_stock(symbol)
        if not result:
            return None

        return {
            "symbol": symbol,
            "bids": [[result['price_vnd'] * 0.99, 1000]],
            "asks": [[result['price_vnd'] * 1.01, 1000]],
            "timestamp": result['timestamp'],
            "note": "Order book not available in VPS basic API"
        }

    async def get_intraday_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get intraday tick data
        Note: VPS basic API doesn't include historical intraday, returns current snapshot
        """
        result = await self.get_single_stock(symbol)
        if not result:
            return None

        now = datetime.now()
        return [
            {
                "time": now.strftime("%H:%M"),
                "price": result['price_vnd'],
                "volume": result['volume'],
                "note": "Intraday history not available in VPS basic API"
            }
        ]


# Module-level singleton
_vps_connector: Optional[VPSMarketConnector] = None


def get_vps_connector() -> VPSMarketConnector:
    """Get or create VPS connector singleton"""
    global _vps_connector
    if _vps_connector is None:
        _vps_connector = VPSMarketConnector()
    return _vps_connector
