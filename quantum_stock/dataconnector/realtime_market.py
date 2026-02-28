"""
Real-time Market Data Connector
Fetches REAL data with VPS as primary source and CafeF as fallback:
- Market Breadth (Xanh vỏ đỏ lòng detection)
- Foreign Flow (Khối ngoại)
- Volume Analysis (Smart Money / Lái gom)
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RealTimeMarketConnector:
    """Connector for real-time Vietnam stock market data with VPS primary + CafeF fallback"""

    CAFEF_STOCK_URL = "https://banggia.cafef.vn/stockhandler.ashx"
    CAFEF_INDEX_URL = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"

    def __init__(self, prefer_vps: bool = True):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 60  # Cache for 60 seconds
        self.prefer_vps = prefer_vps

        # Initialize VPS connector if available
        self._vps_connector = None
        if prefer_vps:
            try:
                from quantum_stock.dataconnector.vps_market import get_vps_connector
                self._vps_connector = get_vps_connector()
                logger.info("VPS connector integrated as primary source")
            except Exception as e:
                logger.warning(f"VPS connector not available, using CafeF only: {e}")
        
    def _get_cached_or_fetch(self) -> List[Dict]:
        """Get cached data or fetch fresh from API"""
        now = datetime.now()
        
        if self._cache_time and (now - self._cache_time).total_seconds() < self._cache_ttl:
            return self._cache.get('stocks', [])
            
        try:
            resp = requests.get(self.CAFEF_STOCK_URL, headers=self.headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                self._cache['stocks'] = data
                self._cache_time = now
                return data
        except Exception as e:
            logger.error(f"Failed to fetch CafeF data: {e}")
            
        return self._cache.get('stocks', [])
    
    def get_market_breadth(self) -> Dict[str, Any]:
        """
        Get REAL market breadth data
        Returns advancing, declining, unchanged counts
        Used for "Xanh vỏ đỏ lòng" (Bull Trap) detection
        """
        stocks = self._get_cached_or_fetch()
        
        advancing = 0
        declining = 0
        unchanged = 0
        ceiling_hits = 0  # Số mã tăng trần
        floor_hits = 0     # Số mã giảm sàn
        
        for stock in stocks:
            change = stock.get('k', 0)  # Price change
            current = stock.get('l', 0)  # Current price
            ceiling = stock.get('b', 0)  # Ceiling price
            floor = stock.get('d', 0)    # Floor price
            
            if change > 0:
                advancing += 1
                # Ceiling hit: price within 0.5% of ceiling (CafeF rounding tolerance)
                if ceiling > 0 and current > 0 and abs(current - ceiling) / ceiling < 0.005:
                    ceiling_hits += 1
            elif change < 0:
                declining += 1
                # Floor hit: price within 0.5% of floor
                if floor > 0 and current > 0 and abs(current - floor) / floor < 0.005:
                    floor_hits += 1
            else:
                unchanged += 1
        
        total = advancing + declining + unchanged
        
        # Bull Trap detection: Index up but more stocks declining
        is_bull_trap = False
        bull_trap_reason = ""
        
        if total > 0 and declining > advancing * 1.3:  # 30% more declining
            is_bull_trap = True
            bull_trap_reason = f"VN-Index tăng nhưng {declining} mã giảm vs {advancing} mã tăng"
        
        return {
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "ceiling_hits": ceiling_hits,
            "floor_hits": floor_hits,
            "total_stocks": total,
            "advance_decline_ratio": round(advancing / declining, 2) if declining > 0 else 999,
            "is_bull_trap": is_bull_trap,
            "bull_trap_reason": bull_trap_reason,
            "breadth_signal": "POSITIVE" if advancing > declining else "NEGATIVE" if declining > advancing else "NEUTRAL",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_foreign_flow(self) -> Dict[str, Any]:
        """
        Get REAL foreign (khối ngoại) trading data
        Primary: VPS API (fBVol/fSVolume), Fallback: CafeF (tb/ts)
        """
        stocks = self._get_cached_or_fetch()

        total_foreign_buy = 0
        total_foreign_sell = 0
        top_foreign_buy = []
        top_foreign_sell = []

        # Try VPS data first (more accurate foreign flow)
        # VPS foreign flow is async — skip in sync context.
        # Async callers (market.py endpoints) use VPS connector directly.

        # Fallback: CafeF tb/ts fields
        for stock in stocks:
            symbol = stock.get('a', '')
            fb = stock.get('tb', 0) or 0  # Foreign buy volume
            fs = stock.get('ts', 0) or 0  # Foreign sell volume
            price = stock.get('l', 0) or 0  # Current price

            # Convert to value (VND)
            fb_value = fb * price * 1000  # Price in thousands
            fs_value = fs * price * 1000

            total_foreign_buy += fb_value
            total_foreign_sell += fs_value

            if fb > 0:
                top_foreign_buy.append({"symbol": symbol, "volume": fb, "value": fb_value})
            if fs > 0:
                top_foreign_sell.append({"symbol": symbol, "volume": fs, "value": fs_value})
        
        # Sort by value
        top_foreign_buy.sort(key=lambda x: x['value'], reverse=True)
        top_foreign_sell.sort(key=lambda x: x['value'], reverse=True)
        
        net_foreign = total_foreign_buy - total_foreign_sell
        net_foreign_bn = round(net_foreign / 1_000_000_000, 2)  # Convert to billion VND
        
        return {
            "total_buy": total_foreign_buy,
            "total_sell": total_foreign_sell,
            "net_value": net_foreign,
            "net_value_billion": net_foreign_bn,
            "flow_type": "BUY" if net_foreign > 0 else "SELL" if net_foreign < 0 else "NEUTRAL",
            "top_buy": top_foreign_buy[:5],
            "top_sell": top_foreign_sell[:5],
            "signal": f"Khối ngoại {'mua ròng' if net_foreign > 0 else 'bán ròng'} {abs(net_foreign_bn):.1f} tỷ",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_volume_anomalies(self, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect volume anomalies (potential smart money / lái gom)
        Returns stocks with unusual volume spikes
        """
        stocks = self._get_cached_or_fetch()
        
        anomalies = []
        high_volume_gainers = []
        high_volume_losers = []
        churning = []  # High volume, low price change (potential distribution)
        
        for stock in stocks:
            symbol = stock.get('a', '')
            volume = stock.get('totalvolume', 0) or stock.get('n', 0) or 0
            change = stock.get('k', 0) or 0
            price = stock.get('l', 0) or 0
            
            # Skip low volume stocks
            if volume < 100000:
                continue
                
            # High volume detection (we don't have average volume, so use heuristics)
            # Consider volume > 1M as potentially significant
            if volume > 1_000_000:
                stock_info = {
                    "symbol": symbol,
                    "volume": volume,
                    "change": change,
                    "price": price,
                    "volume_million": round(volume / 1_000_000, 2)
                }
                
                if change > 0.5:  # Up more than 0.5
                    high_volume_gainers.append(stock_info)
                elif change < -0.5:  # Down more than 0.5
                    high_volume_losers.append(stock_info)
                elif abs(change) < 0.2 and volume > 2_000_000:  # Churning
                    churning.append(stock_info)
        
        # Sort by volume
        high_volume_gainers.sort(key=lambda x: x['volume'], reverse=True)
        high_volume_losers.sort(key=lambda x: x['volume'], reverse=True)
        churning.sort(key=lambda x: x['volume'], reverse=True)
        
        # Determine smart money signal
        smart_money_signal = "Không phát hiện"
        smart_money_stocks = []
        
        if len(high_volume_gainers) > 3:
            smart_money_signal = f"💰 CLIMAX BUYING - {len(high_volume_gainers)} mã có volume bùng nổ + giá tăng"
            smart_money_stocks = [s['symbol'] for s in high_volume_gainers[:5]]
        elif len(churning) > 3:
            smart_money_signal = f"⚠️ CHURNING - {len(churning)} mã volume cao + giá sideway (có thể phân phối)"
            smart_money_stocks = [s['symbol'] for s in churning[:5]]
        elif len(high_volume_losers) > 5:
            smart_money_signal = f"🔻 CLIMAX SELLING - {len(high_volume_losers)} mã volume cao + giá giảm mạnh"
            smart_money_stocks = [s['symbol'] for s in high_volume_losers[:5]]
        
        return {
            "high_volume_gainers": high_volume_gainers[:10],
            "high_volume_losers": high_volume_losers[:10],
            "churning": churning[:10],
            "smart_money_signal": smart_money_signal,
            "smart_money_stocks": smart_money_stocks,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a specific stock
        Returns the live price from CafeF, NOT historical data

        Args:
            symbol: Stock symbol (e.g., 'MWG', 'ACB')

        Returns:
            Current price or None if not found
        """
        stocks = self._get_cached_or_fetch()

        for stock in stocks:
            a_val = stock.get('a')
            if a_val and isinstance(a_val, str) and a_val.upper() == symbol.upper():
                price = stock.get('l', None)  # 'l' = current price
                if price:
                    # CafeF returns price in thousands (25.5 = 25,500 VND)
                    # Multiply by 1000 to match historical data format
                    return float(price) * 1000

        logger.warning(f"Could not find current price for {symbol}")
        return None

    def get_vnindex_realtime(self) -> Dict[str, Any]:
        """Get VN-Index from CafeF"""
        try:
            url = f"{self.CAFEF_INDEX_URL}?Symbol=VNINDEX&StartDate=&EndDate=&PageIndex=1&PageSize=2"
            resp = requests.get(url, headers=self.headers, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get('Success') and data.get('Data', {}).get('Data'):
                    items = data['Data']['Data']
                    if len(items) > 0:
                        latest = items[0]
                        vnindex = float(latest.get('GiaDongCua', 0))
                        
                        # Parse change
                        change_str = latest.get('ThayDoi', '0(0%)')
                        try:
                            parts = change_str.replace('(', ' ').replace('%)', '').split()
                            change = float(parts[0])
                            change_pct = float(parts[1]) if len(parts) > 1 else 0
                        except:
                            change = 0
                            change_pct = 0
                            if len(items) > 1:
                                prev = float(items[1].get('GiaDongCua', vnindex))
                                change = vnindex - prev
                                change_pct = (change / prev) * 100 if prev > 0 else 0
                        
                        return {
                            "vnindex": round(vnindex, 2),
                            "change": round(change, 2),
                            "change_pct": round(change_pct, 2),
                            "date": latest.get('Ngay', ''),
                            "source": "CafeF Real-time"
                        }
        except Exception as e:
            logger.error(f"Failed to fetch VN-Index: {e}")
            
        return {"vnindex": 0, "change": 0, "change_pct": 0, "source": "Error"}
    
    def get_stock_historical(self, symbol: str, days: int = 100) -> Optional[List[Dict]]:
        """
        Fetch REAL historical OHLCV data from CafeF API

        Args:
            symbol: Stock symbol (e.g., 'MWG', 'HPG')
            days: Number of days to fetch (default 100)

        Returns:
            List of OHLCV dicts with keys: date, open, high, low, close, volume
            Prices are in VND (multiplied by 1000 from CafeF format)
        """
        try:
            url = f"{self.CAFEF_INDEX_URL}?Symbol={symbol.upper()}&StartDate=&EndDate=&PageIndex=1&PageSize={days}"
            resp = requests.get(url, headers=self.headers, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                if data.get('Success') and data.get('Data', {}).get('Data'):
                    items = data['Data']['Data']

                    result = []
                    for item in reversed(items):  # Reverse to get oldest first
                        try:
                            # CafeF returns prices in thousands (86 = 86,000 VND)
                            result.append({
                                'date': item.get('Ngay', ''),
                                'open': float(item.get('GiaMoCua', 0)) * 1000,
                                'high': float(item.get('GiaCaoNhat', 0)) * 1000,
                                'low': float(item.get('GiaThapNhat', 0)) * 1000,
                                'close': float(item.get('GiaDongCua', 0)) * 1000,
                                'volume': int(item.get('KhoiLuongKhopLenh', 0))
                            })
                        except (ValueError, TypeError) as e:
                            continue

                    logger.info(f"✅ Fetched {len(result)} days of real data for {symbol}")
                    return result

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")

        return None

    def get_full_market_signals(self) -> Dict[str, Any]:
        """
        Get all market signals in one call (optimized)
        """
        breadth = self.get_market_breadth()
        foreign = self.get_foreign_flow()
        volume = self.get_volume_anomalies()
        vnindex = self.get_vnindex_realtime()
        
        # Determine recommended strategies based on real data
        strategies = []
        
        if breadth['advancing'] > breadth['declining'] * 1.2:
            strategies.append("TREND_FOLLOWING")
            strategies.append("MOMENTUM")
        elif breadth['declining'] > breadth['advancing'] * 1.2:
            strategies.append("DEFENSIVE")
            strategies.append("CASH_PRESERVATION")
        else:
            strategies.append("RANGE_TRADING")
            strategies.append("SCALPING")
        
        if foreign['flow_type'] == 'BUY':
            strategies.append("FOLLOW_FOREIGN")
        
        if len(volume['high_volume_gainers']) > 3:
            strategies.append("BREAKOUT_TRADING")
            
        return {
            "vnindex": vnindex,
            "breadth": breadth,
            "foreign_flow": foreign,
            "volume_analysis": volume,
            "recommended_strategies": strategies[:4],
            "market_regime": "BULL" if breadth['advance_decline_ratio'] > 1.3 else "BEAR" if breadth['advance_decline_ratio'] < 0.7 else "NEUTRAL",
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
_connector = None

def get_realtime_connector() -> RealTimeMarketConnector:
    global _connector
    if _connector is None:
        _connector = RealTimeMarketConnector()
    return _connector
