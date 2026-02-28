"""
Broker API Integration - Level 5 Agentic Platform
Abstract interfaces and implementations for Vietnam stock brokers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type"""
    LIMIT = "LO"          # Limit Order
    MARKET_OPEN = "ATO"   # At The Open
    MARKET_CLOSE = "ATC"  # At The Close
    MARKET = "MP"         # Market Price (VN30 only)
    HIDDEN = "PLO"        # Post Limit Order


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"           # Waiting for market
    QUEUED = "QUEUED"             # Sent to exchange
    PARTIAL = "PARTIAL"           # Partially filled
    FILLED = "FILLED"             # Fully filled
    CANCELLED = "CANCELLED"       # Cancelled
    REJECTED = "REJECTED"         # Rejected by exchange
    EXPIRED = "EXPIRED"           # Order expired


class AccountType(Enum):
    """Account type"""
    CASH = "CASH"
    MARGIN = "MARGIN"
    DERIVATIVES = "DERIVATIVES"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    tax: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'commission': self.commission,
            'tax': self.tax,
            'metadata': self.metadata
        }
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.QUEUED, OrderStatus.PARTIAL]
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees"""
        value = self.filled_quantity * self.filled_price
        return value + self.commission + self.tax


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float = 0.0
    cost_basis: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl': self.realized_pnl,
            'cost_basis': self.cost_basis
        }


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    account_type: AccountType
    name: str
    cash_balance: float
    buying_power: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    nav: float = 0.0  # Net Asset Value
    total_pnl: float = 0.0
    positions: List[Position] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'account_id': self.account_id,
            'account_type': self.account_type.value,
            'name': self.name,
            'cash_balance': self.cash_balance,
            'buying_power': self.buying_power,
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'nav': self.nav,
            'total_pnl': self.total_pnl,
            'positions': [p.to_dict() for p in self.positions]
        }


class BrokerAPI(ABC):
    """
    Abstract base class for broker API integration.
    Implement this for each Vietnam broker (SSI, VPS, VNDirect, etc.)
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_authenticated = False
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with broker API"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information and balances"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, 
                         order_type: OrderType, quantity: int,
                         price: float = None) -> Order:
        """Place a new order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_market_price(self, symbol: str) -> Dict[str, float]:
        """Get current market price for a symbol"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Get order book for a symbol"""
        pass


class PaperTradingBroker(BrokerAPI):
    """
    Paper trading broker for simulation and testing.
    Implements all broker functionality without real money.
    """
    
    # Vietnam market constants
    COMMISSION_RATE = 0.0015  # 0.15% for most brokers
    SELLING_TAX = 0.001       # 0.1% tax on selling
    LOT_SIZE = 100
    PRICE_STEP = 10           # VND
    CEILING_PCT = 0.07        # 7% for HOSE
    FLOOR_PCT = 0.07          # -7% for HOSE
    
    def __init__(self, initial_balance: float = 100_000_000):
        """
        Initialize paper trading broker.
        
        Args:
            initial_balance: Initial cash balance in VND (default 100M)
        """
        super().__init__()
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.order_counter = 0
        self.trade_history: List[Dict[str, Any]] = []
        
        # Simulated market prices (in production, fetch from data provider)
        self.market_prices: Dict[str, Dict[str, float]] = {}
        
        self._load_state()
    
    def _load_state(self):
        """Load state from file if exists"""
        state_file = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'data', 'paper_trading_state.json'
        )
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.cash_balance = state.get('cash_balance', self.initial_balance)
                    self.order_counter = state.get('order_counter', 0)
                    
                    # Restore positions
                    for pos_data in state.get('positions', []):
                        self.positions[pos_data['symbol']] = Position(**pos_data)
                    
                    # Restore orders
                    for order_data in state.get('orders', []):
                        order_data['side'] = OrderSide(order_data['side'])
                        order_data['order_type'] = OrderType(order_data['order_type'])
                        order_data['status'] = OrderStatus(order_data['status'])
                        order_data['created_at'] = datetime.fromisoformat(order_data['created_at'])
                        order_data['updated_at'] = datetime.fromisoformat(order_data['updated_at'])
                        self.orders[order_data['order_id']] = Order(**order_data)
                        
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save state to file"""
        state_file = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'data', 'paper_trading_state.json'
        )
        
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        try:
            state = {
                'cash_balance': self.cash_balance,
                'order_counter': self.order_counter,
                'positions': [pos.to_dict() for pos in self.positions.values()],
                'orders': [order.to_dict() for order in self.orders.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    async def authenticate(self) -> bool:
        """Paper trading is always authenticated"""
        self.is_authenticated = True
        logger.info("Paper trading broker authenticated")
        return True
    
    async def get_account_info(self) -> AccountInfo:
        """Get paper trading account information"""
        positions = await self.get_positions()
        
        # Calculate NAV
        total_position_value = sum(pos.market_value for pos in positions)
        nav = self.cash_balance + total_position_value
        
        # Calculate total P&L
        total_pnl = nav - self.initial_balance
        
        return AccountInfo(
            account_id="PAPER_001",
            account_type=AccountType.CASH,
            name="Paper Trading Account",
            cash_balance=self.cash_balance,
            buying_power=self.cash_balance,  # No margin for paper
            nav=nav,
            total_pnl=total_pnl,
            positions=positions
        )
    
    async def get_positions(self) -> List[Position]:
        """Get current positions with updated P&L"""
        updated_positions = []
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                # Get current price
                market_data = await self.get_market_price(symbol)
                current_price = market_data.get('last', position.current_price)
                
                # Update position
                market_value = position.quantity * current_price
                cost_basis = position.quantity * position.avg_price
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                position.current_price = current_price
                position.market_value = market_value
                position.cost_basis = cost_basis
                position.unrealized_pnl = unrealized_pnl
                position.unrealized_pnl_pct = unrealized_pnl_pct
                
                updated_positions.append(position)
        
        return updated_positions
    
    async def place_order(self, symbol: str, side: OrderSide,
                         order_type: OrderType, quantity: int,
                         price: float = None) -> Order:
        """
        Place a paper trading order.
        Simulates immediate fill at current market price.
        """
        # Validate lot size
        if quantity % self.LOT_SIZE != 0:
            raise ValueError(f"Quantity must be multiple of {self.LOT_SIZE}")
        
        # Get market price
        market_data = await self.get_market_price(symbol)
        market_price = market_data.get('last', price)
        
        if price is None:
            price = market_price
        
        # Validate price limits (ceiling/floor)
        ref_price = market_data.get('reference', market_price)
        ceiling = ref_price * (1 + self.CEILING_PCT)
        floor = ref_price * (1 - self.CEILING_PCT)
        
        if price > ceiling:
            raise ValueError(f"Price {price} exceeds ceiling {ceiling}")
        if price < floor:
            raise ValueError(f"Price {price} below floor {floor}")
        
        # Create order
        self.order_counter += 1
        order_id = f"PAPER_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING
        )
        
        # Validate buying power / position for sell
        order_value = quantity * price
        
        if side == OrderSide.BUY:
            # Check buying power
            commission = order_value * self.COMMISSION_RATE
            total_required = order_value + commission
            
            if total_required > self.cash_balance:
                order.status = OrderStatus.REJECTED
                order.metadata['reject_reason'] = "Insufficient buying power"
                self.orders[order_id] = order
                return order
        else:
            # Check position
            position = self.positions.get(symbol)
            if not position or position.quantity < quantity:
                order.status = OrderStatus.REJECTED
                order.metadata['reject_reason'] = "Insufficient position"
                self.orders[order_id] = order
                return order
        
        # Simulate immediate fill at limit price (for paper trading)
        order = await self._execute_order(order, market_price)
        
        self.orders[order_id] = order
        self._save_state()
        
        logger.info(f"Order placed: {order_id} {side.value} {quantity} {symbol} @ {price}")
        
        return order
    
    async def _execute_order(self, order: Order, market_price: float) -> Order:
        """Execute an order (immediate fill for paper trading)"""
        # Apply slippage simulation (0.1-0.3% for VN market)
        import random
        slippage_pct = random.uniform(0.001, 0.003)  # 0.1% to 0.3%

        if order.side == OrderSide.BUY:
            # Buy orders fill at slightly higher price (slippage against)
            execution_price = market_price * (1 + slippage_pct)
        else:
            # Sell orders fill at slightly lower price (slippage against)
            execution_price = market_price * (1 - slippage_pct)

        # Round to tick size (simplified)
        if execution_price < 10:
            execution_price = round(execution_price, 2)
        elif execution_price < 50:
            execution_price = round(execution_price * 20) / 20  # 0.05 tick
        else:
            execution_price = round(execution_price, 1)

        logger.info(f"Slippage applied: {slippage_pct*100:.2f}% -> {market_price} -> {execution_price}")
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.updated_at = datetime.now()
        
        # Calculate fees
        order_value = order.quantity * execution_price
        order.commission = order_value * self.COMMISSION_RATE
        
        if order.side == OrderSide.SELL:
            order.tax = order_value * self.SELLING_TAX
        
        # Update cash balance and positions
        if order.side == OrderSide.BUY:
            # Deduct cash
            self.cash_balance -= (order_value + order.commission)
            
            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_cost = (pos.quantity * pos.avg_price) + order_value + order.commission
                total_qty = pos.quantity + order.quantity
                pos.avg_price = total_cost / total_qty
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=(order_value + order.commission) / order.quantity,
                    current_price=execution_price,
                    market_value=order_value,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0
                )
        else:
            # Add cash (minus fees and tax)
            self.cash_balance += (order_value - order.commission - order.tax)
            
            # Calculate realized P&L
            pos = self.positions[order.symbol]
            cost = order.quantity * pos.avg_price
            proceeds = order_value - order.commission - order.tax
            realized_pnl = proceeds - cost
            
            pos.realized_pnl += realized_pnl
            pos.quantity -= order.quantity
            
            # Remove position if empty
            if pos.quantity <= 0:
                del self.positions[order.symbol]
        
        # Record trade
        self.trade_history.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'commission': order.commission,
            'tax': order.tax,
            'timestamp': datetime.now().isoformat()
        })
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper trading order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        self._save_state()
        logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status"""
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")
        
        return self.orders[order_id]
    
    async def get_market_price(self, symbol: str) -> Dict[str, float]:
        """Get market price - tries real data from VPS/CafeF, falls back to simulated"""
        if symbol in self.market_prices:
            return self.market_prices[symbol]

        # Try VPS connector first for REAL market data
        try:
            from quantum_stock.dataconnector.vps_market import get_vps_connector
            vps = get_vps_connector()
            price = vps.get_stock_price(symbol)  # Synchronous call, returns VND
            if price and price > 0:
                logger.info(f"✅ VPS real price for {symbol}: {price:,.0f} VND")
                return {
                    'last': price,
                    'bid': price * 0.999,
                    'ask': price * 1.001,
                    'reference': price
                }
        except Exception as e:
            logger.debug(f"VPS price fetch failed for {symbol}: {e}")

        # Try CafeF connector as fallback
        try:
            from quantum_stock.dataconnector.realtime_market import get_realtime_connector
            connector = get_realtime_connector()
            price = connector.get_stock_price(symbol)  # Synchronous, returns VND
            if price and price > 0:
                logger.info(f"✅ CafeF real price for {symbol}: {price:,.0f} VND")
                return {
                    'last': price,
                    'bid': price * 0.999,
                    'ask': price * 1.001,
                    'reference': price
                }
        except Exception as e:
            logger.debug(f"CafeF price fetch failed for {symbol}: {e}")

        # Fall back to hardcoded prices (last resort for testing/demo)
        default_prices = {
            'VNM': {'last': 78500, 'bid': 78400, 'ask': 78600, 'reference': 78000},
            'HPG': {'last': 27800, 'bid': 27700, 'ask': 27900, 'reference': 27500},
            'FPT': {'last': 128000, 'bid': 127900, 'ask': 128200, 'reference': 127500},
            'VCB': {'last': 92500, 'bid': 92400, 'ask': 92600, 'reference': 92000},
            'MBB': {'last': 25300, 'bid': 25200, 'ask': 25400, 'reference': 25000},
            'VIC': {'last': 45600, 'bid': 45500, 'ask': 45700, 'reference': 45000},
            'VHM': {'last': 48200, 'bid': 48100, 'ask': 48300, 'reference': 48000},
            'TCB': {'last': 23500, 'bid': 23400, 'ask': 23600, 'reference': 23000},
            'ACB': {'last': 26500, 'bid': 26400, 'ask': 26600, 'reference': 26000},
            'HDB': {'last': 32800, 'bid': 32700, 'ask': 32900, 'reference': 32500},
            'STB': {'last': 18500, 'bid': 18400, 'ask': 18600, 'reference': 18000},
            'TPB': {'last': 39500, 'bid': 39400, 'ask': 39600, 'reference': 39000},
            'SSI': {'last': 45200, 'bid': 45100, 'ask': 45300, 'reference': 45000},
        }

        if symbol in default_prices:
            logger.warning(f"⚠️ Using fallback price for {symbol} (real data unavailable)")
            return default_prices[symbol]

        # Last resort: return zero price instead of random
        logger.warning(f"⚠️ No price data for {symbol} - returning zero")
        return {
            'last': 0,
            'bid': 0,
            'ask': 0,
            'reference': 0
        }
    
    async def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Get simulated order book"""
        market_data = await self.get_market_price(symbol)
        last_price = market_data['last']
        
        # Generate simulated orderbook
        bids = []
        asks = []
        
        for i in range(1, 4):
            bids.append({
                'price': last_price - i * 100,
                'volume': 1000 * i
            })
            asks.append({
                'price': last_price + i * 100,
                'volume': 1000 * i
            })
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'last': last_price,
            'timestamp': datetime.now().isoformat()
        }
    
    def set_market_price(self, symbol: str, 
                         last: float, 
                         bid: float = None,
                         ask: float = None,
                         reference: float = None):
        """Set simulated market price for testing"""
        self.market_prices[symbol] = {
            'last': last,
            'bid': bid or last - 100,
            'ask': ask or last + 100,
            'reference': reference or last
        }
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get all trade history"""
        return self.trade_history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        nav = self.cash_balance + sum(
            pos.market_value for pos in self.positions.values()
        )
        
        total_pnl = nav - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'initial_balance': self.initial_balance,
            'cash_balance': self.cash_balance,
            'positions_value': nav - self.cash_balance,
            'nav': nav,
            'total_pnl': total_pnl,
            'total_pnl_pct': round(total_pnl_pct, 2),
            'realized_pnl': total_realized,
            'unrealized_pnl': total_unrealized,
            'total_trades': len(self.trade_history),
            'open_positions': len(self.positions)
        }
    
    def reset(self):
        """Reset paper trading account"""
        self.cash_balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.trade_history.clear()
        self.order_counter = 0
        self._save_state()
        logger.info("Paper trading account reset")


class BrokerFactory:
    """Factory for creating broker instances"""
    
    @staticmethod
    def create(broker_type: str = "paper", **kwargs) -> BrokerAPI:
        """
        Create a broker instance.
        
        Args:
            broker_type: Type of broker ("paper", "ssi", "vps", "vndirect")
            **kwargs: Additional broker-specific arguments
            
        Returns:
            BrokerAPI instance
        """
        if broker_type.lower() == "paper":
            return PaperTradingBroker(**kwargs)
        elif broker_type.lower() == "ssi":
            # TODO: Implement SSI broker
            raise NotImplementedError("SSI broker not yet implemented")
        elif broker_type.lower() == "vps":
            # TODO: Implement VPS broker
            raise NotImplementedError("VPS broker not yet implemented")
        elif broker_type.lower() == "vndirect":
            # TODO: Implement VNDirect broker
            raise NotImplementedError("VNDirect broker not yet implemented")
        else:
            raise ValueError(f"Unknown broker type: {broker_type}")
