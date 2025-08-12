"""
Exchange Manager для управління підключеннями до різних бірж
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import ccxt.pro as ccxt
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class ExchangeConfig:
    name: str
    api_key: str
    secret: str
    sandbox: bool = False
    rate_limit: int = 1200  # requests per minute

class ExchangeManager:
    def __init__(self, exchange_configs: List[ExchangeConfig]):
        self.logger = logging.getLogger(__name__)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.rate_limits: Dict[str, int] = {}
        self.last_request_times: Dict[str, datetime] = {}

        for config in exchange_configs:
            self._initialize_exchange(config)

    def _initialize_exchange(self, config: ExchangeConfig):
        """Ініціалізація підключення до біржі"""
        try:
            if config.name.lower() == 'bybit':
                exchange_class = ccxt.bybit
            elif config.name.lower() == 'binance':
                exchange_class = ccxt.binance
            elif config.name.lower() == 'okx':
                exchange_class = ccxt.okx
            else:
                raise ValueError(f"Unsupported exchange: {config.name}")

            exchange = exchange_class({
                'apiKey': config.api_key,
                'secret': config.secret,
                'sandbox': config.sandbox,
                'enableRateLimit': True,
                'rateLimit': config.rate_limit,
            })

            self.exchanges[config.name] = exchange
            self.rate_limits[config.name] = config.rate_limit
            self.logger.info(f"Initialized {config.name} exchange")

        except Exception as e:
            self.logger.error(f"Failed to initialize {config.name}: {e}")
            raise

    async def test_connection(self, exchange_name: str) -> bool:
        """Перевірка підключення до біржі"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return False

            await exchange.load_markets()
            balance = await exchange.fetch_balance()
            self.logger.info(f"Connection to {exchange_name} successful")
            return True

        except Exception as e:
            self.logger.error(f"Connection test failed for {exchange_name}: {e}")
            return False

    async def fetch_ohlcv(self, exchange_name: str, symbol: str,
                         timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Отримання OHLCV даних"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {exchange_name} {symbol}: {e}")
            raise

    async def fetch_ticker(self, exchange_name: str, symbol: str) -> Dict:
        """Отримання поточної ціни"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            ticker = await exchange.fetch_ticker(symbol)
            return ticker

        except Exception as e:
            self.logger.error(f"Error fetching ticker for {exchange_name} {symbol}: {e}")
            raise

    async def create_order(self, exchange_name: str, symbol: str, order_type: str,
                          side: str, amount: float, price: float = None) -> Dict:
        """Створення ордеру"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            order = await exchange.create_order(symbol, order_type, side, amount, price)
            self.logger.info(f"Created {side} order for {amount} {symbol} on {exchange_name}")
            return order

        except Exception as e:
            self.logger.error(f"Error creating order on {exchange_name}: {e}")
            raise

    async def cancel_order(self, exchange_name: str, order_id: str, symbol: str) -> Dict:
        """Скасування ордеру"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            result = await exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Cancelled order {order_id} on {exchange_name}")
            return result

        except Exception as e:
            self.logger.error(f"Error cancelling order on {exchange_name}: {e}")
            raise

    async def fetch_balance(self, exchange_name: str) -> Dict:
        """Отримання балансу"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            balance = await exchange.fetch_balance()
            return balance

        except Exception as e:
            self.logger.error(f"Error fetching balance for {exchange_name}: {e}")
            raise

    async def fetch_positions(self, exchange_name: str, symbol: str = None) -> List[Dict]:
        """Отримання позицій (для ф'ючерсів)"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            if hasattr(exchange, 'fetch_positions'):
                positions = await exchange.fetch_positions(symbol)
                return positions
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error fetching positions for {exchange_name}: {e}")
            return []

    async def watch_ohlcv(self, exchange_name: str, symbol: str, timeframe: str):
        """WebSocket підписка на OHLCV дані"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                raise ValueError(f"Exchange {exchange_name} not found")

            while True:
                ohlcv = await exchange.watch_ohlcv(symbol, timeframe)
                yield ohlcv

        except Exception as e:
            self.logger.error(f"Error watching OHLCV for {exchange_name} {symbol}: {e}")
            raise

    async def close_all(self):
        """Закриття всіх підключень"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self.logger.info(f"Closed connection to {exchange_name}")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_name}: {e}")

    def get_exchange_names(self) -> List[str]:
        """Отримання списку доступних бірж"""
        return list(self.exchanges.keys())

    def get_exchange(self, exchange_name: str) -> Optional[ccxt.Exchange]:
        """Отримання об'єкту біржі"""
        return self.exchanges.get(exchange_name)