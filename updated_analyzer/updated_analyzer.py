
#!/usr/bin/env python3
"""
Розширений модульний аналізатор з генетичним алгоритмом оптимізації
Включає всі режими оптимізації, повні діапазони параметрів та genetic режим
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import concurrent.futures
from functools import lru_cache
import threading
import weakref
import os
import json
import csv
import random
import copy
from itertools import product
import argparse
import time
import gc
import psutil
import aiofiles


@dataclass
class Signal:
    """Структура торгового сигналу"""
    pair: str
    direction: str
    time: datetime
    rsi: float
    rsi_sma: float


@dataclass
class TradeResult:
    """Результат торгівлі"""
    pair: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_percent: float
    hold_time: float
    status: str
    exit_reason: str
    filtered_out: bool = False
    filter_info: str = ""


class DataCache:
    """Високопродуктивний кеш даних з автоматичним управлінням пам'яттю"""

    def __init__(self, max_memory_mb: int = 4096):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._access_times: Dict[str, float] = {}
        self._max_memory = max_memory_mb * 1024 * 1024
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if key in self._cache:
                self._access_times[key] = datetime.now().timestamp()
                return self._cache[key]
        return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._cleanup_if_needed()
            self._cache[key] = df
            self._access_times[key] = datetime.now().timestamp()

    def _cleanup_if_needed(self):
        """Очищення кешу при необхідності"""
        current_memory = sum(df.memory_usage(deep=True).sum() for df in self._cache.values())

        if current_memory > self._max_memory:
            # Видаляємо 30% найменш використовуваних
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
            to_remove = len(sorted_items) // 3

            for key, _ in sorted_items[:to_remove]:
                if key in self._cache:
                    del self._cache[key]
                    del self._access_times[key]


class TechnicalIndicators:
    """Розрахунок технічних індикаторів"""

    @staticmethod
    def calculate_rsi_vectorized(close: pd.Series, period: int = 14) -> pd.Series:
        """ТОЧНА копія логіки з аналізатора"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_sma(series: pd.Series, period: int = 14) -> pd.Series:
        """ТОЧНА копія логіки з аналізатора"""
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Розрахунок ATR для volatility filter"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_price_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Розрахунок волатильності ціни"""
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std() * 100


class DataProvider:
    """Постачальник ринкових даних"""

    def __init__(self, exchange_config: Dict):
        import ccxt
        exchange_name = exchange_config.get('exchange', 'bybit')
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'rateLimit': 30,
            'options': {'defaultType': 'spot', 'recvWindow': 10000}
        })
        self.cache = DataCache()
        self.semaphore = asyncio.Semaphore(20)

    async def fetch_data_batch(self, symbols: List[str], signals: List[Signal],
                               config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Пакетне завантаження даних для всіх символів ОДРАЗУ"""

        # Групуємо сигнали по парах
        pair_signals = {}
        for signal in signals:
            if signal.pair not in pair_signals:
                pair_signals[signal.pair] = []
            pair_signals[signal.pair].append(signal)

        # Створюємо задачі для паралельного завантаження
        tasks = []
        for symbol in symbols:
            if symbol in pair_signals:
                task = self._fetch_symbol_data(symbol, pair_signals[symbol], config)
                tasks.append((symbol, task))

        # Виконуємо ВСІ завантаження паралельно
        results = {}
        completed = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for (symbol, _), result in zip(tasks, completed):
            if not isinstance(result, Exception) and result:
                results[symbol] = result

        return results

    async def _fetch_symbol_data(self, symbol: str, signals: List[Signal],
                                 config: Dict) -> Dict[str, pd.DataFrame]:
        """Завантаження даних для одного символу"""
        async with self.semaphore:

            # Визначаємо діапазон часу
            times = [s.time for s in signals]
            start_time = min(times) - timedelta(hours=config['data_management']['buffer_hours_before'])
            end_time = max(times) + timedelta(hours=config['data_management']['buffer_hours_after'])

            data = {}
            timeframes = [config['timeframe_settings']['primary_timeframe']] + \
                         config['timeframe_settings']['secondary_timeframes']

            for tf in timeframes:
                cache_key = f"{symbol}_{tf}_{int(start_time.timestamp())}_{int(end_time.timestamp())}"

                # Перевіряємо кеш
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    data[tf] = cached_data
                    continue

                # Завантажуємо нові дані
                df = await self._fetch_ohlcv(symbol, tf, start_time, end_time)
                if not df.empty:
                    # Розраховуємо індикатори
                    df = self._add_indicators(df, config)
                    data[tf] = df
                    self.cache.set(cache_key, df)

                await asyncio.sleep(0.02)

            return data

    async def _fetch_ohlcv(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
        """ТОЧНА копія логіки завантаження з аналізатора"""
        api_symbol = symbol.replace("USDT", "/USDT")
        since = int(start.timestamp() * 1000)
        until = int(end.timestamp() * 1000)

        all_data = []
        current = since
        batch_size = 1000
        max_requests = 50
        request_count = 0

        while current < until and request_count < max_requests:
            try:
                ohlcv = self.exchange.fetch_ohlcv(api_symbol, tf, since=current, limit=batch_size)
                if not ohlcv:
                    break

                all_data.extend(ohlcv)
                current = ohlcv[-1][0] + 1
                request_count += 1

                await asyncio.sleep(0.02)

            except Exception as e:
                logging.error(f"Помилка завантаження {symbol} {tf}: {e}")
                break

        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        return pd.DataFrame()

    def _add_indicators(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """ТОЧНА копія розрахунку індикаторів з аналізатора + volatility"""
        rsi_params = config['rsi_parameters']

        df['rsi'] = TechnicalIndicators.calculate_rsi_vectorized(df['close'], rsi_params['rsi_period'])
        df['rsi_sma'] = TechnicalIndicators.calculate_sma(df['rsi'], rsi_params['rsi_sma_period'])

        # Додаткові індикатори для volatility filter
        if 'volatility_filter' in config and config['volatility_filter'].get('enabled', False):
            vol_config = config['volatility_filter']
            method = vol_config.get('volatility_method', 'atr_percentage')
            period = vol_config.get('volatility_period', 20)

            if method == 'atr_percentage':
                df['atr'] = TechnicalIndicators.calculate_atr(df, period)
                df['volatility'] = (df['atr'] / df['close']) * 100
            elif method == 'price_range':
                df['volatility'] = ((df['high'] - df['low']) / df['close']).rolling(window=period).mean() * 100
            elif method == 'close_variance':
                df['volatility'] = TechnicalIndicators.calculate_price_volatility(df, period)

        return df


class VolatilityFilter:
    """Фільтр волатильності"""

    @staticmethod
    def check_volatility(df: pd.DataFrame, signal_time: datetime, config: Dict) -> Tuple[bool, str]:
        """Перевірка волатільності"""
        vol_config = config.get('volatility_filter', {})

        if not vol_config.get('enabled', False):
            return True, ""

        if df.empty or 'volatility' not in df.columns:
            return True, "No volatility data"

        # Знаходимо найближчу свічку
        df_copy = df.copy()
        df_copy['time_diff'] = (df_copy['datetime'] - signal_time).abs()
        signal_idx = df_copy['time_diff'].idxmin()

        current_volatility = df.iloc[signal_idx]['volatility']
        min_vol = vol_config.get('min_volatility', 0.1)
        max_vol = vol_config.get('max_volatility', 8.0)

        if current_volatility < min_vol:
            return False, f"Low volatility: {current_volatility:.2f}% < {min_vol}%"
        elif current_volatility > max_vol:
            return False, f"High volatility: {current_volatility:.2f}% > {max_vol}%"

        return True, f"Volatility OK: {current_volatility:.2f}%"


class TradingEngine:
    """Ядро торговельної логіки - ТОЧНА копія з аналізатора"""

    @staticmethod
    def parse_time(time_str: str) -> datetime:
        """ТОЧНА копія з аналізатора"""
        formats = ['%d.%m.%Y %H:%M', '%Y-%m-%d %H:%M:%S']
        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt)
            except ValueError:
                pass
        return datetime.now()

    @staticmethod
    def check_secondary_tf_trend(secondary_df: pd.DataFrame, signal_time: datetime,
                                 direction: str, config: Dict) -> Tuple[bool, str]:
        """ТОЧНА копія логіки з аналізатора"""
        if secondary_df.empty:
            return True, ""

        secondary_df = secondary_df.copy()
        secondary_df['time_diff'] = (secondary_df['datetime'] - signal_time).abs()
        signal_idx = secondary_df['time_diff'].idxmin()

        filter_config = config['secondary_tf_filter']
        is_long = direction.lower() == 'long'

        # Вся подальша логіка ТОЧНО з аналізатора
        if is_long:
            # Метод 1: Аналіз середньої зміни SMA
            if signal_idx >= filter_config['sma_change_periods']:
                start_idx = signal_idx - filter_config['sma_change_periods'] + 1
                end_idx = signal_idx + 1
                sma_segment = secondary_df.iloc[start_idx:end_idx]['rsi_sma']

                if len(sma_segment) >= 2:
                    sma_changes = sma_segment.diff().dropna()
                    avg_sma_change = sma_changes.mean()

                    rsi_segment = secondary_df.iloc[start_idx:end_idx]['rsi']
                    rsi_extreme_growth = (rsi_segment.diff() > filter_config['rsi_extreme_threshold']).any()

                    if avg_sma_change <= -filter_config['sma_change_threshold'] and not rsi_extreme_growth:
                        return False, f"Downtrend: avg_sma_change={avg_sma_change:.2f}"

            # Метод 2: Пошук перетину RSI/SMA вниз
            lookback_start = max(0, signal_idx - filter_config['cross_lookback_periods'])
            lookback_df = secondary_df.iloc[lookback_start:signal_idx + 1]

            for i in range(1, len(lookback_df)):
                curr = lookback_df.iloc[i]
                prev = lookback_df.iloc[i - 1]

                if (prev['rsi'] >= prev['rsi_sma'] and
                        curr['rsi'] < curr['rsi_sma'] and
                        curr['rsi_sma'] >= filter_config['short_enter_zone_mid_tf']):
                    return False, f"Recent bear cross at RSI_SMA={curr['rsi_sma']:.2f}"

        else:  # short
            # Аналогічна логіка для short (копія з аналізатора)
            if signal_idx >= filter_config['sma_change_periods']:
                start_idx = signal_idx - filter_config['sma_change_periods'] + 1
                end_idx = signal_idx + 1
                sma_segment = secondary_df.iloc[start_idx:end_idx]['rsi_sma']

                if len(sma_segment) >= 2:
                    sma_changes = sma_segment.diff().dropna()
                    avg_sma_change = sma_changes.mean()

                    rsi_segment = secondary_df.iloc[start_idx:end_idx]['rsi']
                    rsi_extreme_decline = (rsi_segment.diff() < -filter_config['rsi_extreme_threshold']).any()

                    if avg_sma_change >= filter_config['sma_change_threshold'] and not rsi_extreme_decline:
                        return False, f"Uptrend: avg_sma_change={avg_sma_change:.2f}"

            lookback_start = max(0, signal_idx - filter_config['cross_lookback_periods'])
            lookback_df = secondary_df.iloc[lookback_start:signal_idx + 1]

            for i in range(1, len(lookback_df)):
                curr = lookback_df.iloc[i]
                prev = lookback_df.iloc[i - 1]

                if (prev['rsi'] <= prev['rsi_sma'] and
                        curr['rsi'] > curr['rsi_sma'] and
                        curr['rsi_sma'] <= filter_config['long_enter_zone_mid_tf']):
                    return False, f"Recent bull cross at RSI_SMA={curr['rsi_sma']:.2f}"

        return True, ""

    @staticmethod
    def simulate_trade(data: Dict[str, pd.DataFrame], signal: Signal, config: Dict) -> TradeResult:
        """ТОЧНА копія логіки симуляції з аналізатора + volatility filter"""
        primary_tf = config['timeframe_settings']['primary_timeframe']
        secondary_tf = config['timeframe_settings']['secondary_timeframes'][0]
        df = data.get(primary_tf)
        secondary_df = data.get(secondary_tf)

        if df is None or df.empty:
            return TradeResult(
                pair=signal.pair, direction=signal.direction,
                entry_time=signal.time, exit_time=signal.time,
                entry_price=0, exit_price=0, pnl_percent=0, hold_time=0,
                status="Filtered", exit_reason="No Data", filtered_out=True
            )

        # Пошук найближчої свічки - ТОЧНА копія
        df = df.copy()
        df['time_diff'] = (df['datetime'] - signal.time).abs()
        entry_idx = df['time_diff'].idxmin()

        entry_candle = df.iloc[entry_idx]
        entry_time = entry_candle['datetime']
        entry_price = entry_candle['close']
        direction = signal.direction
        is_long = direction.lower() == 'long'
        rsi_params = config['rsi_parameters']
        max_hold = config['trading_parameters']['max_hold_hours']
        max_time = entry_time + timedelta(hours=max_hold)

        # Перевірка входу - ТОЧНА копія
        entry_valid = False
        if is_long and signal.rsi_sma < rsi_params['long_enter_zone'] and signal.rsi > signal.rsi_sma:
            entry_valid = True
        elif not is_long and signal.rsi_sma > rsi_params['short_enter_zone'] and signal.rsi < signal.rsi_sma:
            entry_valid = True

        if not entry_valid:
            return TradeResult(
                pair=signal.pair, direction=direction,
                entry_time=signal.time, exit_time=signal.time,
                entry_price=0, exit_price=0, pnl_percent=0, hold_time=0,
                status="Filtered", exit_reason="Weak Entry Signal",
                filtered_out=True, filter_info="Weak Entry Signal"
            )

        # Перевірка тренду на вторинному таймфреймі
        if secondary_df is not None and not secondary_df.empty:
            trend_favorable, trend_reason = TradingEngine.check_secondary_tf_trend(
                secondary_df, signal.time, direction, config)
            if not trend_favorable:
                return TradeResult(
                    pair=signal.pair, direction=direction,
                    entry_time=signal.time, exit_time=signal.time,
                    entry_price=0, exit_price=0, pnl_percent=0, hold_time=0,
                    status="Filtered", exit_reason="Secondary TF Filter",
                    filtered_out=True, filter_info=f"Secondary TF Filter: {trend_reason}"
                )

        # НОВА перевірка волатільності
        if 'volatility_filter' in config:
            vol_ok, vol_reason = VolatilityFilter.check_volatility(df, signal.time, config)
            if not vol_ok:
                return TradeResult(
                    pair=signal.pair, direction=direction,
                    entry_time=signal.time, exit_time=signal.time,
                    entry_price=0, exit_price=0, pnl_percent=0, hold_time=0,
                    status="Filtered", exit_reason="Volatility Filter",
                    filtered_out=True, filter_info=f"Volatility Filter: {vol_reason}"
                )

        # Симуляція торгівлі - ТОЧНА копія всієї логіки з аналізатора
        stop_loss = TradingEngine.calculate_stop_loss(entry_price, direction, config)
        dynamic_sl = stop_loss

        exit_found = False
        exit_reason = 'End of Data'
        exit_price = entry_price
        exit_time = entry_time
        status = 'Loss'

        df_slice = df.iloc[entry_idx + 1:]
        if len(df_slice) > 0:
            for i, (idx, candle) in enumerate(df_slice.iterrows()):
                if i > 0:
                    prev_candle = df_slice.iloc[i - 1]
                    prev_rsi, prev_sma = prev_candle['rsi'], prev_candle['rsi_sma']
                else:
                    prev_candle = entry_candle
                    prev_rsi, prev_sma = entry_candle['rsi'], entry_candle['rsi_sma']

                curr_rsi, curr_sma = candle['rsi'], candle['rsi_sma']

                # Оновлення трейлінг стопу
                stop_loss = TradingEngine.update_trailing_stop(
                    candle['close'], entry_price, stop_loss, direction, config)
                dynamic_sl = stop_loss

                # Перевірка стоп-лоссу
                if TradingEngine.check_stop_loss_hit(candle, stop_loss, direction):
                    exit_reason = 'Stop Loss'
                    exit_price = stop_loss
                    exit_time = candle['datetime']
                    status = 'Loss'
                    exit_found = True
                    break

                # Перевірка зворотного сигналу - ТОЧНА копія
                if is_long:
                    if prev_rsi >= prev_sma and curr_rsi < curr_sma and curr_rsi > rsi_params['long_exit_zone']:
                        exit_reason = 'Reverse Signal (Short)'
                        exit_price = candle['close']
                        exit_time = candle['datetime']
                        status = 'Profit' if exit_price > entry_price else 'Loss'
                        exit_found = True
                        break
                else:
                    if prev_rsi <= prev_sma and curr_rsi > curr_sma and curr_rsi < rsi_params['short_exit_zone']:
                        exit_reason = 'Reverse Signal (Long)'
                        exit_price = candle['close']
                        exit_time = candle['datetime']
                        status = 'Profit' if exit_price < entry_price else 'Loss'
                        exit_found = True
                        break

                # Перевірка таймауту
                if candle['datetime'] > max_time:
                    exit_reason = 'Timeout'
                    exit_price = candle['close']
                    exit_time = candle['datetime']
                    pnl_check = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                            (entry_price - exit_price) / entry_price * 100)
                    status = 'Profit' if pnl_check > 0 else 'Loss'
                    exit_found = True
                    break

        if not exit_found and len(df_slice) > 0:
            last_candle = df_slice.iloc[-1]
            exit_price = last_candle['close']
            exit_time = last_candle['datetime']
            exit_reason = 'End of Data'
            pnl_check = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                    (entry_price - exit_price) / entry_price * 100)
            status = 'Profit' if pnl_check > 0 else 'Loss'

        pnl = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                (entry_price - exit_price) / entry_price * 100)
        hold_time = (exit_time - entry_time).total_seconds() / 3600

        return TradeResult(
            pair=signal.pair, direction=direction,
            entry_time=entry_time, exit_time=exit_time,
            entry_price=entry_price, exit_price=exit_price,
            pnl_percent=pnl, hold_time=hold_time,
            status=status, exit_reason=exit_reason
        )

    @staticmethod
    def calculate_stop_loss(entry_price: float, direction: str, config: Dict) -> float:
        """ТОЧНА копія з аналізатора"""
        stop_loss_pct = config['trading_parameters']['stop_loss']
        is_long = direction.lower() == 'long'

        if is_long:
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)

    @staticmethod
    def update_trailing_stop(current_price: float, entry_price: float,
                             current_sl: float, direction: str, config: Dict) -> float:
        """ТОЧНА копія з аналізатора"""
        if not config['trading_parameters']['use_trailing_stop']:
            return current_sl

        is_long = direction.lower() == 'long'
        activation_pct = config['trading_parameters']['trailing_stop_activation']
        distance_pct = config['trading_parameters']['trailing_stop_distance']

        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= activation_pct:
                new_sl = current_price * (1 - distance_pct)
                return max(current_sl, new_sl)
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= activation_pct:
                new_sl = current_price * (1 + distance_pct)
                return min(current_sl, new_sl)

        return current_sl

    @staticmethod
    def check_stop_loss_hit(candle: pd.Series, stop_loss: float, direction: str) -> bool:
        """ТОЧНА копія з аналізатора"""
        is_long = direction.lower() == 'long'

        if is_long:
            return candle['low'] <= stop_loss
        else:
            return candle['high'] >= stop_loss


class ParameterRanges:
    """Розширені діапазони параметрів для всіх режимів оптимізації"""

    @staticmethod
    def get_parameter_ranges(mode: str = 'balanced') -> Dict[str, List]:
        """Повні діапазони параметрів з усіх попередніх скриптів"""

        base_ranges = {
            # Trading parameters
            'stop_loss': [0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030, 0.035],
            'take_profit': [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070],
            'max_hold_hours': [6, 8, 10, 12, 16, 18, 20, 24, 30, 36, 48],

            # Trailing stop parameters
            'use_trailing_stop': [True, False],
            'trailing_stop_activation': [0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030],
            'trailing_stop_distance': [0.003, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020],

            # RSI parameters
            'long_enter_zone': [25, 28, 30, 32, 35],
            'short_enter_zone': [65, 68, 70, 72, 75],
            'long_exit_zone': [55, 58, 60, 62, 65, 68, 70, 72, 75],
            'short_exit_zone': [25, 28, 30, 32, 35, 38, 40, 42, 45],
            'long_extreme_exit': [75, 78, 80, 82, 85, 88, 90],
            'short_extreme_exit': [10, 12, 15, 18, 20, 22, 25],

            # Timeframe settings
            'secondary_timeframes': [['15m'], ['30m'], ['1h'], ['15m', '30m'], ['15m', '1h'], ['30m', '1h']],

            # Secondary TF filters
            'short_enter_zone_mid_tf': [50, 52, 55, 57, 60, 63, 65, 67, 70, 72, 75],
            'long_enter_zone_mid_tf': [25, 28, 30, 33, 35, 37, 40, 43, 45, 48, 50],
            'sma_change_periods': [3, 4, 5, 6, 7, 8, 9, 10],
            'sma_change_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'rsi_extreme_threshold': [8, 10, 12, 15, 18, 20, 22, 25],
            'cross_lookback_periods': [4, 6, 8, 10, 12, 15, 18, 20],

            # Volatility filter parameters (НОВІ)
            'volatility_enabled': [True, False],
            'min_volatility': [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            'max_volatility': [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
            'volatility_period': [10, 14, 16, 20, 24, 30, 40],
            'volatility_method': ['atr_percentage', 'price_range', 'close_variance'],

            # Buffer parameters
            'buffer_hours_before': [60, 80, 100, 120, 140, 160, 200],
            'buffer_hours_after': [120, 160, 200, 240, 280, 320, 400],
        }

        if mode == 'ultra_fast':
            return {
                'stop_loss': [0.015, 0.018, 0.022],
                'take_profit': [0.030, 0.035, 0.040],
                'max_hold_hours': [16, 20, 24],
                'use_trailing_stop': [True, False],
                'trailing_stop_activation': [0.008, 0.010, 0.012],
                'trailing_stop_distance': [0.005, 0.008, 0.010],
                'long_exit_zone': [60, 65, 70],
                'short_exit_zone': [30, 35, 40],
                'secondary_timeframes': [['15m'], ['30m']],
                'volatility_enabled': [True, False],
                'min_volatility': [0.5, 1.0],
                'max_volatility': [6.0, 8.0],
            }
        elif mode == 'fast':
            return {k: v[:4] if len(v) > 4 else v for k, v in base_ranges.items()}
        elif mode == 'balanced':
            return {k: v[:6] if len(v) > 6 else v for k, v in base_ranges.items()}
        elif mode == 'comprehensive':
            return base_ranges

        return base_ranges


class GeneticOptimizer:
    """Генетичний алгоритм оптимізації"""

    def __init__(self, population_size: int = 50, elite_ratio: float = 0.2,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.param_ranges = {}

    def initialize_population(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Ініціалізація популяції"""
        self.param_ranges = param_ranges
        population = []

        for _ in range(self.population_size):
            individual = {}
            for param, values in param_ranges.items():
                individual[param] = random.choice(values)
            if self._is_valid_individual(individual):
                population.append(individual)

        return population

    def evolve_generation(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Еволюція одного покоління"""
        # Сортуємо за fitness (вищий = кращий)
        sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)

        # Вибираємо еліту
        elite_size = max(1, int(len(population) * self.elite_ratio))
        elite = [ind for ind, _ in sorted_pop[:elite_size]]

        # Нова популяція
        new_population = elite.copy()

        # Генеруємо нащадків
        while len(new_population) < self.population_size:
            # Вибір батьків (турнірна селекція)
            parent1 = self._tournament_selection(sorted_pop)
            parent2 = self._tournament_selection(sorted_pop)

            # Схрещування
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Мутація
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            # Додаємо валідних нащадків
            for child in [child1, child2]:
                if self._is_valid_individual(child) and len(new_population) < self.population_size:
                    new_population.append(child)

        return new_population[:self.population_size]

    def _tournament_selection(self, sorted_population: List[Tuple], tournament_size: int = 3) -> Dict:
        """Турнірна селекція"""
        tournament = random.sample(sorted_population, min(tournament_size, len(sorted_population)))
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Одноточкове схрещування"""
        child1, child2 = parent1.copy(), parent2.copy()

        # Вибираємо випадкові параметри для обміну
        params_to_swap = random.sample(list(parent1.keys()),
                                       random.randint(1, len(parent1.keys()) // 2))

        for param in params_to_swap:
            child1[param], child2[param] = parent2[param], parent1[param]

        return child1, child2

    def _mutate(self, individual: Dict) -> Dict:
        """Мутація індивіда"""
        mutated = individual.copy()

        for param in individual.keys():
            if random.random() < self.mutation_rate:
                if param in self.param_ranges:
                    # З 80% ймовірністю вибираємо близьке значення
                    if random.random() < 0.8:
                        current_val = individual[param]
                        if current_val in self.param_ranges[param]:
                            current_idx = self.param_ranges[param].index(current_val)
                            # Вибираємо сусіднє значення
                            possible_indices = []
                            if current_idx > 0:
                                possible_indices.append(current_idx - 1)
                            if current_idx < len(self.param_ranges[param]) - 1:
                                possible_indices.append(current_idx + 1)

                            if possible_indices:
                                new_idx = random.choice(possible_indices)
                                mutated[param] = self.param_ranges[param][new_idx]
                    else:
                        # З 20% ймовірністю - випадкове значення
                        mutated[param] = random.choice(self.param_ranges[param])

        return mutated

    def _is_valid_individual(self, individual: Dict) -> bool:
        """Перевірка валідності індивіда"""
        try:
            # Базові перевірки
            take_profit = individual.get('take_profit', 0.035)
            stop_loss = individual.get('stop_loss', 0.018)
            if take_profit <= stop_loss:
                return False

            # Trailing stop перевірки
            if individual.get('use_trailing_stop', True):
                activation = individual.get('trailing_stop_activation', 0.010)
                distance = individual.get('trailing_stop_distance', 0.008)
                if activation <= distance or activation >= take_profit:
                    return False

            # RSI зони
            long_exit = individual.get('long_exit_zone', 65)
            short_exit = individual.get('short_exit_zone', 35)
            if long_exit <= short_exit:
                return False

            # Volatility filter
            if individual.get('volatility_enabled', False):
                min_vol = individual.get('min_volatility', 0.5)
                max_vol = individual.get('max_volatility', 8.0)
                if min_vol >= max_vol:
                    return False

            return True
        except:
            return False


class ConfigOptimizer:
    """Оптимізатор конфігурацій з різними режимами"""

    def __init__(self, base_config: Dict, max_concurrent: int = 50):
        self.base_config = base_config
        self.max_concurrent = max_concurrent
        self._shared_data_cache: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        self._signals_cache: Optional[List[Signal]] = None

    def create_default_config(self) -> Dict:
        """Створення стандартного конфігу з усіма параметрами"""
        return {
            "trading_parameters": {
                "stop_loss": 0.018,
                "take_profit": 0.035,
                "max_hold_hours": 24,
                "use_trailing_stop": True,
                "trailing_stop_activation": 0.01,
                "trailing_stop_distance": 0.008,
                "max_volume_per_trade": 1000
            },
            "rsi_parameters": {
                "rsi_period": 14,
                "rsi_sma_period": 14,
                "long_enter_zone": 30,
                "short_enter_zone": 70,
                "long_exit_zone": 65,
                "short_exit_zone": 35,
                "long_extreme_exit": 80,
                "short_extreme_exit": 20
            },
            "timeframe_settings": {
                "primary_timeframe": "5m",
                "secondary_timeframes": ["15m"]
            },
            "secondary_tf_filter": {
                "short_enter_zone_mid_tf": 60,
                "long_enter_zone_mid_tf": 40,
                "sma_change_periods": 5,
                "sma_change_threshold": 0.5,
                "rsi_extreme_threshold": 15,
                "cross_lookback_periods": 10
            },
            "volatility_filter": {
                "enabled": False,
                "min_volatility": 0.5,
                "max_volatility": 8.0,
                "volatility_period": 20,
                "volatility_method": "atr_percentage"
            },
            "data_management": {
                "buffer_hours_before": 120,
                "buffer_hours_after": 240,
                "tf_multiplier": {"5m": 1, "15m": 3, "30m": 6, "1h": 12}
            },
            "exchange": "bybit",
            "banned_pairs": []
        }

    async def optimize(self, signals: List[Signal], mode: str = 'balanced',
                       max_configs: int = 500, generations: int = 10) -> List[Dict]:
        """Універсальна оптимізація з різними режимами"""

        logging.info(f"Початок оптимізації в режимі '{mode}' з {max_configs} конфігурацій")

        # 1. Створюємо спільний кеш даних
        await self._prepare_shared_cache(signals)

        # 2. Вибираємо режим оптимізації
        if mode == 'genetic':
            return await self._genetic_optimization(signals, max_configs, generations)
        elif mode in ['ultra_fast', 'fast', 'balanced', 'comprehensive']:
            return await self._traditional_optimization(signals, mode, max_configs)
        else:
            raise ValueError(f"Невідомий режим оптимізації: {mode}")

    async def _genetic_optimization(self, signals: List[Signal],
                                    population_size: int = 50, generations: int = 10) -> List[Dict]:
        """Генетична оптимізація"""
        logging.info(f"Генетична оптимізація: популяція={population_size}, покоління={generations}")

        # Ініціалізація генетичного алгоритму
        genetic = GeneticOptimizer(population_size=population_size)
        param_ranges = ParameterRanges.get_parameter_ranges('balanced')

        # Початкова популяція
        population = genetic.initialize_population(param_ranges)

        all_results = []
        best_fitness_history = []

        for generation in range(generations):
            logging.info(f"Покоління {generation + 1}/{generations}")

            # Оцінюємо популяцію
            fitness_scores = []
            generation_results = []

            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = []

            for i, individual in enumerate(population):
                config_id = f"gen_{generation + 1}_ind_{i + 1}"
                task = self._test_config_with_shared_cache(individual, config_id, semaphore)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception) or result is None:
                    fitness_scores.append(-999)  # Дуже поганий fitness
                else:
                    fitness = result['metrics']['avg_pnl']  # Використовуємо avg_pnl як fitness
                    fitness_scores.append(fitness)
                    generation_results.append(result)
                    all_results.append(result)

            # Знаходимо найкращий результат покоління
            best_fitness = max(fitness_scores) if fitness_scores else -999
            best_fitness_history.append(best_fitness)

            logging.info(f"Покоління {generation + 1}: найкращий fitness = {best_fitness:.4f}")

            # Еволюція (крім останнього покоління)
            if generation < generations - 1:
                population = genetic.evolve_generation(population, fitness_scores)

        # Сортуємо всі результати
        all_results.sort(key=lambda x: x['metrics']['avg_pnl'], reverse=True)

        logging.info(f"Генетична оптимізація завершена. Найкращі fitness по поколіннях: {best_fitness_history}")

        return all_results

    async def _traditional_optimization(self, signals: List[Signal],
                                        mode: str, max_configs: int) -> List[Dict]:
        """Традиційна оптимізація (grid search / random search)"""

        # Отримуємо діапазони параметрів
        param_ranges = ParameterRanges.get_parameter_ranges(mode)

        # Генеруємо конфігурації
        if mode == 'ultra_fast':
            configs = self._create_grid_search(param_ranges, max_configs)
        else:
            configs = self._create_random_search(param_ranges, max_configs)

        logging.info(f"Створено {len(configs)} конфігурацій для тестування")

        # Тестуємо конфігурації
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []

        for i, config_params in enumerate(configs):
            config_id = f"{mode}_{i + 1:04d}"
            task = self._test_config_with_shared_cache(config_params, config_id, semaphore)
            tasks.append(task)

        # Збираємо результати батчами
        results = []
        batch_size = 100

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if not isinstance(result, Exception) and result is not None:
                    results.append(result)

            progress = min((i + batch_size) / len(tasks) * 100, 100)
            logging.info(f"Прогрес: {progress:.1f}%")

        # Сортуємо результати
        results.sort(key=lambda x: x['metrics']['avg_pnl'], reverse=True)

        return results

    def _create_grid_search(self, param_ranges: Dict[str, List], max_configs: int) -> List[Dict]:
        """Grid search з обмеженнями"""
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        configs = []
        count = 0

        for combination in product(*param_values):
            if count >= max_configs:
                break

            config = dict(zip(param_names, combination))
            if self._is_valid_config(config):
                configs.append(config)
                count += 1

        return configs

    def _create_random_search(self, param_ranges: Dict[str, List], max_configs: int) -> List[Dict]:
        """Random search"""
        configs = []
        param_names = list(param_ranges.keys())

        attempts = 0
        max_attempts = max_configs * 3

        while len(configs) < max_configs and attempts < max_attempts:
            config = {}
            for param_name in param_names:
                config[param_name] = random.choice(param_ranges[param_name])

            if self._is_valid_config(config):
                configs.append(config)

            attempts += 1

        return configs

    def _is_valid_config(self, config: Dict) -> bool:
        """Валідація конфігурації"""
        try:
            # Базові перевірки
            take_profit = config.get('take_profit', 0.035)
            stop_loss = config.get('stop_loss', 0.018)
            if take_profit <= stop_loss:
                return False

            # Trailing stop валідація
            if config.get('use_trailing_stop', True):
                activation = config.get('trailing_stop_activation', 0.010)
                distance = config.get('trailing_stop_distance', 0.008)
                if activation <= distance or activation >= take_profit:
                    return False

            # RSI зони
            long_exit = config.get('long_exit_zone', 65)
            short_exit = config.get('short_exit_zone', 35)
            if long_exit <= short_exit:
                return False

            # Volatility filter
            if config.get('volatility_enabled', False):
                min_vol = config.get('min_volatility', 0.5)
                max_vol = config.get('max_volatility', 8.0)
                if min_vol >= max_vol:
                    return False

            return True
        except:
            return False

    async def _prepare_shared_cache(self, signals: List[Signal]):
        """Підготовка спільного кешу даних"""
        logging.info("Підготовка спільного кешу даних...")

        # Створюємо тимчасовий аналізатор для завантаження даних
        temp_analyzer = SignalAnalyzer(self.base_config)
        unique_pairs = list(set(s.pair for s in signals))

        # Завантажуємо ВСІ дані один раз
        self._shared_data_cache = await temp_analyzer.data_provider.fetch_data_batch(
            unique_pairs, signals, self.base_config
        )
        self._signals_cache = signals

        logging.info(f"Кеш підготовлено для {len(self._shared_data_cache)} пар")

    async def _test_config_with_shared_cache(self, config_params: Dict, config_id: str,
                                             semaphore: asyncio.Semaphore) -> Optional[Dict]:
        """Тестування конфігурації з використанням спільного кешу"""
        async with semaphore:
            try:
                start_time = time.time()
                # Застосовуємо параметри до базової конфігурації
                test_config = self._apply_config_params(config_params)

                # Тестуємо всі сигнали використовуючи спільний кеш
                results = []
                for signal in self._signals_cache:
                    if signal.pair in self._shared_data_cache:
                        result = TradingEngine.simulate_trade(
                            self._shared_data_cache[signal.pair], signal, test_config
                        )
                        results.append(result)

                # Розраховуємо метрики
                metrics = self._calculate_metrics(results)
                execution_time = time.time() - start_time

                return {
                    'config_id': config_id,
                    'parameters': config_params,
                    'metrics': metrics,
                    'execution_time': execution_time
                }

            except Exception as e:
                logging.error(f"Помилка тестування {config_id}: {e}")
                return None

    def _apply_config_params(self, params: Dict) -> Dict:
        """Застосування параметрів до базової конфігурації"""
        config = copy.deepcopy(self.base_config)

        # Застосовуємо параметри торгівлі
        trading_params = config['trading_parameters']
        for param in ['stop_loss', 'take_profit', 'max_hold_hours', 'use_trailing_stop',
                      'trailing_stop_activation', 'trailing_stop_distance']:
            if param in params:
                trading_params[param] = params[param]

        # RSI параметри
        rsi_params = config['rsi_parameters']
        for param in ['long_enter_zone', 'short_enter_zone',
                      'long_exit_zone', 'short_exit_zone', 'long_extreme_exit', 'short_extreme_exit']:
            if param in params:
                rsi_params[param] = params[param]

        # Secondary TF фільтри
        secondary_filter = config['secondary_tf_filter']
        for param in ['short_enter_zone_mid_tf', 'long_enter_zone_mid_tf', 'sma_change_periods',
                      'sma_change_threshold', 'rsi_extreme_threshold', 'cross_lookback_periods']:
            if param in params:
                secondary_filter[param] = params[param]

        # Timeframe налаштування
        if 'secondary_timeframes' in params:
            config['timeframe_settings']['secondary_timeframes'] = params['secondary_timeframes']

        # Volatility filter - НОВІ параметри
        if 'volatility_filter' not in config:
            config['volatility_filter'] = {}

        volatility_filter = config['volatility_filter']
        volatility_filter['enabled'] = params.get('volatility_enabled', False)
        if 'min_volatility' in params:
            volatility_filter['min_volatility'] = params['min_volatility']
        if 'max_volatility' in params:
            volatility_filter['max_volatility'] = params['max_volatility']
        if 'volatility_period' in params:
            volatility_filter['volatility_period'] = params['volatility_period']
        if 'volatility_method' in params:
            volatility_filter['volatility_method'] = params['volatility_method']

        # Buffer параметри
        data_mgmt = config['data_management']
        for param in ['buffer_hours_before', 'buffer_hours_after']:
            if param in params:
                data_mgmt[param] = params[param]

        return config

    def _calculate_metrics(self, results: List[TradeResult]) -> Dict:
        """Розрахунок метрик продуктивності"""
        traded_results = [r for r in results if not r.filtered_out]

        if not traded_results:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0,
                'total_pnl': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0,
                'filtered_out': len(results), 'sharpe_ratio': 0.0,
                'best_trade': 0.0, 'worst_trade': 0.0, 'avg_hold_time': 0.0
            }

        pnls = [r.pnl_percent for r in traded_results]
        hold_times = [r.hold_time for r in traded_results]

        total_trades = len(traded_results)
        profitable_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        total_pnl = sum(pnls)

        # Drawdown розрахунок
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = max(drawdown) if len(drawdown) > 0 else 0

        # Profit factor
        profits = [pnl for pnl in pnls if pnl > 0]
        losses = [abs(pnl) for pnl in pnls if pnl <= 0]
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # Sharpe ratio
        sharpe_ratio = avg_pnl / np.std(pnls) if len(pnls) > 1 and np.std(pnls) > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'filtered_out': len(results) - total_trades,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': max(pnls) if pnls else 0.0,
            'worst_trade': min(pnls) if pnls else 0.0,
            'avg_hold_time': sum(hold_times) / len(hold_times) if hold_times else 0.0
        }


class SignalAnalyzer:
    """Головний клас аналізатора"""

    def __init__(self, config: Dict):
        self.config = config
        self.data_provider = DataProvider(config)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2))

    async def analyze_signals(self, signals: List[Signal]) -> List[TradeResult]:
        """Аналіз сигналів з використанням спільного кешу даних"""

        # 1. Завантажуємо ВСІ дані ОДРАЗУ
        unique_pairs = list(set(s.pair for s in signals))
        logging.info(f"Завантаження даних для {len(unique_pairs)} пар...")

        all_data = await self.data_provider.fetch_data_batch(unique_pairs, signals, self.config)

        # 2. Обробляємо всі сигнали використовуючи кешовані дані
        results = []
        for signal in signals:
            if signal.pair in all_data:
                result = TradingEngine.simulate_trade(all_data[signal.pair], signal, self.config)
                results.append(result)
            else:
                # Створюємо результат для пар без даних
                result = TradeResult(
                    pair=signal.pair, direction=signal.direction,
                    entry_time=signal.time, exit_time=signal.time,
                    entry_price=0, exit_price=0, pnl_percent=0, hold_time=0,
                    status="Filtered", exit_reason="No Data Available", filtered_out=True
                )
                results.append(result)

        return results


# Головні функції для використання

async def run_analysis(config_file: str, signals_file: str, output_file: str):
    """Запуск аналізу сигналів"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Завантажуємо сигнали
    signals = load_signals_from_csv(signals_file)

    # Створюємо аналізатор і запускаємо
    analyzer = SignalAnalyzer(config)
    results = await analyzer.analyze_signals(signals)

    # Зберігаємо результати
    save_results_to_csv(results, output_file)

    logging.info(f"Аналіз завершено. Результати збережено в {output_file}")


async def run_optimization(config_file: str, signals_file: str, output_file: str,
                           mode: str = 'balanced', max_configs: int = 500,
                           generations: int = 10):
    """Запуск оптимізації"""
    with open(config_file, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    # Завантажуємо сигнали
    signals = load_signals_from_csv(signals_file)

    # Створюємо оптимізатор і запускаємо
    optimizer = ConfigOptimizer(base_config)
    results = await optimizer.optimize(signals, mode, max_configs, generations)

    # Зберігаємо результати
    save_optimization_results(results, output_file)

    # Зберігаємо топ-10 в бажаному форматі
    top10_file = output_file.replace('.csv', '_top10.tsv') if output_file.endswith('.csv') else output_file + '_top10.tsv'
    save_top10(results[:10], top10_file)

    logging.info(f"Оптимізація завершена. Результати збережено в {output_file}")
    logging.info(f"Топ-10 конфігурацій збережено в {top10_file}")

    # Виводимо топ результати
    print_top_results(results[:10])


def load_signals_from_csv(filename: str) -> List[Signal]:
    """Завантаження сигналів з CSV"""
    signals = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if row.get('Status') == 'open':
                signals.append(Signal(
                    pair=row['Pair'].replace('/', ''),
                    direction=row['Direction'],
                    time=TradingEngine.parse_time(row['Signal_Time']),
                    rsi=float(row['RSI_5m'].replace(',', '.')),
                    rsi_sma=float(row['RSI_SMA_5m'].replace(',', '.'))
                ))
    return signals


def save_results_to_csv(results: List[TradeResult], filename: str):
    """Збереження результатів в CSV"""
    fieldnames = ['pair', 'direction', 'rsi', 'entry_time', 'exit_time', 'entry_price',
                  'exit_price', 'pnl_percent', 'hold_time', 'status', 'exit_reason',
                  'filtered_out', 'filter_info']

    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'pair': result.pair,
                'direction': result.direction,
                'rsi': 0,  # Placeholder
                'entry_time': result.entry_time.strftime('%d.%m.%Y %H:%M'),
                'exit_time': result.exit_time.strftime('%d.%m.%Y %H:%M'),
                'entry_price': result.entry_price,
                'exit_price': result.exit_price,
                'pnl_percent': result.pnl_percent,
                'hold_time': result.hold_time,
                'status': result.status,
                'exit_reason': result.exit_reason,
                'filtered_out': result.filtered_out,
                'filter_info': getattr(result, 'filter_info', '')
            })


def save_optimization_results(results: List[Dict], filename: str):
    """Збереження результатів оптимізації"""
    if not results:
        return

    fieldnames = ['rank', 'config_id', 'total_trades', 'win_rate', 'avg_pnl',
                  'total_pnl', 'profit_factor', 'max_drawdown', 'filtered_out',
                  'best_trade', 'worst_trade', 'avg_hold_time', 'sharpe_ratio']

    # Додаємо всі параметри конфігурації
    all_params = set()
    for result in results:
        all_params.update(result['parameters'].keys())
    fieldnames.extend(sorted(all_params))

    try:
        save_to_file(filename, results, fieldnames, all_params)
    except Exception as e:
        print(e)
        logging.info(f"The file with name {filename} already exists. Writing to temp.csv")
        filename = "temp.csv"
        save_to_file(filename, results, fieldnames, all_params)


def save_to_file(filename, results, fieldnames, all_params):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=fieldnames)
        writer.writeheader()

        for rank, result in enumerate(results, 1):
            metrics = result['metrics']
            row = {
                'rank': rank,
                'config_id': result['config_id'],
                'total_trades': metrics['total_trades'],
                'win_rate': round(metrics['win_rate'], 2),
                'avg_pnl': round(metrics['avg_pnl'], 4),
                'total_pnl': round(metrics['total_pnl'], 2),
                'profit_factor': round(metrics['profit_factor'], 2),
                'max_drawdown': round(metrics['max_drawdown'], 2),
                'filtered_out': metrics['filtered_out'],
                'best_trade': round(metrics['best_trade'], 2),
                'worst_trade': round(metrics['worst_trade'], 2),
                'avg_hold_time': round(metrics['avg_hold_time'], 2),
                'sharpe_ratio': round(metrics['sharpe_ratio'], 3)
            }

            # Додаємо параметри
            for param in all_params:
                row[param] = result['parameters'].get(param, '')

            writer.writerow(row)


def save_to_file_top10(results, filename, fieldnames):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for rank, result in enumerate(results, 1):
            metrics = result['metrics']
            params = result['parameters']
            config_id_num = ''.join(filter(str.isdigit, result['config_id']))  # Витягуємо цифри для random_
            strategy_name = f"random_{config_id_num}" if config_id_num else f"random_{rank * 100 + random.randint(1, 99)}"

            row = {
                'rank': rank,
                'config_id': result['config_id'],
                'total_trades': metrics['total_trades'],
                'win_rate': round(metrics['win_rate'], 2),
                'avg_pnl': round(metrics['avg_pnl'], 4),
                'total_pnl': round(metrics['total_pnl'], 2),
                'profit_factor': round(metrics['profit_factor'], 2),
                'max_drawdown': round(metrics['max_drawdown'], 2),
                'filtered_out': metrics['filtered_out'],
                'best_trade': round(metrics['best_trade'], 2),
                'worst_trade': round(metrics['worst_trade'], 2),
                'execution_time': round(result.get('execution_time', 0), 3),
                'buffer_hours_after': params.get('buffer_hours_after', 240),
                'buffer_hours_before': params.get('buffer_hours_before', 120),
                'cross_lookback_periods': params.get('cross_lookback_periods', 10),
                'long_enter_zone_mid_tf': params.get('long_enter_zone_mid_tf', 40),
                'long_exit_zone': params.get('long_exit_zone', 65),
                'long_extreme_exit': params.get('long_extreme_exit', 80),
                'max_hold_hours': params.get('max_hold_hours', 24),
                'rsi_extreme_threshold': params.get('rsi_extreme_threshold', 15),
                'secondary_timeframes': str(params.get('secondary_timeframes', ['15m'])),
                'short_enter_zone_mid_tf': params.get('short_enter_zone_mid_tf', 60),
                'short_exit_zone': params.get('short_exit_zone', 35),
                'short_extreme_exit': params.get('short_extreme_exit', 20),
                'sma_change_periods': params.get('sma_change_periods', 5),
                'sma_change_threshold': params.get('sma_change_threshold', 0.5),
                'stop_loss': params.get('stop_loss', 0.018),
                'strategy_name': strategy_name,
                'take_profit': params.get('take_profit', 0.035),
                'tf_multiplier_15m': 3,  # Фіксоване з default config
                'tf_multiplier_1h': 12,
                'tf_multiplier_30m': 6,
                'trailing_stop_activation': params.get('trailing_stop_activation', 0.01),
                'trailing_stop_distance': params.get('trailing_stop_distance', 0.008),
                'use_trailing_stop': 'TRUE' if params.get('use_trailing_stop', True) else 'FALSE'
            }

            writer.writerow(row)


def save_top10(results: List[Dict], filename: str):
    """Збереження топ-10 конфігурацій у вказаному форматі (з табуляцією)"""
    if not results:
        return

    fieldnames = [
        'rank', 'config_id', 'total_trades', 'win_rate', 'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown', 'filtered_out',
        'best_trade', 'worst_trade', 'execution_time', 'buffer_hours_after', 'buffer_hours_before', 'cross_lookback_periods',
        'long_enter_zone_mid_tf', 'long_exit_zone', 'long_extreme_exit', 'max_hold_hours', 'rsi_extreme_threshold',
        'secondary_timeframes', 'short_enter_zone_mid_tf', 'short_exit_zone', 'short_extreme_exit', 'sma_change_periods',
        'sma_change_threshold', 'stop_loss', 'strategy_name', 'take_profit', 'tf_multiplier_15m', 'tf_multiplier_1h',
        'tf_multiplier_30m', 'trailing_stop_activation', 'trailing_stop_distance', 'use_trailing_stop'
    ]

    try:
        save_to_file_top10(results, filename, fieldnames)
    except Exception as e:
        print(e)
        logging.info(f"The file with name {filename} already exists. Writing to temp_top10.csv")
        filename = "temp_top10.csv"
        save_to_file_top10(results, filename, fieldnames)


def print_top_results(results: List[Dict], top_n: int = 10):
    """Виведення топ результатів"""
    if not results:
        return

    print(f"\n{'=' * 80}")
    print(f"ТОП-{top_n} НАЙКРАЩИХ КОНФІГУРАЦІЙ")
    print(f"{'=' * 80}")

    for i, result in enumerate(results[:top_n], 1):
        metrics = result['metrics']
        params = result['parameters']

        print(f"\n#{i} - {result['config_id']}")
        print(f"Угод: {metrics['total_trades']} | "
              f"Відфільтровано: {metrics['filtered_out']} | "
              f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Avg PnL: {metrics['avg_pnl']:.3f}% | "
              f"Total PnL: {metrics['total_pnl']:.2f}% | "
              f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max DD: {metrics['max_drawdown']:.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"Avg Hold: {metrics['avg_hold_time']:.1f}h")

        # Ключові параметри
        key_params = []
        if 'use_trailing_stop' in params:
            ts_status = "ON" if params['use_trailing_stop'] else "OFF"
            key_params.append(f"TrailingStop={ts_status}")

        if 'volatility_enabled' in params:
            vol_status = "ON" if params['volatility_enabled'] else "OFF"
            key_params.append(f"VolFilter={vol_status}")

        key_params.extend([
            f"SL={params.get('stop_loss', 0):.3f}",
            f"TP={params.get('take_profit', 0):.3f}",
            f"MaxHold={params.get('max_hold_hours', 0)}h"
        ])

        if key_params:
            print(f"Параметри: {' | '.join(key_params[:6])}")


def save_best_config(results: List[Dict], output_file: str = None):
    """Збереження найкращої конфігурації"""
    if not results:
        return None

    best_result = results[0]

    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'best_config_{timestamp}.json'

    # Створюємо повну конфігурацію
    optimizer = ConfigOptimizer({})
    base_config = optimizer.create_default_config()

    # Застосовуємо найкращі параметри
    best_config = optimizer._apply_config_params(best_result['parameters'])

    # Додаємо метадані
    best_config['optimization_metadata'] = {
        'config_id': best_result['config_id'],
        'rank': 1,
        'performance_metrics': best_result['metrics'],
        'optimization_date': datetime.now().isoformat(),
        'optimizer_version': "enhanced_modular_with_genetic"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"Найкращу конфігурацію збережено в: {output_file}")
    return output_file


# Приклад використання та головна функція
async def main():
    """Головна функція запуску"""
    parser = argparse.ArgumentParser(description='Розширений аналізатор з генетичним алгоритмом')

    # Основні параметри
    parser.add_argument('mode', choices=['analyze', 'optimize'],
                        help='Режим: analyze або optimize')
    parser.add_argument('config', help='JSON файл конфігурації')
    parser.add_argument('signals', help='CSV файл з сигналами')
    parser.add_argument('output', help='Файл для результатів')

    # Параметри оптимізації
    parser.add_argument('--opt-mode', default='balanced',
                        choices=['ultra_fast', 'fast', 'balanced', 'comprehensive', 'genetic'],
                        help='Режим оптимізації')
    parser.add_argument('--max-configs', type=int, default=500,
                        help='Максимальна кількість конфігурацій')
    parser.add_argument('--generations', type=int, default=10,
                        help='Кількість поколінь для genetic режиму')
    parser.add_argument('--save-best', action='store_true',
                        help='Зберегти найкращу конфігурацію')

    args = parser.parse_args()

    # Налаштування логування
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimizer.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    try:
        start_time = time.time()

        if args.mode == 'analyze':
            await run_analysis(args.config, args.signals, args.output)

        elif args.mode == 'optimize':
            print(f"Початок оптимізації в режимі '{args.opt_mode}'")
            print(f"Максимум конфігурацій: {args.max_configs}")

            if args.opt_mode == 'genetic':
                print(f"Кількість поколінь: {args.generations}")

            # Запускаємо оптимізацію
            await run_optimization(
                args.config, args.signals, args.output,
                args.opt_mode, args.max_configs, args.generations
            )

            # Зберігаємо найкращу конфігурацію якщо потрібно
            if args.save_best:
                # Завантажуємо результати для збереження найкращої конфігурації
                try:
                    results_df = pd.read_csv(args.output, delimiter=';')
                    if not results_df.empty:
                        # Створюємо структуру результатів для save_best_config
                        best_params = {}
                        first_row = results_df.iloc[0]

                        # Витягуємо параметри з першого рядка
                        param_columns = [col for col in results_df.columns
                                         if col not in ['rank', 'config_id', 'total_trades', 'win_rate',
                                                        'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown',
                                                        'filtered_out', 'best_trade', 'worst_trade',
                                                        'avg_hold_time', 'sharpe_ratio']]

                        for param in param_columns:
                            if pd.notna(first_row[param]) and first_row[param] != '':
                                value = first_row[param]
                                # Спробуємо конвертувати типи
                                try:
                                    if value in ['True', 'False']:
                                        best_params[param] = value == 'True'
                                    elif '.' in str(value):
                                        best_params[param] = float(value)
                                    elif str(value).isdigit():
                                        best_params[param] = int(value)
                                    else:
                                        best_params[param] = value
                                except:
                                    best_params[param] = value

                        # Створюємо структуру результату
                        best_result = {
                            'config_id': first_row['config_id'],
                            'parameters': best_params,
                            'metrics': {
                                'total_trades': first_row['total_trades'],
                                'win_rate': first_row['win_rate'],
                                'avg_pnl': first_row['avg_pnl'],
                                'total_pnl': first_row['total_pnl'],
                                'profit_factor': first_row['profit_factor'],
                                'max_drawdown': first_row['max_drawdown'],
                                'filtered_out': first_row['filtered_out'],
                                'sharpe_ratio': first_row['sharpe_ratio']
                            }
                        }

                        save_best_config([best_result])

                except Exception as e:
                    print(f"Помилка збереження найкращої конфігурації: {e}")

        end_time = time.time()
        duration = end_time - start_time

        print(f"\nВиконання завершено за {duration:.2f} секунд")
        print(f"Використання пам'яті: {psutil.virtual_memory().percent:.1f}%")

    except KeyboardInterrupt:
        print("\nОперацію перервано користувачем")
    except Exception as e:
        logging.error(f"Критична помилка: {e}", exc_info=True)
    finally:
        gc.collect()


if __name__ == "__main__":
    # Налаштування для Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Запуск
    asyncio.run(main())
