#!/usr/bin/env python3
"""
Високопродуктивний оптимізатор для Enhanced Analyzer
Основні оптимізації:
1. Кешування даних на рівні пар з розумним управлінням пам'яттю
2. Паралельні обчислення індикаторів з numpy
3. Векторизовані операції для аналізу угод
4. Оптимізовані алгоритми пошуку сигналів
5. Мінімізація I/O операцій
6. Ефективне управління ресурсами
"""

import gc
import asyncio
import json
import pandas as pd
import numpy as np
from itertools import product
import logging
from datetime import datetime, timedelta
import argparse
import os
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import random
import psutil
import aiofiles
from functools import lru_cache
import multiprocessing as mp
import threading
from collections import defaultdict
import numba
from numba import jit, njit
import warnings

warnings.filterwarnings('ignore')

# Імпортуємо базові функції з analyzer
import ccxt


# Оптимізовані функції з numba для швидких обчислень
@njit(fastmath=True)
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Швидке обчислення RSI з numba"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Перший RSI
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi = np.full(n, 50.0)

    for i in range(period, n):
        if i == period:
            rs = avg_gain / (avg_loss + 1e-10)
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            rs = avg_gain / (avg_loss + 1e-10)

        rsi[i] = 100 - (100 / (1 + rs))

    return rsi


@njit(fastmath=True)
def fast_sma(values: np.ndarray, period: int) -> np.ndarray:
    """Швидке обчислення SMA з numba"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    # Перше значення
    result[period - 1] = np.mean(values[:period])

    # Решта значень через rolling sum
    for i in range(period, n):
        result[i] = result[i - 1] + (values[i] - values[i - period]) / period

    return result


@njit(fastmath=True)
def vectorized_trade_analysis(prices: np.ndarray, rsi: np.ndarray, rsi_sma: np.ndarray,
                              entry_idx: int, direction: int, config_params: np.ndarray) -> Tuple[
    float, float, int, str]:
    """Векторизований аналіз угоди
    direction: 1 для long, -1 для short
    config_params: [stop_loss, trailing_activation, trailing_distance, exit_zone, max_hold_idx]
    """
    stop_loss_pct, trailing_activation, trailing_distance, exit_zone, max_hold_idx = config_params
    n = len(prices)

    if entry_idx >= n - 1:
        return 0.0, 0.0, 0, "No data"

    entry_price = prices[entry_idx]

    # Початковий стоп-лос
    if direction == 1:  # Long
        stop_loss = entry_price * (1 - stop_loss_pct)
    else:  # Short
        stop_loss = entry_price * (1 + stop_loss_pct)

    max_search_idx = min(n, entry_idx + int(max_hold_idx))

    for i in range(entry_idx + 1, max_search_idx):
        current_price = prices[i]

        # Оновлення трейлінг стопу
        if direction == 1:  # Long
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= trailing_activation:
                new_sl = current_price * (1 - trailing_distance)
                stop_loss = max(stop_loss, new_sl)

            # Перевірка стоп-лосу (припускаємо, що low ≈ current_price для спрощення)
            if current_price <= stop_loss:
                return stop_loss, (stop_loss - entry_price) / entry_price * 100, i, "Stop Loss"
        else:  # Short
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= trailing_activation:
                new_sl = current_price * (1 + trailing_distance)
                stop_loss = min(stop_loss, new_sl)

            if current_price >= stop_loss:
                return stop_loss, (entry_price - stop_loss) / entry_price * 100, i, "Stop Loss"

        # Перевірка зворотного сигналу
        if i > entry_idx + 1:
            prev_rsi, prev_sma = rsi[i - 1], rsi_sma[i - 1]
            curr_rsi, curr_sma = rsi[i], rsi_sma[i]

            if direction == 1:  # Long
                if prev_rsi >= prev_sma and curr_rsi < curr_sma and curr_rsi > exit_zone:
                    pnl = (current_price - entry_price) / entry_price * 100
                    return current_price, pnl, i, "Reverse Signal"
            else:  # Short
                if prev_rsi <= prev_sma and curr_rsi > curr_sma and curr_rsi < exit_zone:
                    pnl = (entry_price - current_price) / entry_price * 100
                    return current_price, pnl, i, "Reverse Signal"

    # Таймаут
    final_price = prices[max_search_idx - 1]
    if direction == 1:
        pnl = (final_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - final_price) / entry_price * 100

    return final_price, pnl, max_search_idx - 1, "Timeout"


class HighPerformanceDataCache:
    """Високопродуктивний кеш даних з автоматичним управлінням пам'яттю"""

    def __init__(self, max_memory_mb: int = 2048):
        self.cache = {}
        self.access_times = {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.lock = threading.Lock()

    def _estimate_size(self, df: pd.DataFrame) -> int:
        """Оцінка розміру DataFrame в байтах"""
        return df.memory_usage(deep=True).sum()

    def _cleanup_if_needed(self):
        """Очищення кешу при перевищенні ліміту пам'яті"""
        current_memory = sum(self._estimate_size(df) for df in self.cache.values())

        if current_memory > self.max_memory_bytes:
            # Видаляємо найменш використовувані елементи
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            to_remove = len(sorted_items) // 3  # Видаляємо 1/3

            for key, _ in sorted_items[:to_remove]:
                if key in self.cache:
                    del self.cache[key]
                    del self.access_times[key]

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key].copy()
            return None

    def set(self, key: str, df: pd.DataFrame):
        with self.lock:
            self.cache[key] = df.copy()
            self.access_times[key] = time.time()
            self._cleanup_if_needed()

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            gc.collect()


class OptimizedSignalAnalyzer:
    """Оптимізований аналізатор сигналів"""

    def __init__(self, config: Dict, cache_size_mb: int = 2048):
        exchange_name = config.get('exchange', 'bybit')
        self.exchange = ccxt.__getattribute__(exchange_name)({
            'enableRateLimit': True,
            'rateLimit': 50  # Зменшуємо rate limit для швидшості
        })

        self.cache = HighPerformanceDataCache(cache_size_mb)
        self.banned_pairs = set(config.get('banned_pairs', []))

        # Thread pool для паралельних обчислень
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, mp.cpu_count() * 2))

    async def fetch_all_data_batch(self, pairs: List[str], signals: List[Dict],
                                   config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Пакетне завантаження даних для всіх пар одразу"""

        # Групуємо сигнали по парах
        pair_signals = defaultdict(list)
        for signal in signals:
            pair = signal['pair']
            if pair not in self.banned_pairs:
                pair_signals[pair].append(signal)

        # Паралельне завантаження даних
        semaphore = asyncio.Semaphore(20)  # Збільшуємо concurrent requests
        tasks = []

        for pair in pairs:
            if pair in pair_signals:
                task = asyncio.create_task(
                    self._fetch_pair_data_with_semaphore(pair, pair_signals[pair], config, semaphore)
                )
                tasks.append((pair, task))

        # Збираємо результати
        all_data = {}
        for pair, task in tasks:
            try:
                data = await task
                if data:
                    all_data[pair] = data
            except Exception as e:
                logging.warning(f"Помилка завантаження даних для {pair}: {e}")
                continue

        return all_data

    async def _fetch_pair_data_with_semaphore(self, symbol: str, signals: List[Dict],
                                              config: Dict, semaphore: asyncio.Semaphore):
        async with semaphore:
            return await self.fetch_data_with_buffer(symbol, signals, config)

    async def fetch_data_with_buffer(self, symbol: str, signals: List[Dict], config: Dict) -> Dict[str, pd.DataFrame]:
        """Оптимізоване завантаження даних з кешуванням"""
        if not signals:
            return {}

        times = [self._parse_time_fast(s['time']) for s in signals]
        first_time = min(times)
        last_time = max(times)

        data_config = config['data_management']
        tf_settings = config['timeframe_settings']
        primary_tf = tf_settings['primary_timeframe']

        data = {}
        for tf in [primary_tf] + tf_settings['secondary_timeframes']:
            multiplier = data_config['tf_multiplier'].get(tf, 1)
            start = first_time - timedelta(hours=data_config['buffer_hours_before'] * multiplier)
            end = last_time + timedelta(hours=data_config['buffer_hours_after'] * multiplier)

            cache_key = f"{symbol}_{tf}_{start.timestamp():.0f}_{end.timestamp():.0f}"

            # Перевіряємо кеш
            cached_df = self.cache.get(cache_key)
            if cached_df is not None:
                data[tf] = cached_df
                continue

            # Завантажуємо нові дані
            try:
                df = await self._fetch_ohlcv_optimized(symbol, tf, start, end)
                if not df.empty:
                    df = self._calculate_indicators_vectorized(df, config)
                    data[tf] = df
                    self.cache.set(cache_key, df)
            except Exception as e:
                logging.warning(f"Помилка завантаження {symbol} {tf}: {e}")
                continue

            await asyncio.sleep(0.02)  # Зменшуємо затримку

        return data

    def _parse_time_fast(self, time_str: str) -> datetime:
        """Швидкий парсинг часу"""
        try:
            return datetime.strptime(time_str.strip(), '%d.%m.%Y %H:%M')
        except:
            try:
                return datetime.strptime(time_str.strip(), '%Y-%m-%d %H:%M:%S')
            except:
                return datetime.now()

    async def _fetch_ohlcv_optimized(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Оптимізоване завантаження OHLCV даних"""
        api_symbol = symbol.replace("USDT", "/USDT")
        since = int(start.timestamp() * 1000)
        until = int(end.timestamp() * 1000)

        all_data = []
        current = since
        batch_size = 1000

        while current < until:
            try:
                ohlcv = self.exchange.fetch_ohlcv(api_symbol, tf, since=current, limit=batch_size)
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                current = ohlcv[-1][0] + 1

                # Мінімальна затримка
                await asyncio.sleep(0.01)

            except Exception as e:
                logging.warning(f"API помилка для {symbol} {tf}: {e}")
                break

        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        return pd.DataFrame()

    def _calculate_indicators_vectorized(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Векторизоване обчислення індикаторів"""
        if df.empty:
            return df

        rsi_params = config['rsi_parameters']

        # Використовуємо оптимізовані функції
        prices = df['close'].values
        df['rsi'] = fast_rsi(prices, rsi_params['rsi_period'])
        df['rsi_sma'] = fast_sma(df['rsi'].values, rsi_params['rsi_sma_period'])

        return df


class UltraFastOptimizer:
    """Ультра-швидкий оптимізатор з мінімальними накладними витратами"""

    def __init__(self, base_config_file: str = 'updated_analyzer_config.json', max_concurrent: int = 50):
        self.base_config = self._load_config_fast(base_config_file)
        self.max_concurrent = max_concurrent
        self.config_counter = 0
        self.results = []

        # Додаємо banned_pairs з конфігу
        self.banned_pairs = set(self.base_config.get('banned_pairs', []))

        # Статистика
        self.perf_stats = {
            'total_configs': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'avg_execution_time': 0,
            'memory_usage_mb': 0
        }

        logging.warning(f"Ініціалізовано ультра-швидкий оптимізатор (concurrent: {max_concurrent})")
        logging.warning(f"Заборонені пари: {len(self.banned_pairs)}")

    def _load_config_fast(self, config_file: str) -> Dict:
        """Швидке завантаження конфігу"""
        if not os.path.exists(config_file):
            return self._create_default_config()

        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_default_config(self) -> Dict:
        """Створення базового конфігу"""
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
                "long_exit_zone": 60,
                "short_exit_zone": 40,
                "long_extreme_exit": 75,
                "short_extreme_exit": 25
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
            "data_management": {
                "buffer_hours_before": 120,
                "buffer_hours_after": 240,
                "tf_multiplier": {"5m": 1, "15m": 3, "30m": 6, "1h": 12}
            },
            "exchange": "bybit",
            "banned_pairs": []
        }

    def generate_parameter_ranges(self, mode: str = 'balanced') -> Dict[str, List]:
        """Генерація параметрів з урахуванням режиму"""
        if mode == 'fast':
            return {
                'stop_loss': [0.015, 0.018, 0.022, 0.025],
                'take_profit': [0.030, 0.035, 0.040, 0.045],
                'max_hold_hours': [12, 16, 20, 24],
                'use_trailing_stop': [True, False],
                'trailing_stop_activation': [0.008, 0.010, 0.012],
                'trailing_stop_distance': [0.006, 0.008, 0.010],
                'long_exit_zone': [60, 65, 70],
                'short_exit_zone': [30, 35, 40],
                'long_extreme_exit': [75, 80, 85],
                'short_extreme_exit': [15, 20, 25],
                'secondary_timeframes': [['15m'], ['30m'], ['1h'], ['15m', '30m'], ['15m', '1h']],
                'short_enter_zone_mid_tf': [55, 60, 65],
                'long_enter_zone_mid_tf': [35, 40, 45],
                'sma_change_periods': [4, 5, 6],
                'sma_change_threshold': [0.4, 0.5, 0.6],
                'rsi_extreme_threshold': [12, 15, 18],
                'cross_lookback_periods': [8, 10, 12],
                'buffer_hours_before': [100, 120, 140],
                'buffer_hours_after': [200, 240, 280],
            }
        elif mode == 'ultra_fast':
            return {
                'stop_loss': [0.018, 0.022],
                'take_profit': [0.035, 0.040],
                'max_hold_hours': [16, 20],
                'use_trailing_stop': [True],
                'trailing_stop_activation': [0.010],
                'trailing_stop_distance': [0.008],
                'long_exit_zone': [65, 70],
                'short_exit_zone': [30, 35],
                'secondary_timeframes': [['15m'], ['30m']],
                'short_enter_zone_mid_tf': [60],
                'long_enter_zone_mid_tf': [40],
                'sma_change_periods': [5],
                'cross_lookback_periods': [10],
            }
        else:  # balanced или thorough
            return {
                'stop_loss': [0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030],
                'take_profit': [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060],
                'max_hold_hours': [8, 12, 16, 20, 24, 30],
                'use_trailing_stop': [True, False],
                'trailing_stop_activation': [0.008, 0.010, 0.012, 0.015],
                'trailing_stop_distance': [0.005, 0.006, 0.008, 0.010, 0.012],
                'long_exit_zone': [55, 60, 62, 65, 68, 70, 72],
                'short_exit_zone': [28, 30, 32, 35, 38, 40, 45],
                'long_extreme_exit': [75, 78, 80, 82, 85, 88],
                'short_extreme_exit': [12, 15, 18, 20, 22, 25],
                'secondary_timeframes': [['15m'], ['30m'], ['1h'], ['15m', '30m'], ['15m', '1h']],
                'short_enter_zone_mid_tf': [50, 55, 57, 60, 63, 65, 67, 70],
                'long_enter_zone_mid_tf': [30, 35, 37, 40, 43, 45, 47, 50],
                'sma_change_periods': [3, 4, 5, 6, 7, 8, 9],
                'sma_change_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                'rsi_extreme_threshold': [10, 12, 15, 18, 20],
                'cross_lookback_periods': [6, 8, 10, 12, 15],
                'buffer_hours_before': [80, 100, 120, 140, 160],
                'buffer_hours_after': [180, 200, 240, 280, 320],
                'tf_multiplier_15m': [2, 3, 4],
                'tf_multiplier_30m': [4, 6, 8],
                'tf_multiplier_1h': [8, 12, 16],
            }

    def create_config_combinations(self, max_combinations: int = 1000, mode: str = 'balanced') -> List[Dict[str, Any]]:
        """Швидке створення комбінацій параметрів"""
        param_ranges = self.generate_parameter_ranges(mode)

        # Для ultra_fast режиму використовуємо grid search
        if mode == 'ultra_fast':
            return self._create_grid_search(param_ranges, max_combinations)
        else:
            return self._create_random_search(param_ranges, max_combinations)

    def _create_grid_search(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Grid search з обмеженнями"""
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        configs = []
        count = 0

        for combination in product(*param_values):
            if count >= max_combinations:
                break

            config = dict(zip(param_names, combination))
            config['strategy_name'] = f'grid_{count}'

            if self._validate_config_fast(config):
                configs.append(config)
                count += 1

        return configs

    def _create_random_search(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Швидкий random search"""
        configs = []
        param_names = list(param_ranges.keys())

        for i in range(max_combinations * 2):  # Генеруємо з запасом
            if len(configs) >= max_combinations:
                break

            config = {}
            for param_name in param_names:
                config[param_name] = random.choice(param_ranges[param_name])
            config['strategy_name'] = f'random_{len(configs)}'

            if self._validate_config_fast(config):
                configs.append(config)

        return configs

    def _validate_config_fast(self, config: Dict[str, Any]) -> bool:
        """Швидка валідація конфігу з розширеними перевірками"""
        try:
            take_profit = config.get('take_profit', 0.035)
            stop_loss = config.get('stop_loss', 0.018)

            if take_profit <= stop_loss:
                return False

            if config.get('use_trailing_stop', True):
                activation = config.get('trailing_stop_activation', 0.010)
                distance = config.get('trailing_stop_distance', 0.008)
                if activation <= distance:
                    return False

            # Перевірка RSI зон
            long_exit = config.get('long_exit_zone', 65)
            short_exit = config.get('short_exit_zone', 35)
            if long_exit <= short_exit:
                return False

            long_extreme = config.get('long_extreme_exit', 80)
            short_extreme = config.get('short_extreme_exit', 20)
            if long_extreme <= long_exit or short_extreme >= short_exit:
                return False

            # Перевірка secondary TF зон
            short_mid = config.get('short_enter_zone_mid_tf', 60)
            long_mid = config.get('long_enter_zone_mid_tf', 40)
            if short_mid <= long_mid:
                return False

            # Перевірка SMA параметрів
            sma_periods = config.get('sma_change_periods', 5)
            sma_threshold = config.get('sma_change_threshold', 0.5)
            if sma_periods < 2 or sma_threshold <= 0:
                return False

            # Перевірка buffer hours
            buffer_before = config.get('buffer_hours_before', 120)
            buffer_after = config.get('buffer_hours_after', 240)
            if buffer_before < 24 or buffer_after < 48:
                return False

            return True
        except:
            return False

    def apply_config_to_base(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Швидке застосування конфігу"""
        config = copy.deepcopy(self.base_config)

        # Пряме оновлення параметрів
        trading_params = config['trading_parameters']
        rsi_params = config['rsi_parameters']
        tf_settings = config['timeframe_settings']
        secondary_filter = config['secondary_tf_filter']

        # Мапінг параметрів
        param_mapping = {
            'stop_loss': ('trading_parameters', 'stop_loss'),
            'take_profit': ('trading_parameters', 'take_profit'),
            'max_hold_hours': ('trading_parameters', 'max_hold_hours'),
            'use_trailing_stop': ('trading_parameters', 'use_trailing_stop'),
            'trailing_stop_activation': ('trading_parameters', 'trailing_stop_activation'),
            'trailing_stop_distance': ('trading_parameters', 'trailing_stop_distance'),
            'long_exit_zone': ('rsi_parameters', 'long_exit_zone'),
            'short_exit_zone': ('rsi_parameters', 'short_exit_zone'),
            'long_extreme_exit': ('rsi_parameters', 'long_extreme_exit'),
            'short_extreme_exit': ('rsi_parameters', 'short_extreme_exit'),
            'secondary_timeframes': ('timeframe_settings', 'secondary_timeframes'),
            'short_enter_zone_mid_tf': ('secondary_tf_filter', 'short_enter_zone_mid_tf'),
            'long_enter_zone_mid_tf': ('secondary_tf_filter', 'long_enter_zone_mid_tf'),
            'sma_change_periods': ('secondary_tf_filter', 'sma_change_periods'),
            'sma_change_threshold': ('secondary_tf_filter', 'sma_change_threshold'),
            'rsi_extreme_threshold': ('secondary_tf_filter', 'rsi_extreme_threshold'),
            'cross_lookback_periods': ('secondary_tf_filter', 'cross_lookback_periods'),
            'buffer_hours_before': ('data_management', 'buffer_hours_before'),
            'buffer_hours_after': ('data_management', 'buffer_hours_after'),
        }

        for param, value in test_config.items():
            if param in param_mapping:
                section, key = param_mapping[param]
                config[section][key] = value
            elif param.startswith('tf_multiplier_'):
                tf = param.replace('tf_multiplier_', '')
                config['data_management']['tf_multiplier'][tf] = value

        return config

    async def test_single_config_vectorized(self, config: Dict[str, Any], config_id: str,
                                            all_data: Dict[str, Dict[str, pd.DataFrame]],
                                            signals: List[Dict], semaphore: asyncio.Semaphore,
                                            temp_file: str) -> Optional[Dict]:
        """Векторизоване тестування конфігурації"""

        async with semaphore:
            start_time = time.time()

            try:
                full_config = self.apply_config_to_base(config)

                # Паралельне тестування всіх сигналів
                trade_results = []

                # Групуємо сигнали по парах
                pair_signals = defaultdict(list)
                for signal in signals:
                    pair = signal['pair']
                    if pair in all_data:
                        pair_signals[pair].append(signal)

                # Векторизоване тестування для кожної пари
                for pair, pair_signals_list in pair_signals.items():
                    pair_data = all_data[pair]

                    # Тестуємо всі сигнали пари одразу
                    pair_results = self._test_pair_signals_vectorized(
                        pair_data, pair_signals_list, full_config
                    )
                    trade_results.extend(pair_results)

                # Швидкий аналіз результатів
                metrics = self._calculate_metrics_fast(trade_results)

                # Зберігаємо результати в тимчасовий файл
                await self._save_temp_results(temp_file, trade_results)

                result = {
                    'config_id': config_id,
                    'parameters': config,
                    'metrics': metrics,
                    'execution_time': time.time() - start_time
                }

                return result

            except Exception as e:
                logging.warning(f"Помилка тестування конфігу {config_id}: {e}")
                return None

    def _test_pair_signals_vectorized(self, data: Dict[str, pd.DataFrame],
                                      signals: List[Dict], config: Dict) -> List[Dict]:
        """Векторизоване тестування сигналів для пари"""

        primary_tf = config['timeframe_settings']['primary_timeframe']
        df = data.get(primary_tf)

        if df is None or df.empty:
            return []

        results = []

        # Підготовка даних для векторизації
        prices = df['close'].values
        rsi = df['rsi'].values
        rsi_sma = df['rsi_sma'].values
        timestamps = df['datetime'].values

        # Параметри конфігурації
        trading_params = config['trading_parameters']
        rsi_params = config['rsi_parameters']

        config_array = np.array([
            trading_params['stop_loss'],
            trading_params['trailing_stop_activation'],
            trading_params['trailing_stop_distance'],
            rsi_params['long_exit_zone'] if True else rsi_params['short_exit_zone'],
            trading_params['max_hold_hours'] * 12  # 5m свічок в годині
        ])

        for signal in signals:
            signal_time = self._parse_time_fast(signal['time'])
            direction = 1 if signal['direction'].lower() == 'long' else -1

            # Знаходимо індекс входу - конвертуємо все в datetime64
            signal_time_np = np.datetime64(signal_time)
            timestamps_np = timestamps.astype('datetime64[ns]')
            time_diffs = np.abs((timestamps_np - signal_time_np).astype('timedelta64[m]').astype(int))
            entry_idx = np.argmin(time_diffs)

            if entry_idx >= len(prices) - 1:
                continue

            # Налаштовуємо параметри для напряму
            config_for_signal = config_array.copy()
            if direction == 1:  # Long
                config_for_signal[3] = rsi_params['long_exit_zone']
            else:  # Short
                config_for_signal[3] = rsi_params['short_exit_zone']

            # Векторизований аналіз угоди
            exit_price, pnl, exit_idx, exit_reason = vectorized_trade_analysis(
                prices, rsi, rsi_sma, entry_idx, direction, config_for_signal
            )

            # Формуємо результат
            entry_time = timestamps[entry_idx]
            exit_time = timestamps[exit_idx] if exit_idx < len(timestamps) else timestamps[-1]

            # Безпечне конвертування в строки
            try:
                entry_time_str = pd.Timestamp(entry_time).strftime('%d.%m.%Y %H:%M')
                exit_time_str = pd.Timestamp(exit_time).strftime('%d.%m.%Y %H:%M')
            except:
                entry_time_str = str(entry_time)[:16]
                exit_time_str = str(exit_time)[:16]

            result = {
                'pair': signal['pair'],
                'direction': signal['direction'],
                'rsi': signal['rsi'],
                'signal_time': signal['time'],
                'entry_time_str': entry_time_str,
                'exit_time_str': exit_time_str,
                'entry_price': float(prices[entry_idx]),
                'exit_price': float(exit_price),
                'pnl_percent': float(pnl),
                'hold_time': float((exit_idx - entry_idx) * 5 / 60),  # години для 5m свічок
                'status': 'Profit' if pnl > 0 else 'Loss',
                'exit_reason': exit_reason,
                'filtered_out': 0
            }
            results.append(result)

        return results

    def _parse_time_fast(self, time_str: str) -> pd.Timestamp:
        """Швидкий парсинг часу з перевіркою типів"""
        try:
            return pd.to_datetime(time_str, format='%d.%m.%Y %H:%M')
        except:
            try:
                return pd.to_datetime(time_str)
            except:
                return pd.Timestamp.now()

    def _calculate_metrics_fast(self, results: List[Dict]) -> Dict:
        """Швидке обчислення метрик"""
        if not results:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0,
                'total_pnl': 0.0, 'profit_factor': 0.0, 'max_drawdown': 0.0
            }

        pnls = np.array([r['pnl_percent'] for r in results])

        total_trades = len(results)
        profitable_trades = np.sum(pnls > 0)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = np.mean(pnls)
        total_pnl = np.sum(pnls)

        # Швидке обчислення drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Profit factor
        profits = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        total_profit = np.sum(profits) if len(profits) > 0 else 0
        total_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'best_trade': np.max(pnls) if len(pnls) > 0 else 0,
            'worst_trade': np.min(pnls) if len(pnls) > 0 else 0
        }

    async def _save_temp_results(self, temp_file: str, results: List[Dict]):
        """Асинхронне збереження тимчасових результатів"""
        if not results:
            return

        fieldnames = ['pair', 'direction', 'rsi', 'signal_time', 'entry_time_str',
                      'exit_time_str', 'entry_price', 'exit_price', 'pnl_percent',
                      'hold_time', 'status', 'exit_reason', 'filtered_out']

        async with aiofiles.open(temp_file, 'w', encoding='utf-8', newline='') as f:
            writer_data = []
            # Записуємо заголовок
            writer_data.append(';'.join(fieldnames))

            # Записуємо дані
            for result in results:
                row = [str(result.get(field, '')) for field in fieldnames]
                writer_data.append(';'.join(row))

            await f.write('\n'.join(writer_data))

    async def optimize_ultra_fast(self, input_csv: str, max_configs: int = 500,
                                  mode: str = 'ultra_fast', output_file: str = 'results.csv') -> List[Dict]:
        """Ультра-швидка оптимізація"""

        logging.warning(f"Початок ультра-швидкої оптимізації:")
        logging.warning(f"  - Файл сигналів: {input_csv}")
        logging.warning(f"  - Максимум конфігурацій: {max_configs}")
        logging.warning(f"  - Режим: {mode}")

        start_time = time.time()

        # 1. Завантаження сигналів
        signals = self._load_signals_fast(input_csv)
        logging.warning(f"Завантажено {len(signals)} сигналів")

        if not signals:
            logging.error("Немає валідних сигналів для оптимізації")
            return []

        # 2. Пакетне завантаження всіх даних одразу
        logging.warning("Завантаження даних для всіх пар...")
        pairs = list(set(s['pair'] for s in signals))

        analyzer = OptimizedSignalAnalyzer(self.base_config, cache_size_mb=4096)

        try:
            all_data = await analyzer.fetch_all_data_batch(pairs, signals, self.base_config)
            logging.warning(f"Завантажено дані для {len(all_data)} пар")
        except Exception as e:
            logging.error(f"Помилка завантаження даних: {e}")
            return []

        # 3. Генерація конфігурацій
        configs = self.create_config_combinations(max_configs, mode)
        logging.warning(f"Створено {len(configs)} конфігурацій для тестування")

        self.perf_stats['total_configs'] = len(configs)

        # 4. Паралельне тестування конфігурацій
        semaphore = asyncio.Semaphore(self.max_concurrent)
        temp_dir = os.path.splitext(output_file)[0] + '_temp'
        os.makedirs(temp_dir, exist_ok=True)

        tasks = []
        for i, config in enumerate(configs):
            config_id = f"{mode}_{i + 1:04d}"
            temp_file = os.path.join(temp_dir, f"temp_{config_id}.csv")

            task = asyncio.create_task(
                self.test_single_config_vectorized(
                    config, config_id, all_data, signals, semaphore, temp_file
                )
            )
            tasks.append((task, config_id))

        # 5. Збір результатів пакетами
        logging.warning(f"Початок тестування {len(tasks)} конфігурацій...")
        results = []
        batch_size = 100

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(
                *(task for task, _ in batch), return_exceptions=True
            )

            for (task, config_id), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logging.warning(f"Помилка в {config_id}: {result}")
                    self.perf_stats['failed_tests'] += 1
                elif result is not None:
                    self.perf_stats['successful_tests'] += 1
                    results.append(result)

                    # Логування результату
                    metrics = result['metrics']
                    logging.warning(f"{config_id}: Trades={metrics['total_trades']}, "
                                    f"Win={metrics['win_rate']:.1f}%, "
                                    f"AvgPnL={metrics['avg_pnl']:.3f}%")
                else:
                    self.perf_stats['failed_tests'] += 1

            progress = (i + len(batch)) / len(tasks) * 100
            logging.warning(f"Прогрес: {progress:.1f}%")

            # Очищення пам'яті кожні 200 конфігурацій
            if i % 200 == 0:
                gc.collect()

        # 6. Сортування та збереження результатів
        results.sort(key=lambda x: x['metrics']['avg_pnl'], reverse=True)

        total_time = time.time() - start_time
        logging.warning(f"Оптимізація завершена за {total_time:.2f} секунд")
        logging.warning(f"Успішно протестовано: {len(results)} конфігурацій")

        # 7. Збереження топ результатів
        if results:
            await self._save_final_results(results[:50], output_file)
            self._print_top_results(results[:10])

        return results

    def _load_signals_fast(self, filename: str) -> List[Dict]:
        """Швидке завантаження сигналів"""
        signals = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                pair = row['Pair'].replace('/', '')
                if row.get('Status') == 'open' and pair not in self.banned_pairs:
                    signals.append({
                        'pair': pair,
                        'direction': row['Direction'],
                        'time': row['Signal_Time'],
                        'rsi': float(row['RSI_5m'].replace(',', '.')),
                        'rsi_sma': float(row['RSI_SMA_5m'].replace(',', '.'))
                    })
        return signals

    async def _save_final_results(self, results: List[Dict], filename: str):
        """Збереження фінальних результатів"""
        fieldnames = ['rank', 'config_id', 'total_trades', 'win_rate', 'avg_pnl',
                      'total_pnl', 'profit_factor', 'max_drawdown', 'execution_time']

        # Додаємо параметри
        all_params = set()
        for result in results:
            all_params.update(result['parameters'].keys())
        fieldnames.extend(sorted(all_params))

        async with aiofiles.open(filename, 'w', encoding='utf-8', newline='') as f:
            await f.write(';'.join(fieldnames) + '\n')

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
                    'execution_time': round(result['execution_time'], 3)
                }

                # Додаємо параметри
                for param in all_params:
                    row[param] = result['parameters'].get(param, '')

                row_values = [str(row.get(field, '')) for field in fieldnames]
                await f.write(';'.join(row_values) + '\n')

    def _print_top_results(self, results: List[Dict], top_n: int = 10):
        """Виведення топ результатів"""
        logging.warning(f"\n{'=' * 80}")
        logging.warning(f"ТОП-{top_n} НАЙКРАЩИХ КОНФІГУРАЦІЙ")
        logging.warning(f"{'=' * 80}")

        for i, result in enumerate(results[:top_n], 1):
            metrics = result['metrics']
            params = result['parameters']

            logging.warning(f"\n#{i} - {result['config_id']}")
            logging.warning(f"Угод: {metrics['total_trades']} | "
                            f"Win Rate: {metrics['win_rate']:.1f}% | "
                            f"Avg PnL: {metrics['avg_pnl']:.3f}%")
            logging.warning(f"Total PnL: {metrics['total_pnl']:.2f}% | "
                            f"Profit Factor: {metrics['profit_factor']:.2f} | "
                            f"Max DD: {metrics['max_drawdown']:.2f}%")

            # Ключові параметри
            key_params = ['stop_loss', 'take_profit', 'use_trailing_stop', 'long_exit_zone']
            param_str = []
            for param in key_params:
                if param in params:
                    param_str.append(f"{param}={params[param]}")

            if param_str:
                logging.warning(f"Параметри: {' | '.join(param_str[:4])}")


# Головна функція для запуску
async def main():
    """Головна функція запуску оптимізатора"""
    parser = argparse.ArgumentParser(description='Ультра-швидкий оптимізатор Enhanced Analyzer')
    parser.add_argument('input_csv', help='CSV файл з сигналами')
    parser.add_argument('-c', '--config', default='updated_analyzer_config.json',
                        help='Базовий конфіг файл')
    parser.add_argument('-n', '--max-configs', type=int, default=500,
                        help='Максимальна кількість конфігурацій')
    parser.add_argument('-m', '--mode', default='ultra_fast',
                        choices=['ultra_fast', 'balanced'],
                        help='Режим оптимізації')
    parser.add_argument('-o', '--output', default='ultra_fast_results.csv',
                        help='Файл для результатів')
    parser.add_argument('--concurrent', type=int, default=50,
                        help='Максимальна кількість одночасних завдань')

    args = parser.parse_args()

    # Налаштування логування
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ultra_fast_optimizer.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    if not os.path.exists(args.input_csv):
        logging.error(f"Файл сигналів не знайдено: {args.input_csv}")
        return

    # Ініціалізація оптимізатора
    optimizer = UltraFastOptimizer(
        base_config_file=args.config,
        max_concurrent=args.concurrent
    )

    logging.warning("Початок ультра-швидкої оптимізації")
    logging.warning(f"CPU cores: {psutil.cpu_count()}")
    logging.warning(f"Доступна RAM: {psutil.virtual_memory().total // (1024 ** 3)} GB")
    logging.warning(f"Concurrent tasks: {args.concurrent}")

    try:
        results = await optimizer.optimize_ultra_fast(
            input_csv=args.input_csv,
            max_configs=args.max_configs,
            mode=args.mode,
            output_file=args.output
        )

        if results:
            logging.warning("\nОптимізація успішно завершена!")
            logging.warning(f"Найкращий результат: Avg PnL = {results[0]['metrics']['avg_pnl']:.3f}%")
            logging.warning(f"Результати збережено в: {args.output}")
        else:
            logging.error("Оптимізація не дала результатів")

    except KeyboardInterrupt:
        logging.warning("Оптимізацію перервано користувачем")
    except Exception as e:
        logging.error(f"Критична помилка: {e}", exc_info=True)
    finally:
        gc.collect()
        final_memory = psutil.virtual_memory().percent
        logging.warning(f"Використання пам'яті: {final_memory:.1f}%")


if __name__ == "__main__":
    # Налаштування для Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Запуск
    asyncio.run(main())