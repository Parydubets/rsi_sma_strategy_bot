
# !/usr/bin/env python3
"""
Модифікований інтегрований оптимізатор з покращеним розпаралеленням
"""

import asyncio
import json
import pandas as pd
import numpy as np
from itertools import product, islice
import logging
from datetime import datetime
import argparse
import os
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional, Set
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, shared_memory, Manager, Queue, Process, Lock
import time
import random
import hashlib
import pickle
import heapq
import psutil
from functools import lru_cache, partial
import gc
from threading import Lock
from logging.handlers import QueueHandler, QueueListener

# Імпортуємо наш аналізатор
from enhanced_analyzer import TrendSignalAnalyzer

# Глобальний лок для запису в файл
csv_lock = Lock()


def configure_logging(log_queue: Optional[Queue] = None):
    """Конфігурація логування з підтримкою черги для multiprocess"""
    logger = logging.getLogger(__name__)

    # Видаляємо існуючі handlers, щоб уникнути дублювання
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_queue:
        # Для worker процесів - тільки QueueHandler
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        logger.setLevel(logging.ERROR)
    else:
        # Для main процесу - файл і консоль (тільки помилки в консоль)
        file_handler = logging.FileHandler('optimizer.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Все в файл

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.ERROR)  # Тільки помилки в консоль

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)
    return logger


@dataclass
class OptimizationResult:
    """Результат одного тесту оптимізації з оптимізованою структурою"""
    config_id: str
    parameters: Dict[str, Any]
    total_trades: int
    profitable_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    avg_hold_time: float
    max_drawdown: float
    profit_factor: float
    filtered_out: int
    best_trade: float
    worst_trade: float
    sharpe_ratio: float = 0.0
    strategy_name: str = ""
    execution_time: float = 0.0

    def __lt__(self, other):
        """Для використання в heap структурах"""
        return self.total_pnl < other.total_pnl

    def to_dict(self) -> Dict:
        """Конвертація в словник для серіалізації"""
        return asdict(self)


def test_config_worker(config_batch: List[Tuple[Dict[str, Any], str]],
                       signals_shm_name: str,
                       base_config_shm_name: str,
                       log_queue: Queue,
                       output_file: str) -> List[OptimizationResult]:
    """Робоча функція для тестування батча конфігурацій в окремому процесі"""

    # Налаштовуємо логування з чергою
    logger = configure_logging(log_queue)

    # Відновлюємо дані з shared memory
    try:
        signals_shm = shared_memory.SharedMemory(name=signals_shm_name)
        signals_data = pickle.loads(bytes(signals_shm.buf))

        base_config_shm = shared_memory.SharedMemory(name=base_config_shm_name)
        base_config = pickle.loads(bytes(base_config_shm.buf))
    except Exception as e:
        logger.error(f"Помилка відновлення shared memory: {e}")
        return []

    results = []

    # Створюємо аналізатор для цього процесу
    analyzer = TrendSignalAnalyzer()

    for test_config, config_id in config_batch:
        start_time = time.time()

        try:
            # Застосовуємо конфігурацію до базової
            full_config = apply_config_to_base_static(base_config, test_config)

            # Аналізуємо сигнали синхронно (в процесі)
            analysis_results = analyzer.analyze_signals_sync(signals_data, full_config)

            if not analysis_results:
                continue

            # Розрахунок метрик
            traded_results = [r for r in analysis_results if not r.get('filtered_out', False)]
            filtered_out = len(analysis_results) - len(traded_results)

            if not traded_results:
                result = OptimizationResult(
                    config_id=config_id,
                    parameters=test_config,
                    total_trades=0,
                    profitable_trades=0,
                    win_rate=0.0,
                    avg_pnl=0.0,
                    total_pnl=0.0,
                    avg_hold_time=0.0,
                    max_drawdown=0.0,
                    profit_factor=0.0,
                    filtered_out=filtered_out,
                    best_trade=0.0,
                    worst_trade=0.0,
                    sharpe_ratio=0.0,
                    strategy_name=test_config.get('strategy_name', 'unknown'),
                    execution_time=time.time() - start_time
                )
                # Виводимо результат в консоль
                print(
                    f"Тестовано {config_id}: Avg PnL={result.avg_pnl:.3f}%, Trades={result.total_trades}, Win Rate={result.win_rate:.1f}%")

                # Одразу записуємо в файл з локом
                append_result_to_csv(result, output_file)

                results.append(result)
                continue

            # Векторизовані обчислення
            pnl_values = np.array([float(r.get('pnl_percent', 0)) for r in traded_results])
            hold_times = np.array([float(r.get('hold_time_hours', 0)) for r in traded_results])

            total_trades = len(traded_results)
            profitable_trades = np.sum(pnl_values > 0)
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = np.mean(pnl_values)
            total_pnl = np.sum(pnl_values)
            avg_hold_time = np.mean(hold_times)

            # Максимальна просадка
            cumulative_pnl = np.cumsum(pnl_values)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            # Профіт-фактор
            profitable_pnl = pnl_values[pnl_values > 0]
            loss_pnl = pnl_values[pnl_values <= 0]

            total_profit = np.sum(profitable_pnl) if len(profitable_pnl) > 0 else 0
            total_loss = abs(np.sum(loss_pnl)) if len(loss_pnl) > 0 else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

            # Sharpe ratio
            sharpe_ratio = avg_pnl / np.std(pnl_values) if len(pnl_values) > 1 and np.std(pnl_values) > 0 else 0

            best_trade = np.max(pnl_values) if len(pnl_values) > 0 else 0
            worst_trade = np.min(pnl_values) if len(pnl_values) > 0 else 0

            result = OptimizationResult(
                config_id=config_id,
                parameters=test_config,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                avg_hold_time=avg_hold_time,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                filtered_out=filtered_out,
                best_trade=best_trade,
                worst_trade=worst_trade,
                sharpe_ratio=sharpe_ratio,
                strategy_name=test_config.get('strategy_name', 'unknown'),
                execution_time=time.time() - start_time
            )

            # Виводимо результат в консоль
            print(
                f"Тестовано {config_id}: Avg PnL={result.avg_pnl:.3f}%, Trades={result.total_trades}, Win Rate={result.win_rate:.1f}%")

            # Одразу записуємо в файл з локом
            append_result_to_csv(result, output_file)

            results.append(result)

        except Exception as e:
            logger.error(f"Помилка тестування конфігурації {config_id}: {e}")
            continue

    # Звільняємо shared memory
    signals_shm.close()
    base_config_shm.close()

    return results


def append_result_to_csv(result: OptimizationResult, filename: str):
    """Додає один результат до CSV файлу з локуванням"""
    with csv_lock:
        fieldnames = [
                         'config_id', 'strategy_name', 'total_trades', 'win_rate',
                         'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown', 'sharpe_ratio',
                         'avg_hold_time', 'filtered_out', 'best_trade', 'worst_trade', 'execution_time'
                     ] + list(result.parameters.keys())  # Додаємо всі параметри динамічно

        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            row = {
                'config_id': result.config_id,
                'strategy_name': result.strategy_name,
                'total_trades': result.total_trades,
                'win_rate': round(result.win_rate, 2),
                'avg_pnl': round(result.avg_pnl, 4),
                'total_pnl': round(result.total_pnl, 2),
                'profit_factor': round(result.profit_factor, 2),
                'max_drawdown': round(result.max_drawdown, 2),
                'sharpe_ratio': round(result.sharpe_ratio, 3),
                'avg_hold_time': round(result.avg_hold_time, 2),
                'filtered_out': result.filtered_out,
                'best_trade': round(result.best_trade, 2),
                'worst_trade': round(result.worst_trade, 2),
                'execution_time': round(result.execution_time, 3),
            }
            row.update(result.parameters)  # Додаємо параметри

            writer.writerow(row)


def apply_config_to_base_static(base_config: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
    """Покращена функція застосування тестової конфігурації з підтримкою всіх фільтрів"""
    config = copy.deepcopy(base_config)

    # Торгові параметри
    for param in ['stop_loss', 'take_profit', 'max_hold_hours', 'use_trailing_stop']:
        if param in test_config:
            config['trading_parameters'][param] = test_config[param]

    # RSI параметри
    rsi_params = config['rsi_parameters']
    for param in ['long_exit_zone', 'short_exit_zone', 'long_extreme_exit', 'short_extreme_exit']:
        if param in test_config:
            rsi_params[param] = test_config[param]

    # Основні фільтри
    filters = config['filters']
    filter_params = ['use_filters', 'min_filters_required', 'use_trend_filter',
                     'use_adx_filter', 'use_volatility_filter', 'use_volume_filter',
                     'use_momentum_filter']
    for param in filter_params:
        if param in test_config:
            filters[param] = test_config[param]

    # Пороги основних фільтрів
    thresholds = config['filter_thresholds']
    threshold_params = ['min_adx', 'min_volatility', 'max_volatility', 'min_volume_ratio']
    for param in threshold_params:
        if param in test_config:
            thresholds[param] = test_config[param]

    # НОВІ: Розширені параметри трендової фільтрації
    trend_filter_params = ['trend_strength_threshold', 'trend_confirmation_periods',
                           'use_multi_timeframe_trend', 'trend_divergence_threshold',
                           'min_trend_consistency', 'max_trend_age_hours',
                           'require_all_timeframes_trend', 'allow_weak_trend_on_strong_signal',
                           'trend_weight_multiplier']

    # Додаємо секцію для трендових фільтрів якщо її немає
    if 'trend_filter_settings' not in config:
        config['trend_filter_settings'] = {}

    trend_settings = config['trend_filter_settings']
    for param in trend_filter_params:
        if param in test_config:
            trend_settings[param] = test_config[param]

    # Параметри momentum фільтра
    momentum_params = ['momentum_threshold', 'momentum_periods', 'use_roc_momentum', 'roc_threshold']
    if 'momentum_filter_settings' not in config:
        config['momentum_filter_settings'] = {}

    momentum_settings = config['momentum_filter_settings']
    for param in momentum_params:
        if param in test_config:
            momentum_settings[param] = test_config[param]

    # Ваги таймфреймів для загальної обробки
    if any(k in test_config for k in ['weight_5m', 'weight_15m', 'weight_1h']):
        weights = config['timeframe_settings']['trend_weights']
        for tf in ['5m', '15m', '1h']:
            weight_key = f'weight_{tf}'
            if weight_key in test_config:
                weights[tf] = test_config[weight_key]

        # Нормалізація
        total = sum(weights.values())
        for k in weights:
            weights[k] = weights[k] / total

    # Ваги таймфреймів для трендової фільтрації
    if any(k in test_config for k in ['trend_weight_5m', 'trend_weight_15m', 'trend_weight_1h']):
        if 'trend_timeframe_weights' not in config:
            config['trend_timeframe_weights'] = {'5m': 0.4, '15m': 0.35, '1h': 0.25}

        trend_weights = config['trend_timeframe_weights']
        for tf in ['5m', '15m', '1h']:
            weight_key = f'trend_weight_{tf}'
            if weight_key in test_config:
                trend_weights[tf] = test_config[weight_key]

        # Нормалізація трендових ваг
        total = sum(trend_weights.values())
        for k in trend_weights:
            trend_weights[k] = trend_weights[k] / total

    # Технічні індикатори (без змін)
    indicators_5m = config['technical_indicators']['5m']
    indicator_params = ['rsi_period', 'sma_fast', 'sma_slow', 'adx_period',
                        'ema_fast', 'ema_slow', 'atr_period', 'bb_period']
    for param in indicator_params:
        param_5m = f'{param}_5m'
        if param_5m in test_config:
            indicators_5m[param] = test_config[param_5m]

    return config


class IntegratedConfigOptimizer:
    """Модифікований інтегрований оптимізатор конфігурацій з покращеним розпаралеленням"""

    def __init__(self, base_config_file: str = 'enhanced_analyzer_config.json',
                 max_workers: Optional[int] = None,
                 cache_size: int = 10000,
                 use_adaptive_batching: bool = True,
                 early_stop_threshold: float = -5.0,
                 use_fast_mode: bool = True,
                 batch_size_per_worker: int = 5):

        # Ініціалізація логера перед використанням
        self.logger = configure_logging()

        self.base_config = TrendSignalAnalyzer.load_config(base_config_file)
        self.results = []
        self.top_results_heap = []

        # Оптимальна кількість воркерів для CPU-bound задач
        cpu_cores = cpu_count()
        self.max_workers = max_workers or min(cpu_cores - 1, 16)  # Залишаємо 1 ядро для системи

        self.cache_size = cache_size
        self.use_adaptive_batching = use_adaptive_batching
        self.early_stop_threshold = early_stop_threshold
        self.use_fast_mode = use_fast_mode
        self.batch_size_per_worker = batch_size_per_worker

        # Статистика продуктивності
        self.perf_stats = {
            'total_configs': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'early_stops': 0,
            'memory_usage_mb': 0,
            'avg_batch_time': 0,
            'validation_failures': {},
            'parallel_efficiency': 0.0
        }

        # Блокування для thread-safe операцій
        self.results_lock = Lock()
        self.progress_lock = Lock()

        self.logger.info(f"Ініціалізовано паралельний оптимізатор:")
        self.logger.info(f"  - Швидкий режим: {use_fast_mode}")
        self.logger.info(f"  - CPU ядер: {cpu_cores}, воркерів: {self.max_workers}")
        self.logger.info(f"  - Розмір батчу на воркер: {batch_size_per_worker}")
        self.logger.info(f"  - Розмір кешу: {cache_size}")
        self.logger.info(f"  - Адаптивне батчування: {use_adaptive_batching}")

    def _validate_config_logic_with_debug(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Валідація конфігурації з детальною діагностикою"""
        try:
            # Перевірка ваг таймфреймів
            weight_5m = config.get('weight_5m', 0.4)
            weight_15m = config.get('weight_15m', 0.35)
            weight_1h = config.get('weight_1h', 0.25)
            total_weight = weight_5m + weight_15m + weight_1h

            if abs(total_weight - 1.0) > 0.15:
                return False, f"Invalid weights sum: {total_weight:.3f} (should be ~1.0)"

            # Перевірка take_profit та stop_loss
            take_profit = config.get('take_profit', 0.035)
            stop_loss = config.get('stop_loss', 0.018)
            if take_profit <= stop_loss:
                return False, f"Take profit ({take_profit}) <= Stop loss ({stop_loss})"

            # Перевірка RSI зон
            long_exit = config.get('long_exit_zone', 65)
            short_exit = config.get('short_exit_zone', 35)
            if long_exit <= short_exit:
                return False, f"Long exit ({long_exit}) <= Short exit ({short_exit})"

            # Перевірка волатільності
            min_vol = config.get('min_volatility', 0.15)
            max_vol = config.get('max_volatility', 2.5)
            if min_vol >= max_vol:
                return False, f"Min volatility ({min_vol}) >= Max volatility ({max_vol})"

            # Перевірка SMA
            sma_fast = config.get('sma_fast_5m', 20)
            sma_slow = config.get('sma_slow_5m', 50)
            if sma_fast >= sma_slow:
                return False, f"Fast SMA ({sma_fast}) >= Slow SMA ({sma_slow})"

            # Перевірка EMA
            ema_fast = config.get('ema_fast_5m', 12)
            ema_slow = config.get('ema_slow_5m', 26)
            if ema_fast >= ema_slow:
                return False, f"Fast EMA ({ema_fast}) >= Slow EMA ({ema_slow})"

            # Перевірка extreme zones
            long_extreme = config.get('long_extreme_exit', 80)
            short_extreme = config.get('short_extreme_exit', 20)
            if long_extreme <= long_exit or short_extreme >= short_exit:
                return False, f"Invalid extreme zones: long_extreme={long_extreme}, short_extreme={short_extreme}"

            return True, "Valid"

        except Exception as e:
            return False, f"Exception: {str(e)}"

    def _validate_config_logic(self, config: Dict[str, Any]) -> bool:
        """Спрощена валідація без діагностики для продуктивності"""
        valid, _ = self._validate_config_logic_with_debug(config)
        return valid

    def generate_parameter_ranges(self, mode: str = 'balanced') -> Dict[str, List]:
        """Генерація валідних діапазонів параметрів з повним покриттям фільтрів"""
        if mode == 'fast':
            return {
                # Базові торгові параметри
                'stop_loss': [0.015, 0.020, 0.025],
                'take_profit': [0.030, 0.040, 0.050],
                'max_hold_hours': [8, 12, 20],
                'use_trailing_stop': [True, False],

                # RSI параметри
                'long_exit_zone': [65, 70, 75],
                'short_exit_zone': [25, 30, 35],
                'long_extreme_exit': [80, 85],
                'short_extreme_exit': [15, 20],

                # Основні фільтри
                'use_filters': [True, False],
                'min_filters_required': [2, 3],
                'use_trend_filter': [True, False],
                'use_adx_filter': [True, False],
                'use_volatility_filter': [True, False],
                'use_volume_filter': [True, False],
                'use_momentum_filter': [True, False],

                # Пороги фільтрів
                'min_adx': [15, 20, 25],
                'min_volatility': [0.12, 0.18],
                'max_volatility': [2.5, 3.5],
                'min_volume_ratio': [0.8, 1.0, 1.2],

                # НОВІ: Параметри трендової фільтрації
                'trend_strength_threshold': [0.6, 0.7, 0.8],
                'trend_confirmation_periods': [3, 5, 7],
                'use_multi_timeframe_trend': [True, False],
                'trend_divergence_threshold': [0.3, 0.5],
                'min_trend_consistency': [0.6, 0.8],

                # Ваги таймфреймів
                'weight_5m': [0.4, 0.5],
                'weight_15m': [0.3, 0.35],
                'weight_1h': [0.15, 0.25],

                # Технічні індикатори
                'sma_fast_5m': [15, 20],
                'sma_slow_5m': [50, 60],
                'ema_fast_5m': [10, 12],
                'ema_slow_5m': [26, 30],
            }

        elif mode == 'ultra_fast':
            return {
                'stop_loss': [0.018, 0.022],
                'take_profit': [0.035, 0.045],
                'max_hold_hours': [12, 16],
                'use_trailing_stop': [True],
                'long_exit_zone': [65, 70],
                'short_exit_zone': [30, 35],
                'use_filters': [True],
                'min_filters_required': [3],
                'use_trend_filter': [True],
                'use_adx_filter': [True],
                'use_volatility_filter': [True],
                'trend_strength_threshold': [0.7],
                'min_trend_consistency': [0.7],
                'min_adx': [20],
                'min_volatility': [0.15],
                'max_volatility': [3.0],
            }

        else:  # balanced або thorough режим
            return {
                # Базові торгові параметри
                'stop_loss': [0.012, 0.015, 0.018, 0.022, 0.025, 0.030],
                'take_profit': [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060],
                'max_hold_hours': [6, 8, 10, 12, 16, 20, 24],
                'use_trailing_stop': [True, False],

                # RSI параметри
                'long_exit_zone': [60, 62, 65, 68, 70, 72],
                'short_exit_zone': [28, 30, 32, 35, 38, 40],
                'long_extreme_exit': [75, 78, 80, 82, 85],
                'short_extreme_exit': [15, 18, 20, 22, 25],

                # Основні фільтри
                'use_filters': [True, False],
                'min_filters_required': [1, 2, 3, 4],
                'use_trend_filter': [True, False],
                'use_adx_filter': [True, False],
                'use_volatility_filter': [True, False],
                'use_volume_filter': [True, False],
                'use_momentum_filter': [True, False],

                # Пороги основних фільтрів
                'min_adx': [12, 15, 18, 20, 22, 25, 28],
                'min_volatility': [0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25],
                'max_volatility': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                'min_volume_ratio': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5],

                # РОЗШИРЕНІ ПАРАМЕТРИ ТРЕНДОВОЇ ФІЛЬТРАЦІ Ї
                'trend_strength_threshold': [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
                'trend_confirmation_periods': [2, 3, 4, 5, 6, 7, 8],
                'use_multi_timeframe_trend': [True, False],
                'trend_divergence_threshold': [0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
                'min_trend_consistency': [0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
                'max_trend_age_hours': [4, 6, 8, 12, 16, 20, 24],

                # Параметри momentum фільтра
                'momentum_threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
                'momentum_periods': [10, 14, 20, 26],
                'use_roc_momentum': [True, False],
                'roc_threshold': [0.5, 1.0, 1.5, 2.0],

                # Параметри комбінованих фільтрів
                'require_all_timeframes_trend': [True, False],
                'allow_weak_trend_on_strong_signal': [True, False],
                'trend_weight_multiplier': [1.0, 1.2, 1.5, 2.0],

                # Ваги таймфреймів для трендової фільтрації
                'trend_weight_5m': [0.3, 0.4, 0.5, 0.6],
                'trend_weight_15m': [0.25, 0.3, 0.35, 0.4],
                'trend_weight_1h': [0.15, 0.2, 0.25, 0.3, 0.35],

                # Ваги загальні
                'weight_5m': [0.35, 0.4, 0.45, 0.5, 0.55],
                'weight_15m': [0.25, 0.3, 0.35, 0.4, 0.45],
                'weight_1h': [0.15, 0.2, 0.25, 0.3],

                # Технічні індикатори
                'rsi_period_5m': [12, 14, 16, 18],
                'sma_fast_5m': [15, 20, 25],
                'sma_slow_5m': [40, 50, 60],
                'adx_period_5m': [12, 14, 16],
                'ema_fast_5m': [10, 12, 15],
                'ema_slow_5m': [21, 26, 30],
                'atr_period_5m': [12, 14, 16],
                'bb_period_5m': [18, 20, 22],
            }

    def _create_grid_search_optimized(self, param_ranges: Dict[str, List], max_combinations: int) -> List[
        Dict[str, Any]]:
        """Оптимізований повний перебір з валідацією та батчуванням"""
        self.logger.info(f"Запуск оптимізованого Grid Search з максимумом {max_combinations} комбінацій")

        # Розрахуємо загальну кількість комбінацій
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)

        self.logger.info(f"Загальна кількість можливих комбінацій: {total_combinations:,}")

        if total_combinations > max_combinations:
            self.logger.warning(f"Комбінацій ({total_combinations:,}) більше максимуму ({max_combinations:,})")
            self.logger.info("Буде використано випадкове семплування з перевіркою валідності")
            return self._create_sampled_grid_search(param_ranges, max_combinations)

        valid_configs = []
        total_checked = 0
        validation_failures = {}

        # Генеруємо комбінації батчами для економії пам'яті
        batch_size = min(10000, max_combinations // 10) if max_combinations > 100 else max_combinations

        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        self.logger.info(f"Генерація комбінацій батчами по {batch_size}")

        for batch_start in range(0, min(total_combinations, max_combinations * 5), batch_size):
            batch_combinations = list(islice(product(*param_values), batch_start, batch_start + batch_size))

            if not batch_combinations:
                break

            # Паралельна валідація батчу
            batch_configs = []
            for i, combination in enumerate(batch_combinations):
                if len(valid_configs) >= max_combinations:
                    break

                config = dict(zip(param_names, combination))
                config['strategy_name'] = f'grid_batch{batch_start // batch_size}_{i}'
                batch_configs.append(config)

            # Валідація в паралельних потоках
            with ThreadPoolExecutor(max_workers=min(8, len(batch_configs))) as executor:
                validation_futures = {
                    executor.submit(self._validate_config_logic_with_debug, config): config
                    for config in batch_configs
                }

                for future in as_completed(validation_futures):
                    config = validation_futures[future]
                    total_checked += 1

                    try:
                        is_valid, reason = future.result()
                        if is_valid:
                            valid_configs.append(config)
                        else:
                            if reason not in validation_failures:
                                validation_failures[reason] = 0
                            validation_failures[reason] += 1
                    except Exception as e:
                        self.logger.error(f"Помилка валідації: {e}")

            # Прогрес
            if total_checked % 50000 == 0:
                self.logger.info(f"Перевірено: {total_checked:,}, валідних: {len(valid_configs):,}")

            if len(valid_configs) >= max_combinations:
                break

        # Статистика валідації
        if validation_failures:
            self.logger.info("Статистика помилок валідації:")
            for reason, count in sorted(validation_failures.items(), key=lambda x: x[1], reverse=True)[:5]:
                self.logger.info(f"  {reason}: {count:,} разів")

        self.logger.info(
            f"Grid Search завершено: {len(valid_configs)} валідних конфігурацій з {total_checked:,} перевірених")
        return valid_configs[:max_combinations]

    def _create_sampled_grid_search(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Паралельне випадкове семплування з валідацією"""
        self.logger.info("Використовуємо паралельне випадкове семплування з валідацією")

        valid_configs = []
        attempts = 0
        max_attempts = max_combinations * 20
        validation_failures = {}

        # Генеруємо батчі кандидатів
        batch_size = min(1000, max_combinations // 4)
        param_names = list(param_ranges.keys())

        while len(valid_configs) < max_combinations and attempts < max_attempts:
            # Створюємо батч випадкових комбінацій
            batch_configs = []
            for i in range(min(batch_size, max_combinations - len(valid_configs))):
                config = {}
                for param_name in param_names:
                    config[param_name] = random.choice(param_ranges[param_name])
                config['strategy_name'] = f'sampled_grid_{len(valid_configs) + len(batch_configs)}'
                batch_configs.append(config)

            # Паралельна валідація батчу
            with ThreadPoolExecutor(max_workers=min(8, len(batch_configs))) as executor:
                validation_futures = {
                    executor.submit(self._validate_config_logic_with_debug, config): config
                    for config in batch_configs
                }

                batch_valid = []
                for future in as_completed(validation_futures):
                    config = validation_futures[future]
                    attempts += 1

                    try:
                        is_valid, reason = future.result()
                        if is_valid:
                            batch_valid.append(config)
                        else:
                            if reason not in validation_failures:
                                validation_failures[reason] = 0
                            validation_failures[reason] += 1
                    except Exception as e:
                        self.logger.error(f"Помилка валідації: {e}")

            valid_configs.extend(batch_valid)

            # Прогрес
            if attempts % 5000 == 0:
                success_rate = len(valid_configs) / attempts * 100
                self.logger.info(f"Спроб: {attempts:,}, валідних: {len(valid_configs):,} ({success_rate:.1f}%)")

        final_success_rate = len(valid_configs) / attempts * 100 if attempts > 0 else 0
        self.logger.info(
            f"Паралельне семплування завершено: {len(valid_configs)} конфігурацій за {attempts} спроб ({final_success_rate:.1f}% успіх)")
        return valid_configs

    def create_config_combinations(self, max_combinations: int = 1000,
                                   strategy_type: str = 'smart_mixed',
                                   mode: str = 'balanced') -> List[Dict[str, Any]]:
        """Створення комбінацій конфігурацій з паралельною валідацією"""
        param_ranges = self.generate_parameter_ranges(mode)
        self.logger.info(f"Генерація конфігурацій: стратегія={strategy_type}, режим={mode}")

        if strategy_type == 'genetic':
            strategies = self._create_genetic_search_fixed(param_ranges, max_combinations)
        elif strategy_type == 'grid_search' or strategy_type == 'full_grid':
            strategies = self._create_grid_search_optimized(param_ranges, max_combinations)
        else:
            strategies = self._create_sampled_grid_search(param_ranges, max_combinations)

        self.logger.info(f"Створено {len(strategies)} валідних конфігурацій")
        return strategies[:max_combinations]

    def _create_batches(self, configurations: List[Dict[str, Any]]) -> List[List[Tuple[Dict[str, Any], str]]]:
        """Створення оптимальних батчів для паралельної обробки"""
        total_configs = len(configurations)

        # Адаптивний розмір батчу
        if self.use_adaptive_batching:
            # Базуємо розмір батчу на кількості воркерів та складності
            base_batch_size = max(1, self.batch_size_per_worker)

            # Адаптуємо під кількість конфігурацій
            if total_configs < 100:
                batch_size = max(1, total_configs // self.max_workers)
            elif total_configs < 1000:
                batch_size = base_batch_size * 2
            else:
                batch_size = base_batch_size * 4
        else:
            batch_size = self.batch_size_per_worker

        self.logger.info(f"Створення батчів розміром {batch_size} для {total_configs} конфігурацій")

        batches = []
        for i in range(0, total_configs, batch_size):
            batch_configs = []
            for j, config in enumerate(configurations[i:i + batch_size], i):
                config_id = f"config_{j:04d}_{config.get('strategy_name', 'unknown')}"
                batch_configs.append((config, config_id))
            batches.append(batch_configs)

        self.logger.info(f"Створено {len(batches)} батчів для паралельної обробки")
        return batches

    async def optimize_parallel(self, signals_file: str, max_configs: int = 500,
                                output_file: str = 'optimization_results.csv',
                                strategy_type: str = 'smart_mixed',
                                mode: str = 'balanced') -> List[OptimizationResult]:
        """Основний метод паралельної оптимізації"""
        self.logger.info(f"Початок паралельної оптимізації з файлом сигналів: {signals_file}")
        self.logger.info(f"Максимальна кількість конфігурацій: {max_configs}")
        self.logger.info(f"Стратегія: {strategy_type}, Режим: {mode}")
        self.logger.info(f"Воркерів: {self.max_workers}")

        start_time = time.time()

        # Завантажуємо сигнали
        analyzer = TrendSignalAnalyzer()
        signals = analyzer.read_signals_csv(signals_file)

        if not signals:
            self.logger.error("Не вдалося завантажити сигнали")
            return []

        self.logger.info(f"Завантажено {len(signals)} сигналів")

        # Створюємо shared memory для signals_data
        serialized_signals = pickle.dumps(signals)
        signals_shm = shared_memory.SharedMemory(create=True, size=len(serialized_signals))
        signals_shm.buf[:] = serialized_signals

        # Shared memory для base_config
        serialized_base = pickle.dumps(self.base_config)
        base_shm = shared_memory.SharedMemory(create=True, size=len(serialized_base))
        base_shm.buf[:] = serialized_base

        # Генеруємо конфігурації
        configurations = self.create_config_combinations(max_configs, strategy_type, mode)

        if not configurations:
            self.logger.error("Не створено жодної валідної конфігурації")
            signals_shm.close()
            signals_shm.unlink()
            base_shm.close()
            base_shm.unlink()
            return []

        self.logger.info(f"Створено {len(configurations)} конфігурацій")

        # Створюємо батчі для паралельної обробки
        batches = self._create_batches(configurations)

        # Статистика для моніторингу
        total_batches = len(batches)
        completed_batches = 0
        all_results = []

        print(f"\nПочинаємо паралельне тестування {len(configurations)} конфігурацій у {total_batches} батчах...")
        print(f"Використовуємо {self.max_workers} паралельних процесів")

        # Паралельна обробка батчів
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Відправляємо батчі на обробку
            future_to_batch = {}

            for batch_idx, batch in enumerate(batches):
                future = executor.submit(test_config_worker, batch, signals_shm.name, base_shm.name, self.log_queue,
                                         output_file)
                future_to_batch[future] = batch_idx

            # Збираємо результати по мірі завершення
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]

                try:
                    batch_results = future.result(timeout=300)  # 5 хвилин таймаут на батч
                    all_results.extend(batch_results)
                    completed_batches += 1

                    # Прогрес
                    progress = completed_batches / total_batches * 100
                    configs_processed = completed_batches * self.batch_size_per_worker

                    with self.progress_lock:
                        print(f"Прогрес: {progress:.1f}% ({completed_batches}/{total_batches} батчів)")

                        if all_results:
                            # Показуємо поточний найкращий результат
                            current_best = max(all_results, key=lambda x: x.avg_pnl)
                            print(f"Найкращий поки що: {current_best.avg_pnl:.3f}% (угод: {current_best.total_trades})")

                except Exception as e:
                    self.logger.error(f"Помилка обробки батчу {batch_idx}: {e}")
                    continue

        if not all_results:
            self.logger.error("Немає результатів паралельної оптимізації")
            signals_shm.close()
            signals_shm.unlink()
            base_shm.close()
            base_shm.unlink()
            return []

        # Сортуємо результати
        self.results = sorted(all_results, key=lambda x: x.avg_pnl, reverse=True)

        # Розрахунок ефективності паралелізації
        execution_time = time.time() - start_time
        theoretical_sequential_time = sum(r.execution_time for r in all_results)
        parallel_efficiency = (
                                      theoretical_sequential_time / execution_time) / self.max_workers * 100 if execution_time > 0 else 0

        self.perf_stats['parallel_efficiency'] = parallel_efficiency

        # Оновлюємо CSV з рангами (оскільки записували без рангів)
        self.update_csv_with_ranks(self.results, output_file)

        self.print_top_results(self.results, top_n=10)
        self._print_performance_stats(execution_time, theoretical_sequential_time)

        self.logger.info(f"Паралельна оптимізація завершена за {execution_time:.2f} секунд")
        self.logger.info(f"Ефективність паралелізації: {parallel_efficiency:.1f}%")
        self.logger.info(f"Результати збережено в {output_file}")

        # Звільняємо shared memory
        signals_shm.close()
        signals_shm.unlink()
        base_shm.close()
        base_shm.unlink()

        return self.results

    def update_csv_with_ranks(self, results: List[OptimizationResult], filename: str):
        """Оновлює CSV з додаванням ранків (після всіх тестів)"""
        df = pd.read_csv(filename)
        df['rank'] = range(1, len(df) + 1)  # Додаємо ранк
        df.to_csv(filename, index=False, encoding='utf-8')

    def _print_performance_stats(self, actual_time: float, theoretical_sequential_time: float):
        """Виведення статистики продуктивності"""
        print(f"\n{'=' * 60}")
        print("СТАТИСТИКА ПРОДУКТИВНОСТІ")
        print(f"{'=' * 60}")
        print(f"Фактичний час виконання: {actual_time:.2f} секунд")
        print(f"Теоретичний послідовний час: {theoretical_sequential_time:.2f} секунд")
        print(f"Прискорення: {theoretical_sequential_time / actual_time:.2f}x")
        print(f"Ефективність паралелізації: {self.perf_stats['parallel_efficiency']:.1f}%")
        print(f"Використано CPU ядер: {self.max_workers}")

        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Використання пам'яті: {memory_usage:.1f} MB")

    def _create_genetic_search_fixed(self, param_ranges: Dict[str, List], max_combinations: int) -> List[
        Dict[str, Any]]:
        """Виправлений генетичний алгоритм з забезпеченням валідності"""
        population_size = min(500, max_combinations // 10)
        max_generations = max_combinations // population_size

        def create_valid_individual():
            """Створення валідної особини"""
            max_attempts = 100
            for _ in range(max_attempts):
                config = {}

                # Спочатку встановлюємо критичні зв'язані параметри
                stop_loss = random.choice(param_ranges['stop_loss'])
                valid_tp = [tp for tp in param_ranges['take_profit'] if tp > stop_loss]
                if not valid_tp:
                    continue
                take_profit = random.choice(valid_tp)

                config['stop_loss'] = stop_loss
                config['take_profit'] = take_profit

                # RSI зони
                short_exit = random.choice(param_ranges['short_exit_zone'])
                valid_long = [le for le in param_ranges['long_exit_zone'] if le > short_exit]
                if not valid_long:
                    continue
                long_exit = random.choice(valid_long)

                config['short_exit_zone'] = short_exit
                config['long_exit_zone'] = long_exit

                # Extreme zones
                if 'long_extreme_exit' in param_ranges:
                    valid_long_extreme = [le for le in param_ranges['long_extreme_exit'] if le > long_exit]
                    valid_short_extreme = [se for se in param_ranges['short_extreme_exit'] if se < short_exit]

                    if not valid_long_extreme or not valid_short_extreme:
                        continue

                    config['long_extreme_exit'] = random.choice(valid_long_extreme)
                    config['short_extreme_exit'] = random.choice(valid_short_extreme)

                # SMA параметри
                if 'sma_fast_5m' in param_ranges and 'sma_slow_5m' in param_ranges:
                    sma_fast = random.choice(param_ranges['sma_fast_5m'])
                    valid_slow_sma = [ss for ss in param_ranges['sma_slow_5m'] if ss > sma_fast]
                    if not valid_slow_sma:
                        continue

                    config['sma_fast_5m'] = sma_fast
                    config['sma_slow_5m'] = random.choice(valid_slow_sma)

                # EMA параметри
                if 'ema_fast_5m' in param_ranges and 'ema_slow_5m' in param_ranges:
                    ema_fast = random.choice(param_ranges['ema_fast_5m'])
                    valid_slow_ema = [se for se in param_ranges['ema_slow_5m'] if se > ema_fast]
                    if not valid_slow_ema:
                        continue

                    config['ema_fast_5m'] = ema_fast
                    config['ema_slow_5m'] = random.choice(valid_slow_ema)

                # Волатільність
                if 'min_volatility' in param_ranges and 'max_volatility' in param_ranges:
                    min_vol = random.choice(param_ranges['min_volatility'])
                    valid_max_vol = [mv for mv in param_ranges['max_volatility'] if mv > min_vol]
                    if not valid_max_vol:
                        continue

                    config['min_volatility'] = min_vol
                    config['max_volatility'] = random.choice(valid_max_vol)

                # Ваги (нормалізуємо)
                if all(k in param_ranges for k in ['weight_5m', 'weight_15m', 'weight_1h']):
                    w5 = random.choice(param_ranges['weight_5m'])
                    w15 = random.choice(param_ranges['weight_15m'])
                    w1h = random.choice(param_ranges['weight_1h'])

                    total = w5 + w15 + w1h
                    config['weight_5m'] = w5 / total
                    config['weight_15m'] = w15 / total
                    config['weight_1h'] = w1h / total

                # Інші параметри
                for param, values in param_ranges.items():
                    if param not in config:
                        config[param] = random.choice(values)

                # Перевіряємо валідність
                if self._validate_config_logic(config):
                    return config

            return None

        # Створюємо початкову популяцію
        population = []
        for i in range(population_size * 3):
            individual = create_valid_individual()
            if individual:
                individual['strategy_name'] = f'genetic_gen0_{len(population)}'
                population.append(individual)
                if len(population) >= population_size:
                    break

        if not population:
            self.logger.error("Не вдалося створити жодної валідної конфігурації")
            return []

        self.logger.info(f"Створено початкову популяцію: {len(population)} особин")
        all_strategies = population.copy()

        # Еволюція
        for gen in range(1, min(max_generations, 10)):
            new_population = []

            # Елітизм - зберігаємо кращих
            elite_size = max(1, len(population) // 5)
            population.sort(key=lambda x: x.get('take_profit', 0) - x.get('stop_loss', 0), reverse=True)
            new_population.extend(population[:elite_size])

            # Генеруємо нових особин
            attempts = 0
            max_attempts = population_size * 10

            while len(new_population) < population_size and attempts < max_attempts:
                attempts += 1

                if random.random() < 0.7:  # Схрещування
                    if len(population) >= 2:
                        parent1 = random.choice(population)
                        parent2 = random.choice(population)
                        child = create_valid_individual()
                        if child:
                            # Копіюємо деякі гени від батьків
                            for param in random.sample(list(parent1.keys()), min(5, len(parent1))):
                                if param in parent2 and random.random() < 0.5:
                                    child[param] = random.choice([parent1[param], parent2[param]])

                            child['strategy_name'] = f'genetic_gen{gen}_cross_{len(new_population)}'
                            if self._validate_config_logic(child):
                                new_population.append(child)
                else:  # Мутація
                    if population:
                        child = create_valid_individual()
                        if child:
                            child['strategy_name'] = f'genetic_gen{gen}_mut_{len(new_population)}'
                            new_population.append(child)

            population = new_population
            all_strategies.extend(population)
            self.logger.info(f"Покоління {gen}: {len(population)} особин")

        self.logger.info(f"Генетичний алгоритм створив {len(all_strategies)} конфігурацій")
        return all_strategies

    def apply_config_to_base(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Застосування тестової конфігурації до базової"""
        return apply_config_to_base_static(self.base_config, test_config)

    def save_optimization_results(self, results: List[OptimizationResult], filename: str):
        """Покращене збереження результатів з повним набором параметрів фільтрації"""
        self.logger.info(f"Збереження {len(results)} результатів у {filename}")

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            # Розширений список полів включаючи всі параметри фільтрів
            fieldnames = [
                'rank', 'config_id', 'strategy_name', 'total_trades', 'win_rate',
                'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown', 'sharpe_ratio',
                'avg_hold_time', 'filtered_out', 'best_trade', 'worst_trade', 'execution_time',

                # Торгові параметри
                'stop_loss', 'take_profit', 'max_hold_hours', 'use_trailing_stop',

                # RSI параметри
                'long_exit_zone', 'short_exit_zone', 'long_extreme_exit', 'short_extreme_exit',

                # Основні фільтри
                'use_filters', 'min_filters_required', 'use_trend_filter',
                'use_adx_filter', 'use_volatility_filter', 'use_volume_filter', 'use_momentum_filter',

                # Пороги фільтрів
                'min_adx', 'min_volatility', 'max_volatility', 'min_volume_ratio',

                # НОВІ: Параметри трендової фільтрації
                'trend_strength_threshold', 'trend_confirmation_periods', 'use_multi_timeframe_trend',
                'trend_divergence_threshold', 'min_trend_consistency', 'max_trend_age_hours',
                'require_all_timeframes_trend', 'allow_weak_trend_on_strong_signal',
                'trend_weight_multiplier',

                # Параметри momentum фільтра
                'momentum_threshold', 'momentum_periods', 'use_roc_momentum', 'roc_threshold',

                # Ваги таймфреймів
                'weight_5m', 'weight_15m', 'weight_1h',
                'trend_weight_5m', 'trend_weight_15m', 'trend_weight_1h',

                # Технічні індикатори
                'rsi_period_5m', 'sma_fast_5m', 'sma_slow_5m', 'adx_period_5m',
                'ema_fast_5m', 'ema_slow_5m', 'atr_period_5m', 'bb_period_5m'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rank, result in enumerate(results, 1):
                params = result.parameters
                row = {
                    'rank': rank,
                    'config_id': result.config_id,
                    'strategy_name': result.strategy_name,
                    'total_trades': result.total_trades,
                    'win_rate': round(result.win_rate, 2),
                    'avg_pnl': round(result.avg_pnl, 4),
                    'total_pnl': round(result.total_pnl, 2),
                    'profit_factor': round(result.profit_factor, 2),
                    'max_drawdown': round(result.max_drawdown, 2),
                    'sharpe_ratio': round(result.sharpe_ratio, 3),
                    'avg_hold_time': round(result.avg_hold_time, 2),
                    'filtered_out': result.filtered_out,
                    'best_trade': round(result.best_trade, 2),
                    'worst_trade': round(result.worst_trade, 2),
                    'execution_time': round(result.execution_time, 3),
                }

                # Додаємо всі параметри з конфігурації
                for field in fieldnames[15:]:  # пропускаємо базові метрики
                    row[field] = params.get(field, '')

                writer.writerow(row)

        # Зберігаємо топ-10 повних конфігурацій
        top_configs = results[:10]
        configs_data = {}

        for i, result in enumerate(top_configs, 1):
            full_config = self.apply_config_to_base(result.parameters)
            configs_data[f'top_{i}_config'] = {
                'rank': i,
                'avg_pnl': result.avg_pnl,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'config': full_config
            }

        config_filename = filename.replace('.csv', '_top_configs.json')
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(configs_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Топ-10 повних конфігурацій збережено в {config_filename}")

    def print_top_results(self, results: List[OptimizationResult], top_n: int = 10):
        """Покращений вивід результатів з параметрами фільтрації"""
        print(f"\n{'=' * 80}")
        print(f"ТОП-{top_n} НАЙКРАЩИХ КОНФІГУРАЦІЙ")
        print(f"{'=' * 80}")

        for i, result in enumerate(results[:top_n], 1):
            params = result.parameters
            print(f"\n#{i} - {result.config_id}")
            print(f"Стратегія: {result.strategy_name}")
            print(f"Угод: {result.total_trades} | Win Rate: {result.win_rate:.1f}% | "
                  f"Avg PnL: {result.avg_pnl:.3f}% | Total PnL: {result.total_pnl:.2f}%")
            print(f"Profit Factor: {result.profit_factor:.2f} | Max DD: {result.max_drawdown:.2f}% | "
                  f"Sharpe: {result.sharpe_ratio:.3f}")
            print(f"Час виконання: {result.execution_time:.3f}с | Відфільтровано: {result.filtered_out}")

            # Торгові параметри
            if 'stop_loss' in params and 'take_profit' in params:
                print(f"Торгові: SL={params['stop_loss']:.3f} | TP={params['take_profit']:.3f} | "
                      f"Hold={params.get('max_hold_hours', 'N/A')}h | "
                      f"Trailing={params.get('use_trailing_stop', 'N/A')}")

            # Фільтри
            filters_info = []
            if params.get('use_filters'):
                if params.get('use_trend_filter'):
                    trend_strength = params.get('trend_strength_threshold', 'N/A')
                    filters_info.append(f"Trend(>{trend_strength})")
                if params.get('use_adx_filter'):
                    min_adx = params.get('min_adx', 'N/A')
                    filters_info.append(f"ADX(>{min_adx})")
                if params.get('use_volatility_filter'):
                    min_vol = params.get('min_volatility', 'N/A')
                    max_vol = params.get('max_volatility', 'N/A')
                    filters_info.append(f"Vol({min_vol}-{max_vol})")
                if params.get('use_volume_filter'):
                    vol_ratio = params.get('min_volume_ratio', 'N/A')
                    filters_info.append(f"Volume(>{vol_ratio})")
                if params.get('use_momentum_filter'):
                    mom_threshold = params.get('momentum_threshold', 'N/A')
                    filters_info.append(f"Momentum(>{mom_threshold})")

                min_required = params.get('min_filters_required', 'N/A')
                print(f"Фільтри (потрібно {min_required}): {' | '.join(filters_info) if filters_info else 'Немає'}")

            # Трендові параметри
            trend_params = []
            if params.get('trend_confirmation_periods'):
                trend_params.append(f"Conf={params['trend_confirmation_periods']}")
            if params.get('min_trend_consistency'):
                trend_params.append(f"Consist={params['min_trend_consistency']:.2f}")
            if params.get('use_multi_timeframe_trend'):
                trend_params.append("MultiTF=Yes")
            if trend_params:
                print(f"Тренд: {' | '.join(trend_params)}")

            print("-" * 80)

    async def optimize(self, signals_file: str, max_configs: int = 500,
                       output_file: str = 'optimization_results.csv',
                       strategy_type: str = 'smart_mixed',
                       mode: str = 'balanced') -> List[OptimizationResult]:
        """Основний метод оптимізації (зворотна сумісність)"""
        return await self.optimize_parallel(signals_file, max_configs, output_file, strategy_type, mode)


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(
        description='Паралельний інтегрований оптимізатор з підтримкою повного перебору параметрів')
    parser.add_argument('signals_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='optimization_results.csv',
                        help='Файл результатів оптимізації')
    parser.add_argument('-c', '--config', default='enhanced_analyzer_config.json',
                        help='Базовий файл конфігурації')
    parser.add_argument('-n', '--max-configs', type=int, default=500,
                        help='Максимальна кількість конфігурацій для тестування')
    parser.add_argument('--strategy', choices=['genetic', 'simple', 'grid_search', 'full_grid'],
                        default='genetic', help='Стратегія оптимізації')
    parser.add_argument('--mode', choices=['fast', 'balanced', 'thorough', 'ultra_fast'],
                        default='balanced', help='Режим оптимізації')
    parser.add_argument('--workers', type=int, default=None,
                        help='Кількість паралельних воркерів (за замовчуванням: CPU ядра - 1)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Розмір батчу на воркер')

    args = parser.parse_args()

    # Перевіряємо файли
    if not os.path.exists(args.signals_file):
        print(f"Помилка: файл сигналів {args.signals_file} не знайдено")
        return

    if not os.path.exists(args.config):
        print(f"Попередження: базовий конфіг {args.config} не знайдено")

    print(f"Паралельний інтегрований оптимізатор з підтримкою повного перебору параметрів")
    print(f"Файл сигналів: {args.signals_file}")
    print(f"Стратегія: {args.strategy}")
    print(f"Режим: {args.mode}")
    print(f"Максимальна кількість конфігурацій: {args.max_configs}")
    print(f"Воркерів: {args.workers or 'auto'}")
    print(f"Розмір батчу: {args.batch_size}")

    # Попереджуємо про повний перебір
    if args.strategy in ['grid_search', 'full_grid']:
        print(f"\n⚠️  УВАГА: Обрано повний перебір параметрів!")
        print(f"Це може займати багато часу при великій кількості параметрів.")
        print(f"Паралелізація значно пришвидшить процес.")

        if args.mode not in ['fast', 'ultra_fast'] and args.max_configs > 10000:
            response = input(f"Продовжити з режимом '{args.mode}' та лімітом {args.max_configs} конфігурацій? (y/N): ")
            if response.lower() != 'y':
                print("Оптимізацію скасовано.")
                return

    # Створюємо оптимізатор
    optimizer = IntegratedConfigOptimizer(
        base_config_file=args.config,
        use_fast_mode=False,
        max_workers=args.workers,
        batch_size_per_worker=args.batch_size
    )

    # Налаштовуємо логування для main
    logger = configure_logging()

    # Створюємо чергу для логів
    manager = Manager()
    log_queue = manager.Queue(-1)
    optimizer.log_queue = log_queue  # Зберігаємо в об'єкті для доступу

    # Налаштовуємо listener для обробки логів
    file_handler = logging.FileHandler('optimizer.log', encoding='utf-8')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    listener = QueueListener(log_queue, file_handler, stream_handler)
    listener.start()

    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("ЗАПУСК ПАРАЛЕЛЬНОЇ ОПТИМІЗАЦІ Ї")
        logger.info("=" * 60)

        results = await optimizer.optimize_parallel(
            signals_file=args.signals_file,
            max_configs=args.max_configs,
            output_file=args.output,
            strategy_type=args.strategy,
            mode=args.mode
        )

        if results:
            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print("ПІДСУМОК ПАРАЛЕЛЬНО Ї ОПТИМІЗАЦІ Ї")
            print(f"{'=' * 60}")
            print(f"Протестовано конфігурацій: {len(results)}")
            print(f"Час виконання: {duration}")
            print(f"Найкращий результат: {results[0].avg_pnl:.3f}% середній PnL")
            print(f"Результати збережено в: {args.output}")

            logger.info("Паралельна оптимізація завершена успішно")

        else:
            print("Не отримано результатів оптимізації")
            logger.error("Оптимізація не дала результатів")

    except KeyboardInterrupt:
        print("\nОптимізацію перервано користувачем")
        logger.info("Оптимізацію перервано користувачем")
    except Exception as e:
        print(f"Критична помилка: {e}")
        logger.error(f"Критична помилка під час оптимізації: {e}", exc_info=True)
    finally:
        listener.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограму перервано")
    except Exception as e:
        print(f"Критична помилка: {e}")
        logging.getLogger(__name__).error(f"Критична помилка: {e}", exc_info=True)