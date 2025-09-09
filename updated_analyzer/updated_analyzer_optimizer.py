#!/usr/bin/env python3
"""
Асинхронний оптимізатор для Enhanced Analyzer з динамічним трейлінг стопом
"""

import asyncio
import json
import pandas as pd
import numpy as np
from itertools import product, islice
import logging
from datetime import datetime, timedelta
import argparse
import os
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import hashlib
import heapq
import psutil
from functools import lru_cache
import gc
from threading import Lock
import aiofiles

# Імпортуємо аналізатор
from updated_analyzer import SignalAnalyzer


# Налаштування логування
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('optimizer.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


@dataclass
class OptimizationResult:
    """Результат оптимізації одної конфігурації"""
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
        return self.avg_pnl < other.avg_pnl

    def to_dict(self) -> Dict:
        return asdict(self)


class AsyncConfigOptimizer:
    """Асинхронний оптимізатор конфігурацій для Enhanced Analyzer"""

    def __init__(self, base_config_file: str = 'updated_analyzer_config.json',
                 max_concurrent: int = 20,
                 use_fast_mode: bool = True):

        self.base_config = SignalAnalyzer.load_config(base_config_file)
        self.results = []
        self.max_concurrent = max_concurrent
        self.use_fast_mode = use_fast_mode
        self.results_lock = Lock()

        # Статистика
        self.perf_stats = {
            'total_configs': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'avg_execution_time': 0,
            'memory_usage_mb': 0
        }

        logger.info(f"Ініціалізовано асинхронний оптимізатор:")
        logger.info(f"  - Максимум одночасних завдань: {max_concurrent}")
        logger.info(f"  - Швидкий режим: {use_fast_mode}")

    def generate_parameter_ranges(self, mode: str = 'balanced') -> Dict[str, List]:
        """Генерація діапазонів параметрів для Enhanced Analyzer"""

        if mode == 'fast':
            return {
                # Торгові параметри
                'stop_loss': [0.015, 0.018, 0.022, 0.025],
                'take_profit': [0.030, 0.035, 0.040, 0.045],
                'max_hold_hours': [12, 16, 20, 24],
                'use_trailing_stop': [True, False],
                'trailing_stop_activation': [0.008, 0.010, 0.012],
                'trailing_stop_distance': [0.006, 0.008, 0.010],

                # RSI зони виходу
                'long_exit_zone': [60, 65, 70],
                'short_exit_zone': [30, 35, 40],
                'long_extreme_exit': [75, 80, 85],
                'short_extreme_exit': [15, 20, 25],

                # Вторинні таймфрейми
                'secondary_timeframes': [['15m'], ['30m'], ['1h'], ['15m', '30m'], ['15m', '1h']],

                # Параметри вторинного TF фільтру
                'short_enter_zone_mid_tf': [55, 60, 65],
                'long_enter_zone_mid_tf': [35, 40, 45],
                'sma_change_periods': [4, 5, 6],
                'sma_change_threshold': [0.4, 0.5, 0.6],
                'rsi_extreme_threshold': [12, 15, 18],
                'cross_lookback_periods': [8, 10, 12],

                # Буфери даних
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

        else:  # balanced або thorough
            return {
                # Торгові параметри
                'stop_loss': [0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030],
                'take_profit': [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060],
                'max_hold_hours': [8, 12, 16, 20, 24, 30],
                'use_trailing_stop': [True, False],
                'trailing_stop_activation': [0.008, 0.010, 0.012, 0.015],
                'trailing_stop_distance': [0.005, 0.006, 0.008, 0.010, 0.012],

                # RSI зони
                'long_exit_zone': [55, 60, 62, 65, 68, 70, 72],
                'short_exit_zone': [28, 30, 32, 35, 38, 40, 45],
                'long_extreme_exit': [75, 78, 80, 82, 85, 88],
                'short_extreme_exit': [12, 15, 18, 20, 22, 25],

                # Вторинні таймфрейми (включаючи 30m, 1h)
                'secondary_timeframes': [
                    ['15m'], ['30m'], ['1h'], ['15m', '30m'], ['15m', '1h']
                ],

                # Параметри фільтру вторинного TF
                'short_enter_zone_mid_tf': [50, 55, 57, 60, 63, 65, 67, 70],
                'long_enter_zone_mid_tf': [30, 35, 37, 40, 43, 45, 47, 50],
                'sma_change_periods': [3, 4, 5, 6, 7, 8, 9],
                'sma_change_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                'rsi_extreme_threshold': [10, 12, 15, 18, 20],
                'cross_lookback_periods': [6, 8, 10, 12, 15],

                # Буфери даних
                'buffer_hours_before': [80, 100, 120, 140, 160],
                'buffer_hours_after': [180, 200, 240, 280, 320],

                # Множники для TF
                'tf_multiplier_15m': [2, 3, 4],
                'tf_multiplier_30m': [4, 6, 8],
                'tf_multiplier_1h': [8, 12, 16],
            }

    def _validate_config_logic(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Валідація логіки конфігурації"""
        try:
            # Перевірка take_profit > stop_loss
            take_profit = config.get('take_profit', 0.035)
            stop_loss = config.get('stop_loss', 0.018)
            if take_profit <= stop_loss:
                return False, f"Take profit ({take_profit}) <= Stop loss ({stop_loss})"

            # Перевірка trailing stop параметрів
            if config.get('use_trailing_stop', True):
                activation = config.get('trailing_stop_activation', 0.010)
                distance = config.get('trailing_stop_distance', 0.008)
                if activation <= distance:
                    return False, f"Trailing activation ({activation}) <= distance ({distance})"

            # Перевірка RSI зон
            long_exit = config.get('long_exit_zone', 65)
            short_exit = config.get('short_exit_zone', 35)
            if long_exit <= short_exit:
                return False, f"Long exit ({long_exit}) <= Short exit ({short_exit})"

            # Перевірка extreme zones
            long_extreme = config.get('long_extreme_exit', 80)
            short_extreme = config.get('short_extreme_exit', 20)
            if long_extreme <= long_exit or short_extreme >= short_exit:
                return False, f"Invalid extreme zones"

            # Перевірка параметрів вторинного TF
            short_mid = config.get('short_enter_zone_mid_tf', 60)
            long_mid = config.get('long_enter_zone_mid_tf', 40)
            if short_mid <= long_mid:
                return False, f"Secondary TF zones invalid"

            return True, "Valid"

        except Exception as e:
            return False, f"Exception: {str(e)}"

    def apply_config_to_base(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Застосування тестової конфігурації до базової"""
        config = copy.deepcopy(self.base_config)

        # Торгові параметри
        trading_params = config['trading_parameters']
        for param in ['stop_loss', 'take_profit', 'max_hold_hours', 'use_trailing_stop',
                      'trailing_stop_activation', 'trailing_stop_distance']:
            if param in test_config:
                trading_params[param] = test_config[param]

        # RSI параметри
        rsi_params = config['rsi_parameters']
        for param in ['long_exit_zone', 'short_exit_zone', 'long_extreme_exit', 'short_extreme_exit']:
            if param in test_config:
                rsi_params[param] = test_config[param]

        # Налаштування таймфреймів
        tf_settings = config['timeframe_settings']
        if 'secondary_timeframes' in test_config:
            tf_settings['secondary_timeframes'] = test_config['secondary_timeframes']

        # Фільтр вторинного TF
        secondary_filter = config['secondary_tf_filter']
        for param in ['short_enter_zone_mid_tf', 'long_enter_zone_mid_tf', 'sma_change_periods',
                      'sma_change_threshold', 'rsi_extreme_threshold', 'cross_lookback_periods']:
            if param in test_config:
                secondary_filter[param] = test_config[param]

        # Управління даними
        data_mgmt = config['data_management']
        for param in ['buffer_hours_before', 'buffer_hours_after']:
            if param in test_config:
                data_mgmt[param] = test_config[param]

        # Множники TF
        tf_multiplier = data_mgmt['tf_multiplier']
        if 'tf_multiplier_15m' in test_config:
            tf_multiplier['15m'] = test_config['tf_multiplier_15m']
        if 'tf_multiplier_30m' in test_config:
            tf_multiplier['30m'] = test_config['tf_multiplier_30m']
        if 'tf_multiplier_1h' in test_config:
            tf_multiplier['1h'] = test_config['tf_multiplier_1h']

        return config

    async def test_single_config(self, config: Dict[str, Any], config_id: str,
                                 signals: List[Dict], semaphore: asyncio.Semaphore) -> Optional[OptimizationResult]:
        """Тестування однієї конфігурації"""
        async with semaphore:
            start_time = time.time()

            try:
                # Валідація
                is_valid, reason = self._validate_config_logic(config)
                if not is_valid:
                    logger.debug(f"Конфігурація {config_id} невалідна: {reason}")
                    return None

                # Застосовуємо конфігурацію
                full_config = self.apply_config_to_base(config)

                # Створюємо аналізатор
                analyzer = SignalAnalyzer(full_config)

                # Аналізуємо сигнали
                results = []
                processed_pairs = set()

                for signal in signals:
                    pair = signal['pair']
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)

                    # Отримуємо сигнали для цієї пари
                    pair_signals = [s for s in signals if s['pair'] == pair]

                    try:
                        # Отримуємо дані з буфером
                        data = await analyzer.fetch_data_with_buffer(pair, pair_signals, full_config)

                        if not data:
                            continue

                        # Симулюємо торгівлю для кожного сигналу пари
                        for pair_signal in pair_signals:
                            await analyzer.simulate_trade(data, pair_signal, full_config, f"temp_{config_id}.csv")

                    except Exception as e:
                        logger.debug(f"Помилка обробки пари {pair} для конфігурації {config_id}: {e}")
                        continue

                # Читаємо результати з тимчасового CSV
                temp_file = f"temp_{config_id}.csv"
                if os.path.exists(temp_file):
                    df = pd.read_csv(temp_file, delimiter=';')
                    results = df.to_dict('records')
                    os.remove(temp_file)  # Видаляємо тимчасовий файл

                if not results:
                    return OptimizationResult(
                        config_id=config_id,
                        parameters=config,
                        total_trades=0, profitable_trades=0, win_rate=0.0,
                        avg_pnl=0.0, total_pnl=0.0, avg_hold_time=0.0,
                        max_drawdown=0.0, profit_factor=0.0, filtered_out=0,
                        best_trade=0.0, worst_trade=0.0, sharpe_ratio=0.0,
                        strategy_name=config.get('strategy_name', 'test'),
                        execution_time=time.time() - start_time
                    )

                # Обчислення метрик
                traded_results = [r for r in results if not r.get('filtered_out', False)]
                filtered_out = len(results) - len(traded_results)

                if not traded_results:
                    return OptimizationResult(
                        config_id=config_id, parameters=config, total_trades=0,
                        profitable_trades=0, win_rate=0.0, avg_pnl=0.0, total_pnl=0.0,
                        avg_hold_time=0.0, max_drawdown=0.0, profit_factor=0.0,
                        filtered_out=filtered_out, best_trade=0.0, worst_trade=0.0,
                        sharpe_ratio=0.0, strategy_name=config.get('strategy_name', 'test'),
                        execution_time=time.time() - start_time
                    )

                # Векторизовані обчислення
                pnl_values = np.array([float(r.get('pnl_percent', 0)) for r in traded_results])
                hold_times = np.array([float(r.get('hold_time', 0)) for r in traded_results])

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

                # Profit factor
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
                    parameters=config,
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
                    strategy_name=config.get('strategy_name', 'test'),
                    execution_time=time.time() - start_time
                )

                # Виводимо прогрес
                print(f"Тестовано {config_id}: PnL={avg_pnl:.3f}%, Угод={total_trades}, Win={win_rate:.1f}%")

                return result

            except Exception as e:
                logger.error(f"Помилка тестування конфігурації {config_id}: {e}")
                return None

    def create_config_combinations(self, max_combinations: int = 1000,
                                   strategy_type: str = 'smart_mixed',
                                   mode: str = 'balanced') -> List[Dict[str, Any]]:
        """Створення комбінацій конфігурацій"""
        param_ranges = self.generate_parameter_ranges(mode)

        logger.info(f"Генерація конфігурацій: стратегія={strategy_type}, режим={mode}")

        if strategy_type == 'grid_search':
            return self._create_grid_search(param_ranges, max_combinations)
        elif strategy_type == 'genetic':
            return self._create_genetic_search(param_ranges, max_combinations)
        else:  # random sampling
            return self._create_random_search(param_ranges, max_combinations)

    def _create_grid_search(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Повний перебір з обмеженням"""
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)

        logger.info(f"Загальна кількість комбінацій: {total_combinations:,}")

        if total_combinations <= max_combinations:
            # Повний перебір
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())

            valid_configs = []
            for i, combination in enumerate(product(*param_values)):
                config = dict(zip(param_names, combination))
                config['strategy_name'] = f'grid_{i}'

                is_valid, _ = self._validate_config_logic(config)
                if is_valid:
                    valid_configs.append(config)
                    if len(valid_configs) >= max_combinations:
                        break

            return valid_configs
        else:
            # Випадкове семплування
            return self._create_random_search(param_ranges, max_combinations)

    def _create_random_search(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Випадкове семплування"""
        valid_configs = []
        attempts = 0
        max_attempts = max_combinations * 10

        param_names = list(param_ranges.keys())

        while len(valid_configs) < max_combinations and attempts < max_attempts:
            config = {}
            for param_name in param_names:
                config[param_name] = random.choice(param_ranges[param_name])
            config['strategy_name'] = f'random_{len(valid_configs)}'

            is_valid, _ = self._validate_config_logic(config)
            if is_valid:
                valid_configs.append(config)

            attempts += 1

        logger.info(f"Створено {len(valid_configs)} валідних конфігурацій за {attempts} спроб")
        return valid_configs

    def _create_genetic_search(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Генетичний алгоритм"""
        population_size = min(100, max_combinations // 5)
        generations = max_combinations // population_size

        def create_individual():
            config = {}
            for param_name, values in param_ranges.items():
                config[param_name] = random.choice(values)
            return config

        # Початкова популяція
        population = []
        for i in range(population_size * 2):
            individual = create_individual()
            individual['strategy_name'] = f'genetic_gen0_{len(population)}'
            is_valid, _ = self._validate_config_logic(individual)
            if is_valid:
                population.append(individual)
                if len(population) >= population_size:
                    break

        all_configs = population.copy()

        # Еволюція
        for gen in range(1, min(generations, 5)):
            new_population = []

            # Елітизм
            elite_size = max(1, len(population) // 5)
            new_population.extend(population[:elite_size])

            # Генерація нових особин
            while len(new_population) < population_size:
                if random.random() < 0.7:  # Схрещування
                    if len(population) >= 2:
                        parent1 = random.choice(population)
                        parent2 = random.choice(population)
                        child = create_individual()
                        # Копіюємо деякі гени
                        for param in random.sample(list(parent1.keys()), min(3, len(parent1))):
                            if param in parent2:
                                child[param] = random.choice([parent1[param], parent2[param]])
                        child['strategy_name'] = f'genetic_gen{gen}_cross_{len(new_population)}'
                else:  # Мутація
                    child = create_individual()
                    child['strategy_name'] = f'genetic_gen{gen}_mut_{len(new_population)}'

                is_valid, _ = self._validate_config_logic(child)
                if is_valid:
                    new_population.append(child)

            population = new_population
            all_configs.extend(population)

        return all_configs

    async def save_results_async(self, results: List[OptimizationResult], filename: str):
        """Асинхронне збереження результатів"""
        fieldnames = [
            'rank', 'config_id', 'strategy_name', 'total_trades', 'win_rate',
            'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown', 'sharpe_ratio',
            'avg_hold_time', 'filtered_out', 'best_trade', 'worst_trade', 'execution_time'
        ]

        # Додаємо всі можливі параметри
        all_param_names = set()
        for result in results:
            all_param_names.update(result.parameters.keys())

        fieldnames.extend(sorted(all_param_names))

        async with aiofiles.open(filename, 'w', encoding='utf-8', newline='') as f:
            # Записуємо заголовок
            await f.write(';'.join(fieldnames) + '\n')

            # Записуємо результати
            for rank, result in enumerate(results, 1):
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
                    'execution_time': round(result.execution_time, 3)
                }

                # Додаємо параметри
                for param_name in all_param_names:
                    row[param_name] = result.parameters.get(param_name, '')

                # Формуємо рядок
                row_values = []
                for field in fieldnames:
                    value = row.get(field, '')
                    if isinstance(value, list):
                        value = str(value)
                    row_values.append(str(value))

                await f.write(';'.join(row_values) + '\n')

    def print_top_results(self, results: List[OptimizationResult], top_n: int = 10):
        """Вивід топ результатів"""
        print(f"\n{'=' * 80}")
        print(f"ТОП-{top_n} НАЙКРАЩИХ КОНФІГУРАЦІЙ")
        print(f"{'=' * 80}")

        for i, result in enumerate(results[:top_n], 1):
            params = result.parameters
            print(f"\n#{i} - {result.config_id}")
            print(f"Угод: {result.total_trades} | Win Rate: {result.win_rate:.1f}% | "
                  f"Avg PnL: {result.avg_pnl:.3f}% | Total PnL: {result.total_pnl:.2f}%")
            print(f"Profit Factor: {result.profit_factor:.2f} | Max DD: {result.max_drawdown:.2f}% | "
                  f"Sharpe: {result.sharpe_ratio:.3f}")
            print(f"Час виконання: {result.execution_time:.3f}с | Відфільтровано: {result.filtered_out}")

            # Топ параметри
            important_params = ['stop_loss', 'take_profit', 'use_trailing_stop',
                                'trailing_stop_activation', 'trailing_stop_distance',
                                'long_exit_zone', 'short_exit_zone', 'secondary_timeframes']

            param_str = []
            for param in important_params:
                if param in params:
                    param_str.append(f"{param}={params[param]}")

            if param_str:
                print(f"Ключові параметри: {' | '.join(param_str[:4])}")  # Показуємо перші 4

    def save_best_config_to_file(self, best_result: OptimizationResult, filename: str = None):
        """Зберігає найкращу конфігурацію у файл"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'best_config_{timestamp}.json'

        # Застосовуємо найкращі параметри до базової конфігурації
        best_config = self.apply_config_to_base(best_result.parameters)

        # Додаємо метадані
        best_config['optimization_metadata'] = {
            'config_id': best_result.config_id,
            'strategy_name': best_result.strategy_name,
            'performance_metrics': {
                'avg_pnl': best_result.avg_pnl,
                'win_rate': best_result.win_rate,
                'total_trades': best_result.total_trades,
                'profit_factor': best_result.profit_factor,
                'max_drawdown': best_result.max_drawdown,
                'sharpe_ratio': best_result.sharpe_ratio
            },
            'optimization_date': datetime.now().isoformat(),
            'execution_time': best_result.execution_time
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Найкращу конфігурацію збережено у файл: {filename}")
        return filename

    def analyze_parameter_impact(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """Аналіз впливу параметрів на результат"""
        if not results:
            return {}

        # Збираємо всі параметри
        all_params = set()
        for result in results:
            all_params.update(result.parameters.keys())

        param_analysis = {}

        for param in all_params:
            # Групуємо результати по значенням параметра
            groups = {}
            for result in results:
                if param in result.parameters:
                    value = str(result.parameters[param])
                    if value not in groups:
                        groups[value] = []
                    groups[value].append(result.avg_pnl)

            if len(groups) > 1:  # Має сенс аналізувати тільки якщо є різні значення
                # Обчислюємо статистику для кожного значення
                value_stats = {}
                for value, pnls in groups.items():
                    if pnls:
                        value_stats[value] = {
                            'avg_pnl': np.mean(pnls),
                            'std_pnl': np.std(pnls),
                            'count': len(pnls),
                            'max_pnl': np.max(pnls),
                            'min_pnl': np.min(pnls)
                        }

                # Знаходимо найкраще і найгірше значення
                if value_stats:
                    best_value = max(value_stats.keys(), key=lambda x: value_stats[x]['avg_pnl'])
                    worst_value = min(value_stats.keys(), key=lambda x: value_stats[x]['avg_pnl'])

                    param_analysis[param] = {
                        'impact_range': value_stats[best_value]['avg_pnl'] - value_stats[worst_value]['avg_pnl'],
                        'best_value': best_value,
                        'worst_value': worst_value,
                        'best_avg_pnl': value_stats[best_value]['avg_pnl'],
                        'worst_avg_pnl': value_stats[worst_value]['avg_pnl'],
                        'value_stats': value_stats
                    }

        # Сортуємо параметри за впливом
        sorted_params = sorted(param_analysis.items(),
                               key=lambda x: x[1]['impact_range'], reverse=True)

        return {
            'parameter_impact': dict(sorted_params),
            'most_impactful': sorted_params[0] if sorted_params else None,
            'analysis_summary': {
                'total_parameters_analyzed': len(param_analysis),
                'total_configurations': len(results),
                'avg_performance_range': np.mean([p[1]['impact_range'] for p in sorted_params]) if sorted_params else 0
            }
        }

    def print_parameter_analysis(self, analysis: Dict[str, Any], top_n: int = 5):
        """Виводить аналіз впливу параметрів"""
        if not analysis or 'parameter_impact' not in analysis:
            print("Недостатньо даних для аналізу параметрів")
            return

        print(f"\n{'=' * 80}")
        print("АНАЛІЗ ВПЛИВУ ПАРАМЕТРІВ")
        print(f"{'=' * 80}")

        param_impact = analysis['parameter_impact']
        summary = analysis['analysis_summary']

        print(f"Проаналізовано {summary['total_parameters_analyzed']} параметрів")
        print(f"на основі {summary['total_configurations']} конфігурацій")
        print(f"Середній діапазон впливу: {summary['avg_performance_range']:.3f}%")

        print(f"\nТОП-{min(top_n, len(param_impact))} НАЙВПЛИВОВІШИХ ПАРАМЕТРІВ:")
        print("-" * 80)

        for i, (param, data) in enumerate(list(param_impact.items())[:top_n], 1):
            print(f"\n#{i}. {param}")
            print(f"   Діапазон впливу: {data['impact_range']:.3f}%")
            print(f"   Найкраще значення: {data['best_value']} (PnL: {data['best_avg_pnl']:.3f}%)")
            print(f"   Найгірше значення: {data['worst_value']} (PnL: {data['worst_avg_pnl']:.3f}%)")

            # Показуємо топ-3 значення для цього параметра
            value_stats = data['value_stats']
            sorted_values = sorted(value_stats.items(),
                                   key=lambda x: x[1]['avg_pnl'], reverse=True)[:3]

            print(f"   Топ значення:")
            for j, (value, stats) in enumerate(sorted_values, 1):
                print(f"     {j}. {value}: {stats['avg_pnl']:.3f}% (тестів: {stats['count']})")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Отримання статистики виконання"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        self.perf_stats['memory_usage_mb'] = memory_mb

        return {
            'total_configs_tested': self.perf_stats['total_configs'],
            'successful_tests': self.perf_stats['successful_tests'],
            'failed_tests': self.perf_stats['failed_tests'],
            'success_rate': (self.perf_stats['successful_tests'] / max(1, self.perf_stats['total_configs'])) * 100,
            'avg_execution_time': self.perf_stats['avg_execution_time'],
            'memory_usage_mb': memory_mb,
            'cpu_percent': psutil.cpu_percent(),
        }

    async def optimize_async(self, input_csv: str, max_configs: int = 500,
                             strategy_type: str = 'random', mode: str = 'balanced',
                             output_file: str = 'optimization_results.csv',
                             top_n_save: int = 50) -> List[OptimizationResult]:
        """Головний метод асинхронної оптимізації"""

        logger.info(f"Початок оптимізації:")
        logger.info(f"  - Файл сигналів: {input_csv}")
        logger.info(f"  - Максимум конфігурацій: {max_configs}")
        logger.info(f"  - Стратегія: {strategy_type}")
        logger.info(f"  - Режим: {mode}")
        logger.info(f"  - Файл результатів: {output_file}")

        start_time = time.time()

        # Завантажуємо сигнали
        try:
            analyzer = SignalAnalyzer(self.base_config)
            signals = analyzer.read_signals_csv(input_csv)
            logger.info(f"Завантажено {len(signals)} сигналів")

            if not signals:
                logger.error("Немає валідних сигналів для оптимізації")
                return []

        except Exception as e:
            logger.error(f"Помилка завантаження сигналів: {e}")
            return []

        # Генеруємо конфігурації
        configs = self.create_config_combinations(max_configs, strategy_type, mode)
        logger.info(f"Створено {len(configs)} конфігурацій для тестування")

        if not configs:
            logger.error("Немає валідних конфігурацій для тестування")
            return []

        # Оновлюємо статистику
        self.perf_stats['total_configs'] = len(configs)

        # Створюємо семафор для обмеження одночасних завдань
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Створюємо завдання для тестування
        tasks = []
        for i, config in enumerate(configs):
            config_id = f"{strategy_type}_{mode}_{i:04d}"
            task = asyncio.create_task(
                self.test_single_config(config, config_id, signals, semaphore)
            )
            tasks.append(task)

        # Виконуємо тестування з прогрес-баром
        logger.info(f"Початок тестування {len(tasks)} конфігурацій...")

        results = []
        completed = 0
        batch_size = 50  # Оброблюємо батчами для контролю пам'яті

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Помилка в батчі: {result}")
                    self.perf_stats['failed_tests'] += 1
                elif result is not None:
                    results.append(result)
                    self.perf_stats['successful_tests'] += 1
                else:
                    self.perf_stats['failed_tests'] += 1

                completed += 1

            # Показуємо прогрес
            progress = (completed / len(tasks)) * 100
            logger.info(f"Прогрес: {progress:.1f}% ({completed}/{len(tasks)})")

            # Очищуємо пам'ять
            if i % (batch_size * 2) == 0:
                gc.collect()

        # Сортуємо результати за середнім PnL
        results.sort(key=lambda x: x.avg_pnl, reverse=True)

        # Обчислюємо статистику виконання
        total_time = time.time() - start_time
        self.perf_stats['avg_execution_time'] = total_time / len(results) if results else 0

        logger.info(f"Оптимізація завершена за {total_time:.2f} секунд")
        logger.info(f"Успішно протестовано: {len(results)} конфігурацій")

        # Зберігаємо результати
        if results:
            top_results = results[:top_n_save]
            await self.save_results_async(top_results, output_file)
            logger.info(f"Збережено топ-{len(top_results)} результатів у {output_file}")

            # Виводимо топ результати
            self.print_top_results(results, min(10, len(results)))

            # Виводимо статистику
            perf_summary = self.get_performance_summary()
            logger.info("Статистика виконання:")
            for key, value in perf_summary.items():
                logger.info(f"  {key}: {value}")

        return results

    async def run_multi_strategy_optimization(self, input_csv: str,
                                              strategies: List[str] = None,
                                              modes: List[str] = None,
                                              configs_per_strategy: int = 200) -> Dict[str, List[OptimizationResult]]:
        """Запуск оптимізації з кількома стратегіями"""

        if strategies is None:
            strategies = ['random', 'genetic', 'grid_search']
        if modes is None:
            modes = ['fast', 'balanced']

        all_results = {}

        for strategy in strategies:
            for mode in modes:
                key = f"{strategy}_{mode}"
                logger.info(f"Запуск оптимізації: {key}")

                output_file = f"optimization_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                results = await self.optimize_async(
                    input_csv=input_csv,
                    max_configs=configs_per_strategy,
                    strategy_type=strategy,
                    mode=mode,
                    output_file=output_file,
                    top_n_save=50
                )

                all_results[key] = results

                # Пауза між стратегіями
                await asyncio.sleep(2)
                gc.collect()

        # Зберігаємо зведений звіт
        await self.save_combined_report(all_results)

        return all_results

    async def save_combined_report(self, all_results: Dict[str, List[OptimizationResult]]):
        """Збереження зведеного звіту"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"combined_optimization_report_{timestamp}.csv"

        # Об'єднуємо всі результати
        combined_results = []
        for strategy_mode, results in all_results.items():
            for result in results[:10]:  # Топ-10 з кожної стратегії
                result.strategy_name = f"{strategy_mode}_{result.strategy_name}"
                combined_results.append(result)

        # Сортуємо за avg_pnl
        combined_results.sort(key=lambda x: x.avg_pnl, reverse=True)

        await self.save_results_async(combined_results, filename)
        logger.info(f"Зведений звіт збережено: {filename}")

        # Виводимо найкращі результати по кожній стратегії
        print(f"\n{'=' * 100}")
        print("ЗВЕДЕНИЙ ЗВІТ ПО СТРАТЕГІЯМ")
        print(f"{'=' * 100}")

        for strategy_mode, results in all_results.items():
            if results:
                best = results[0]
                print(f"\n{strategy_mode.upper()}:")
                print(f"  Найкращий результат: Avg PnL={best.avg_pnl:.3f}%, "
                      f"Win Rate={best.win_rate:.1f}%, Угод={best.total_trades}")
                print(f"  Всього протестовано: {len(results)} конфігурацій")


async def main():
    """Головна функція запуску оптимізатора"""
    parser = argparse.ArgumentParser(description='Асинхронний оптимізатор для Enhanced Signal Analyzer')

    parser.add_argument('input_csv', help='CSV файл з сигналами')
    parser.add_argument('-c', '--config', default='updated_analyzer_config.json',
                        help='Базовий конфіг файл')
    parser.add_argument('-n', '--max-configs', type=int, default=500,
                        help='Максимальна кількість конфігурацій для тестування')
    parser.add_argument('-s', '--strategy', default='random',
                        choices=['random', 'genetic', 'grid_search'],
                        help='Стратегія генерації конфігурацій')
    parser.add_argument('-m', '--mode', default='balanced',
                        choices=['fast', 'balanced', 'thorough', 'ultra_fast'],
                        help='Режим оптимізації (швидкість vs точність)')
    parser.add_argument('-o', '--output', default='optimization_results.csv',
                        help='Файл для збереження результатів')
    parser.add_argument('--concurrent', type=int, default=20,
                        help='Максимальна кількість одночасних завдань')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Кількість топ результатів для збереження')
    parser.add_argument('--multi-strategy', action='store_true',
                        help='Запустити оптимізацію з кількома стратегіями')

    args = parser.parse_args()

    # Перевіряємо існування файлів
    if not os.path.exists(args.input_csv):
        logger.error(f"Файл сигналів не знайдено: {args.input_csv}")
        return

    if not os.path.exists(args.config):
        logger.warning(f"Конфіг файл не знайдено: {args.config}, буде створено стандартний")

    # Створюємо оптимізатор
    optimizer = AsyncConfigOptimizer(
        base_config_file=args.config,
        max_concurrent=args.concurrent,
        use_fast_mode=args.mode in ['fast', 'ultra_fast']
    )

    logger.info("Початок оптимізації Enhanced Signal Analyzer")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Доступна RAM: {psutil.virtual_memory().total // (1024 ** 3)} GB")

    try:
        if args.multi_strategy:
            # Багатостратегійна оптимізація
            strategies = ['random', 'genetic']
            modes = ['fast', 'balanced'] if args.mode != 'ultra_fast' else ['ultra_fast']

            configs_per_strategy = max(100, args.max_configs // (len(strategies) * len(modes)))

            results = await optimizer.run_multi_strategy_optimization(
                input_csv=args.input_csv,
                strategies=strategies,
                modes=modes,
                configs_per_strategy=configs_per_strategy
            )

            # Виводимо загальну статистику
            total_configs = sum(len(r) for r in results.values())
            logger.info(f"Загалом протестовано {total_configs} конфігурацій")

        else:
            # Одностратегійна оптимізація
            results = await optimizer.optimize_async(
                input_csv=args.input_csv,
                max_configs=args.max_configs,
                strategy_type=args.strategy,
                mode=args.mode,
                output_file=args.output,
                top_n_save=args.top_n
            )

            if results:
                logger.info("Оптимізація успішно завершена!")
                logger.info(f"Найкращий результат: Avg PnL = {results[0].avg_pnl:.3f}%")
            else:
                logger.error("Оптимізація не дала результатів")

    except KeyboardInterrupt:
        logger.info("Оптимізацію перервано користувачем")
    except Exception as e:
        logger.error(f"Критична помилка оптимізації: {e}", exc_info=True)
    finally:
        # Очищення пам'яті
        gc.collect()

        # Фінальна статистика
        final_stats = optimizer.get_performance_summary()
        logger.info("Фінальна статистика:")
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    # Налаштування asyncio для Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())