#!/usr/bin/env python3
"""
Розширений модульний аналізатор з генетичним алгоритмом оптимізації
Включає всі режими оптимізації, повні діапазони параметрів та genetic режим
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import concurrent.futures
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
from analyzer_optimizer_data_management import Signal, TradeResult, DataProvider
from analyzer_optimizer_data_management import load_signals_from_csv, save_results_to_csv, save_optimization_results, \
                                                save_top10, print_top_results, save_best_config


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
        max_vol = vol_config.get('max_volatility', 3.0)

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
            #'rsi_period': [10, 12, 14, 16, 18, 20, 22],
            #'rsi_sma_period': [10, 12, 14, 16, 18, 20, 22],
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
            'long_enter_zone_mid_tf': [25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50],
            'sma_change_periods': [3, 4, 5, 6, 7, 8, 10],
            'sma_change_threshold': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            'rsi_extreme_threshold': [5, 8, 10, 12, 15, 18, 20],
            'cross_lookback_periods': [4, 5, 6, 7, 8, 10, 12],

            # Volatility filter
            'volatility_enabled': [True, False],
            'min_volatility': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            'max_volatility': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0],
            'volatility_period': [10, 14, 20, 25, 30, 40, 50],
            'volatility_method': ['atr_percentage', 'price_range', 'close_variance'],

            # Data management
            'buffer_hours_before': [48, 60, 72, 84, 96, 120, 144, 168],
            'buffer_hours_after': [180, 200, 220, 240, 260, 280, 300, 360]
        }

        if mode == 'ultra_fast':
            # Скорочуємо вдвічі
            for key in base_ranges:
                if isinstance(base_ranges[key][0], (int, float)):
                    base_ranges[key] = base_ranges[key][::2]

        elif mode == 'fast':
            # Скоротити на 30%
            for key in base_ranges:
                if len(base_ranges[key]) > 3 and isinstance(base_ranges[key][0], (int, float)):
                    to_keep = int(len(base_ranges[key]) * 0.7)
                    base_ranges[key] = base_ranges[key][:to_keep]

        elif mode == 'comprehensive':
            # Додаємо більше значень
            for key in list(base_ranges.keys()):
                if isinstance(base_ranges[key][0], float):
                    new_values = [v * 1.1 for v in base_ranges[key]] + [v * 0.9 for v in base_ranges[key]]
                    base_ranges[key].extend(new_values)
                    base_ranges[key] = sorted(set(base_ranges[key]))
                elif isinstance(base_ranges[key][0], int):
                    new_values = [v + 1 for v in base_ranges[key]] + [v - 1 for v in base_ranges[key] if v > 1]
                    base_ranges[key].extend(new_values)
                    base_ranges[key] = sorted(set(base_ranges[key]))

        return base_ranges


class ConfigOptimizer:
    """Оптимізатор конфігурацій з підтримкою різних режимів"""

    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.data_provider = DataProvider(base_config)
        self._shared_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._signals_cache: List[Signal] = []

    async def optimize(self, signals: List[Signal], mode: str = 'balanced',
                       max_configs: int = 500, generations: int = 10) -> List[Dict]:
        """Запуск оптимізації в заданому режимі"""
        logging.info(f"Початок оптимізації в режимі {mode}")

        ranges = ParameterRanges.get_parameter_ranges(mode)

        if mode == 'genetic':
            results = await self._genetic_optimization(signals, ranges, generations, max_configs)
        else:
            results = await self._grid_optimization(signals, ranges, max_configs, mode)

        # Сортуємо за composite score
        for res in results:
            metrics = res['metrics']
            res['composite_score'] = (
                metrics['profit_factor'] * 0.4 +
                metrics['win_rate'] / 100 * 0.2 +
                metrics['sharpe_ratio'] * 0.2 -
                metrics['max_drawdown'] / 100 * 0.1 +
                metrics['total_trades'] / 1000 * 0.1
            )

        results.sort(key=lambda x: x['composite_score'], reverse=True)

        logging.info(f"Оптимізація завершена. Тестовано {len(results)} конфігурацій")
        return results

    async def _grid_optimization(self, signals: List[Signal], ranges: Dict,
                                 max_configs: int, mode: str) -> List[Dict]:
        """Grid search оптимізація"""
        param_combinations = list(product(*ranges.values()))
        param_names = list(ranges.keys())

        if mode in ['ultra_fast', 'fast']:
            random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_configs]
        else:
            param_combinations = random.sample(param_combinations, min(max_configs, len(param_combinations)))

        configs = []
        for i, combo in enumerate(param_combinations):
            params = dict(zip(param_names, combo))
            configs.append((f"grid_{i+1}", params))

        return await self._test_configs(signals, configs)

    async def _genetic_optimization(self, signals: List[Signal], ranges: Dict,
                                    generations: int, population_size: int) -> List[Dict]:
        """Генетична оптимізація"""
        param_names = list(ranges.keys())
        all_results = []

        # Ініціалізація популяції
        population = []
        for i in range(population_size):
            individual = {name: random.choice(ranges[name]) for name in param_names}
            population.append((f"gen0_{i+1}", individual))

        for gen in range(generations):
            logging.info(f"Покоління {gen+1}/{generations}")

            # Тестуємо поточну популяцію
            results = await self._test_configs(signals, population)
            all_results.extend(results)

            # Сортуємо за composite score
            results.sort(key=lambda x: x['composite_score'], reverse=True)

            # Еліта - топ 20%
            elite_size = max(1, population_size // 5)
            elite = results[:elite_size]

            # Нова популяція
            new_population = []
            for i in range(population_size - elite_size):
                # Турнірний селект
                parent1 = random.choice(results[:population_size//2])['parameters']
                parent2 = random.choice(results[:population_size//2])['parameters']

                # Кроссовер
                child = {}
                for name in param_names:
                    child[name] = random.choice([parent1[name], parent2[name]])

                # Мутація 10%
                if random.random() < 0.1:
                    mutate_key = random.choice(param_names)
                    child[mutate_key] = random.choice(ranges[mutate_key])

                new_population.append((f"gen{gen+1}_{i+1}", child))

            # Додаємо еліту в нову популяцію
            for i, elite_ind in enumerate(elite):
                new_population.append((f"gen{gen+1}_elite{i+1}", elite_ind['parameters']))

            population = new_population

        return all_results

    async def _test_configs(self, signals: List[Signal], configs: List[Tuple[str, Dict]]) -> List[Dict]:
        """Паралельне тестування конфігурацій з спільним кешем"""

        # Підготовка спільного кешу якщо потрібно
        if not self._shared_data_cache:
            unique_pairs = list(set(s.pair for s in signals))
            self._shared_data_cache = await self.data_provider.fetch_data_batch(
                unique_pairs, signals, self.base_config
            )
            self._signals_cache = signals

        logging.info(f"Кеш підготовлено для {len(self._shared_data_cache)} пар")

        # Створюємо семафор для контролю паралелізму
        semaphore = asyncio.Semaphore(20)

        # Запускаємо паралельні задачі
        tasks = []
        for config_id, params in configs:
            task = self._test_config_with_shared_cache(params, config_id, semaphore)
            tasks.append(task)

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = [res for res in completed if not isinstance(res, Exception) and res is not None]

        return results

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
                trade_percentage = metrics['total_trades'] / len(self._signals_cache) if self._signals_cache else 0
                if trade_percentage < 0.005:
                    logging.info(f"Конфіг {config_id} пропущено: лише {trade_percentage*100:.2f}% трейдів пройшли фільтрацію")
                    return None

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
        adjusted_pnls = [pnl - 0.2 for pnl in pnls]
        hold_times = [r.hold_time for r in traded_results]

        total_trades = len(traded_results)
        profitable_trades = sum(1 for pnl in adjusted_pnls if pnl > 0)
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = sum(adjusted_pnls) / len(adjusted_pnls) if adjusted_pnls else 0
        total_pnl = sum(adjusted_pnls)

        # Drawdown розрахунок
        cumulative = np.cumsum(adjusted_pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = max(drawdown) if len(drawdown) > 0 else 0

        # Profit factor
        profits = [pnl for pnl in adjusted_pnls if pnl > 0]
        losses = [abs(pnl) for pnl in adjusted_pnls if pnl <= 0]
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # Sharpe ratio
        sharpe_ratio = avg_pnl / np.std(adjusted_pnls) if len(adjusted_pnls) > 1 and np.std(adjusted_pnls) > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'filtered_out': len(results) - total_trades,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': max(adjusted_pnls) if adjusted_pnls else 0.0,
            'worst_trade': min(adjusted_pnls) if adjusted_pnls else 0.0,
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

    # Збереження топ-10 конфігурацій у папку
    output_basename = os.path.splitext(os.path.basename(output_file))[0]
    top_configs_dir = f"{output_basename}_top_configs"
    os.makedirs(top_configs_dir, exist_ok=True)

    for i, result in enumerate(results[:10], 1):
        config = optimizer._apply_config_params(result['parameters'])
        config['optimization_metadata'] = {
            'rank': i,
            'config_id': result['config_id'],
            'performance_metrics': result['metrics'],
            'optimization_date': datetime.now().isoformat(),
            'optimizer_version': "enhanced_modular_with_genetic"
        }
        config_filename = os.path.join(top_configs_dir, f"config_rank_{i}_{result['config_id']}.json")
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"Збережено конфігурацію топ-{i}: {config_filename}")
        print(f"Збережено конфігурацію топ-{i}: {config_filename}")
    # Виводимо топ результати
    print_top_results(results[:10])


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