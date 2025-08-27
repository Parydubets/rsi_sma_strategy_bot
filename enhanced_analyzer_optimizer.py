#!/usr/bin/env python3
"""
Оптимізатор налаштувань для торгового аналізатора сигналів
Перебирає різні комбінації параметрів для максимізації PnL
"""

import asyncio
import json
import pandas as pd
import numpy as np
from itertools import product
import logging
from datetime import datetime
import argparse
import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import copy

# Імпортуємо наш аналізатор
from enhanced_analyzer import TrendSignalAnalyzer

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Результат одного тесту оптимізації"""
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


class ConfigOptimizer:
    """Оптимізатор конфігурацій аналізатора"""

    def __init__(self, base_config_file: str = 'enhanced_analyzer_config.json'):
        self.base_config = TrendSignalAnalyzer.load_config(base_config_file)
        self.results = []

    def generate_parameter_ranges(self) -> Dict[str, List]:
        """Генерація діапазонів параметрів для оптимізації"""
        return {
            # Торгові параметри
            'stop_loss': [0.012, 0.015, 0.018, 0.022, 0.025, 0.030, 0.035],
            'take_profit': [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070],
            'max_hold_hours': [6, 8, 10, 12, 16, 20, 24],
            'use_trailing_stop': [True, False],

            # RSI параметри
            'long_exit_zone': [60, 62, 65, 68, 70, 72, 75],
            'short_exit_zone': [25, 28, 30, 32, 35, 38, 40],
            'long_extreme_exit': [75, 78, 80, 82, 85],
            'short_extreme_exit': [15, 18, 20, 22, 25],

            # Фільтри
            'use_filters': [True, False],
            'min_filters_required': [2, 3, 4, 5],
            'use_trend_filter': [True, False],
            'use_adx_filter': [True, False],
            'use_volatility_filter': [True, False],
            'use_volume_filter': [True, False],
            'use_momentum_filter': [True, False],

            # Пороги фільтрів
            'min_adx': [12, 15, 18, 20, 22, 25],
            'min_volatility': [0.1, 0.12, 0.15, 0.18, 0.2, 0.25],
            'max_volatility': [2.0, 2.2, 2.5, 3.0, 3.5, 4.0],
            'min_volume_ratio': [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],

            # Вага таймфреймів
            'weight_5m': [0.3, 0.35, 0.4, 0.45, 0.5],
            'weight_15m': [0.25, 0.3, 0.35, 0.4],
            'weight_1h': [0.15, 0.2, 0.25, 0.3],

            # Технічні індикатори для 5m
            'rsi_period_5m': [12, 14, 16, 18],
            'sma_fast_5m': [15, 20, 25],
            'sma_slow_5m': [40, 50, 60],
            'adx_period_5m': [12, 14, 16],
            'ema_fast_5m': [10, 12, 15],
            'ema_slow_5m': [21, 26, 30],
            'atr_period_5m': [12, 14, 16],
            'bb_period_5m': [18, 20, 22],
        }

    def create_config_combinations(self, max_combinations: int = 1000) -> List[Dict[str, Any]]:
        """Створення комбінацій конфігурацій"""
        param_ranges = self.generate_parameter_ranges()

        # Створюємо стратегії тестування
        strategies = self._create_test_strategies(param_ranges, max_combinations)

        logger.info(f"Створено {len(strategies)} конфігурацій для тестування")
        return strategies

    def _create_test_strategies(self, param_ranges: Dict[str, List], max_combinations: int) -> List[Dict[str, Any]]:
        """Створення тестових стратегій"""
        strategies = []

        # 1. Базові стратегії (різні підходи)
        base_strategies = [
            # Консервативна стратегія
            {
                'name': 'conservative',
                'stop_loss': [0.012, 0.015],
                'take_profit': [0.025, 0.030],
                'max_hold_hours': [6, 8],
                'use_trailing_stop': [False],
                'use_filters': [True],
                'min_filters_required': [4, 5],
                'min_adx': [20, 22, 25],
                'min_volatility': [0.12, 0.15],
                'max_volatility': [2.0, 2.2]
            },
            # Агресивна стратегія
            {
                'name': 'aggressive',
                'stop_loss': [0.025, 0.030, 0.035],
                'take_profit': [0.050, 0.060, 0.070],
                'max_hold_hours': [12, 16, 20],
                'use_trailing_stop': [True],
                'use_filters': [False, True],
                'min_filters_required': [2, 3],
                'min_adx': [12, 15],
                'min_volatility': [0.2, 0.25],
                'max_volatility': [3.0, 3.5]
            },
            # Трендова стратегія
            {
                'name': 'trend_following',
                'stop_loss': [0.018, 0.022],
                'take_profit': [0.035, 0.040, 0.045],
                'max_hold_hours': [10, 12, 16],
                'use_trailing_stop': [True],
                'use_trend_filter': [True],
                'use_adx_filter': [True],
                'min_adx': [18, 20],
                'weight_5m': [0.35, 0.4],
                'weight_15m': [0.35, 0.4],
                'weight_1h': [0.2, 0.25]
            },
            # Скальпинговая стратегия
            {
                'name': 'scalping',
                'stop_loss': [0.012, 0.015, 0.018],
                'take_profit': [0.025, 0.030, 0.035],
                'max_hold_hours': [4, 6, 8],
                'use_trailing_stop': [True, False],
                'long_exit_zone': [62, 65, 68],
                'short_exit_zone': [32, 35, 38],
                'use_momentum_filter': [True],
                'weight_5m': [0.45, 0.5]
            },
            # Волатільність стратегія
            {
                'name': 'volatility',
                'stop_loss': [0.020, 0.025, 0.030],
                'take_profit': [0.040, 0.050, 0.060],
                'use_volatility_filter': [True],
                'min_volatility': [0.18, 0.20, 0.25],
                'max_volatility': [2.5, 3.0],
                'use_trailing_stop': [True],
                'max_hold_hours': [8, 12, 16]
            }
        ]

        # Генеруємо комбінації для кожної стратегії
        for strategy in base_strategies:
            strategy_name = strategy.pop('name')

            # Заповнюємо відсутні параметри базовими значеннями
            full_params = {}
            for key, values in param_ranges.items():
                if key in strategy:
                    full_params[key] = strategy[key]
                else:
                    # Беремо середні або базові значення
                    if len(values) == 1:
                        full_params[key] = values
                    elif key in ['use_trailing_stop', 'use_filters', 'use_trend_filter',
                                 'use_adx_filter', 'use_volatility_filter', 'use_volume_filter',
                                 'use_momentum_filter']:
                        full_params[key] = [True]  # По умолчанию включаем фильтры
                    else:
                        # Берем средние значения
                        mid_idx = len(values) // 2
                        full_params[key] = [values[mid_idx]]

            # Создаем комбинации для стратегии
            keys = list(full_params.keys())
            values = list(full_params.values())

            # Ограничиваем количество комбинаций для каждой стратегии
            max_per_strategy = max_combinations // len(base_strategies)
            combinations = list(product(*values))[:max_per_strategy]

            for combo in combinations:
                config = dict(zip(keys, combo))
                config['strategy_name'] = strategy_name
                strategies.append(config)

        # 2. Добавляем случайные комбинации
        remaining = max_combinations - len(strategies)
        if remaining > 0:
            random_strategies = self._generate_random_combinations(param_ranges, remaining)
            strategies.extend(random_strategies)

        return strategies[:max_combinations]

    def _generate_random_combinations(self, param_ranges: Dict[str, List], count: int) -> List[Dict[str, Any]]:
        """Генерация случайных комбинаций параметров"""
        import random

        strategies = []
        for i in range(count):
            config = {}
            for key, values in param_ranges.items():
                config[key] = random.choice(values)
            config['strategy_name'] = f'random_{i}'

            # Проверяем логические ограничения
            if self._validate_config_logic(config):
                strategies.append(config)

        return strategies

    def _validate_config_logic(self, config: Dict[str, Any]) -> bool:
        """Проверка логических ограничений конфигурации"""
        # Проверяем, что веса таймфреймов в сумме дают примерно 1
        total_weight = config.get('weight_5m', 0.4) + config.get('weight_15m', 0.35) + config.get('weight_1h', 0.25)
        if abs(total_weight - 1.0) > 0.1:
            return False

        # TP должен быть больше SL
        if config.get('take_profit', 0.035) <= config.get('stop_loss', 0.018):
            return False

        # Зоны выхода должны быть логичными
        if config.get('long_exit_zone', 65) <= config.get('short_exit_zone', 35):
            return False

        # Минимальная волатильность должна быть меньше максимальной
        if config.get('min_volatility', 0.15) >= config.get('max_volatility', 2.5):
            return False

        return True

    def apply_config_to_base(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Применение тестовой конфигурации к базовой"""
        config = copy.deepcopy(self.base_config)

        # Торговые параметры
        if 'stop_loss' in test_config:
            config['trading_parameters']['stop_loss'] = test_config['stop_loss']
        if 'take_profit' in test_config:
            config['trading_parameters']['take_profit'] = test_config['take_profit']
        if 'max_hold_hours' in test_config:
            config['trading_parameters']['max_hold_hours'] = test_config['max_hold_hours']
        if 'use_trailing_stop' in test_config:
            config['trading_parameters']['use_trailing_stop'] = test_config['use_trailing_stop']

        # RSI параметры
        rsi_params = config['rsi_parameters']
        if 'long_exit_zone' in test_config:
            rsi_params['long_exit_zone'] = test_config['long_exit_zone']
        if 'short_exit_zone' in test_config:
            rsi_params['short_exit_zone'] = test_config['short_exit_zone']
        if 'long_extreme_exit' in test_config:
            rsi_params['long_extreme_exit'] = test_config['long_extreme_exit']
        if 'short_extreme_exit' in test_config:
            rsi_params['short_extreme_exit'] = test_config['short_extreme_exit']

        # Фильтры
        filters = config['filters']
        if 'use_filters' in test_config:
            filters['use_filters'] = test_config['use_filters']
        if 'min_filters_required' in test_config:
            filters['min_filters_required'] = test_config['min_filters_required']
        if 'use_trend_filter' in test_config:
            filters['use_trend_filter'] = test_config['use_trend_filter']
        if 'use_adx_filter' in test_config:
            filters['use_adx_filter'] = test_config['use_adx_filter']
        if 'use_volatility_filter' in test_config:
            filters['use_volatility_filter'] = test_config['use_volatility_filter']
        if 'use_volume_filter' in test_config:
            filters['use_volume_filter'] = test_config['use_volume_filter']
        if 'use_momentum_filter' in test_config:
            filters['use_momentum_filter'] = test_config['use_momentum_filter']

        # Пороги фильтров
        thresholds = config['filter_thresholds']
        if 'min_adx' in test_config:
            thresholds['min_adx'] = test_config['min_adx']
        if 'min_volatility' in test_config:
            thresholds['min_volatility'] = test_config['min_volatility']
        if 'max_volatility' in test_config:
            thresholds['max_volatility'] = test_config['max_volatility']
        if 'min_volume_ratio' in test_config:
            thresholds['min_volume_ratio'] = test_config['min_volume_ratio']

        # Веса таймфреймов
        if any(k in test_config for k in ['weight_5m', 'weight_15m', 'weight_1h']):
            weights = config['timeframe_settings']['trend_weights']
            if 'weight_5m' in test_config:
                weights['5m'] = test_config['weight_5m']
            if 'weight_15m' in test_config:
                weights['15m'] = test_config['weight_15m']
            if 'weight_1h' in test_config:
                weights['1h'] = test_config['weight_1h']

            # Нормализуем веса
            total = sum(weights.values())
            for k in weights:
                weights[k] = weights[k] / total

        # Технические индикаторы для 5m
        indicators_5m = config['technical_indicators']['5m']
        if 'rsi_period_5m' in test_config:
            indicators_5m['rsi_period'] = test_config['rsi_period_5m']
        if 'sma_fast_5m' in test_config:
            indicators_5m['sma_fast'] = test_config['sma_fast_5m']
        if 'sma_slow_5m' in test_config:
            indicators_5m['sma_slow'] = test_config['sma_slow_5m']
        if 'adx_period_5m' in test_config:
            indicators_5m['adx_period'] = test_config['adx_period_5m']
        if 'ema_fast_5m' in test_config:
            indicators_5m['ema_fast'] = test_config['ema_fast_5m']
        if 'ema_slow_5m' in test_config:
            indicators_5m['ema_slow'] = test_config['ema_slow_5m']
        if 'atr_period_5m' in test_config:
            indicators_5m['atr_period'] = test_config['atr_period_5m']
        if 'bb_period_5m' in test_config:
            indicators_5m['bb_period'] = test_config['bb_period_5m']

        return config

    async def test_configuration(self, test_config: Dict[str, Any], signals: List[Dict],
                                 config_id: str) -> OptimizationResult:
        """Тестирование одной конфигурации"""
        try:
            # Применяем конфигурацию
            full_config = self.apply_config_to_base(test_config)

            # Создаем анализатор и тестируем
            analyzer = TrendSignalAnalyzer()
            results = await analyzer.analyze_signals(signals, full_config)

            if not results:
                logger.warning(f"Нет результатов для конфигурации {config_id}")
                return None

            # Анализируем результаты
            traded_results = [r for r in results if not r.get('filtered_out', False)]

            if not traded_results:
                logger.warning(f"Все сигналы отфильтрованы для конфигурации {config_id}")
                return OptimizationResult(
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
                    filtered_out=len(results),
                    best_trade=0.0,
                    worst_trade=0.0
                )

            # Расчет метрик
            pnl_values = [float(r.get('pnl_percent', 0)) for r in traded_results]
            profitable = [pnl for pnl in pnl_values if pnl > 0]
            losses = [pnl for pnl in pnl_values if pnl <= 0]

            total_trades = len(traded_results)
            profitable_trades = len(profitable)
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = np.mean(pnl_values) if pnl_values else 0
            total_pnl = sum(pnl_values)
            avg_hold_time = np.mean([float(r.get('hold_time_hours', 0)) for r in traded_results])

            # Максимальная просадка
            cumulative_pnl = np.cumsum(pnl_values)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            # Профит-фактор
            total_profit = sum(profitable) if profitable else 0
            total_loss = abs(sum(losses)) if losses else 1  # Избегаем деления на 0
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

            best_trade = max(pnl_values) if pnl_values else 0
            worst_trade = min(pnl_values) if pnl_values else 0

            filtered_out = len(results) - len(traded_results)

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
                worst_trade=worst_trade
            )

            logger.debug(f"Конфигурация {config_id}: {total_trades} торгов, "
                         f"Win Rate: {win_rate:.1f}%, Avg PnL: {avg_pnl:.3f}%")

            return result

        except Exception as e:
            logger.error(f"Ошибка тестирования конфигурации {config_id}: {e}")
            return None

    async def optimize(self, signals_file: str, max_configs: int = 500,
                       output_file: str = 'optimization_results.csv') -> List[OptimizationResult]:
        """Основной метод оптимизации"""
        logger.info(f"Начало оптимизации с файлом сигналов: {signals_file}")
        logger.info(f"Максимальное количество конфигураций: {max_configs}")

        # Загружаем сигналы
        analyzer = TrendSignalAnalyzer()
        signals = analyzer.read_signals_csv(signals_file)

        if not signals:
            logger.error("Не удалось загрузить сигналы")
            return []

        logger.info(f"Загружено {len(signals)} сигналов")

        # Генерируем конфигурации
        test_configs = self.create_config_combinations(max_configs)
        logger.info(f"Создано {len(test_configs)} тестовых конфигураций")

        # Тестируем конфигурации
        results = []
        completed = 0

        print(f"\nНачинаем тестирование {len(test_configs)} конфигураций...")
        print("Прогресс: ", end="", flush=True)

        for i, test_config in enumerate(test_configs):
            config_id = f"config_{i:04d}_{test_config.get('strategy_name', 'unknown')}"

            try:
                result = await self.test_configuration(test_config, signals, config_id)
                if result:
                    results.append(result)

                completed += 1

                # Показываем прогресс
                if completed % 10 == 0 or completed == len(test_configs):
                    progress = completed / len(test_configs) * 100
                    print(f"{progress:.1f}% ", end="", flush=True)

                    # Показываем лучший результат пока что
                    if results:
                        best_result = max(results, key=lambda x: x.avg_pnl)
                        print(f"(Лучший: {best_result.avg_pnl:.3f}%) ", end="", flush=True)

                # Небольшая задержка между тестами
                if i % 50 == 0:
                    await asyncio.sleep(0.05)

            except Exception as e:
                logger.error(f"Ошибка тестирования конфигурации {config_id}: {e}")
                continue

        print(f"\n\nТестирование завершено! Протестировано {completed} конфигураций")

        if not results:
            logger.error("Нет результатов оптимизации")
            return []

        # Сортируем результаты по среднему PnL
        results.sort(key=lambda x: x.avg_pnl, reverse=True)

        # Сохраняем результаты
        self.save_optimization_results(results, output_file)

        # Выводим топ результаты
        self.print_top_results(results, top_n=10)

        logger.info(f"Оптимизация завершена. Результаты сохранены в {output_file}")
        return results

    def save_optimization_results(self, results: List[OptimizationResult], filename: str):
        """Сохранение результатов оптимизации"""
        logger.info(f"Сохранение {len(results)} результатов в {filename}")

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'rank', 'config_id', 'strategy_name', 'total_trades', 'win_rate',
                'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown',
                'avg_hold_time', 'filtered_out', 'best_trade', 'worst_trade',
                'stop_loss', 'take_profit', 'max_hold_hours', 'use_trailing_stop',
                'long_exit_zone', 'short_exit_zone', 'use_filters', 'min_filters_required',
                'min_adx', 'min_volatility', 'max_volatility', 'min_volume_ratio'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rank, result in enumerate(results, 1):
                params = result.parameters
                row = {
                    'rank': rank,
                    'config_id': result.config_id,
                    'strategy_name': params.get('strategy_name', 'unknown'),
                    'total_trades': result.total_trades,
                    'win_rate': round(result.win_rate, 2),
                    'avg_pnl': round(result.avg_pnl, 4),
                    'total_pnl': round(result.total_pnl, 2),
                    'profit_factor': round(result.profit_factor, 2),
                    'max_drawdown': round(result.max_drawdown, 2),
                    'avg_hold_time': round(result.avg_hold_time, 2),
                    'filtered_out': result.filtered_out,
                    'best_trade': round(result.best_trade, 2),
                    'worst_trade': round(result.worst_trade, 2),
                    'stop_loss': params.get('stop_loss', ''),
                    'take_profit': params.get('take_profit', ''),
                    'max_hold_hours': params.get('max_hold_hours', ''),
                    'use_trailing_stop': params.get('use_trailing_stop', ''),
                    'long_exit_zone': params.get('long_exit_zone', ''),
                    'short_exit_zone': params.get('short_exit_zone', ''),
                    'use_filters': params.get('use_filters', ''),
                    'min_filters_required': params.get('min_filters_required', ''),
                    'min_adx': params.get('min_adx', ''),
                    'min_volatility': params.get('min_volatility', ''),
                    'max_volatility': params.get('max_volatility', ''),
                    'min_volume_ratio': params.get('min_volume_ratio', ''),
                }
                writer.writerow(row)

        # Сохраняем также полные конфигурации топ-10
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

        logger.info(f"Топ-10 полных конфигураций сохранено в {config_filename}")

    def print_top_results(self, results: List[OptimizationResult], top_n: int = 10):
        """Вывод топ результатов"""
        print(f"\n{'=' * 80}")
        print(f"ТОП-{top_n} НАЙКРАЩИХ КОНФІГУРАЦІЙ")
        print(f"{'=' * 80}")

        for i, result in enumerate(results[:top_n], 1):
            params = result.parameters
            print(f"\n#{i} - {result.config_id}")
            print(f"Стратегія: {params.get('strategy_name', 'unknown')}")
            print(f"Угод: {result.total_trades} | Win Rate: {result.win_rate:.1f}% | "
                  f"Avg PnL: {result.avg_pnl:.3f}% | Total PnL: {result.total_pnl:.2f}%")
            print(f"Profit Factor: {result.profit_factor:.2f} | Max DD: {result.max_drawdown:.2f}% | "
                  f"Avg Hold: {result.avg_hold_time:.1f}h")
            print(f"Відфільтровано: {result.filtered_out} | "
                  f"Найкраща угода: {result.best_trade:.2f}% | Найгірша: {result.worst_trade:.2f}%")

            # Основні параметри
            key_params = []
            if 'stop_loss' in params:
                key_params.append(f"SL: {params['stop_loss']:.3f}")
            if 'take_profit' in params:
                key_params.append(f"TP: {params['take_profit']:.3f}")
            if 'max_hold_hours' in params:
                key_params.append(f"Hold: {params['max_hold_hours']}h")
            if 'use_trailing_stop' in params:
                key_params.append(f"Trail: {params['use_trailing_stop']}")
            if 'use_filters' in params:
                key_params.append(f"Filters: {params['use_filters']}")
            if 'min_adx' in params:
                key_params.append(f"ADX≥{params['min_adx']}")

            print(f"Ключові параметри: {' | '.join(key_params)}")
            print("-" * 80)

    def analyze_best_parameters(self, results: List[OptimizationResult], top_n: int = 50):
        """Аналіз найкращих параметрів"""
        if not results:
            return

        top_results = results[:top_n]
        print(f"\n{'=' * 60}")
        print(f"АНАЛІЗ НАЙКРАЩИХ ПАРАМЕТРІВ (топ-{top_n})")
        print(f"{'=' * 60}")

        # Аналіз параметрів
        param_analysis = {}

        for result in top_results:
            for param, value in result.parameters.items():
                if param not in param_analysis:
                    param_analysis[param] = {}

                if value not in param_analysis[param]:
                    param_analysis[param][value] = []

                param_analysis[param][value].append(result.avg_pnl)

        # Виводимо найкращі значення для кожного параметру
        for param, values in param_analysis.items():
            if len(values) > 1:  # Тільки якщо є варіативність
                avg_pnl_by_value = {}
                for value, pnls in values.items():
                    avg_pnl_by_value[value] = np.mean(pnls)

                # Сортуємо за середнім PnL
                sorted_values = sorted(avg_pnl_by_value.items(),
                                       key=lambda x: x[1], reverse=True)

                print(f"\n{param}:")
                for value, avg_pnl in sorted_values[:3]:  # Топ-3 значення
                    count = len(param_analysis[param][value])
                    print(f"  {value}: {avg_pnl:.3f}% (з {count} конфігурацій)")


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description='Оптимізатор налаштувань торгового аналізатора')
    parser.add_argument('signals_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='optimization_results.csv',
                        help='Файл результатів оптимізації')
    parser.add_argument('-c', '--config', default='enhanced_analyzer_config.json',
                        help='Базовий файл конфігурації')
    parser.add_argument('-n', '--max-configs', type=int, default=500,
                        help='Максимальна кількість конфігурацій для тестування')
    parser.add_argument('--top', type=int, default=10,
                        help='Кількість топ результатів для виведення')
    parser.add_argument('--analysis-depth', type=int, default=50,
                        help='Глибина аналізу параметрів (топ-N конфігурацій)')

    # Режими оптимізації
    parser.add_argument('--mode', choices=['full', 'quick', 'aggressive', 'conservative'],
                        default='full', help='Режим оптимізації')
    parser.add_argument('--focus', choices=['pnl', 'winrate', 'profit_factor', 'balanced'],
                        default='pnl', help='Фокус оптимізації')

    args = parser.parse_args()

    # Перевіряємо існування файлів
    if not os.path.exists(args.signals_file):
        print(f"Помилка: файл сигналів {args.signals_file} не знайдено")
        return

    if not os.path.exists(args.config):
        print(f"Попередження: базовий конфіг {args.config} не знайдено, створюємо стандартний")
        TrendSignalAnalyzer.create_default_config()

    # Налаштування кількості конфігурацій залежно від режиму
    mode_configs = {
        'quick': 100,
        'full': 500,
        'aggressive': 1000,
        'conservative': 250
    }

    max_configs = mode_configs.get(args.mode, args.max_configs)

    print(f"Оптимізатор налаштувань торгового аналізатора")
    print(f"Файл сигналів: {args.signals_file}")
    print(f"Базовий конфіг: {args.config}")
    print(f"Режим: {args.mode} (до {max_configs} конфігурацій)")
    print(f"Фокус оптимізації: {args.focus}")
    print(f"Файл результатів: {args.output}")
    print(f"Лог файл: optimizer.log")

    # Запускаємо оптимізацію
    optimizer = ConfigOptimizer(args.config)

    try:
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("ЗАПУСК ОПТИМІЗАЦІЇ НАЛАШТУВАНЬ")
        logger.info("=" * 60)
        logger.info(f"Файл сигналів: {args.signals_file}")
        logger.info(f"Максимум конфігурацій: {max_configs}")
        logger.info(f"Режим: {args.mode}")
        logger.info(f"Фокус: {args.focus}")

        results = await optimizer.optimize(
            signals_file=args.signals_file,
            max_configs=max_configs,
            output_file=args.output
        )

        if results:
            # Додатковий аналіз
            optimizer.analyze_best_parameters(results, args.analysis_depth)

            # Статистика по стратегіям
            strategies_stats = {}
            for result in results:
                strategy = result.parameters.get('strategy_name', 'unknown')
                if strategy not in strategies_stats:
                    strategies_stats[strategy] = []
                strategies_stats[strategy].append(result.avg_pnl)

            print(f"\n{'=' * 60}")
            print("СТАТИСТИКА ПО СТРАТЕГІЯМ")
            print(f"{'=' * 60}")

            for strategy, pnls in strategies_stats.items():
                avg_pnl = np.mean(pnls)
                max_pnl = max(pnls)
                count = len(pnls)
                print(f"{strategy}: {count} конф., середній PnL: {avg_pnl:.3f}%, "
                      f"максимальний: {max_pnl:.3f}%")

            end_time = datetime.now()
            duration = end_time - start_time

            print(f"\n{'=' * 60}")
            print("ПІДСУМОК ОПТИМІЗАЦІЇ")
            print(f"{'=' * 60}")
            print(f"Протестовано конфігурацій: {len(results)}")
            print(f"Час виконання: {duration}")
            print(f"Найкращий результат: {results[0].avg_pnl:.3f}% середній PnL")
            print(f"Результати збережено в: {args.output}")
            print(f"Топ-10 повних конфігурацій: {args.output.replace('.csv', '_top_configs.json')}")

            logger.info("Оптимізація завершена успішно")
            logger.info(f"Найкращий результат: {results[0].avg_pnl:.3f}% (конфіг {results[0].config_id})")

        else:
            print("Не отримано результатів оптимізації")
            logger.error("Оптимізація не дала результатів")

    except KeyboardInterrupt:
        print("\nОптимізацію перервано користувачем")
        logger.info("Оптимізацію перервано користувачем")
    except Exception as e:
        print(f"Критична помилка: {e}")
        logger.error(f"Критична помилка під час оптимізації: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограму перервано")
    except Exception as e:
        print(f"Критична помилка: {e}")
        logger.error(f"Критична помилка: {e}", exc_info=True)