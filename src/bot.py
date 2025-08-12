"""
Основний клас торгового бота RSI-SMA
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict

from .config import Config
from .exchanges.exchange_manager import ExchangeManager
from .indicators.rsi_sma_strategy import RSISMAStrategy
from .data_manager import DataManager
from .utils.csv_logger import CSVLogger
from .utils.delay_manager import DelayManager


class TradingBot:
    """Основний клас торгового бота"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Компоненти бота
        self.exchange_manager: Optional[ExchangeManager] = None
        self.strategy: Optional[RSISMAStrategy] = None
        self.data_manager: Optional[DataManager] = None
        self.csv_logger: Optional[CSVLogger] = None
        self.delay_manager: Optional[DelayManager] = None

        # Стан бота
        self.running = False
        self.dry_run = False
        self.tasks: List[asyncio.Task] = []

        # Налаштування
        self.pairs: List[str] = []
        self.exchanges: List[str] = []
        self.active_pairs: Set[str] = set()

        # Статистика
        self.stats = {
            'signals_generated': 0,
            'signals_by_pair': defaultdict(int),
            'signals_by_direction': defaultdict(int),
            'start_time': None,
            'last_signal_time': None
        }

    async def initialize(self):
        """Ініціалізація всіх компонентів бота"""
        try:
            self.logger.info("Ініціалізація торгового бота...")

            # Ініціалізація менеджера бірж
            self.exchange_manager = ExchangeManager(self.config)
            await self.exchange_manager.initialize()

            # Ініціалізація стратегії
            strategy_config = self.config.get('strategy', {})
            self.strategy = RSISMAStrategy(strategy_config)

            # Ініціалізація менеджера даних
            self.data_manager = DataManager(self.config)
            await self.data_manager.initialize()

            # Ініціалізація CSV логгера
            csv_config = self.config.get('logging.csv_files', {})
            self.csv_logger = CSVLogger(csv_config)

            # Ініціалізація менеджера делею
            delay_config = self.config.get('delay_system', {})
            self.delay_manager = DelayManager(delay_config)

            # Завантаження активних пар
            await self._load_active_pairs()

            self.logger.info("Бот успішно ініціалізований")

        except Exception as e:
            self.logger.error(f"Помилка ініціалізації бота: {e}")
            raise

    async def _load_active_pairs(self):
        """Завантаження активних торгових пар"""
        pairs_config = self.config.get('pairs', [])

        for pair_config in pairs_config:
            if pair_config.get('enabled', True):
                symbol = pair_config['symbol']
                self.active_pairs.add(symbol)

                # Якщо не вказані конкретні пари, використовуємо всі активні
                if not self.pairs:
                    self.pairs = list(self.active_pairs)

    def set_pairs(self, pairs: List[str]):
        """Встановлення конкретних пар для торгівлі"""
        self.pairs = [pair.upper() for pair in pairs]
        self.logger.info(f"Встановлені пари для торгівлі: {self.pairs}")

    def set_exchanges(self, exchanges: List[str]):
        """Встановлення конкретних бірж для торгівлі"""
        self.exchanges = [exchange.lower() for exchange in exchanges]
        self.logger.info(f"Встановлені біржі для торгівлі: {self.exchanges}")

    def set_dry_run(self, dry_run: bool):
        """Встановлення режиму без реальних угод"""
        self.dry_run = dry_run
        if dry_run:
            self.logger.info("Режим DRY RUN активований")

    async def start(self):
        """Запуск бота"""
        if self.running:
            self.logger.warning("Бот вже запущений")
            return

        try:
            self.running = True
            self.stats['start_time'] = datetime.now()

            self.logger.info("Запуск торгового бота...")

            # Створення тасків для кожної пари на кожній біржі
            await self._create_trading_tasks()

            # Запуск основного циклу
            await self._run_main_loop()

        except Exception as e:
            self.logger.error(f"Помилка запуску бота: {e}")
            raise
        finally:
            await self.stop()

    async def _create_trading_tasks(self):
        """Створення асинхронних тасків для торгівлі"""
        active_exchanges = self.exchanges if self.exchanges else self.exchange_manager.get_active_exchanges()
        active_pairs = self.pairs if self.pairs else list(self.active_pairs)

        self.logger.info(f"Створення тасків для {len(active_exchanges)} бірж і {len(active_pairs)} пар")

        for exchange_name in active_exchanges:
            for pair in active_pairs:
                if self.exchange_manager.is_pair_available(exchange_name, pair):
                    task = asyncio.create_task(
                        self._trading_task(exchange_name, pair),
                        name=f"trading-{exchange_name}-{pair}"
                    )
                    self.tasks.append(task)
                else:
                    self.logger.warning(f"Пара {pair} недоступна на біржі {exchange_name}")

    async def _trading_task(self, exchange_name: str, pair: str):
        """Основний таск для торгівлі однією парою на одній біржі"""
        logger = logging.getLogger(f"{__name__}.{exchange_name}.{pair}")

        try:
            logger.info(f"Запуск торгівлі парою {pair} на біржі {exchange_name}")

            # Отримання об'єкту біржі
            exchange = self.exchange_manager.get_exchange(exchange_name)
            if not exchange:
                logger.error(f"Біржа {exchange_name} недоступна")
                return

            # Підписка на дані
            await exchange.subscribe_to_klines(pair, ['1m', '5m'])

            while self.running:
                try:
                    # Отримання даних для аналізу
                    klines_1m = await exchange.get_klines(pair, '1m', limit=100)
                    klines_5m = await exchange.get_klines(pair, '5m', limit=100)

                    if not klines_1m or not klines_5m:
                        logger.warning(f"Немає даних для {pair}")
                        await asyncio.sleep(5)
                        continue

                    # Аналіз стратегії
                    signal = await self._analyze_strategy(
                        exchange_name, pair, klines_1m, klines_5m
                    )

                    if signal:
                        await self._process_signal(exchange_name, pair, signal, klines_1m[-1])

                    # Затримка перед наступною ітерацією
                    await asyncio.sleep(1)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Помилка в торговому циклі {pair}: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Критична помилка в торговому таску {pair}: {e}")
        finally:
            logger.info(f"Торговий таск {pair} на біржі {exchange_name} завершений")

    async def _analyze_strategy(self, exchange_name: str, pair: str,
                               klines_1m: List[Dict], klines_5m: List[Dict]) -> Optional[Dict]:
        """Аналіз стратегії та генерація сигналу"""
        try:
            # Розрахунок індикаторів для 1m таймфрейму
            indicators_1m = self.strategy.calculate_indicators(klines_1m, '1m')

            # Розрахунок індикаторів для 5m таймфрейму
            indicators_5m = self.strategy.calculate_indicators(klines_5m, '5m')

            if not indicators_1m or not indicators_5m:
                return None

            # Перевірка умов для LONG
            long_signal = self.strategy.check_long_conditions(
                indicators_1m, indicators_5m, exchange_name, pair
            )

            # Перевірка умов для SHORT
            short_signal = self.strategy.check_short_conditions(
                indicators_1m, indicators_5m, exchange_name, pair
            )

            # Перевірка делею
            if long_signal and self.delay_manager.check_delay(exchange_name, pair, 'LONG'):
                return {
                    'direction': 'LONG',
                    'indicators_1m': indicators_1m,
                    'indicators_5m': indicators_5m,
                    'confidence': long_signal.get('confidence', 0.5)
                }

            if short_signal and self.delay_manager.check_delay(exchange_name, pair, 'SHORT'):
                return {
                    'direction': 'SHORT',
                    'indicators_1m': indicators_1m,
                    'indicators_5m': indicators_5m,
                    'confidence': short_signal.get('confidence', 0.5)
                }

            return None

        except Exception as e:
            self.logger.error(f"Помилка аналізу стратегії {exchange_name}-{pair}: {e}")
            return None

    async def _process_signal(self, exchange_name: str, pair: str,
                             signal: Dict, current_kline: Dict):
        """Обробка торгового сигналу"""
        try:
            direction = signal['direction']
            indicators_1m = signal['indicators_1m']
            indicators_5m = signal['indicators_5m']

            current_price = float(current_kline['close'])
            timestamp = datetime.now()

            # Логування основного сигналу
            await self.csv_logger.log_signal(
                exchange=exchange_name,
                pair=pair,
                direction=direction,
                rsi=indicators_1m['rsi'][-1],
                sma=indicators_1m['rsi_sma'][-1],
                timestamp=timestamp,
                comment=f"Signal generated - confidence: {signal['confidence']:.2f}"
            )

            # Детальне логування
            await self.csv_logger.log_detailed(
                exchange=exchange_name,
                pair=pair,
                direction=direction,
                action="SIGNAL",
                rsi_1m=indicators_1m['rsi'][-1],
                sma_1m=indicators_1m['rsi_sma'][-1],
                rsi_5m=indicators_5m['rsi'][-1],
                sma_5m=indicators_5m['rsi_sma'][-1],
                price=current_price,
                volume=float(current_kline.get('volume', 0)),
                diff=indicators_1m['diff'][-1],
                trend_confirmed=True,
                delay_ok=True,
                comment=f"Generated signal with confidence {signal['confidence']:.2f}",
                timestamp=timestamp
            )

            # Оновлення делею
            self.delay_manager.update_delay(exchange_name, pair, direction, timestamp)

            # Оновлення статистики
            self._update_stats(pair, direction, timestamp)

            # Лог про згенерований сигнал
            self.logger.info(
                f"🎯 Сигнал: {exchange_name} {pair} {direction} | "
                f"RSI: {indicators_1m['rsi'][-1]:.2f} | "
                f"SMA: {indicators_1m['rsi_sma'][-1]:.2f} | "
                f"Price: {current_price:.6f} | "
                f"Conf: {signal['confidence']:.2f}"
            )

            if self.dry_run:
                self.logger.info("DRY RUN режим - реальна угода не виконується")

        except Exception as e:
            self.logger.error(f"Помилка обробки сигналу {exchange_name}-{pair}: {e}")

    def _update_stats(self, pair: str, direction: str, timestamp: datetime):
        """Оновлення статистики"""
        self.stats['signals_generated'] += 1
        self.stats['signals_by_pair'][pair] += 1
        self.stats['signals_by_direction'][direction] += 1
        self.stats['last_signal_time'] = timestamp

    async def _run_main_loop(self):
        """Основний цикл бота"""
        try:
            self.logger.info("Запуск основного циклу...")

            # Запуск всіх торгових тасків
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            else:
                self.logger.error("Немає активних торгових тасків")

        except Exception as e:
            self.logger.error(f"Помилка основного циклу: {e}")

    async def stop(self):
        """Зупинка бота"""
        if not self.running:
            return

        try:
            self.logger.info("Зупинка торгового бота...")
            self.running = False

            # Скасування всіх тасків
            for task in self.tasks:
                if not task.done():
                    task.cancel()

            # Очікування завершення тасків
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)

            # Закриття з'єднань
            if self.exchange_manager:
                await self.exchange_manager.close_all()

            # Збереження фінальної статистики
            await self._save_final_stats()

            self.logger.info("Бот зупинений")

        except Exception as e:
            self.logger.error(f"Помилка при зупинці бота: {e}")

    async def test_connections(self) -> bool:
        """Тестування підключення до всіх бірж"""
        if not self.exchange_manager:
            self.logger.error("Exchange manager не ініціалізований")
            return False

        return await self.exchange_manager.test_all_connections()

    async def get_status(self) -> Dict:
        """Отримання поточного статусу бота"""
        status = {
            'running': self.running,
            'dry_run': self.dry_run,
            'start_time': self.stats.get('start_time'),
            'uptime': None,
            'active_tasks': len([task for task in self.tasks if not task.done()]),
            'total_tasks': len(self.tasks),
            'stats': dict(self.stats),
            'pairs': self.pairs,
            'exchanges': self.exchanges
        }

        if status['start_time']:
            uptime = datetime.now() - status['start_time']
            status['uptime'] = str(uptime).split('.')[0]  # Без мікросекунд

        return status

    async def _save_final_stats(self):
        """Збереження фінальної статистики"""
        try:
            status = await self.get_status()

            # Логування статистики
            self.logger.info("=== Фінальна статистика ===")
            self.logger.info(f"Час роботи: {status.get('uptime', 'N/A')}")
            self.logger.info(f"Згенеровано сигналів: {self.stats['signals_generated']}")
            self.logger.info(f"Сигнали по парах: {dict(self.stats['signals_by_pair'])}")
            self.logger.info(f"Сигнали по напрямках: {dict(self.stats['signals_by_direction'])}")

            # Збереження в файл статистики (опціонально)
            stats_file = self.config.get('logging.csv_files.stats', 'data/stats.json')
            if stats_file:
                import json
                from pathlib import Path

                stats_data = {
                    'session': {
                        'start_time': status['start_time'].isoformat() if status['start_time'] else None,
                        'end_time': datetime.now().isoformat(),
                        'uptime': status['uptime'],
                        'mode': 'dry_run' if self.dry_run else 'live'
                    },
                    'stats': dict(self.stats),
                    'config': {
                        'pairs': self.pairs,
                        'exchanges': self.exchanges
                    }
                }

                # Конвертація datetime об'єктів в строки
                for key, value in stats_data['stats'].items():
                    if isinstance(value, datetime):
                        stats_data['stats'][key] = value.isoformat()

                Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
                with open(stats_file, 'w') as f:
                    json.dump(stats_data, f, indent=2, default=str)

                self.logger.info(f"Статистика збережена в {stats_file}")

        except Exception as e:
            self.logger.error(f"Помилка збереження статистики: {e}")