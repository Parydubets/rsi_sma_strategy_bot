#!/usr/bin/env python3
"""
RSI-SMA Trading Bot
Торговий бот на базі RSI та SMA індикаторів з підтримкою хеджування та історичних даних
"""

import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.bot import TradingBot
from src.config import Config
from src.data_manager import DataManager
from src.backtest import BacktestEngine
#from src.utils.logger import setup_logging


class BotManager:
    """Менеджер для управління торговим ботом"""

    def __init__(self):
        self.bot: Optional[TradingBot] = None
        self.config: Optional[Config] = None
        self.running = False

    async def initialize(self, config_path: str = "config/config.yaml"):
        """Ініціалізація бота"""
        try:
            # Завантаження конфігурації
            self.config = Config(config_path)
            await self.config.load()

            # Налаштування логування
            #setup_logging(self.config.get('logging', {}))

            # Створення директорій
            self._create_directories()

            # Ініціалізація бота
            self.bot = TradingBot(self.config)
            await self.bot.initialize()

            logging.info("Бот успішно ініціалізований")
            return True

        except Exception as e:
            logging.error(f"Помилка ініціалізації: {e}")
            return False

    def _create_directories(self):
        """Створення необхідних директорій"""
        directories = ["data", "data/history", "data/backtest", "logs"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def run_live(self, pairs: Optional[List[str]] = None,
                       exchanges: Optional[List[str]] = None, dry_run: bool = False):
        """Запуск в live режимі"""
        if not self.bot:
            logging.error("Бот не ініціалізований")
            return

        try:
            self.running = True
            logging.info(f"Запуск бота в {'DRY RUN' if dry_run else 'LIVE'} режимі")

            # Встановлення параметрів
            if pairs:
                self.bot.set_pairs(pairs)
            if exchanges:
                self.bot.set_exchanges(exchanges)

            self.bot.set_dry_run(dry_run)

            # Запуск бота
            await self.bot.start()

        except KeyboardInterrupt:
            logging.info("Отримано сигнал переривання")
        except Exception as e:
            logging.error(f"Помилка під час роботи: {e}")
        finally:
            await self.shutdown()

    async def run_backtest(self, days: int = 30, pairs: Optional[List[str]] = None):
        """Запуск бектесту"""
        try:
            logging.info(f"Запуск бектесту за {days} днів")

            backtest_engine = BacktestEngine(self.config)
            await backtest_engine.run(days=days, pairs=pairs)

        except Exception as e:
            logging.error(f"Помилка бектесту: {e}")

    async def download_history(self, days: int = 90,
                               timeframes: List[str] = None,
                               pairs: Optional[List[str]] = None):
        """Завантаження історичних даних"""
        try:
            if timeframes is None:
                timeframes = ['1m', '5m']

            logging.info(f"Завантаження історичних даних за {days} днів")

            data_manager = DataManager(self.config)
            await data_manager.download_historical_data(
                days=days,
                timeframes=timeframes,
                pairs=pairs
            )

        except Exception as e:
            logging.error(f"Помилка завантаження історичних даних: {e}")

    async def export_signals(self, from_date: str, to_date: str):
        """Експорт сигналів за період"""
        try:
            logging.info(f"Експорт сигналів з {from_date} до {to_date}")

            data_manager = DataManager(self.config)
            await data_manager.export_signals(from_date, to_date)

        except Exception as e:
            logging.error(f"Помилка експорту сигналів: {e}")

    async def validate_config(self):
        """Валідація конфігурації"""
        try:
            if await self.config.validate():
                logging.info("Конфігурація валідна")
                return True
            else:
                logging.error("Конфігурація містить помилки")
                return False
        except Exception as e:
            logging.error(f"Помилка валідації конфігурації: {e}")
            return False

    async def test_connections(self):
        """Тестування підключення до бірж"""
        try:
            if not self.bot:
                logging.error("Бот не ініціалізований")
                return False

            logging.info("Тестування підключення до бірж...")
            result = await self.bot.test_connections()

            if result:
                logging.info("Всі підключення успішні")
            else:
                logging.error("Є проблеми з підключенням")

            return result

        except Exception as e:
            logging.error(f"Помилка тестування підключення: {e}")
            return False

    async def get_status(self):
        """Отримання статусу бота"""
        try:
            if not self.bot:
                logging.info("Бот не запущений")
                return

            status = await self.bot.get_status()
            logging.info(f"Статус бота: {status}")

        except Exception as e:
            logging.error(f"Помилка отримання статусу: {e}")

    async def stop_all_positions(self):
        """Закриття всіх позицій (заглушка - реалізація в окремому скрипті)"""
        logging.info("Команда закриття всіх позицій - реалізована в окремому скрипті")

    async def shutdown(self):
        """Завершення роботи бота"""
        try:
            self.running = False
            if self.bot:
                logging.info("Завершення роботи бота...")
                await self.bot.stop()
                logging.info("Бот зупинений")
        except Exception as e:
            logging.error(f"Помилка при завершенні: {e}")


def setup_signal_handlers(bot_manager: BotManager):
    """Налаштування обробників сигналів"""

    def signal_handler(signum, frame):
        logging.info(f"Отримано сигнал {signum}")
        if bot_manager.running:
            asyncio.create_task(bot_manager.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Основна функція"""
    parser = argparse.ArgumentParser(description='RSI-SMA Trading Bot')

    # Основні команди
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Шлях до конфігураційного файлу')
    parser.add_argument('--dry-run', action='store_true',
                        help='Запуск без реальних угод')
    parser.add_argument('--pairs', type=str,
                        help='Торгові пари через кому (BTCUSDT,ETHUSDT)')
    parser.add_argument('--exchange', type=str,
                        help='Конкретна біржа (bybit, binance)')

    # Сервісні команди
    parser.add_argument('--status', action='store_true',
                        help='Показати поточний статус')
    parser.add_argument('--stop-all', action='store_true',
                        help='Закрити всі позиції')
    parser.add_argument('--validate-config', action='store_true',
                        help='Перевірити конфігурацію')
    parser.add_argument('--test-connection', action='store_true',
                        help='Тестувати підключення до бірж')

    # Команди для роботи з даними
    parser.add_argument('--backtest', action='store_true',
                        help='Запустити бектест')
    parser.add_argument('--days', type=int, default=30,
                        help='Кількість днів для бектесту або завантаження')
    parser.add_argument('--download-history', action='store_true',
                        help='Завантажити історичні дані')
    parser.add_argument('--timeframes', type=str, default='1m,5m',
                        help='Таймфрейми через кому')
    parser.add_argument('--export', action='store_true',
                        help='Експортувати сигнали')
    parser.add_argument('--from', dest='from_date', type=str,
                        help='Початкова дата (YYYY-MM-DD)')
    parser.add_argument('--to', dest='to_date', type=str,
                        help='Кінцева дата (YYYY-MM-DD)')
    parser.add_argument('--stats', action='store_true',
                        help='Показати статистику')
    parser.add_argument('--period', type=str, default='7d',
                        help='Період для статистики')

    args = parser.parse_args()

    # Створення менеджера бота
    bot_manager = BotManager()
    setup_signal_handlers(bot_manager)

    # Ініціалізація
    if not await bot_manager.initialize(args.config):
        sys.exit(1)

    try:
        # Обробка різних команд
        if args.validate_config:
            success = await bot_manager.validate_config()
            sys.exit(0 if success else 1)

        elif args.test_connection:
            success = await bot_manager.test_connections()
            sys.exit(0 if success else 1)

        elif args.status:
            await bot_manager.get_status()

        elif args.stop_all:
            await bot_manager.stop_all_positions()

        elif args.download_history:
            pairs = args.pairs.split(',') if args.pairs else None
            timeframes = args.timeframes.split(',')
            await bot_manager.download_history(args.days, timeframes, pairs)

        elif args.backtest:
            pairs = args.pairs.split(',') if args.pairs else None
            await bot_manager.run_backtest(args.days, pairs)

        elif args.export:
            if not args.from_date or not args.to_date:
                logging.error("Для експорту потрібно вказати --from і --to")
                sys.exit(1)
            await bot_manager.export_signals(args.from_date, args.to_date)

        elif args.stats:
            # Заглушка для статистики
            logging.info(f"Статистика за період {args.period} - буде реалізована")

        else:
            # Запуск в live режимі
            pairs = args.pairs.split(',') if args.pairs else None
            exchanges = [args.exchange] if args.exchange else None
            await bot_manager.run_live(pairs, exchanges, args.dry_run)

    except Exception as e:
        logging.error(f"Помилка виконання: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Програма перервана користувачем")
    except Exception as e:
        logging.error(f"Критична помилка: {e}")
        sys.exit(1)