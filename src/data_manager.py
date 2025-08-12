"""
Data Manager для збереження та завантаження історичних даних
"""
import os
import pandas as pd
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import csv
import json
from dataclasses import asdict
from pathlib import Path


class DataManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data.directory', 'data')
        self.logger = logging.getLogger(__name__)

        # Створюємо необхідні директорії
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "history"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "backtest"), exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Шляхи до файлів
        self.signals_file = os.path.join(self.data_dir, "signals.csv")
        self.detailed_log_file = os.path.join(self.data_dir, "detailed_log.csv")
        self.stats_file = os.path.join(self.data_dir, "stats.json")
        self.db_file = os.path.join(self.data_dir, "trading_data.db")

        # Ініціалізація бази даних
        self._init_database()

        # Ініціалізація CSV файлів
        self._init_csv_files()

    def _init_database(self):
        """Ініціалізація SQLite бази даних"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            # Таблиця для OHLCV даних
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(exchange, symbol, timeframe, timestamp)
                )
            ''')

            # Таблиця для сигналів
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    action TEXT NOT NULL,
                    rsi_1m REAL,
                    sma_1m REAL,
                    rsi_5m REAL,
                    sma_5m REAL,
                    price REAL,
                    volume REAL,
                    diff REAL,
                    trend_confirmed BOOLEAN,
                    delay_ok BOOLEAN,
                    comment TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Таблиця для позицій
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    pnl REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    comment TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Індекси для оптимізації
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup ON ohlcv_data(exchange, symbol, timeframe, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_lookup ON signals(exchange, symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_lookup ON positions(exchange, symbol, status)')

            conn.commit()
            conn.close()

            self.logger.info("База даних SQLite ініціалізована")

        except Exception as e:
            self.logger.error(f"Помилка ініціалізації бази даних: {e}")
            raise

    def _init_csv_files(self):
        """Ініціалізація CSV файлів з заголовками"""
        try:
            # Ініціалізація signals.csv
            if not os.path.exists(self.signals_file):
                with open(self.signals_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(['exchange', 'pair', 'direction', 'comment', 'rsi', 'timestamp'])

            # Ініціалізація detailed_log.csv
            if not os.path.exists(self.detailed_log_file):
                with open(self.detailed_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow([
                        'exchange', 'pair', 'direction', 'action', 'rsi_1m', 'sma_1m',
                        'rsi_5m', 'sma_5m', 'price', 'volume', 'diff', 'trend_confirmed',
                        'delay_ok', 'comment', 'timestamp'
                    ])

            self.logger.info("CSV файли ініціалізовані")

        except Exception as e:
            self.logger.error(f"Помилка ініціалізації CSV файлів: {e}")
            raise

    async def initialize(self):
        """Асинхронна ініціалізація"""
        self.logger.info("DataManager ініціалізований")

    async def save_ohlcv_data(self, exchange: str, symbol: str, timeframe: str,
                             data: List[Dict]) -> bool:
        """Збереження OHLCV даних в базу"""
        try:
            if not data:
                return True

            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            # Підготовка даних для вставки
            records = []
            for candle in data:
                records.append((
                    exchange,
                    symbol,
                    timeframe,
                    datetime.fromtimestamp(candle['timestamp'] / 1000),
                    float(candle['open']),
                    float(candle['high']),
                    float(candle['low']),
                    float(candle['close']),
                    float(candle['volume'])
                ))

            # Вставка з обробкою дублікатів
            cursor.executemany('''
                INSERT OR IGNORE INTO ohlcv_data 
                (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)

            conn.commit()
            rows_affected = cursor.rowcount
            conn.close()

            self.logger.debug(f"Збережено {rows_affected} свічок для {exchange}-{symbol}-{timeframe}")
            return True

        except Exception as e:
            self.logger.error(f"Помилка збереження OHLCV даних: {e}")
            return False

    async def get_ohlcv_data(self, exchange: str, symbol: str, timeframe: str,
                            limit: Optional[int] = None,
                            start_time: Optional[datetime] = None) -> List[Dict]:
        """Отримання OHLCV даних з бази"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
            '''
            params = [exchange, symbol, timeframe]

            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)

            query += ' ORDER BY timestamp DESC'

            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            # Перетворення в формат словників
            data = []
            for row in rows:
                data.append({
                    'timestamp': int(datetime.fromisoformat(str(row[0])).timestamp() * 1000),
                    'open': str(row[1]),
                    'high': str(row[2]),
                    'low': str(row[3]),
                    'close': str(row[4]),
                    'volume': str(row[5])
                })

            # Повертаємо в хронологічному порядку
            return list(reversed(data))

        except Exception as e:
            self.logger.error(f"Помилка отримання OHLCV даних: {e}")
            return []

    async def save_signal_to_db(self, signal_data: Dict) -> bool:
        """Збереження сигналу в базу даних"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO signals 
                (timestamp, exchange, symbol, direction, action, rsi_1m, sma_1m,
                 rsi_5m, sma_5m, price, volume, diff, trend_confirmed, delay_ok, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get('timestamp'),
                signal_data.get('exchange'),
                signal_data.get('symbol'),
                signal_data.get('direction'),
                signal_data.get('action', 'SIGNAL'),
                signal_data.get('rsi_1m'),
                signal_data.get('sma_1m'),
                signal_data.get('rsi_5m'),
                signal_data.get('sma_5m'),
                signal_data.get('price'),
                signal_data.get('volume'),
                signal_data.get('diff'),
                signal_data.get('trend_confirmed', True),
                signal_data.get('delay_ok', True),
                signal_data.get('comment', '')
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            self.logger.error(f"Помилка збереження сигналу в БД: {e}")
            return False

    async def get_historical_signals(self, days: int = 30,
                                   exchange: Optional[str] = None,
                                   symbol: Optional[str] = None) -> List[Dict]:
        """Отримання історичних сигналів"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            start_date = datetime.now() - timedelta(days=days)

            query = 'SELECT * FROM signals WHERE timestamp >= ?'
            params = [start_date]

            if exchange:
                query += ' AND exchange = ?'
                params.append(exchange)

            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)

            query += ' ORDER BY timestamp DESC'

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            # Отримання назв колонок
            columns = [description[0] for description in cursor.description]

            signals = []
            for row in rows:
                signals.append(dict(zip(columns, row)))

            return signals

        except Exception as e:
            self.logger.error(f"Помилка отримання історичних сигналів: {e}")
            return []

    async def export_signals_to_csv(self, output_file: str, days: int = 30) -> bool:
        """Експорт сигналів в CSV файл"""
        try:
            signals = await self.get_historical_signals(days)

            if not signals:
                self.logger.warning("Немає сигналів для експорту")
                return False

            df = pd.DataFrame(signals)
            df.to_csv(output_file, index=False, sep=';')

            self.logger.info(f"Експортовано {len(signals)} сигналів в {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Помилка експорту сигналів: {e}")
            return False

    async def save_stats(self, stats: Dict) -> bool:
        """Збереження статистики в JSON файл"""
        try:
            # Конвертація datetime об'єктів в строки
            stats_copy = {}
            for key, value in stats.items():
                if isinstance(value, datetime):
                    stats_copy[key] = value.isoformat()
                elif isinstance(value, dict):
                    stats_copy[key] = {}
                    for k, v in value.items():
                        if isinstance(v, datetime):
                            stats_copy[key][k] = v.isoformat()
                        else:
                            stats_copy[key][k] = v
                else:
                    stats_copy[key] = value

            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_copy, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            self.logger.error(f"Помилка збереження статистики: {e}")
            return False

    async def load_stats(self) -> Dict:
        """Завантаження статистики з JSON файлу"""
        try:
            if not os.path.exists(self.stats_file):
                return {}

            with open(self.stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            return stats

        except Exception as e:
            self.logger.error(f"Помилка завантаження статистики: {e}")
            return {}

    async def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """Очищення старих даних"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            # Очищення старих OHLCV даних
            cursor.execute('DELETE FROM ohlcv_data WHERE timestamp < ?', (cutoff_date,))
            ohlcv_deleted = cursor.rowcount

            # Очищення старих сигналів
            cursor.execute('DELETE FROM signals WHERE timestamp < ?', (cutoff_date,))
            signals_deleted = cursor.rowcount

            conn.commit()
            conn.close()

            self.logger.info(f"Видалено {ohlcv_deleted} OHLCV записів та {signals_deleted} сигналів")
            return True

        except Exception as e:
            self.logger.error(f"Помилка очищення старих даних: {e}")
            return False

    async def get_database_stats(self) -> Dict:
        """Отримання статистики бази даних"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()

            stats = {}

            # Кількість записів в таблицях
            cursor.execute('SELECT COUNT(*) FROM ohlcv_data')
            stats['ohlcv_records'] = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM signals')
            stats['signals_records'] = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM positions')
            stats['positions_records'] = cursor.fetchone()[0]

            # Розмір файлу бази даних
            stats['db_size_mb'] = round(os.path.getsize(self.db_file) / (1024 * 1024), 2)

            # Останні записи
            cursor.execute('SELECT MAX(timestamp) FROM ohlcv_data')
            last_ohlcv = cursor.fetchone()[0]
            if last_ohlcv:
                stats['last_ohlcv_update'] = last_ohlcv

            cursor.execute('SELECT MAX(timestamp) FROM signals')
            last_signal = cursor.fetchone()[0]
            if last_signal:
                stats['last_signal'] = last_signal

            conn.close()
            return stats

        except Exception as e:
            self.logger.error(f"Помилка отримання статистики БД: {e}")
            return {}

    async def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Створення резервної копії бази даних"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.data_dir}/backups/trading_data_backup_{timestamp}.db"

            # Створення директорії для бекапів
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # Копіювання файлу бази даних
            import shutil
            shutil.copy2(self.db_file, backup_path)

            self.logger.info(f"Резервна копія створена: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Помилка створення резервної копії: {e}")
            return False

    def get_csv_files_paths(self) -> Dict[str, str]:
        """Отримання шляхів до CSV файлів"""
        return {
            'signals': self.signals_file,
            'detailed_log': self.detailed_log_file,
            'stats': self.stats_file
        }