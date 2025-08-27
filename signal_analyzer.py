#!/usr/bin/env python3
"""
Асинхронний процесор торгових сигналів для криптовалютного трейдингу
Читає CSV файл з сигналами та обробляє кожен сигнал з реальними ринковими даними

ОНОВЛЕННЯ v2.3:
1. Виправлено FutureWarning для fillna() - використано bfill() та ffill()
2. Покращено логіку кешування для зменшення кількості API запитів
3. Додано обробку помилок при розрахунку індикаторів
4. Оптимізовано використання пам'яті
"""

import ccxt
import pandas as pd
import asyncio
import csv
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
import logging
from typing import Dict, List, Optional, Tuple
import argparse
import os
import aiofiles
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Аналізатор тренду SMA з гнучкими налаштуваннями"""

    def __init__(self, trend_periods: int = 3, deviation_threshold: float = 0.5):
        self.trend_periods = trend_periods
        self.deviation_threshold = deviation_threshold

    def analyze_sma_trend_with_values(self, sma_values: List[float]) -> Dict:
        """
        Аналіз тренду на основі значень SMA з поверненням детальної інформації

        Args:
            sma_values: Список значень SMA

        Returns:
            Dict з трендом та значеннями
        """
        if not sma_values or len(sma_values) < 2:
            return {
                'trend': 'neutral',
                'sma_values': sma_values,
                'sma_changes': [],
                'sma_values_formatted': '',
                'significant_changes': [],
                'changes_count': 0,
                'significant_count': 0
            }

        # Розраховуємо зміни SMA
        sma_changes = []
        for i in range(1, len(sma_values)):
            change = sma_values[i] - sma_values[i - 1]
            sma_changes.append(round(change, 3))

        # Відфільтровуємо зміни які менші за поріг похибки
        significant_changes = [change for change in sma_changes if abs(change) > self.deviation_threshold]

        # Визначення тренду
        trend = 'neutral'
        if significant_changes:
            positive_changes = sum(1 for change in significant_changes if change > 0)
            negative_changes = sum(1 for change in significant_changes if change < 0)

            if positive_changes > negative_changes:
                trend = 'uptrend'
            elif negative_changes > positive_changes:
                trend = 'downtrend'

        # Форматування значень SMA для виводу
        sma_values_formatted = ' -> '.join([f"{val:.2f}" for val in sma_values])

        return {
            'trend': trend,
            'sma_values': sma_values,
            'sma_changes': sma_changes,
            'sma_values_formatted': sma_values_formatted,
            'significant_changes': significant_changes,
            'changes_count': len(sma_changes),
            'significant_count': len(significant_changes),
            'deviation_threshold': self.deviation_threshold
        }


class SignalProcessor:
    """Асинхронний процесор торгових сигналів з покращеною логікою"""

    def __init__(self, config: Dict):
        self.config = config
        self.exchange = self._init_exchange()
        self.processed_signals = []
        self.data_cache = {}
        self.semaphore = asyncio.Semaphore(5)
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Ініціалізація аналізатора тренду
        self.trend_analyzer = TrendAnalyzer(
            trend_periods=config.get('trend_periods', 3),
            deviation_threshold=config.get('deviation_threshold', 0.5)
        )

    def _init_exchange(self):
        """Ініціалізація біржі"""
        try:
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'sandbox': False,
                'rateLimit': 100,
                'timeout': 30000,
            })
            return exchange
        except Exception as e:
            logger.error(f"Помилка ініціалізації біржі: {e}")
            return None

    async def read_signals_csv(self, filename: str) -> List[Dict]:
        """Асинхронне читання CSV файлу з сигналами"""
        try:
            async with aiofiles.open(filename, mode='r', encoding='utf-8') as file:
                content = await file.read()

            signals = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._parse_csv_content, content
            )

            logger.info(f"Успішно завантажено {len(signals)} сигналів з {filename}")
            return signals

        except Exception as e:
            logger.error(f"Помилка читання файлу {filename}: {e}")
            return []

    def _parse_csv_content(self, content: str) -> List[Dict]:
        """Парсинг контенту CSV файлу"""
        signals = []
        lines = content.strip().split('\n')

        if not lines:
            return signals

        # Визначаємо роздільник
        first_line = lines[0]
        separators = [';', '\t', ',']
        separator = ';'

        for sep in separators:
            if sep in first_line:
                test_columns = first_line.split(sep)
                if len(test_columns) > 5:
                    separator = sep
                    break

        logger.info(f"Використовуємо роздільник: '{separator}'")

        # Читаємо з pandas
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(content), sep=separator)
            logger.info(f"Знайдено колонки: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Помилка читання CSV: {e}")
            return signals

        # Мапінг колонок
        column_mapping = {
            'Pair': ['Pair'],
            'Direction': ['Direction'],
            'Status': ['Status'],
            'Signal_Time': ['Signal_Time'],
            'Entry_Time': ['Entry_Time'],
            'RSI_1m': ['RSI_1m'],
            'RSI_SMA_1m': ['RSI_SMA_1m'],
            'RSI_5m': ['RSI_5m'],
            'RSI_SMA_5m': ['RSI_SMA_5m'],
            'Signal_Price': ['Signal_Price'],
            'Entry_Price': ['Entry_Price'],
            'Quality': ['Quality'],
            'Confidence': ['Confidence'],
            'Comment': ['Comment']
        }

        mapped_cols = {}
        available_columns = list(df.columns)

        for target, variants in column_mapping.items():
            for variant in variants:
                if variant in available_columns:
                    mapped_cols[target] = variant
                    break

        # Обробка кожного рядка
        for idx, row in df.iterrows():
            try:
                signal = {
                    'pair': self._clean_string(row.get(mapped_cols.get('Pair', ''), '')),
                    'direction': self._clean_string(row.get(mapped_cols.get('Direction', ''), '')).lower(),
                    'status': self._clean_string(row.get(mapped_cols.get('Status', ''), '')).lower(),
                    'signal_time': self._clean_string(row.get(mapped_cols.get('Signal_Time', ''), '')),
                    'entry_time': self._clean_string(row.get(mapped_cols.get('Entry_Time', ''), '')),
                    'rsi_1m': self._safe_float(row.get(mapped_cols.get('RSI_1m', ''), 0)),
                    'rsi_sma_1m': self._safe_float(row.get(mapped_cols.get('RSI_SMA_1m', ''), 0)),
                    'rsi_5m': self._safe_float(row.get(mapped_cols.get('RSI_5m', ''), 0)),
                    'rsi_sma_5m': self._safe_float(row.get(mapped_cols.get('RSI_SMA_5m', ''), 0)),
                    'signal_price': self._safe_float(row.get(mapped_cols.get('Signal_Price', ''), 0)),
                    'entry_price': self._safe_float(row.get(mapped_cols.get('Entry_Price', ''), 0)),
                    'quality': self._clean_string(row.get(mapped_cols.get('Quality', ''), '')),
                    'confidence': self._safe_float(row.get(mapped_cols.get('Confidence', ''), 0)),
                    'comment': self._clean_string(row.get(mapped_cols.get('Comment', ''), ''))
                }

                if (signal['pair'] and signal['direction'] in ['long', 'short'] and
                        signal['status'] in ['open'] and signal['signal_time']):
                    signals.append(signal)

            except Exception as e:
                logger.warning(f"Помилка обробки рядка {idx}: {e}")
                continue

        return signals

    def _clean_string(self, value) -> str:
        """Очищення строкових значень"""
        if pd.isna(value):
            return ''
        return str(value).strip()

    def _safe_float(self, value) -> float:
        """Безпечне перетворення в float"""
        try:
            if pd.isna(value) or value == '':
                return 0.0
            if isinstance(value, str):
                value = value.replace(',', '.').strip()
            return float(value)
        except:
            return 0.0

    def parse_signal_time(self, time_str: str) -> datetime:
        """Парсинг часу сигналу з різними форматами"""
        if not time_str:
            return datetime.now()

        if ':' in time_str and len(time_str.split()) == 1:
            try:
                today = datetime.now().date()
                if len(time_str.split(':')) == 2:
                    time_str += ":00"
                time_part = datetime.strptime(time_str, '%H:%M:%S').time()
                return datetime.combine(today, time_part)
            except:
                pass

        formats = [
            '%d.%m.%Y %H:%M',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y %H:%M',
            '%d.%m.%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%d-%m-%Y %H:%M',
            '%m/%d/%Y %H:%M',
            '%H:%M:%S',
            '%H:%M'
        ]

        for fmt in formats:
            try:
                parsed_time = datetime.strptime(time_str.strip(), fmt)
                if fmt in ['%H:%M:%S', '%H:%M']:
                    today = datetime.now().date()
                    parsed_time = datetime.combine(today, parsed_time.time())
                return parsed_time
            except ValueError:
                continue

        logger.warning(f"Не вдалося розпарсити час: {time_str}")
        return datetime.now()

    async def check_sma_trend_filter(self, df_5m: pd.DataFrame, signal_time: datetime,
                                     direction: str) -> Tuple[bool, Dict]:
        """Перевірка SMA тренду на 5-хвилинному таймфреймі"""
        try:
            # Знаходимо свічку що відповідає часу сигналу
            signal_candles = df_5m[df_5m['datetime'] <= signal_time]
            if len(signal_candles) < self.trend_analyzer.trend_periods + 1:
                return False, {'error': 'Insufficient data for trend analysis', 'sma_values_formatted': ''}

            # Беремо останні свічки для аналізу тренду
            recent_candles = signal_candles.tail(self.trend_analyzer.trend_periods + 1)

            # Збираємо значення SMA
            sma_values = []
            for i in range(len(recent_candles)):
                sma_val = recent_candles.iloc[i]['rsi_sma']
                sma_values.append(round(sma_val, 2))

            # Аналізуємо тренд з значеннями SMA
            trend_info = self.trend_analyzer.analyze_sma_trend_with_values(sma_values)
            trend = trend_info['trend']

            # Логіка фільтрації
            should_skip = False
            filter_reason = ""

            if direction.lower() == 'short' and trend == 'uptrend':
                should_skip = True
                filter_reason = f"SHORT signal skipped: SMA uptrend detected"
            elif direction.lower() == 'long' and trend == 'downtrend':
                should_skip = True
                filter_reason = f"LONG signal skipped: SMA downtrend detected"

            trend_info['filter_reason'] = filter_reason
            trend_info['should_skip'] = should_skip

            return should_skip, trend_info

        except Exception as e:
            logger.error(f"Помилка перевірки SMA тренду: {e}")
            return False, {'error': str(e), 'sma_values_formatted': ''}

    async def fetch_market_data_with_cache(self, symbol: str, timeframe: str,
                                           start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Завантаження ринкових даних з покращеним кешуванням"""
        normalized_symbol = symbol.replace('//', '/').replace('/', '').strip()

        # Створюємо ключ кешу базуючись на дні, щоб зменшити дублювання запитів
        cache_key = f"{normalized_symbol}_{timeframe}_{start_time.date()}"

        if cache_key in self.data_cache:
            df = self.data_cache[cache_key]
            # Перевіряємо чи покривають кешовані дані потрібний діапазон
            if (not df.empty and
                    df['datetime'].min() <= start_time and
                    df['datetime'].max() >= end_time):

                mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
                cached_result = df[mask].copy()
                if not cached_result.empty:
                    logger.debug(f"Використано кешовані дані для {symbol}")
                    return cached_result

        async with self.semaphore:
            df = await self.fetch_market_data(symbol, timeframe, start_time, end_time)

        # Кешуємо дані на весь день для повторного використання
        if not df.empty:
            # Розширюємо кеш на весь день
            day_start = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)

            try:
                # Завантажуємо дані на весь день для кешу (якщо потрібно)
                if len(df) < 100:  # Якщо мало даних, завантажуємо більше
                    full_day_df = await self.fetch_market_data(symbol, timeframe, day_start, day_end)
                    if not full_day_df.empty:
                        self.data_cache[cache_key] = full_day_df
                else:
                    self.data_cache[cache_key] = df
            except:
                # Якщо не вдається завантажити на весь день, кешуємо що є
                self.data_cache[cache_key] = df

        return df

    async def fetch_market_data(self, symbol: str, timeframe: str,
                                start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Завантаження ринкових даних з біржі"""
        try:
            if not self.exchange:
                raise Exception("Біржа не ініціалізована")

            original_symbol = symbol.strip()
            clean_symbol = original_symbol.replace('//', '/').replace('/', '')

            if clean_symbol.endswith('USDT'):
                base_currency = clean_symbol[:-4]
                api_symbol = f"{base_currency}/USDT"
            else:
                api_symbol = f"{clean_symbol}/USDT"

            logger.debug(f"Символ: {original_symbol} -> API символ: {api_symbol}")

            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)

            all_data = []
            current_since = since
            limit = 1000
            requests_count = 0

            while current_since < until:
                try:
                    requests_count += 1
                    ohlcv = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.exchange.fetch_ohlcv(api_symbol, timeframe, current_since, limit)
                    )

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1

                    await asyncio.sleep(0.1)

                    if len(ohlcv) < limit:
                        break

                except Exception as e:
                    logger.warning(f"Помилка завантаження даних для {api_symbol} (запит {requests_count}): {e}")
                    await asyncio.sleep(1)
                    break

            if not all_data:
                logger.warning(f"Немає даних для {original_symbol} (API: {api_symbol})")
                return pd.DataFrame()

            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            df = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.calculate_indicators, df
            )

            logger.debug(f"Завантажено {len(df)} свічок для {original_symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Помилка завантаження даних для {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ВИПРАВЛЕНО: Розрахунок технічних індикаторів без FutureWarning"""
        if len(df) < 30:
            logger.warning(f"Недостатньо даних для розрахунку індикаторів: {len(df)} свічок")
            df = df.copy()
            df['rsi'] = 50.0
            df['rsi_sma'] = 50.0
            return df

        try:
            df = df.copy()

            # Розрахунок RSI з періодом 14
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            df['rsi'] = rsi_indicator.rsi()

            # Розрахунок RSI SMA з періодом 14
            df['rsi_sma'] = df['rsi'].rolling(window=14, min_periods=1).mean()

            # ВИПРАВЛЕНО: Заповнення пропущених значень без FutureWarning
            df['rsi'] = df['rsi'].bfill().fillna(50.0)
            df['rsi_sma'] = df['rsi_sma'].bfill().fillna(50.0)

            # Логування для дебагу
            if len(df) > 0:
                logger.debug(
                    f"Індикатори розраховані. Останні значення: RSI={df['rsi'].iloc[-1]:.2f}, RSI_SMA={df['rsi_sma'].iloc[-1]:.2f}")

        except Exception as e:
            logger.warning(f"Помилка розрахунку індикаторів: {e}")
            df = df.copy()
            df['rsi'] = 50.0
            df['rsi_sma'] = 50.0

        return df

    def check_exit_conditions(self, df: pd.DataFrame, current_idx: int, direction: str) -> Tuple[bool, str, bool]:
        """Перевірка умов виходу з позиції з правильною логікою"""
        if current_idx < 1 or current_idx >= len(df):
            return False, "", False

        current_candle = df.iloc[current_idx]
        prev_candle = df.iloc[current_idx - 1]

        current_rsi = current_candle.get('rsi', 50)
        current_rsi_sma = current_candle.get('rsi_sma', 50)
        prev_rsi = prev_candle.get('rsi', 50)
        prev_rsi_sma = prev_candle.get('rsi_sma', 50)

        if pd.isna(current_rsi) or pd.isna(current_rsi_sma) or pd.isna(prev_rsi) or pd.isna(prev_rsi_sma):
            return False, "", False

        is_long = direction.lower() == 'long'

        if is_long:
            # ВИХІД З ЛОНГА:
            # 1. Перекупленість (RSI досягає 70+)
            if current_rsi >= 68:
                return True, f'Long Overbought Zone (RSI: {current_rsi:.1f} >= 70)', True

            # 2. Перетин RSI вниз через RSI SMA у зоні 60+
            if prev_rsi >= 59 and (current_rsi_sma >= 59 or prev_rsi_sma>=59):  # Обидві свічки в зоні 60+
                if prev_rsi > prev_rsi_sma and current_rsi <= current_rsi_sma:  # Перетин вниз
                    return True, f'Long RSI Cross Down (RSI: {prev_rsi:.1f}>{prev_rsi_sma:.1f} -> {current_rsi:.1f}<={current_rsi_sma:.1f} at 60+ zone)', False

        else:  # Short
            # ВИХІД З ШОРТА:
            # 1. Перепроданість (RSI досягає 30-)
            if current_rsi <= 32:
                return True, f'Short Oversold Zone (RSI: {current_rsi:.1f} <= 30)', True

            # 2. Перетин RSI вгору через RSI SMA у зоні 40-
            if prev_rsi <= 41 and (current_rsi_sma <= 41 or prev_rsi_sma <= 41):  # Обидві свічки в зоні 40-
                if prev_rsi < prev_rsi_sma and current_rsi >= current_rsi_sma:  # Перетин вгору
                    return True, f'Short RSI Cross Up (RSI: {prev_rsi:.1f}<{prev_rsi_sma:.1f} -> {current_rsi:.1f}>={current_rsi_sma:.1f} at 40- zone)', False

        return False, "", False

    def collect_rsi_history(self, df: pd.DataFrame, entry_idx: int, exit_idx: int) -> Dict:
        """Збір історії RSI від входу до виходу"""
        try:
            if entry_idx >= len(df) or exit_idx >= len(df) or entry_idx > exit_idx:
                return {'error': 'Invalid indices'}

            rsi_history = []
            time_history = []

            for i in range(entry_idx, min(exit_idx + 1, len(df))):
                candle = df.iloc[i]
                rsi_history.append(round(candle.get('rsi', 0), 2))
                time_history.append(candle['datetime'].strftime('%H:%M'))

            # Статистика по RSI
            if rsi_history:
                rsi_stats = {
                    'min': min(rsi_history),
                    'max': max(rsi_history),
                    'start': rsi_history[0] if rsi_history else 0,
                    'end': rsi_history[-1] if rsi_history else 0,
                    'avg': round(sum(rsi_history) / len(rsi_history), 2) if rsi_history else 0,
                    'volatility': round(np.std(rsi_history), 2) if len(rsi_history) > 1 else 0
                }
            else:
                rsi_stats = {}

            return {
                'rsi_history': rsi_history,
                'time_history': time_history,
                'rsi_stats': rsi_stats,
                'history_length': len(rsi_history)
            }

        except Exception as e:
            logger.error(f"Помилка збору історії RSI: {e}")
            return {'error': str(e)}

    async def process_single_signal(self, signal: Dict) -> Optional[Dict]:
        """Асинхронна обробка одного сигналу"""
        try:
            pair = signal['pair']
            direction = signal['direction']
            signal_time = self.parse_signal_time(signal['signal_time'])

            entry_time = signal_time + timedelta(minutes=1)
            start_time = signal_time - timedelta(hours=2)
            end_time = entry_time + timedelta(hours=self.config.get('max_hold_hours', 24))

            logger.debug(f"Обробка сигналу: {pair} {direction.upper()} о {signal_time.strftime('%H:%M:%S')}")

            # Завантажуємо дані для обох таймфреймів
            df_1m = await self.fetch_market_data_with_cache(pair, '1m', start_time, end_time)
            df_5m = await self.fetch_market_data_with_cache(pair, '5m', start_time, end_time)

            if df_1m.empty:
                logger.warning(f"Немає 1-хвилинних даних для {pair}")
                return None

            # Перевірка SMA тренд фільтра
            sma_trend_values = ""
            trend_info = {}

            if not df_5m.empty:
                should_skip, trend_info = await self.check_sma_trend_filter(df_5m, signal_time, direction)
                sma_trend_values = trend_info.get('sma_values_formatted', '')

                if should_skip:
                    logger.info(
                        f"🚫 {pair} {direction.upper()}: {trend_info['filter_reason']} (SMA: {sma_trend_values})")
                    return {
                        **signal,
                        'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'sma_trend_values': sma_trend_values,
                        'status': 'filtered_by_sma_trend',
                        'filter_reason': trend_info['filter_reason'],
                        'sma_trend_analysis': trend_info,
                        'pnl_percent': 0,
                        'hold_time_hours': 0
                    }
                else:
                    logger.debug(
                        f"✅ {pair} {direction.upper()}: SMA trend check passed - {trend_info['trend']} (SMA: {sma_trend_values})")

            # Обробка сигналу
            entry_candles = df_1m[df_1m['datetime'] >= entry_time]
            if entry_candles.empty:
                logger.warning(f"Немає даних для входу в {pair}")
                return None

            entry_candle = entry_candles.iloc[0]
            entry_price = entry_candle['close']
            actual_entry_time = entry_candle['datetime']
            entry_idx = entry_candles.index[0]

            # Пошук точки виходу
            exit_found = False
            max_exit_time = actual_entry_time + timedelta(hours=self.config.get('max_hold_hours', 24))
            exit_idx = entry_idx
            pending_exit = False
            pending_exit_reason = ""

            for i in range(entry_idx + 1, len(df_1m)):
                candle = df_1m.iloc[i]
                current_time = candle['datetime']

                if current_time > max_exit_time:
                    exit_price = candle['close']
                    exit_time = current_time
                    exit_reason = f'Max Hold Time ({self.config.get("max_hold_hours", 24)}h)'
                    exit_idx = i
                    exit_found = True
                    break

                if pending_exit:
                    exit_price = candle['close']
                    exit_time = current_time
                    exit_reason = pending_exit_reason + ' [Next Candle]'
                    exit_idx = i
                    exit_found = True
                    break

                should_exit, exit_reason, exit_next_candle = self.check_exit_conditions(df_1m, i, direction)

                if should_exit:
                    if exit_next_candle:
                        pending_exit = True
                        pending_exit_reason = exit_reason
                        continue
                    else:
                        exit_price = candle['close']
                        exit_time = current_time
                        exit_idx = i
                        exit_found = True
                        break

            if not exit_found:
                last_candle = df_1m.iloc[-1]
                exit_price = last_candle['close']
                exit_time = last_candle['datetime']
                if pending_exit:
                    exit_reason = pending_exit_reason + ' [Next Candle - Data End]'
                else:
                    exit_reason = 'Data End'
                exit_idx = len(df_1m) - 1

            # Розрахунок результату
            if direction == 'long':
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100

            hold_time = exit_time - actual_entry_time
            hold_time_hours = hold_time.total_seconds() / 3600

            # Збір історії RSI
            rsi_history_data = self.collect_rsi_history(df_1m, entry_idx, exit_idx)

            result = {
                **signal,
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'actual_entry_time': actual_entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'actual_entry_price': round(entry_price, 6),
                'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_price': round(exit_price, 6),
                'pnl_percent': round(pnl_percent, 3),
                'hold_time_hours': round(hold_time_hours, 2),
                'exit_reason': exit_reason,
                'sma_trend_values': sma_trend_values,
                'status': 'profit' if pnl_percent > 0 else 'loss',
                'rsi_history': rsi_history_data,
                'sma_trend_analysis': trend_info if not df_5m.empty else None
            }

            # Логування результату
            if rsi_history_data.get('rsi_history'):
                rsi_history_str = ' -> '.join([str(rsi) for rsi in rsi_history_data['rsi_history']])
                logger.info(
                    f"✓ {pair} {direction.upper()}: Signal: {signal_time.strftime('%H:%M:%S')} → Exit: {exit_time.strftime('%H:%M:%S')}, "
                    f"PnL: {pnl_percent:.2f}%, Hold: {hold_time_hours:.1f}h")
                logger.info(f"  Exit Reason: {exit_reason}")
                logger.info(f"📊 RSI History: {rsi_history_str}")
                logger.info(f"📊 SMA Values (5m): {sma_trend_values}")

                rsi_stats = rsi_history_data.get('rsi_stats', {})
                if rsi_stats:
                    logger.info(f"📈 RSI Stats: Min={rsi_stats.get('min', 0)}, "
                                f"Max={rsi_stats.get('max', 0)}, "
                                f"Avg={rsi_stats.get('avg', 0)}")
            else:
                logger.info(
                    f"✓ {pair} {direction.upper()}: Signal: {signal_time.strftime('%H:%M:%S')} → Exit: {exit_time.strftime('%H:%M:%S')}, "
                    f"PnL: {pnl_percent:.2f}%, Hold: {hold_time_hours:.1f}h")
                logger.info(f"  Exit Reason: {exit_reason}")
                logger.info(f"📊 SMA Values (5m): {sma_trend_values}")

            return result

        except Exception as e:
            logger.error(f"Помилка обробки сигналу {signal.get('pair', 'Unknown')}: {e}")
            return None

    async def process_signals_batch(self, signals: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Асинхронна обробка сигналів батчами з покращеною звітністю"""
        results = []
        total_signals = len(signals)

        logger.info(f"Початок обробки {total_signals} сигналів (батч розмір: {batch_size})")

        for i in range(0, total_signals, batch_size):
            batch = signals[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_signals + batch_size - 1) // batch_size

            logger.info(f"Обробка батчу {batch_num}/{total_batches} ({len(batch)} сигналів)")

            tasks = [self.process_single_signal(signal) for signal in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results = 0
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                    successful_results += 1
                elif isinstance(result, Exception):
                    logger.error(f"Помилка в батчі: {result}")

            logger.info(f"Батч {batch_num} оброблено: {successful_results} успішних з {len(batch)}")

            # Невелика пауза між батчами для стабільності
            if batch_num < total_batches:
                await asyncio.sleep(0.5)

        logger.info(f"Обробка завершена: {len(results)} успішних результатів з {total_signals}")
        return results

    async def save_results_async(self, results: List[Dict], filename: str):
        """Асинхронне збереження результатів"""
        if not results:
            logger.warning("Немає результатів для збереження")
            return

        try:
            fieldnames = [
                'pair', 'direction', 'signal_time', 'actual_entry_time', 'exit_time',
                'actual_entry_price', 'exit_price', 'pnl_percent', 'hold_time_hours',
                'exit_reason', 'sma_trend_values', 'status', 'rsi_1m', 'rsi_sma_1m', 'rsi_5m', 'rsi_sma_5m',
                'quality', 'confidence', 'comment',
                'rsi_history', 'rsi_stats', 'sma_trend_info',
                'filter_reason'
            ]

            content_lines = []
            content_lines.append(';'.join(fieldnames))

            for result in results:
                row_data = []
                for field in fieldnames:
                    value = result.get(field, '')

                    # Спеціальна обробка для складних полів
                    if field == 'rsi_history' and isinstance(value, dict):
                        if 'rsi_history' in value and value['rsi_history']:
                            value = ' -> '.join([str(rsi) for rsi in value['rsi_history']])
                        else:
                            value = ''
                    elif field == 'rsi_stats' and isinstance(value, dict):
                        if 'rsi_stats' in value and value['rsi_stats']:
                            stats = value['rsi_stats']
                            value = f"Min:{stats.get('min', 0)} Max:{stats.get('max', 0)} Avg:{stats.get('avg', 0)}"
                        else:
                            value = ''
                    elif field == 'sma_trend_info' and isinstance(value, dict):
                        if value and 'trend' in value:
                            value = f"{value['trend']} ({value.get('sma_changes', [])})"
                        else:
                            value = ''

                    row_data.append(str(value))

                content_lines.append(';'.join(row_data))

            content = '\n'.join(content_lines)

            async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
                await f.write(content)

            logger.info(f"Результати збережено в {filename}")
            await self.print_statistics_async(results)

        except Exception as e:
            logger.error(f"Помилка збереження результатів: {e}")

    async def print_statistics_async(self, results: List[Dict]):
        """Покращена статистика з детальним аналізом"""
        if not results:
            return

        # Розділяємо результати на оброблені та відфільтровані
        processed_results = [r for r in results if r.get('status') not in ['filtered_by_sma_trend']]
        filtered_results = [r for r in results if r.get('status') == 'filtered_by_sma_trend']

        total_trades = len(processed_results)
        total_filtered = len(filtered_results)

        print("\n" + "=" * 80)
        print("📊 СТАТИСТИКА ОБРОБКИ СИГНАЛІВ (З SMA ФІЛЬТРОМ) v2.3")
        print("=" * 80)

        if total_trades > 0:
            profitable_trades = len([r for r in processed_results if float(r.get('pnl_percent', 0)) > 0])
            losing_trades = total_trades - profitable_trades
            total_pnl = sum(float(r.get('pnl_percent', 0)) for r in processed_results)
            win_rate = (profitable_trades / total_trades) * 100
            avg_hold_time = sum(float(r.get('hold_time_hours', 0)) for r in processed_results) / total_trades

            # Додаткова статистика
            pnl_values = [float(r.get('pnl_percent', 0)) for r in processed_results]
            max_profit = max(pnl_values) if pnl_values else 0
            max_loss = min(pnl_values) if pnl_values else 0
            avg_profit = sum([p for p in pnl_values if p > 0]) / profitable_trades if profitable_trades > 0 else 0
            avg_loss = sum([p for p in pnl_values if p < 0]) / losing_trades if losing_trades > 0 else 0

            print(f"📈 Загальна статистика:")
            print(f"   Всього сигналів: {len(results)}")
            print(f"   Оброблених угод: {total_trades}")
            print(f"   Відфільтровано SMA: {total_filtered} ({total_filtered / len(results) * 100:.1f}%)")
            print(f"   Прибуткових: {profitable_trades} ({win_rate:.1f}%)")
            print(f"   Збиткових: {losing_trades} ({100 - win_rate:.1f}%)")
            print(f"   Сумарний PnL: {total_pnl:.2f}%")
            print(f"   Середній PnL: {total_pnl / total_trades:.2f}%")
            print(f"   Макс прибуток: {max_profit:.2f}%")
            print(f"   Макс збиток: {max_loss:.2f}%")
            print(f"   Середній прибуток: {avg_profit:.2f}%")
            print(f"   Середній збиток: {avg_loss:.2f}%")
            print(f"   Середній час утримання: {avg_hold_time:.1f} годин")

            # Статистика по напрямках
            direction_stats = {'long': {'count': 0, 'pnl': 0, 'wins': 0},
                               'short': {'count': 0, 'pnl': 0, 'wins': 0}}

            for result in processed_results:
                direction = result.get('direction', 'unknown').lower()
                pnl = float(result.get('pnl_percent', 0))

                if direction in direction_stats:
                    direction_stats[direction]['count'] += 1
                    direction_stats[direction]['pnl'] += pnl
                    if pnl > 0:
                        direction_stats[direction]['wins'] += 1

            print(f"\n📊 Статистика по напрямках:")
            for direction, stats in direction_stats.items():
                if stats['count'] > 0:
                    avg_pnl = stats['pnl'] / stats['count']
                    win_rate_dir = (stats['wins'] / stats['count']) * 100
                    print(
                        f"   {direction.upper()}: {stats['count']} угод, середній PnL: {avg_pnl:.2f}%, винрейт: {win_rate_dir:.1f}%")

            # Топ пари
            pair_stats = {}
            for result in processed_results:
                pair = result.get('pair', 'Unknown')
                pnl = float(result.get('pnl_percent', 0))

                if pair not in pair_stats:
                    pair_stats[pair] = {'count': 0, 'pnl': 0, 'wins': 0}
                pair_stats[pair]['count'] += 1
                pair_stats[pair]['pnl'] += pnl
                if pnl > 0:
                    pair_stats[pair]['wins'] += 1

            print(f"\n🏆 Топ-10 пар по прибутковості:")
            sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]
            for i, (pair, stats) in enumerate(sorted_pairs, 1):
                avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
                win_rate_pair = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
                print(
                    f"   {i:2d}. {pair}: {stats['pnl']:.2f}% ({stats['count']} угод, сер.: {avg_pnl:.2f}%, WR: {win_rate_pair:.0f}%)")

        # Статистика відфільтрованих сигналів
        if total_filtered > 0:
            print(f"\n🚫 Статистика відфільтрованих сигналів:")
            filter_stats = {'long_downtrend': 0, 'short_uptrend': 0}

            for result in filtered_results:
                direction = result.get('direction', 'unknown').lower()
                reason = result.get('filter_reason', '')

                if direction == 'long' and 'downtrend' in reason:
                    filter_stats['long_downtrend'] += 1
                elif direction == 'short' and 'uptrend' in reason:
                    filter_stats['short_uptrend'] += 1

            print(f"   LONG сигналів відфільтровано через SMA downtrend: {filter_stats['long_downtrend']}")
            print(f"   SHORT сигналів відфільтровано через SMA uptrend: {filter_stats['short_uptrend']}")

            if total_filtered > 0:
                effectiveness = (total_trades / (total_trades + total_filtered)) * 100
                print(f"   Ефективність фільтра: пропущено {100 - effectiveness:.1f}% сигналів")

        else:
            print("\n📊 ВСІ СИГНАЛИ БУЛИ ОБРОБЛЕНІ (жоден не відфільтровано)")

        print("=" * 80)


async def main():
    """Головна функція з покращеною обробкою помилок"""
    parser = argparse.ArgumentParser(description='Асинхронний процесор торгових сигналів з SMA фільтром v2.3')
    parser.add_argument('input_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='processed_signals_v23.csv', help='Файл для результатів')
    parser.add_argument('--batch-size', type=int, default=10, help='Розмір батчу для обробки')
    parser.add_argument('--max-hold', type=int, default=24, help='Макс час утримання (години)')
    parser.add_argument('--trend-periods', type=int, default=3, help='Кількість періодів для аналізу тренду SMA')
    parser.add_argument('--deviation-threshold', type=float, default=0.5, help='Поріг відхилення для визначення тренду')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Рівень логування')

    args = parser.parse_args()

    # Налаштування рівня логування
    logger.setLevel(getattr(logging, args.log_level))

    if not os.path.exists(args.input_file):
        print(f"❌ Файл {args.input_file} не знайдено")
        return

    config = {
        'max_hold_hours': args.max_hold,
        'batch_size': args.batch_size,
        'trend_periods': args.trend_periods,
        'deviation_threshold': args.deviation_threshold
    }

    print("🚀 АСИНХРОННИЙ ПРОЦЕСОР ТОРГОВИХ СИГНАЛІВ З SMA ФІЛЬТРОМ v2.3")
    print(f"📁 Вхідний файл: {args.input_file}")
    print(f"📊 Налаштування:")
    print(f"   Розмір батчу: {args.batch_size}")
    print(f"   Максимальний час утримання: {args.max_hold} годин")
    print(f"   SMA тренд періодів: {args.trend_periods}")
    print(f"   Поріг відхилення SMA: {args.deviation_threshold}")
    print(f"   Рівень логування: {args.log_level}")
    print("-" * 60)

    processor = None
    try:
        processor = SignalProcessor(config)

        signals = await processor.read_signals_csv(args.input_file)
        if not signals:
            print("❌ Не вдалося прочитати сигнали з файлу")
            return

        print(f"✅ Завантажено {len(signals)} сигналів")

        open_signals = [s for s in signals if s.get('status', '').lower() == 'open']
        print(f"📌 Знайдено {len(open_signals)} відкритих сигналів для обробки")

        if not open_signals:
            print("⚠️ Немає відкритих сигналів для обробки")
            return

        results = await processor.process_signals_batch(open_signals, args.batch_size)

        if results:
            await processor.save_results_async(results, args.output)
            print(f"\n✅ Обробка завершена успішно!")
            print(f"📄 Результати збережено в {args.output}")
            print(f"💡 Використовуйте --log-level DEBUG для детального логування")
        else:
            print("❌ Не вдалося отримати результати обробки")

    except KeyboardInterrupt:
        print("\n⏹️ Обробка перервана користувачем")
    except Exception as e:
        logger.error(f"Критична помилка: {e}")
        print(f"❌ Критична помилка: {e}")
    finally:
        if processor and hasattr(processor, 'executor'):
            processor.executor.shutdown(wait=True)
            logger.info("ThreadPoolExecutor закрито")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Програма перервана користувачем")
    except Exception as e:
        print(f"❌ Помилка запуску: {e}")