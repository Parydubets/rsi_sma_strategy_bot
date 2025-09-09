#!/usr/bin/env python3
"""
Модифікований аналізатор з розумним буфером та SMA_RSI трендовим фільтром
"""

import ccxt
import pandas as pd
import asyncio
import csv
import json
import os
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import logging
from logging.handlers import RotatingFileHandler
import argparse
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union


# Налаштування логування з записом у файл
def setup_logging():
    """Налаштування логування з записом у файл enhanced_analyzer.log"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Очищення попередніх handlers
    logger.handlers.clear()

    # Форматування
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler з ротацією
    file_handler = RotatingFileHandler(
        'enhanced_analyzer.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Додавання handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


class TrendSignalAnalyzer:
    def __init__(self):
        self.exchange = ccxt.bybit({'enableRateLimit': True, 'rateLimit': 100})
        self.cache = {}
        self.pair_data_cache = {}  # Кеш для даних по парах

    @staticmethod
    def create_optimized_default_config():
        """Створення оптимізованого стандартного конфігу"""
        default_config = {
            "trading_parameters": {
                "stop_loss": 0.018,
                "take_profit": 0.035,
                "max_hold_hours": 24,
                "use_trailing_stop": True
            },
            "rsi_parameters": {
                "long_exit_zone": 60,
                "short_exit_zone": 40,
                "long_extreme_exit": 75,
                "short_extreme_exit": 25
            },
            "filters": {
                "use_filters": True,
                "min_filters_required": 3,
                "use_trend_filter": True,
                "use_adx_filter": True,
                "use_volatility_filter": True,
                "use_volume_filter": True,
                "use_momentum_filter": True,
                "use_divergence_filter": False  # Новий фільтр
            },
            # ОПТИМІЗОВАНІ ПАРАМЕТРИ для SMA_RSI трендового фільтру
            "trend_filter": {
                "medium_timeframe": "15m",
                "lookback_candles": 12,  # Збільшено для кращої точності
                "trend_strength_threshold": 0.4,  # Знижено для більшої чутливості
                "sideways_threshold": 0.8,
                "recent_cross_candles": 6,  # Збільшено діапазон пошуку
                "cross_zone_buffer": 8,  # Збільшено буфер зон
                "enable_sideways_trading": True,
                "enable_recent_cross": True,
                # НОВІ параметри для оптимізації
                "volatility_adaptation": True,  # Адаптація до волатільності
                "confidence_weighting": True,  # Врахування достовірності
                "multi_layer_analysis": True  # Багатошаровий аналіз
            },
            "filter_thresholds": {
                "min_adx": 15,  # Знижено для більшої гнучкості
                "min_volatility": 0.12,
                "max_volatility": 3.0,  # Збільшено для високоволатильних ринків
                "min_volume_ratio": 0.7  # Знижено для кращого покриття
            },
            "smart_exit": {  # НОВА секція для розумного закриття
                "enable_smart_exit": True,
                "min_trend_strength_for_hold": 30,
                "min_trend_confidence_for_hold": 40,
                "max_trend_holds": 3,
                "critical_rsi_high": 75,
                "critical_rsi_low": 25,
                "sideways_max_hold_hours": 12,
                "force_exit_on_strong_reversal": True
            },
            "timeframe_settings": {
                "primary_timeframe": "5m",
                "secondary_timeframes": ["15m"],
                "trend_weights": {
                    "5m": 0.4,  # Знижено вагу короткого ТФ
                    "15m": 0.6,  # Збільшено вагу середнього ТФ
                }
            },
            "technical_indicators": {
                "5m": {
                    "rsi_period": 14,
                    "rsi_sma_period": 14,
                    "sma_fast": 20,
                    "sma_slow": 50,
                    "adx_period": 14,
                    "ema_fast": 12,
                    "ema_slow": 26,
                    "atr_period": 14,
                    "bb_period": 20,
                    "volume_window": 20
                },
                "15m": {
                    "rsi_period": 14,
                    "rsi_sma_period": 14,  # Збільшено для кращого згладжування
                    "sma_fast": 20,
                    "sma_slow": 50,
                    "adx_period": 14,
                    "ema_fast": 21,
                    "ema_slow": 50,
                    "atr_period": 14,
                    "bb_period": 20,
                    "volume_window": 40
                },
                "1h": {
                    "rsi_period": 14,
                    "rsi_sma_period": 14,
                    "sma_fast": 50,
                    "sma_slow": 200,
                    "adx_period": 14,
                    "ema_fast": 50,
                    "ema_slow": 200,
                    "atr_period": 20,
                    "bb_period": 20,
                    "volume_window": 50
                }
            },
            "data_management": {
                "buffer_hours_after": 240,
                "min_buffer_hours": 48,
                "reload_threshold_hours": 24,
                "smart_buffer_multiplier": 1.5,
                "max_buffer_hours": 720
            }
        }

        return default_config

    @staticmethod
    def load_config(config_file='enhanced_analyzer_config.json'):
        """Завантаження конфігурації з файлу"""
        if not os.path.exists(config_file):
            logger.info(f"Конфіг файл {config_file} не знайдено, створюємо стандартний")
            config = TrendSignalAnalyzer.create_optimized_default_config()

            # Зберігаємо стандартний конфіг у файл
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                logger.info(f"Створено стандартний конфіг файл: {config_file}")
            except Exception as e:
                logger.warning(f"Не вдалося створити конфіг файл {config_file}: {e}")

            return config

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Завантажено конфіг з {config_file}")
            return config
        except Exception as e:
            logger.error(f"Помилка читання конфігу {config_file}: {e}")
            logger.info("Використовується стандартний конфіг")
            return TrendSignalAnalyzer.create_optimized_default_config()

    def calculate_optimal_buffer(self, signals: list, config: dict) -> dict:
        """
        НОВА ФУНКЦІЯ: Розрахунок оптимального буфера для кожної пари
        на основі аналізу всіх сигналів
        """
        logger.info("Розрахунок оптимального буфера для кожної пари...")

        data_config = config.get('data_management', {})
        base_buffer = data_config.get('buffer_hours_after', 240)
        max_hold_hours = config.get('trading_parameters', {}).get('max_hold_hours', 24)
        max_buffer = data_config.get('max_buffer_hours', 720)

        pair_requirements = {}

        # Групуємо сигнали по парах
        signals_by_pair = {}
        for signal in signals:
            signal_time = self.parse_time(signal['time'])
            pair = signal['pair']
            if pair not in signals_by_pair:
                signals_by_pair[pair] = []
            signals_by_pair[pair].append((signal, signal_time))

        # Розраховуємо потрібний буфер для кожної пари
        for pair, pair_signals in signals_by_pair.items():
            # Сортуємо сигнали по часу
            pair_signals.sort(key=lambda x: x[1])

            if not pair_signals:
                continue

            first_signal_time = pair_signals[0][1]
            last_signal_time = pair_signals[-1][1]

            # Розраховуємо необхідний діапазон
            # Потрібні дані від першого сигналу до останнього + max_hold_hours
            required_start = first_signal_time - timedelta(hours=120)  # 5 днів до першого сигналу
            required_end = last_signal_time + timedelta(hours=max_hold_hours + 48)  # + запас

            total_hours_needed = (required_end - required_start).total_seconds() / 3600

            # Визначаємо оптимальний буфер
            optimal_buffer = min(max(int(total_hours_needed), base_buffer), max_buffer)

            pair_requirements[pair] = {
                'start_time': required_start,
                'end_time': required_end,
                'buffer_hours': optimal_buffer,
                'total_signals': len(pair_signals),
                'time_span_hours': int((last_signal_time - first_signal_time).total_seconds() / 3600)
            }

            logger.info(
                f"Пара {pair}: {len(pair_signals)} сигналів, діапазон {pair_requirements[pair]['time_span_hours']}h, "
                f"буфер {optimal_buffer}h")

        logger.info(f"Розраховано оптимальні буфери для {len(pair_requirements)} пар")
        return pair_requirements

    def read_signals_csv(self, filename: str) -> list:
        """Читання CSV з автовизначенням формату"""
        logger.info(f"Читання сигналів з файлу: {filename}")

        for encoding in ['utf-8', 'cp1251', 'windows-1251']:
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(filename, encoding=encoding, sep=sep)
                    if len(df.columns) > 3:
                        signals = self._parse_signals(df)
                        logger.info(f"Успішно прочитано файл з encoding={encoding}, separator='{sep}'")
                        return signals
                except Exception as e:
                    logger.debug(f"Спроба читання з encoding={encoding}, sep='{sep}' неуспішна: {e}")
                    continue

        logger.error(f"Не вдалося прочитати {filename} з жодним форматом")
        return []

    def _parse_signals(self, df: pd.DataFrame) -> list:
        """Парсинг сигналів з DataFrame"""
        signals = []
        mapping = {
            'time': ['Time', 'Час', 'DateTime'],
            'pair': ['Pair', 'Symbol', 'Пара'],
            'direction': ['Direction', 'Напрямок', 'Type'],
            'rsi': ['RSI']
        }

        columns = {}
        for key, names in mapping.items():
            for col in df.columns:
                if any(name.lower() in col.lower() for name in names):
                    columns[key] = col
                    break

        logger.info(f"Виявлені колонки: {columns}")

        for idx, row in df.iterrows():
            try:
                signal = {
                    'time': str(row.get(columns.get('time', ''), '')).strip(),
                    'pair': str(row.get(columns.get('pair', ''), '')).strip().replace('/', ''),
                    'direction': str(row.get(columns.get('direction', ''), '')).strip(),
                    'rsi': float(str(row.get(columns.get('rsi', ''), 0)).replace(',', '.'))
                }
                if signal['pair'] and signal['direction'] and signal['rsi'] > 0:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Помилка парсингу рядка {idx}: {e}")
                continue

        logger.info(f"Завантажено {len(signals)} валідних сигналів з {len(df)} рядків")
        return signals

    def parse_time(self, time_str: str) -> datetime:
        """Парсинг часу"""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d.%m.%Y %H:%M',
            '%d/%m/%Y %H:%M'
        ]

        clean_time = time_str.strip()

        for fmt in formats:
            try:
                parsed_time = datetime.strptime(clean_time, fmt)
                logger.debug(f"Успішно парсили час: '{time_str}' -> {parsed_time}")
                return parsed_time
            except ValueError:
                continue

        logger.warning(f"Не вдалося парсити час: '{time_str}'. Використовується поточний час.")
        return datetime.now()

    async def fetch_multi_data_smart_buffer(self, symbol: str, pair_requirements: dict,
                                            config: dict = None) -> dict:
        """
        МОДИФІКОВАНА ФУНКЦІЯ: Завантаження даних з розумним буфером на основі всіх сигналів пари
        """
        if config is None:
            config = self.create_optimized_default_config()

        if symbol not in pair_requirements:
            # Фоллбек до старого методу
            logger.warning(f"Немає інформації про буфер для {symbol}, використовуємо стандартний підхід")
            return await self.fetch_multi_data_fallback(symbol, config)

        req = pair_requirements[symbol]
        logger.info(f"Завантаження даних для {symbol} з розумним буфером: "
                    f"{req['buffer_hours']}h для {req['total_signals']} сигналів")

        # Розрахунок періодів для різних таймфреймів з урахуванням оптимального буфера
        buffer_hours = req['buffer_hours']
        start_time = req['start_time']
        end_time = req['end_time']

        timeframes = {
            '5m': {'start': start_time, 'end': end_time},
            '15m': {'start': start_time, 'end': end_time}
        }

        data = {}
        logger.debug(f"Завантаження розумного буфера для {symbol}")

        for tf, tf_config in timeframes.items():
            cache_key = f"{symbol}_{tf}_{tf_config['start']}_{tf_config['end']}_smart"

            if cache_key in self.cache:
                data[tf] = self.cache[cache_key]
                logger.debug(f"Використано кеш для {symbol} {tf}")
                continue

            df = await self._fetch_data(symbol, tf, tf_config['start'], tf_config['end'])
            if not df.empty:
                df = self._calc_indicators(df, tf)
                data[tf] = df
                self.cache[cache_key] = df

                # Перевіряємо покриття
                data_hours = (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
                logger.info(f"Завантажено {len(df)} свічок для {symbol} {tf}, покриття: {data_hours:.0f}h")

                await asyncio.sleep(0.05)
            else:
                logger.warning(f"Немає даних для {symbol} {tf}")

        return data

    async def fetch_multi_data_fallback(self, symbol: str, config: dict) -> dict:
        """Фоллбек метод завантаження (старий підхід)"""
        data_config = config.get('data_management', {})
        buffer_after = data_config.get('buffer_hours_after', 240)

        timeframes = {
            '5m': {'hours_before': 120, 'hours_after': buffer_after},
            '15m': {'hours_before': 360, 'hours_after': buffer_after // 2}
        }

        current_time = datetime.now()
        data = {}

        for tf, tf_config in timeframes.items():
            start = current_time - timedelta(hours=tf_config['hours_before'])
            end = current_time + timedelta(hours=tf_config['hours_after'])

            df = await self._fetch_data(symbol, tf, start, end)
            if not df.empty:
                df = self._calc_indicators(df, tf)
                data[tf] = df
                await asyncio.sleep(0.05)

        return data

    async def fetch_multi_data(self, symbol: str, signal_time: datetime,
                               config: dict = None, pair_requirements: dict = None) -> dict:
        """
        МОДИФІКОВАНА ОСНОВНА ФУНКЦІЯ: Використовує розумний буфер якщо доступний
        """
        if pair_requirements and symbol in pair_requirements:
            return await self.fetch_multi_data_smart_buffer(symbol, pair_requirements, config)
        else:
            # Стара логіка як фоллбек
            return await self.fetch_multi_data_original(symbol, signal_time, config)

    async def fetch_multi_data_original(self, symbol: str, signal_time: datetime,
                                        config: dict = None) -> dict:
        """Оригінальна логіка завантаження"""
        if config is None:
            config = self.create_optimized_default_config()

        data_config = config.get('data_management', {})
        buffer_after = data_config.get('buffer_hours_after', 240)

        timeframes = {
            '5m': {'hours_before': 120, 'hours_after': buffer_after},
            '15m': {'hours_before': 360, 'hours_after': buffer_after // 2}
        }

        data = {}
        logger.debug(f"Завантаження даних для {symbol}, час сигналу: {signal_time}")

        for tf, tf_config in timeframes.items():
            start = signal_time - timedelta(hours=tf_config['hours_before'])
            end = signal_time + timedelta(hours=tf_config['hours_after'])

            cache_key = f"{symbol}_{tf}_{start}_{end}"
            if cache_key in self.cache:
                data[tf] = self.cache[cache_key]
                logger.debug(f"Використано кеш для {symbol} {tf}")
                continue

            df = await self._fetch_data(symbol, tf, start, end)
            if not df.empty:
                df = self._calc_indicators(df, tf)
                data[tf] = df
                self.cache[cache_key] = df
                logger.debug(f"Завантажено {len(df)} свічок для {symbol} {tf}")
                await asyncio.sleep(0.05)
            else:
                logger.warning(f"Немає даних для {symbol} {tf}")

        return data

    async def _fetch_data(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Завантаження OHLCV даних"""
        try:
            api_symbol = str(symbol).replace("USDT", "/USDT")
            since = int(start.timestamp() * 1000)
            until = int(end.timestamp() * 1000)
            all_data = []

            logger.debug(f"Завантаження {api_symbol} {tf} з {start} до {end}")

            current = since
            requests_count = 0
            while current < until:
                ohlcv = self.exchange.fetch_ohlcv(api_symbol, tf, since=current, limit=1000)
                requests_count += 1

                if not ohlcv:
                    logger.debug(f"Немає більше даних для {api_symbol} {tf}")
                    break

                all_data.extend(ohlcv)
                current = ohlcv[-1][0] + 1
                await asyncio.sleep(0.05)

            if all_data:
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                result_df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
                logger.debug(f"Завантажено {len(result_df)} свічок для {api_symbol} {tf} ({requests_count} запитів)")
                return result_df

        except Exception as e:
            logger.error(f"Помилка завантаження {symbol} {tf}: {e}")

        return pd.DataFrame()

    def _calc_indicators(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Розрахунок технічних індикаторів для різних таймфреймів"""
        if len(df) < 50:
            logger.warning(f"Недостатньо даних для розрахунку індикаторів {tf}: {len(df)} свічок")
            return df

        try:
            logger.debug(f"Розрахунок індикаторів для {tf}")

            # Адаптовані параметри для різних таймфреймів
            if tf == '5m':
                periods = {'rsi': 14, 'rsi_sma': 10, 'sma_fast': 20, 'sma_slow': 50, 'adx': 14}
                ema_fast, ema_slow = 12, 26
                macd_fast, macd_slow = 12, 26
            elif tf == '15m':
                periods = {'rsi': 14, 'rsi_sma': 10, 'sma_fast': 20, 'sma_slow': 50, 'adx': 14}
                ema_fast, ema_slow = 21, 50
                macd_fast, macd_slow = 12, 26

            # RSI та його згладжена лінія
            df['rsi'] = RSIIndicator(df['close'], periods['rsi']).rsi()
            df['rsi_sma'] = df['rsi'].rolling(periods['rsi_sma']).mean()

            # MACD
            macd = MACD(df['close'], macd_fast, macd_slow)
            df['macd_diff'] = macd.macd_diff()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff() - macd.macd_signal()

            # ADX для визначення сили тренду
            adx = ADXIndicator(df['high'], df['low'], df['close'], periods['adx'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()

            # Трендові індикатори
            df['ema_fast'] = EMAIndicator(df['close'], ema_fast).ema_indicator()
            df['ema_slow'] = EMAIndicator(df['close'], ema_slow).ema_indicator()
            df['sma_fast'] = SMAIndicator(df['close'], periods['sma_fast']).sma_indicator()
            df['sma_slow'] = SMAIndicator(df['close'], periods['sma_slow']).sma_indicator()

            # ATR для волатільності
            atr_period = 14 if tf == '5m' else 20
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], atr_period).average_true_range()
            df['atr_pct'] = (df['atr'] / df['close']) * 100

            # Bollinger Bands
            bb_period = 20
            bb = BollingerBands(df['close'], bb_period, 2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['close']) * 100

            # Об'єм
            volume_window = 20 if tf == '5m' else (40 if tf == '15m' else 50)
            df['volume_sma'] = df['volume'].rolling(volume_window).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Трендові сигнали
            df['trend_ema'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
            df['trend_sma'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)

            # Momentum для 5m
            if tf == '5m':
                df['momentum'] = df['close'].pct_change(periods=5) * 100
                df['momentum_sma'] = df['momentum'].rolling(5).mean()

            logger.debug(f"Індикатори розраховано для {tf}")

        except Exception as e:
            logger.error(f"Помилка розрахунку індикаторів для {tf}: {e}")

        return df

    def analyze_sma_rsi_trend_optimized(self, df: pd.DataFrame, signal_time: datetime, config: dict) -> dict:
        """
        ОПТИМІЗОВАНА ФУНКЦІЯ: Аналіз тренду SMA_RSI з адаптивними параметрами

        Автоматично визначає оптимальну кількість свічок залежно від волатільності
        та використовує багатошарову оцінку тренду
        """

        # Знаходимо найближчу свічку до часу сигналу
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        current_idx = df['time_diff'].idxmin()

        if pd.isna(current_idx):
            logger.warning("Не можна знайти відповідну свічку для аналізу тренду")
            return {'trend_direction': 'unknown', 'trend_strength': 0, 'trend_confidence': 0}

        # Базові параметри
        trend_config = config.get('trend_filter', {})
        base_lookback = trend_config.get('lookback_candles', 10)

        # Розрахунок волатільності для адаптації параметрів
        if current_idx >= 20:
            recent_data = df.iloc[max(0, current_idx - 20):current_idx + 1]
            if 'atr_pct' in recent_data.columns:
                avg_volatility = recent_data['atr_pct'].mean()
            else:
                # Альтернативний розрахунок волатільності
                returns = recent_data['close'].pct_change().abs()
                avg_volatility = returns.mean() * 100
        else:
            avg_volatility = 1.0

        # Адаптивні параметри на основі волатільності
        if avg_volatility < 0.5:  # Низька волатільність
            optimal_lookback = min(base_lookback * 2, 25)  # Більше свічок
            trend_sensitivity = 0.3  # Більш чутливий до слабких трендів
            strength_multiplier = 1.5
        elif avg_volatility > 2.0:  # Висока волатільність
            optimal_lookback = max(base_lookback // 2, 5)  # Менше свічок
            trend_sensitivity = 0.8  # Менш чутливий до шуму
            strength_multiplier = 0.8
        else:  # Середня волатільність
            optimal_lookback = base_lookback
            trend_sensitivity = 0.5
            strength_multiplier = 1.0

        # Діапазон для аналізу
        start_idx = max(0, current_idx - optimal_lookback)
        end_idx = min(len(df), current_idx + 1)

        if end_idx - start_idx < 5:
            logger.warning("Недостатньо даних для оптимізованого аналізу тренду")
            return {'trend_direction': 'unknown', 'trend_strength': 0, 'trend_confidence': 0}

        analysis_data = df.iloc[start_idx:end_idx].copy()

        if 'rsi_sma' not in analysis_data.columns:
            logger.warning("Відсутній індикатор rsi_sma")
            return {'trend_direction': 'unknown', 'trend_strength': 0, 'trend_confidence': 0}

        # БАГАТОШАРОВИЙ АНАЛІЗ ТРЕНДУ

        # 1. Основний тренд SMA_RSI (лінійна регресія)
        rsi_sma_values = analysis_data['rsi_sma'].dropna()
        if len(rsi_sma_values) < 3:
            return {'trend_direction': 'unknown', 'trend_strength': 0, 'trend_confidence': 0}

        x = np.arange(len(rsi_sma_values))
        slope, intercept = np.polyfit(x, rsi_sma_values, 1)

        # R² для оцінки якості тренду
        y_pred = slope * x + intercept
        ss_res = np.sum((rsi_sma_values - y_pred) ** 2)
        ss_tot = np.sum((rsi_sma_values - np.mean(rsi_sma_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # 2. Короткостроковий тренд (останні 5 свічок)
        if len(rsi_sma_values) >= 5:
            short_x = np.arange(5)
            short_slope, _ = np.polyfit(short_x, rsi_sma_values[-5:], 1)
        else:
            short_slope = slope

        # 3. Momentum аналіз (зміна швидкості)
        if len(rsi_sma_values) >= 6:
            first_half = np.mean(rsi_sma_values[:len(rsi_sma_values) // 2])
            second_half = np.mean(rsi_sma_values[len(rsi_sma_values) // 2:])
            momentum_direction = 1 if second_half > first_half else -1
        else:
            momentum_direction = 1 if slope > 0 else -1

        # 4. Визначення основного напрямку тренду
        primary_trend = 0
        if slope > trend_sensitivity:
            primary_trend = 1  # Вгору
        elif slope < -trend_sensitivity:
            primary_trend = -1  # Вниз

        # 5. Зважена оцінка тренду
        # Основний тренд має вагу 50%, короткостроковий 30%, momentum 20%
        trend_score = (slope * 0.5 + short_slope * 0.3 + momentum_direction * 0.2)

        # Визначення фінального напрямку з урахуванням всіх факторів
        if trend_score > trend_sensitivity:
            final_direction = 'up'
        elif trend_score < -trend_sensitivity:
            final_direction = 'down'
        else:
            final_direction = 'sideways'

        # Розрахунок сили тренду (0-100)
        base_strength = min(abs(trend_score) * 10 * strength_multiplier, 100)

        # Коригування сили на основі R²
        confidence_adjusted_strength = base_strength * (0.5 + r_squared * 0.5)

        # Розрахунок достовірності тренду
        trend_confidence = min(r_squared * 100 + (abs(slope) * 10), 100)

        # Пошук нещодавніх перетинів (оптимізовано)
        recent_cross_info = self._find_recent_crosses_optimized(df, current_idx, config, optimal_lookback)

        result = {
            'trend_direction': final_direction,
            'trend_strength': confidence_adjusted_strength,
            'trend_confidence': trend_confidence,
            'primary_slope': slope,
            'short_term_slope': short_slope,
            'r_squared': r_squared,
            'volatility_adjusted': True,
            'optimal_lookback': optimal_lookback,
            'recent_cross_info': recent_cross_info,
            'trend_score': trend_score,
            'momentum_direction': momentum_direction
        }

        logger.debug(f"Оптимізований SMA_RSI тренд: напрямок={final_direction}, "
                     f"сила={confidence_adjusted_strength:.1f}, достовірність={trend_confidence:.1f}, "
                     f"lookback={optimal_lookback}, R²={r_squared:.3f}")

        return result

    def analyze_signals_sync(self, signals_data: List[Dict], config: Dict) -> List[Dict]:
        """
        Synchronous version of signal analysis for use in multiprocessing
        This method should be added to your TrendSignalAnalyzer class
        """
        try:
            # If you have an async analyze_signals method, you can run it synchronously
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Assuming you have an async method like analyze_signals
                result = loop.run_until_complete(self.analyze_signals(signals_data, config))
                return result
            finally:
                loop.close()
        except Exception as e:
            print(f"Error in analyze_signals_sync: {e}")
            return []

    def _find_recent_crosses_optimized(self, df: pd.DataFrame, current_idx: int,
                                       config: dict, lookback_period: int) -> dict:
        """
        ОПТИМІЗОВАНА ФУНКЦІЯ: Пошук нещодавніх перетинів RSI та RSI_SMA
        """
        trend_config = config.get('trend_filter', {})
        recent_cross_candles = min(trend_config.get('recent_cross_candles', 5), lookback_period)
        cross_zone_buffer = trend_config.get('cross_zone_buffer', 5)

        rsi_params = config.get('rsi_parameters', {})
        long_zone = rsi_params.get('long_exit_zone', 40)
        short_zone = rsi_params.get('short_exit_zone', 60)

        # Розширені зони з урахуванням волатільності
        if 'atr_pct' in df.columns and current_idx > 0:
            current_vol = df.iloc[current_idx].get('atr_pct', 1.0)
            vol_buffer = min(current_vol * 2, 10)  # Максимум 10 пунктів RSI
        else:
            vol_buffer = cross_zone_buffer

        recent_start = max(0, current_idx - recent_cross_candles)
        recent_data = df.iloc[recent_start:current_idx + 1]

        crosses = []

        for i in range(1, len(recent_data)):
            prev_candle = recent_data.iloc[i - 1]
            curr_candle = recent_data.iloc[i]

            prev_rsi = prev_candle.get('rsi', 0)
            curr_rsi = curr_candle.get('rsi', 0)
            prev_rsi_sma = prev_candle.get('rsi_sma', 0)
            curr_rsi_sma = curr_candle.get('rsi_sma', 0)

            # Перетин вниз
            if prev_rsi > prev_rsi_sma and curr_rsi <= curr_rsi_sma:
                cross_strength = abs(prev_rsi - prev_rsi_sma)  # Сила перетину
                in_short_zone = prev_rsi >= short_zone - vol_buffer

                crosses.append({
                    'type': 'cross_down',
                    'candles_ago': current_idx - (recent_start + i),
                    'rsi_level': curr_rsi,
                    'cross_strength': cross_strength,
                    'in_short_zone': in_short_zone,
                    'zone_relevance': max(0, prev_rsi - short_zone) if in_short_zone else 0
                })

            # Перетин вгору
            elif prev_rsi < prev_rsi_sma and curr_rsi >= curr_rsi_sma:
                cross_strength = abs(prev_rsi - prev_rsi_sma)
                in_long_zone = prev_rsi <= long_zone + vol_buffer

                crosses.append({
                    'type': 'cross_up',
                    'candles_ago': current_idx - (recent_start + i),
                    'rsi_level': curr_rsi,
                    'cross_strength': cross_strength,
                    'in_long_zone': in_long_zone,
                    'zone_relevance': max(0, long_zone - prev_rsi) if in_long_zone else 0
                })

        # Повертаємо найсильніший та найсвіжіший перетин
        if crosses:
            # Сортуємо за силою перетину та свіжістю
            crosses.sort(key=lambda x: (x['cross_strength'], -x['candles_ago']), reverse=True)
            return crosses[0]  # Найкращий перетин

        return None

    def check_smart_exit_conditions(self, multi_data: dict, current_time: datetime,
                                    direction: str, entry_time: datetime, config: dict) -> tuple:
        """
        НОВА ФУНКЦІЯ: Розумна перевірка умов виходу з урахуванням тренду

        Логіка:
        1. Перевіряє звичайний зворотний RSI сигнал
        2. Якщо є зворотний сигнал, але тренд на середньому ТФ ще зберігається в бік позиції:
           - НЕ закриває позицію
           - Очікує підтвердження зміни тренду або новий сигнал
        3. Закриває тільки при справжній зміні тренду або сильному зворотному сигналі
        """

        if '5m' not in multi_data:
            return False, ""

        # Отримуємо трендовий аналіз на середньому ТФ
        trend_config = config.get('trend_filter', {})
        medium_tf = trend_config.get('medium_timeframe', '15m')

        if medium_tf not in multi_data:
            # Фоллбек до звичайної логіки
            return self.check_reverse_signal_exit(multi_data, current_time, direction, config)

        # Аналіз поточного тренду
        trend_analysis = self.analyze_sma_rsi_trend_optimized(
            multi_data[medium_tf], current_time, config
        )

        # Перевірка базового зворотного сигналу на 5m
        base_reverse_signal, base_reason = self.check_reverse_signal_exit(
            multi_data, current_time, direction, config
        )

        if not base_reverse_signal:
            return False, ""

        # Якщо є базовий зворотний сигнал, аналізуємо тренд
        is_long = direction.lower() == 'long'
        trend_direction = trend_analysis['trend_direction']
        trend_strength = trend_analysis['trend_strength']
        trend_confidence = trend_analysis['trend_confidence']

        # Мінімальні пороги для блокування виходу
        min_trend_strength = 30  # Мінімальна сила тренду для блокування
        min_confidence = 40  # Мінімальна достовірність тренду

        logger.debug(f"Розумна перевірка виходу: {direction}, базовий сигнал={base_reverse_signal}, "
                     f"тренд={trend_direction} (сила={trend_strength:.1f}, достовірність={trend_confidence:.1f})")

        # ЛОГІКА ДЛЯ LONG ПОЗИЦІЇ
        if is_long:
            # Якщо тренд все ще вгору і достатньо сильний - НЕ закриваємо
            if (trend_direction == 'up' and
                    trend_strength >= min_trend_strength and
                    trend_confidence >= min_confidence):

                logger.info(f"LONG позицію НЕ закрито: зберігається висхідний тренд "
                            f"(сила={trend_strength:.1f}, достовірність={trend_confidence:.1f})")
                return False, f"Trend Still UP (str:{trend_strength:.0f}, conf:{trend_confidence:.0f})"

            # Закриваємо якщо:
            # 1. Тренд змінився на спадаючий
            # 2. Тренд слабкий або недостовірний
            # 3. Тренд боковий, але сигнал сильний
            elif trend_direction == 'down':
                logger.info(f"LONG закрито: тренд змінився на спадаючий")
                return True, f"Trend Change to DOWN + {base_reason}"
            elif trend_strength < min_trend_strength or trend_confidence < min_confidence:
                logger.info(
                    f"LONG закрито: тренд ослаб (сила={trend_strength:.1f}, достовірність={trend_confidence:.1f})")
                return True, f"Weak Trend + {base_reason}"
            else:
                # Боковий рух - додаткова перевірка сили сигналу
                df = multi_data['5m']
                df['time_diff'] = (df['datetime'] - current_time).abs()
                idx = df['time_diff'].idxmin()

                if not pd.isna(idx) and idx > 0:
                    curr_rsi = df.iloc[idx].get('rsi', 50)
                    # Якщо RSI дуже високий (>70) - закриваємо навіть при боковику
                    if curr_rsi > 70:
                        logger.info(f"LONG закрито: RSI дуже високий ({curr_rsi:.1f}) навіть при боковому тренді")
                        return True, f"High RSI({curr_rsi:.0f}) + {base_reason}"

                logger.info(f"LONG НЕ закрито: боковий тренд, RSI не критичний")
                return False, f"Sideways Trend, RSI OK"

        # ЛОГІКА ДЛЯ SHORT ПОЗИЦІЇ
        else:
            # Якщо тренд все ще вниз і достатньо сильний - НЕ закриваємо
            if (trend_direction == 'down' and
                    trend_strength >= min_trend_strength and
                    trend_confidence >= min_confidence):

                logger.info(f"SHORT позицію НЕ закрито: зберігається спадний тренд "
                            f"(сила={trend_strength:.1f}, достовірність={trend_confidence:.1f})")
                return False, f"Trend Still DOWN (str:{trend_strength:.0f}, conf:{trend_confidence:.0f})"

            # Закриваємо якщо тренд змінився або ослаб
            elif trend_direction == 'up':
                logger.info(f"SHORT закрито: тренд змінився на висхідний")
                return True, f"Trend Change to UP + {base_reason}"
            elif trend_strength < min_trend_strength or trend_confidence < min_confidence:
                logger.info(
                    f"SHORT закрито: тренд ослаб (сила={trend_strength:.1f}, достовірність={trend_confidence:.1f})")
                return True, f"Weak Trend + {base_reason}"
            else:
                # Боковий рух - додаткова перевірка
                df = multi_data['5m']
                df['time_diff'] = (df['datetime'] - current_time).abs()
                idx = df['time_diff'].idxmin()

                if not pd.isna(idx) and idx > 0:
                    curr_rsi = df.iloc[idx].get('rsi', 50)
                    # Якщо RSI дуже низький (<30) - закриваємо навіть при боковику
                    if curr_rsi < 30:
                        logger.info(f"SHORT закрито: RSI дуже низький ({curr_rsi:.1f}) навіть при боковому тренді")
                        return True, f"Low RSI({curr_rsi:.0f}) + {base_reason}"

                logger.info(f"SHORT НЕ закрито: боковий тренд, RSI не критичний")
                return False, f"Sideways Trend, RSI OK"

    def check_trend_continuation_signal(self, multi_data: dict, current_time: datetime,
                                        direction: str, entry_time: datetime, config: dict) -> tuple:
        """
        НОВА ФУНКЦІЯ: Перевірка сигналу на продовження тренду

        Використовується для позицій, які не були закриті через збереження тренду
        Шукає додаткові підтвердження для тримання позиції або сигнали для закриття
        """

        trend_config = config.get('trend_filter', {})
        medium_tf = trend_config.get('medium_timeframe', '15m')

        if medium_tf not in multi_data or '5m' not in multi_data:
            return False, ""

        # Аналіз тренду
        trend_analysis = self.analyze_sma_rsi_trend_optimized(
            multi_data[medium_tf], current_time, config
        )

        df_5m = multi_data['5m']
        df_5m['time_diff'] = (df_5m['datetime'] - current_time).abs()
        idx = df_5m['time_diff'].idxmin()

        if pd.isna(idx) or idx < 1:
            return False, ""

        curr_candle = df_5m.iloc[idx]
        curr_rsi = curr_candle.get('rsi', 50)
        curr_rsi_sma = curr_candle.get('rsi_sma', 50)

        is_long = direction.lower() == 'long'
        trend_direction = trend_analysis['trend_direction']
        trend_strength = trend_analysis['trend_strength']

        # Час утримання позиції
        hold_time = (current_time - entry_time).total_seconds() / 3600

        # Критичні рівні RSI для форсованого закриття
        critical_high = 75
        critical_low = 25

        logger.debug(f"Перевірка продовження тренду: {direction}, тренд={trend_direction}, "
                     f"RSI={curr_rsi:.1f}, час утримання={hold_time:.1f}h")

        if is_long:
            # Форсоване закриття при критичних рівнях
            if curr_rsi >= critical_high:
                return True, f"Critical RSI High ({curr_rsi:.0f})"

            # Закриття при зміні тренду на спадний з підтвердженням
            if (trend_direction == 'down' and trend_strength > 40 and
                    curr_rsi < curr_rsi_sma):
                return True, f"Strong Trend Reversal (RSI:{curr_rsi:.0f})"

            # Тривале утримання в боковику
            if trend_direction == 'sideways' and hold_time > 12 and curr_rsi > 65:
                return True, f"Long Sideways Hold ({hold_time:.0f}h, RSI:{curr_rsi:.0f})"

        else:  # SHORT
            # Форсоване закриття при критичних рівнях
            if curr_rsi <= critical_low:
                return True, f"Critical RSI Low ({curr_rsi:.0f})"

            # Закриття при зміні тренду на висхідний з підтвердженням
            if (trend_direction == 'up' and trend_strength > 40 and
                    curr_rsi > curr_rsi_sma):
                return True, f"Strong Trend Reversal (RSI:{curr_rsi:.0f})"

            # Тривале утримання в боковику
            if trend_direction == 'sideways' and hold_time > 12 and curr_rsi < 35:
                return True, f"Long Sideways Hold ({hold_time:.0f}h, RSI:{curr_rsi:.0f})"

        return False, ""


    def check_sma_rsi_trend_filter(self, multi_data: dict, signal_time: datetime,
                                   direction: str, config: dict) -> tuple:
        """
        НОВА ФУНКЦІЯ: Трендовий фільтр на базі SMA_RSI середнього таймфрейму

        Логіка:
        - Для SHORT: дозволяє вхід якщо на 15m є спадання SMA_RSI, боковик, або нещодавній перетин вниз в зоні шорту
        - Для LONG: дозволяє вхід якщо на 15m є зростання SMA_RSI, боковик, або нещодавній перетин вгору в зоні лонгу
        """

        trend_config = config.get('trend_filter', {})
        medium_tf = trend_config.get('medium_timeframe', '15m')
        enable_sideways = trend_config.get('enable_sideways_trading', True)
        enable_recent_cross = trend_config.get('enable_recent_cross', True)

        if medium_tf not in multi_data:
            logger.warning(f"Немає даних для середнього таймфрейму {medium_tf}")
            return False, f"No {medium_tf} data"

        # Аналіз SMA_RSI тренду
        trend_analysis = self.analyze_sma_rsi_trend_optimized(multi_data[medium_tf], signal_time, config)

        trend_direction = trend_analysis['trend_direction']
        trend_strength = trend_analysis['trend_strength']
        recent_cross = trend_analysis['recent_cross_info']

        is_long = direction.lower() == 'long'

        logger.debug(
            f"Перевірка SMA_RSI тренд фільтру: {direction}, тренд={trend_direction}, сила={trend_strength:.1f}")

        # Логіка для SHORT сигналу
        if not is_long:
            # 1. Основний спадаючий тренд SMA_RSI
            if trend_direction == 'down':
                reason = f"SMA_RSI↓({trend_strength:.1f})"
                logger.debug(f"SHORT дозволено: спадаючий тренд SMA_RSI")
                return True, reason

            # 2. Боковий рух (якщо дозволено)
            elif trend_direction == 'sideways' and enable_sideways:
                reason = f"SMA_RSI→({trend_strength:.1f})"
                logger.debug(f"SHORT дозволено: боковий рух SMA_RSI")
                return True, reason

            # 3. Нещодавній перетин вниз в зоні шорту
            elif (enable_recent_cross and recent_cross and
                  recent_cross['type'] == 'cross_down' and recent_cross['in_short_zone']):
                candles_ago = recent_cross['candles_ago']
                rsi_level = recent_cross['rsi_level']
                reason = f"RSI↓x{candles_ago}ago@{rsi_level:.0f}"
                logger.debug(
                    f"SHORT дозволено: нещодавній перетин вниз {candles_ago} свічок тому на рівні {rsi_level:.1f}")
                return True, reason

            else:
                reason = f"SMA_RSI↑({trend_strength:.1f})"
                logger.debug(f"SHORT заборонено: зростаючий тренд SMA_RSI")
                return False, reason

        # Логіка для LONG сигналу
        else:
            # 1. Основний зростаючий тренд SMA_RSI
            if trend_direction == 'up':
                reason = f"SMA_RSI↑({trend_strength:.1f})"
                logger.debug(f"LONG дозволено: зростаючий тренд SMA_RSI")
                return True, reason

            # 2. Боковий рух (якщо дозволено)
            elif trend_direction == 'sideways' and enable_sideways:
                reason = f"SMA_RSI→({trend_strength:.1f})"
                logger.debug(f"LONG дозволено: боковий рух SMA_RSI")
                return True, reason

            # 3. Нещодавній перетин вгору в зоні лонгу
            elif (enable_recent_cross and recent_cross and
                  recent_cross['type'] == 'cross_up' and recent_cross['in_long_zone']):
                candles_ago = recent_cross['candles_ago']
                rsi_level = recent_cross['rsi_level']
                reason = f"RSI↑x{candles_ago}ago@{rsi_level:.0f}"
                logger.debug(
                    f"LONG дозволено: нещодавній перетин вгору {candles_ago} свічок тому на рівні {rsi_level:.1f}")
                return True, reason

            else:
                reason = f"SMA_RSI↓({trend_strength:.1f})"
                logger.debug(f"LONG заборонено: спадаючий тренд SMA_RSI")
                return False, reason

    def apply_filters(self, multi_data: dict, signal_time: datetime, direction: str, config: dict) -> tuple:
        """
        МОДИФІКОВАНА ФУНКЦІЯ: Застосування фільтрів з новим SMA_RSI трендовим фільтром
        """
        if '5m' not in multi_data:
            logger.warning("Немає даних 5m для фільтрації")
            return False, "No 5m data"

        df = multi_data['5m']
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        idx = df['time_diff'].idxmin()

        if pd.isna(idx):
            logger.warning("Невалідний індекс для фільтрації")
            return False, "Invalid index"

        candle = df.iloc[idx]
        is_long = direction.lower() == 'long'
        passed, failed = [], []

        logger.debug(f"Застосування фільтрів для {direction} на час {signal_time}")

        # Отримуємо параметри з конфігу
        filter_config = config.get('filters', {})
        threshold_config = config.get('filter_thresholds', {})

        # 1. НОВИЙ SMA_RSI трендовий фільтр (замість старого)
        if filter_config.get('use_trend_filter', True):
            trend_passed, trend_reason = self.check_sma_rsi_trend_filter(multi_data, signal_time, direction, config)
            if trend_passed:
                passed.append(f"Trend:{trend_reason}")
            else:
                failed.append(f"Trend:{trend_reason}")

        # 2. ADX фільтр (без змін)
        adx = candle.get('adx', 0)
        min_adx = threshold_config.get('min_adx', 18)
        if filter_config.get('use_adx_filter', True):
            if adx >= min_adx:
                passed.append(f"ADX:{adx:.0f}")
            else:
                failed.append(f"ADX:{adx:.0f}")

        # 3. Волатільність (без змін)
        atr_pct = candle.get('atr_pct', 0)
        bb_width = candle.get('bb_width', 0)

        if filter_config.get('use_volatility_filter', True):
            min_vol = threshold_config.get('min_volatility', 0.15)
            max_vol = threshold_config.get('max_volatility', 2.5)

            volatility = max(atr_pct, bb_width * 0.5)

            if min_vol <= volatility <= max_vol:
                passed.append(f"Vol:{volatility:.2f}%")
            else:
                failed.append(f"Vol:{volatility:.2f}%")

        # 4. Об'єм (без змін)
        vol_ratio = candle.get('volume_ratio', 1)
        min_vol_ratio = threshold_config.get('min_volume_ratio', 0.8)
        if filter_config.get('use_volume_filter', True):
            if vol_ratio >= min_vol_ratio:
                passed.append(f"Vol:{vol_ratio:.1f}x")
            else:
                failed.append(f"Vol:{vol_ratio:.1f}x")

        # 5. Momentum фільтр (без змін)
        if filter_config.get('use_momentum_filter', True):
            momentum = candle.get('momentum', 0)
            momentum_sma = candle.get('momentum_sma', 0)

            if (is_long and momentum > momentum_sma and momentum > -0.5) or \
                    (not is_long and momentum < momentum_sma and momentum < 0.5):
                passed.append(f"Mom:{momentum:.2f}")
            else:
                failed.append(f"Mom:{momentum:.2f}")

        min_required = filter_config.get('min_filters_required', 3)
        filter_passed = len(passed) >= min_required
        filter_info = f"✓{','.join(passed)} ✗{','.join(failed)}"

        logger.debug(f"Фільтри: пройдено {len(passed)}/{min_required}, результат: {filter_passed}")
        return filter_passed, filter_info

    def apply_filters_optimized(self, multi_data: dict, signal_time: datetime,
                                direction: str, config: dict) -> tuple:
        """
        МОДИФІКОВАНА ФУНКЦІЯ: Застосування фільтрів з оптимізованим SMA_RSI трендовим фільтром
        """
        if '5m' not in multi_data:
            logger.warning("Немає даних 5m для фільтрації")
            return False, "No 5m data"

        df = multi_data['5m']
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        idx = df['time_diff'].idxmin()

        if pd.isna(idx):
            logger.warning("Невалідний індекс для фільтрації")
            return False, "Invalid index"

        candle = df.iloc[idx]
        is_long = direction.lower() == 'long'
        passed, failed = [], []

        logger.debug(f"Застосування оптимізованих фільтрів для {direction} на час {signal_time}")

        # Отримуємо параметри з конфігу
        filter_config = config.get('filters', {})
        threshold_config = config.get('filter_thresholds', {})

        # 1. ОПТИМІЗОВАНИЙ SMA_RSI трендовий фільтр
        if filter_config.get('use_trend_filter', True):
            trend_passed, trend_reason = self.check_optimized_trend_filter(
                multi_data, signal_time, direction, config
            )
            if trend_passed:
                passed.append(f"Trend:{trend_reason}")
            else:
                failed.append(f"Trend:{trend_reason}")

        # 2. Адаптивний ADX фільтр
        adx = candle.get('adx', 0)
        min_adx = threshold_config.get('min_adx', 18)

        # Адаптація ADX порогу на основі волатільності
        atr_pct = candle.get('atr_pct', 1.0)
        if atr_pct > 2.0:  # Висока волатільність
            adjusted_min_adx = min_adx * 0.8  # Знижуємо вимоги
        elif atr_pct < 0.5:  # Низька волатільність
            adjusted_min_adx = min_adx * 1.2  # Підвищуємо вимоги
        else:
            adjusted_min_adx = min_adx

        if filter_config.get('use_adx_filter', True):
            if adx >= adjusted_min_adx:
                passed.append(f"ADX:{adx:.0f}")
            else:
                failed.append(f"ADX:{adx:.0f}({adjusted_min_adx:.0f})")

        # 3. Розширений волатільність фільтр
        bb_width = candle.get('bb_width', 0)
        if filter_config.get('use_volatility_filter', True):
            min_vol = threshold_config.get('min_volatility', 0.15)
            max_vol = threshold_config.get('max_volatility', 2.5)

            volatility = max(atr_pct, bb_width * 0.5)

            # Адаптивні пороги волатільності
            if min_vol <= volatility <= max_vol:
                vol_quality = "good" if 0.5 <= volatility <= 1.5 else "ok"
                passed.append(f"Vol:{volatility:.2f}%({vol_quality})")
            else:
                failed.append(f"Vol:{volatility:.2f}%")

        # 4. Покращений об'ємний фільтр
        vol_ratio = candle.get('volume_ratio', 1)
        min_vol_ratio = threshold_config.get('min_volume_ratio', 0.8)

        if filter_config.get('use_volume_filter', True):
            # Бонус за високий об'єм
            if vol_ratio >= min_vol_ratio * 1.5:
                passed.append(f"Vol:{vol_ratio:.1f}x(high)")
            elif vol_ratio >= min_vol_ratio:
                passed.append(f"Vol:{vol_ratio:.1f}x")
            else:
                failed.append(f"Vol:{vol_ratio:.1f}x")

        # 5. Покращений momentum фільтр
        if filter_config.get('use_momentum_filter', True):
            momentum = candle.get('momentum', 0)
            momentum_sma = candle.get('momentum_sma', 0)

            # Розширена логіка momentum
            momentum_strength = abs(momentum - momentum_sma)

            if is_long:
                momentum_good = (momentum > momentum_sma and momentum > -0.5)
                if momentum_good:
                    strength_label = "strong" if momentum_strength > 0.3 else "weak"
                    passed.append(f"Mom:{momentum:.2f}({strength_label})")
                else:
                    failed.append(f"Mom:{momentum:.2f}")
            else:
                momentum_good = (momentum < momentum_sma and momentum < 0.5)
                if momentum_good:
                    strength_label = "strong" if momentum_strength > 0.3 else "weak"
                    passed.append(f"Mom:{momentum:.2f}({strength_label})")
                else:
                    failed.append(f"Mom:{momentum:.2f}")

        # 6. Додатковий фільтр: перевірка RSI дивергенції (новий)
        if filter_config.get('use_divergence_filter', False):
            divergence_signal = self._check_rsi_price_divergence(df, idx, direction)
            if divergence_signal:
                passed.append("Div:Yes")
            else:
                failed.append("Div:No")

        min_required = filter_config.get('min_filters_required', 3)
        filter_passed = len(passed) >= min_required
        filter_info = f"✓{','.join(passed)} ✗{','.join(failed)}"

        logger.debug(f"Оптимізовані фільтри: пройдено {len(passed)}/{min_required}, результат: {filter_passed}")
        return filter_passed, filter_info

    def check_optimized_trend_filter(self, multi_data: dict, signal_time: datetime,
                                     direction: str, config: dict) -> tuple:
        """
        ОПТИМІЗОВАНИЙ ТРЕНДОВИЙ ФІЛЬТР: Використовує покращений аналіз тренду
        """
        trend_config = config.get('trend_filter', {})
        medium_tf = trend_config.get('medium_timeframe', '15m')
        enable_sideways = trend_config.get('enable_sideways_trading', True)
        enable_recent_cross = trend_config.get('enable_recent_cross', True)

        if medium_tf not in multi_data:
            logger.warning(f"Немає даних для середнього таймфрейму {medium_tf}")
            return False, f"No {medium_tf} data"

        # Використовуємо оптимізований аналіз тренду
        trend_analysis = self.analyze_sma_rsi_trend_optimized(multi_data[medium_tf], signal_time, config)

        trend_direction = trend_analysis['trend_direction']
        trend_strength = trend_analysis['trend_strength']
        trend_confidence = trend_analysis['trend_confidence']
        recent_cross = trend_analysis['recent_cross_info']

        is_long = direction.lower() == 'long'

        # Адаптивні пороги на основі достовірності тренду
        base_strength_threshold = 25
        if trend_confidence > 70:
            strength_threshold = base_strength_threshold * 0.8  # Знижуємо поріг при високій достовірності
        elif trend_confidence < 40:
            strength_threshold = base_strength_threshold * 1.3  # Підвищуємо поріг при низькій достовірності
        else:
            strength_threshold = base_strength_threshold

        logger.debug(f"Оптимізована перевірка трендового фільтру: {direction}, "
                     f"тренд={trend_direction}, сила={trend_strength:.1f}, "
                     f"достовірність={trend_confidence:.1f}, поріг={strength_threshold:.1f}")

        # ЛОГІКА ДЛЯ SHORT СИГНАЛУ
        if not is_long:
            # 1. Сильний спадаючий тренд
            if trend_direction == 'down' and trend_strength >= strength_threshold:
                reason = f"SMA_RSI↓({trend_strength:.0f}|{trend_confidence:.0f})"
                return True, reason

            # 2. Боковий рух з високою достовірністю
            elif (trend_direction == 'sideways' and enable_sideways and
                  trend_confidence > 50):
                reason = f"SMA_RSI→({trend_strength:.0f}|{trend_confidence:.0f})"
                return True, reason

            # 3. Нещодавній перетин вниз в релевантній зоні
            elif (enable_recent_cross and recent_cross and
                  recent_cross['type'] == 'cross_down' and recent_cross['in_short_zone'] and
                  recent_cross['cross_strength'] > 2):  # Мінімальна сила перетину
                candles_ago = recent_cross['candles_ago']
                rsi_level = recent_cross['rsi_level']
                cross_str = recent_cross['cross_strength']
                reason = f"RSI↓x{candles_ago}@{rsi_level:.0f}({cross_str:.1f})"
                return True, reason

            # 4. Слабкий висхідний тренд може бути прийнятним для шорту
            elif (trend_direction == 'up' and trend_strength < strength_threshold * 0.7 and
                  trend_confidence < 60):
                reason = f"WeakUP({trend_strength:.0f}|{trend_confidence:.0f})"
                return True, reason

            else:
                reason = f"StrongUP({trend_strength:.0f}|{trend_confidence:.0f})"
                return False, reason

        # ЛОГІКА ДЛЯ LONG СИГНАЛУ
        else:
            # 1. Сильний висхідний тренд
            if trend_direction == 'up' and trend_strength >= strength_threshold:
                reason = f"SMA_RSI↑({trend_strength:.0f}|{trend_confidence:.0f})"
                return True, reason

            # 2. Боковий рух з високою достовірністю
            elif (trend_direction == 'sideways' and enable_sideways and
                  trend_confidence > 50):
                reason = f"SMA_RSI→({trend_strength:.0f}|{trend_confidence:.0f})"
                return True, reason

            # 3. Нещодавній перетин вгору в релевантній зоні
            elif (enable_recent_cross and recent_cross and
                  recent_cross['type'] == 'cross_up' and recent_cross['in_long_zone'] and
                  recent_cross['cross_strength'] > 2):
                candles_ago = recent_cross['candles_ago']
                rsi_level = recent_cross['rsi_level']
                cross_str = recent_cross['cross_strength']
                reason = f"RSI↑x{candles_ago}@{rsi_level:.0f}({cross_str:.1f})"
                return True, reason

            # 4. Слабкий спадний тренд може бути прийнятним для лонгу
            elif (trend_direction == 'down' and trend_strength < strength_threshold * 0.7 and
                  trend_confidence < 60):
                reason = f"WeakDN({trend_strength:.0f}|{trend_confidence:.0f})"
                return True, reason

            else:
                reason = f"StrongDN({trend_strength:.0f}|{trend_confidence:.0f})"
                return False, reason

    def _check_rsi_price_divergence(self, df: pd.DataFrame, current_idx: int, direction: str) -> bool:
        """
        НОВА ФУНКЦІЯ: Перевірка дивергенції між RSI та ціною
        """
        if current_idx < 10:
            return False

        # Аналізуємо останні 10 свічок
        recent_data = df.iloc[max(0, current_idx - 10):current_idx + 1]

        if len(recent_data) < 5 or 'rsi' not in recent_data.columns:
            return False

        prices = recent_data['close'].values
        rsi_values = recent_data['rsi'].values

        # Розраховуємо тренди
        x = np.arange(len(prices))
        price_slope, _ = np.polyfit(x, prices, 1)
        rsi_slope, _ = np.polyfit(x, rsi_values, 1)

        is_long = direction.lower() == 'long'

        # Бульська дивергенція: ціна падає, RSI зростає
        if is_long and price_slope < -0.001 and rsi_slope > 0.5:
            return True

        # Ведмежа дивергенція: ціна зростає, RSI падає
        elif not is_long and price_slope > 0.001 and rsi_slope < -0.5:
            return True

        return False

    def _create_data_coverage_result(self, df: pd.DataFrame, entry_time: datetime,
                                     entry_price: float, direction: str,
                                     filter_info: str, dynamic_sl_pct: float) -> dict:
        """
        ДОПОМІЖНА ФУНКЦІЯ: Створення результату при недостатньому покритті даних
        """
        return {
            'entry_time': entry_time,
            'entry_time_str': entry_time.strftime('%d.%m.%Y %H:%M'),
            'exit_time': df['datetime'].max(),
            'exit_time_str': df['datetime'].max().strftime('%d.%m.%Y %H:%M'),
            'entry_price': entry_price,
            'exit_price': df.iloc[-1]['close'],
            'pnl_percent': 0,
            'hold_time_hours': 0,
            'exit_reason': 'Insufficient Data Coverage',
            'status': 'Data Error',
            'filter_info': filter_info,
            'dynamic_sl_pct': dynamic_sl_pct * 100,
            'filtered_out': False,
            'trend_holds': 0
        }


    def calc_stop_loss(self, multi_data: dict, entry_time: datetime, direction: str, config: dict) -> float:
        """Розрахунок динамічного стоп-лосу в процентах"""
        if '5m' not in multi_data:
            return config.get('trading_parameters', {}).get('stop_loss', 0.018)

        df = multi_data['5m']
        df['time_diff'] = (df['datetime'] - entry_time).abs()
        idx = df['time_diff'].idxmin()

        if pd.isna(idx):
            return config.get('trading_parameters', {}).get('stop_loss', 0.018)

        candle = df.iloc[idx]
        atr_pct = candle.get('atr_pct', 1.0)
        bb_width = candle.get('bb_width', 2.0)

        # Базовий стоп-лос з конфігу
        base_sl = config.get('trading_parameters', {}).get('stop_loss', 0.018)

        # Коригування на волатільність
        volatility = max(atr_pct, bb_width * 0.3)
        vol_mult = min(max(volatility / 1.5, 0.8), 2.2)

        # Трендове коригування (спрощено для SMA_RSI)
        trend_config = config.get('trend_filter', {})
        medium_tf = trend_config.get('medium_timeframe', '15m')

        if medium_tf in multi_data:
            trend_analysis = self.analyze_sma_rsi_trend_optimized(multi_data[medium_tf], entry_time, config)
            if trend_analysis['trend_direction'] != 'unknown' and trend_analysis['trend_strength'] > 40:
                sl_mult = 0.85  # Менший SL при сильному тренді SMA_RSI
            else:
                sl_mult = 1.15  # Більший SL при слабкому тренді
        else:
            sl_mult = 1.0

        dynamic_sl = base_sl * vol_mult * sl_mult
        logger.debug(f"Динамічний SL: {dynamic_sl * 100:.2f}% (base: {base_sl * 100:.2f}%)")
        return dynamic_sl

    def check_reverse_signal_exit(self, multi_data: dict, current_time: datetime, direction: str,
                                  config: dict) -> tuple:
        """
        Перевірка зворотнього RSI сигналу для виходу
        Для LONG: RSI перетинає свою SMA вниз в зоні шорту (< short_exit_zone)
        Для SHORT: RSI перетинає свою SMA вгору в зоні лонгу (> long_exit_zone)
        """
        if '5m' not in multi_data:
            return False, ""

        df = multi_data['5m']
        df['time_diff'] = (df['datetime'] - current_time).abs()
        idx = df['time_diff'].idxmin()

        if pd.isna(idx) or idx < 1:
            return False, ""

        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        curr_rsi = curr.get('rsi', 0)
        curr_rsi_sma = curr.get('rsi_sma', 0)
        prev_rsi = prev.get('rsi', 0)
        prev_rsi_sma = prev.get('rsi_sma', 0)

        if any(pd.isna(x) for x in [curr_rsi, curr_rsi_sma, prev_rsi, prev_rsi_sma]):
            return False, ""

        is_long = direction.lower() == 'long'

        # Отримуємо зони з конфігу
        rsi_params = config.get('rsi_parameters', {})
        long_exit_zone = rsi_params.get('long_exit_zone', 65)
        short_exit_zone = rsi_params.get('short_exit_zone', 35)

        logger.debug(f"Перевірка зворотнього сигналу: {direction}, RSI={curr_rsi:.1f}, RSI_SMA={curr_rsi_sma:.1f}")

        if is_long:
            # Для LONG позиції: шукаємо перетин RSI вниз через SMA в зоні шорту
            if curr_rsi >= long_exit_zone or prev_rsi >= long_exit_zone:  # RSI в зоні шорту
                if prev_rsi > prev_rsi_sma and curr_rsi <= curr_rsi_sma:  # Перетин вниз
                    logger.info(
                        f"Зворотний сигнал LONG->SHORT: RSI {curr_rsi:.1f} перетнув SMA {curr_rsi_sma:.1f} вниз в зоні шорту")
                    return True, f'Reverse Signal: RSI Cross Down ({curr_rsi:.1f})'
        else:
            # Для SHORT позиції: шукаємо перетин RSI вгору через SMA в зоні лонгу
            if curr_rsi <= short_exit_zone or prev_rsi <= short_exit_zone:  # RSI в зоні лонгу
                if prev_rsi < prev_rsi_sma and curr_rsi >= curr_rsi_sma:  # Перетин вгору
                    logger.info(
                        f"Зворотний сигнал SHORT->LONG: RSI {curr_rsi:.1f} перетнув SMA {curr_rsi_sma:.1f} вгору в зоні лонгу")
                    return True, f'Reverse Signal: RSI Cross Up ({curr_rsi:.1f})'

        return False, ""

    def check_data_coverage(self, df: pd.DataFrame, required_end_time: datetime) -> bool:
        """
        НОВА ФУНКЦІЯ: Перевірка чи покривають дані необхідний період
        """
        if df.empty:
            return False

        last_data_time = df['datetime'].max()
        coverage_sufficient = last_data_time >= required_end_time

        if not coverage_sufficient:
            hours_short = (required_end_time - last_data_time).total_seconds() / 3600
            logger.debug(f"Недостатнє покриття даних: нестача {hours_short:.1f} годин")

        return coverage_sufficient

    async def find_entry_exit_with_smart_closure(self, multi_data: dict, signal_time: datetime,
                                                 direction: str, entry_rsi: float, config: dict,
                                                 pair_requirements: dict = None) -> dict:
        """
        МОДИФІКОВАНА ФУНКЦІЯ: Логіка входу та виходу з розумним закриттям позицій
        """

        if '5m' not in multi_data:
            logger.warning("Немає даних 5m для аналізу")
            return None

        # Застосування фільтрів (з оптимізованим SMA_RSI трендовим фільтром)
        filter_passed, filter_info = self.apply_filters_optimized(multi_data, signal_time, direction, config)

        filters_config = config.get('filters', {})
        if not filter_passed and filters_config.get('use_filters', True):
            logger.debug(f"Сигнал відфільтровано: {filter_info}")
            return {
                'filtered_out': True,
                'filter_info': filter_info,
                'pnl_percent': 0,
                'status': 'Filtered'
            }

        df = multi_data['5m']
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        entry_idx = df['time_diff'].idxmin()

        if pd.isna(entry_idx):
            logger.warning("Невалідний індекс входу")
            return None

        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']
        is_long = direction.lower() == 'long'

        logger.debug(f"Вхід: {direction} {entry_price} в {entry_time}")

        # Розрахунок динамічного стоп-лосу
        dynamic_sl_pct = self.calc_stop_loss(multi_data, entry_time, direction, config)

        if is_long:
            sl_price = entry_price * (1 - dynamic_sl_pct)
        else:
            sl_price = entry_price * (1 + dynamic_sl_pct)

        # Параметри
        trading_params = config.get('trading_parameters', {})
        max_hours = trading_params.get('max_hold_hours', 72)
        max_time = entry_time + timedelta(hours=max_hours)

        # Перевірка покриття даних
        if not self.check_data_coverage(df, max_time):
            logger.warning(f"Недостатнє покриття даних для {direction} в {entry_time}")
            return self._create_data_coverage_result(df, entry_time, entry_price, direction,
                                                     filter_info, dynamic_sl_pct)

        # Трейлінг стоп
        use_trailing = trading_params.get('use_trailing_stop', True)
        best_price = entry_price
        trailing_sl = sl_price

        # Лічильники для розумного закриття
        trend_hold_count = 0  # Скільки разів утримували через тренд
        max_trend_holds = 3  # Максимум утримань через тренд

        # Пошук виходу з новою логікою
        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i]
            curr_time = candle['datetime']

            # 1. РОЗУМНА перевірка зворотного RSI сигналу з урахуванням тренду
            smart_exit, smart_reason = self.check_smart_exit_conditions(
                multi_data, curr_time, direction, entry_time, config
            )

            if smart_exit:
                logger.info(f"Розумний вихід: {smart_reason}")
                return self._create_result(entry_time, entry_price, curr_time, candle['close'],
                                           direction, smart_reason, filter_info, dynamic_sl_pct)

            # 2. Перевірка на продовження тренду (для довготривалих позицій)
            if not smart_exit and trend_hold_count < max_trend_holds:
                continuation_exit, continuation_reason = self.check_trend_continuation_signal(
                    multi_data, curr_time, direction, entry_time, config
                )

                if continuation_exit:
                    logger.info(f"Вихід за продовженням тренду: {continuation_reason}")
                    return self._create_result(entry_time, entry_price, curr_time, candle['close'],
                                               direction, continuation_reason, filter_info, dynamic_sl_pct)

            # 3. Якщо була спроба закриття, але тренд зберігся - збільшуємо лічильник
            if "Trend Still" in smart_reason:
                trend_hold_count += 1
                if trend_hold_count >= max_trend_holds:
                    logger.info(f"Досягнуто максимум утримань через тренд ({max_trend_holds})")

            # 4. Оновлення трейлінга
            if use_trailing:
                if is_long and candle['high'] > best_price:
                    best_price = candle['high']
                    trailing_sl = best_price * (1 - dynamic_sl_pct)
                elif not is_long and candle['low'] < best_price:
                    best_price = candle['low']
                    trailing_sl = best_price * (1 + dynamic_sl_pct)

            # 5. Перевірка таймауту
            if curr_time > max_time:
                logger.debug(f"Таймаут {max_hours}h")
                return self._create_result(entry_time, entry_price, curr_time, candle['close'],
                                           direction, f'Time Exit ({max_hours}h)', filter_info, dynamic_sl_pct)

            # 6. Перевірка стоп-лосу
            current_sl = trailing_sl if use_trailing else sl_price

            if is_long:
                if candle['low'] <= current_sl:
                    reason = 'Trailing Stop' if use_trailing else 'Stop Loss'
                    logger.debug(f"SL спрацював: {current_sl}")
                    return self._create_result(entry_time, entry_price, curr_time, current_sl,
                                               direction, reason, filter_info, dynamic_sl_pct)
            else:
                if candle['high'] >= current_sl:
                    reason = 'Trailing Stop' if use_trailing else 'Stop Loss'
                    logger.debug(f"SL спрацював: {current_sl}")
                    return self._create_result(entry_time, entry_price, curr_time, current_sl,
                                               direction, reason, filter_info, dynamic_sl_pct)

        # Якщо не знайдено вихід - остання свічка
        last = df.iloc[-1]
        logger.warning(f"Досягнуто кінця даних для {direction} позиції")
        return self._create_result(entry_time, entry_price, last['datetime'], last['close'],
                                   direction, 'Data End', filter_info, dynamic_sl_pct)

    def _create_result(self, entry_time, entry_price, exit_time, exit_price,
                       direction, reason, filter_info, dynamic_sl_pct) -> dict:
        """Створення результату торгівлі з додаванням часу входу та виходу"""
        is_long = direction.lower() == 'long'
        if is_long:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        hold_hours = (exit_time - entry_time).total_seconds() / 3600

        result = {
            'entry_time': entry_time,
            'entry_time_str': entry_time.strftime('%d.%m.%Y %H:%M'),
            'exit_time': exit_time,
            'exit_time_str': exit_time.strftime('%d.%m.%Y %H:%M'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_pct,
            'hold_time_hours': hold_hours,
            'exit_reason': reason,
            'status': 'Profit' if pnl_pct > 0 else 'Loss',
            'filter_info': filter_info,
            'dynamic_sl_pct': dynamic_sl_pct * 100,
            'filtered_out': False
        }

        logger.debug(f"Результат: PnL={pnl_pct:.2f}%, час={hold_hours:.1f}h, причина={reason}")
        return result

    async def analyze_signals(self, signals: list, config: dict) -> list:
        """
        МОДИФІКОВАНА ФУНКЦІЯ: Аналіз сигналів з розумним буфером
        """
        results = []

        # КРОК 1: Розрахунок оптимальних буферів для всіх пар
        logger.info("Етап 1: Розрахунок оптимальних буферів для кожної пари")
        pair_requirements = self.calculate_optimal_buffer(signals, config)

        # Сортуємо сигнали по часу для оптимізації
        signals_with_time = []
        for signal in signals:
            signal_time = self.parse_time(signal['time'])
            signals_with_time.append((signal, signal_time))

        # Сортуємо по парі та часу
        signals_with_time.sort(key=lambda x: (x[0]['pair'], x[1]))

        logger.info(f"Етап 2: Аналіз {len(signals)} сигналів з розумними буферами та SMA_RSI фільтром")

        # КРОК 2: Обробка кожного сигналу з розумним буфером
        for signal_idx, (signal, signal_time) in enumerate(signals_with_time):
            pair = signal['pair']
            logger.info(f"Аналіз сигналу {signal_idx + 1}/{len(signals)}: {pair} {signal['direction']} в {signal_time}")

            # Завантаження даних з розумним буфером
            multi_data = await self.fetch_multi_data(pair, signal_time, config, pair_requirements)
            if not multi_data or '5m' not in multi_data:
                logger.warning(f"Немає даних для {pair} {signal['direction']} в {signal_time}")
                continue

            # Аналіз сигналу з перевіркою покриття
            result = await self.find_entry_exit_with_smart_closure(multi_data, signal_time, signal['direction'],
                                                signal['rsi'], config, pair_requirements)

            if result:
                result.update({
                    **signal,
                    'signal_time': signal_time.strftime('%d.%m.%Y %H:%M')
                })
                results.append(result)

            # Невеличка пауза між сигналами
            if signal_idx < len(signals_with_time) - 1:
                await asyncio.sleep(0.1)

        logger.info(f"Аналіз завершено. Отримано {len(results)} результатів")
        return results

    def save_results(self, results: list, filename: str):
        """Збереження результатів з часом входу та виходу"""
        if not results:
            logger.warning("Немає результатів для збереження")
            return

        # Розширюємо список полів для включення часу входу та виходу
        fields = ['pair', 'direction', 'rsi', 'signal_time', 'entry_time_str', 'exit_time_str',
                  'entry_price', 'exit_price', 'pnl_percent', 'hold_time_hours', 'dynamic_sl_pct',
                  'status', 'exit_reason', 'filter_info', 'filtered_out']

        logger.info(f"Збереження {len(results)} результатів у {filename}")

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, delimiter=';', fieldnames=fields)
            writer.writeheader()

            for result in results:
                row = {}
                for field in fields:
                    value = result.get(field, '')
                    if field in ['rsi', 'entry_price', 'exit_price', 'pnl_percent',
                                 'hold_time_hours', 'dynamic_sl_pct']:
                        try:
                            row[field] = str(round(float(value), 4) if value != '' else 0).replace('.', ',')
                        except:
                            row[field] = 0
                    elif field == 'filtered_out':
                        row[field] = 1 if value else 0
                    else:
                        row[field] = str(value)

                writer.writerow(row)

        logger.info(f"Результати збережено в {filename}")
        self.print_stats(results)

    def print_stats(self, results: list):
        """Виведення статистики з аналізом покриття даних та SMA_RSI фільтру"""
        if not results:
            logger.info("Немає результатів для статистики")
            return

        traded = [r for r in results if not r.get('filtered_out', False)]
        filtered = [r for r in results if r.get('filtered_out', False)]

        print(f"\n{'=' * 70}")
        print(f"СТАТИСТИКА ТОРГІВЛІ (SMA_RSI трендовий фільтр)")
        print(f"{'=' * 70}")
        print(f"Всього сигналів: {len(results)}")
        print(f"Відфільтровано: {len(filtered)} ({len(filtered) / len(results) * 100:.1f}%)")
        print(f"Торговано: {len(traded)} ({len(traded) / len(results) * 100:.1f}%)")

        if not traded:
            print("Немає торгованих результатів")
            logger.info("Немає торгованих результатів для аналізу")
            return

        profitable = [r for r in traded if float(r.get('pnl_percent', 0)) > 0]
        total_pnl = sum(float(r.get('pnl_percent', 0)) for r in traded)
        avg_pnl = total_pnl / len(traded)
        win_rate = len(profitable) / len(traded) * 100
        avg_hold_time = np.mean([float(r.get('hold_time_hours', 0)) for r in traded])

        print(f"\nТОРГОВА СТАТИСТИКА:")
        print(f"Прибуткових: {len(profitable)} ({win_rate:.1f}%)")
        print(f"Середній PnL: {avg_pnl:.2f}%")
        print(f"Загальний PnL: {total_pnl:.2f}%")
        print(f"Середній час утримання: {avg_hold_time:.1f}h")

        # Аналіз динамічних SL
        avg_sl = np.mean([float(r.get('dynamic_sl_pct', 0)) for r in traded])
        print(f"Середній динамічний SL: {avg_sl:.2f}%")

        # Аналіз за типами виходу
        exit_stats = {}
        for result in traded:
            reason = result.get('exit_reason', 'Unknown')
            if reason not in exit_stats:
                exit_stats[reason] = {'count': 0, 'pnl': 0}
            exit_stats[reason]['count'] += 1
            exit_stats[reason]['pnl'] += float(result.get('pnl_percent', 0))

        print(f"\nАНАЛІЗ ЗА ТИПАМИ ВИХОДУ:")
        reverse_signals = 0
        data_end_count = 0
        insufficient_data_count = 0

        for reason, stats in exit_stats.items():
            avg_pnl = stats['pnl'] / stats['count']
            pct = stats['count'] / len(traded) * 100
            print(f"{reason}: {stats['count']} ({pct:.1f}%) - Avg PnL: {avg_pnl:.2f}%")
            if 'Reverse Signal' in reason:
                reverse_signals += stats['count']
            elif 'Data End' in reason:
                data_end_count += stats['count']
            elif 'Insufficient Data Coverage' in reason:
                insufficient_data_count += stats['count']

        if reverse_signals > 0:
            print(f"\nЗВОРОТНІ СИГНАЛИ:")
            print(f"Всього зворотних виходів: {reverse_signals} ({reverse_signals / len(traded) * 100:.1f}%)")

        # Аналіз покриття даних
        total_data_issues = data_end_count + insufficient_data_count
        if total_data_issues > 0:
            print(f"\nАНАЛІЗ ПОКРИТТЯ ДАНИХ:")
            print(f"Data End: {data_end_count} ({data_end_count / len(traded) * 100:.1f}%)")
            print(
                f"Insufficient Coverage: {insufficient_data_count} ({insufficient_data_count / len(traded) * 100:.1f}%)")
            print(f"Всього проблем з даними: {total_data_issues} ({total_data_issues / len(traded) * 100:.1f}%)")

            if total_data_issues > len(traded) * 0.1:  # Більше 10%
                print(f"⚠️  УВАГА: Високий відсоток проблем з покриттям даних!")
                print(f"   Рекомендується збільшити buffer_hours_after в конфігурації")
            else:
                print(f"✅ Покриття даних в межах норми")
        else:
            print(f"\n✅ ВІДМІННО: Жодних проблем з покриттям даних!")

        # Аналіз фільтрації за трендовим фільтром
        trend_filtered = 0
        for result in results:
            if result.get('filtered_out', False):
                filter_info = result.get('filter_info', '')
                if 'Trend:SMA_RSI' in filter_info and '✗' in filter_info:
                    trend_filtered += 1

        if trend_filtered > 0:
            print(f"\nАНАЛІЗ SMA_RSI ТРЕНДОВОГО ФІЛЬТРУ:")
            print(f"Відфільтровано за трендом: {trend_filtered} ({trend_filtered / len(results) * 100:.1f}%)")

        logger.info(f"Статистика: {len(profitable)}/{len(traded)} прибуткових ({win_rate:.1f}%), "
                    f"середній PnL: {avg_pnl:.2f}%, загальний PnL: {total_pnl:.2f}%, "
                    f"проблеми з даними: {total_data_issues}, трендова фільтрація: {trend_filtered}")


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description='Аналізатор з SMA_RSI трендовим фільтром')
    parser.add_argument('input_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='results_sma_rsi_filter.csv', help='Файл результатів')
    parser.add_argument('-c', '--config', default='enhanced_analyzer_config.json', help='Файл конфігурації')

    # Опції для перевизначення конфігу через командний рядок
    parser.add_argument('--stop-loss', type=float, help='Stop Loss % (базовий)')
    parser.add_argument('--max-hold', type=int, help='Макс час утримання (години)')
    parser.add_argument('--long-exit', type=int, help='RSI зона виходу лонг')
    parser.add_argument('--short-exit', type=int, help='RSI зона виходу шорт')
    parser.add_argument('--min-adx', type=int, help='Мін ADX')
    parser.add_argument('--min-vol', type=float, help='Мін волатільність %')
    parser.add_argument('--max-vol', type=float, help='Макс волатільність %')
    parser.add_argument('--min-vol-ratio', type=float, help='Мінімальне відношення об\'єму')
    parser.add_argument('--buffer-hours', type=int, help='Базовий буфер після сигналу (години)')
    parser.add_argument('--max-buffer', type=int, help='Максимальний буфер (години)')
    parser.add_argument('--no-filters', action='store_true', help='Відключити всі фільтри')

    # НОВІ опції для SMA_RSI трендового фільтру
    parser.add_argument('--medium-tf', type=str, help='Середній таймфрейм для аналізу тренду (15m)')
    parser.add_argument('--lookback-candles', type=int, help='Кількість свічок для оцінки тренду')
    parser.add_argument('--trend-threshold', type=float, help='Поріг сили тренду')
    parser.add_argument('--sideways-threshold', type=float, help='Поріг бокового руху')
    parser.add_argument('--cross-candles', type=int, help='Свічки для пошуку нещодавнього перетину')
    parser.add_argument('--no-sideways', action='store_true', help='Заборонити торгівлю в боковику')
    parser.add_argument('--no-recent-cross', action='store_true', help='Заборонити торгівлю при нещодавньому перетині')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Файл {args.input_file} не знайдено")
        return

    analyzer = TrendSignalAnalyzer()

    # Завантаження або створення конфігурації
    config = analyzer.load_config(args.config)

    # Перевизначення параметрів з командного рядка
    if args.stop_loss:
        config['trading_parameters']['stop_loss'] = args.stop_loss / 100
    if args.max_hold:
        config['trading_parameters']['max_hold_hours'] = args.max_hold
    if args.long_exit:
        config['rsi_parameters']['long_exit_zone'] = args.long_exit
    if args.short_exit:
        config['rsi_parameters']['short_exit_zone'] = args.short_exit
    if args.min_adx:
        config['filter_thresholds']['min_adx'] = args.min_adx
    if args.min_vol:
        config['filter_thresholds']['min_volatility'] = args.min_vol / 100
    if args.max_vol:
        config['filter_thresholds']['max_volatility'] = args.max_vol / 100
    if args.min_vol_ratio:
        config['filter_thresholds']['min_volume_ratio'] = args.min_vol_ratio
    if args.buffer_hours:
        config['data_management']['buffer_hours_after'] = args.buffer_hours
    if args.max_buffer:
        config['data_management']['max_buffer_hours'] = args.max_buffer
    if args.no_filters:
        config['filters']['use_filters'] = False

    # Нові параметри SMA_RSI трендового фільтру
    if args.medium_tf:
        config['trend_filter']['medium_timeframe'] = args.medium_tf
    if args.lookback_candles:
        config['trend_filter']['lookback_candles'] = args.lookback_candles
    if args.trend_threshold:
        config['trend_filter']['trend_strength_threshold'] = args.trend_threshold
    if args.sideways_threshold:
        config['trend_filter']['sideways_threshold'] = args.sideways_threshold
    if args.cross_candles:
        config['trend_filter']['recent_cross_candles'] = args.cross_candles
    if args.no_sideways:
        config['trend_filter']['enable_sideways_trading'] = False
    if args.no_recent_cross:
        config['trend_filter']['enable_recent_cross'] = False

    try:
        # Читання сигналів
        signals = analyzer.read_signals_csv(args.input_file)
        if not signals:
            logger.error("Не вдалося завантажити жодного валідного сигналу")
            return

        logger.info(f"Завантажено {len(signals)} сигналів")
        logger.info("Початок аналізу з SMA_RSI трендовим фільтром...")

        # Виведення конфігурації
        print(f"\n{'=' * 60}")
        print("КОНФІГУРАЦІЯ АНАЛІЗАТОРА (SMA_RSI ТРЕНДОВИЙ ФІЛЬТР)")
        print(f"{'=' * 60}")
        print(f"Базовий Stop Loss: {config['trading_parameters']['stop_loss'] * 100:.2f}%")
        print(f"Макс час утримання: {config['trading_parameters']['max_hold_hours']}h")
        print(f"Використання трейлінг стопу: {config['trading_parameters']['use_trailing_stop']}")
        print(f"RSI зона виходу Long: {config['rsi_parameters']['long_exit_zone']}")
        print(f"RSI зона виходу Short: {config['rsi_parameters']['short_exit_zone']}")

        print(f"\nSMA_RSI ТРЕНДОВИЙ ФІЛЬТР:")
        trend_config = config.get('trend_filter', {})
        print(f"Середній таймфрейм: {trend_config.get('medium_timeframe', '15m')}")
        print(f"Свічки для аналізу тренду: {trend_config.get('lookback_candles', 10)}")
        print(f"Поріг сили тренду: {trend_config.get('trend_strength_threshold', 0.5)}")
        print(f"Поріг бокового руху: {trend_config.get('sideways_threshold', 1.0)}")
        print(f"Пошук перетинів (свічки): {trend_config.get('recent_cross_candles', 5)}")
        print(f"Буфер зон перетину: ±{trend_config.get('cross_zone_buffer', 5)}")
        print(f"Дозволена торгівля в боковику: {trend_config.get('enable_sideways_trading', True)}")
        print(f"Дозволені нещодавні перетини: {trend_config.get('enable_recent_cross', True)}")

        print(f"\nІНШІ ФІЛЬТРИ:")
        print(f"Фільтри увімкнені: {config['filters']['use_filters']}")
        if config['filters']['use_filters']:
            print(f"Мінімум фільтрів: {config['filters']['min_filters_required']}")
            print(f"Мін ADX: {config['filter_thresholds']['min_adx']}")
            print(f"Мін волатільність: {config['filter_thresholds']['min_volatility'] * 100:.2f}%")
            print(f"Макс волатільність: {config['filter_thresholds']['max_volatility'] * 100:.2f}%")
            print(f"Мін об'єм ratio: {config['filter_thresholds']['min_volume_ratio']}")
        print(f"Базовий буфер: {config['data_management']['buffer_hours_after']}h")
        print(f"Максимальний буфер: {config['data_management']['max_buffer_hours']}h")
        print(f"{'=' * 60}")

        # Основний аналіз
        start_time = datetime.now()
        results = await analyzer.analyze_signals(signals, config)
        end_time = datetime.now()

        analysis_time = (end_time - start_time).total_seconds()
        logger.info(f"Аналіз завершено за {analysis_time:.1f} секунд")

        if results:
            # Збереження результатів
            analyzer.save_results(results, args.output)

            # Додаткова статистика
            print(f"\nАНАЛІЗ ЕФЕКТИВНОСТІ:")
            print(f"Час аналізу: {analysis_time:.1f}с")
            print(f"Швидкість: {len(signals) / analysis_time:.2f} сигналів/с")
            print(f"Результат збережено в: {args.output}")

            # Інформація про використання буферів
            unique_pairs = len(set(signal['pair'] for signal in signals))
            print(f"Унікальних пар: {unique_pairs}")
            print(f"Середня кількість сигналів на пару: {len(signals) / unique_pairs:.1f}")
        else:
            logger.warning("Не отримано жодного результату аналізу")

    except KeyboardInterrupt:
        logger.info("Аналіз перервано користувачем")
    except Exception as e:
        logger.error(f"Помилка під час аналізу: {e}")


if __name__ == "__main__":
    # Запуск основної функції
    asyncio.run(main())