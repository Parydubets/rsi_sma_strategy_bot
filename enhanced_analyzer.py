#!/usr/bin/env python3
"""
Модифікований аналізатор з виходом тільки по зворотному RSI сигналу
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

    @staticmethod
    def create_default_config():
        """Створення стандартного конфігу в файлі enhanced_analyzer_config.json"""
        default_config = {
            "trading_parameters": {
                "stop_loss": 0.018,
                "take_profit": 0.035,  # Залишаємо для сумісності, але не використовуємо
                "max_hold_hours": 24,  # Збільшуємо час для утримання до зворотнього сигналу
                "use_trailing_stop": True
            },
            "rsi_parameters": {
                "long_exit_zone": 65,  # RSI зона для пошуку виходу з лонгу
                "short_exit_zone": 35,  # RSI зона для пошуку виходу з шорту
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
                "use_momentum_filter": True
            },
            "filter_thresholds": {
                "min_adx": 18,
                "min_volatility": 0.15,
                "max_volatility": 2.5,
                "min_volume_ratio": 0.8
            },
            "timeframe_settings": {
                "primary_timeframe": "5m",
                "secondary_timeframes": ["15m", "1h"],
                "trend_weights": {
                    "5m": 0.4,
                    "15m": 0.35,
                    "1h": 0.25
                }
            },
            "technical_indicators": {
                "5m": {
                    "rsi_period": 14,
                    "rsi_sma_period": 14,  # Додаємо період для SMA RSI
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
                    "rsi_sma_period": 10,
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
                    "rsi_sma_period": 10,
                    "sma_fast": 50,
                    "sma_slow": 200,
                    "adx_period": 14,
                    "ema_fast": 50,
                    "ema_slow": 200,
                    "atr_period": 20,
                    "bb_period": 20,
                    "volume_window": 50
                }
            }
        }

        config_file = 'enhanced_analyzer_config.json'

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)

        logger.info(f"Створено стандартний конфіг: {config_file}")
        return default_config

    @staticmethod
    def load_config(config_file='enhanced_analyzer_config.json'):
        """Завантаження конфігурації з файлу"""
        if not os.path.exists(config_file):
            logger.info(f"Конфіг файл {config_file} не знайдено, створюємо стандартний")
            return TrendSignalAnalyzer.create_default_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Завантажено конфіг з {config_file}")
            return config
        except Exception as e:
            logger.error(f"Помилка читання конфігу {config_file}: {e}")
            logger.info("Використовується стандартний конфіг")
            return TrendSignalAnalyzer.create_default_config()

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

    async def fetch_multi_data(self, symbol: str, signal_time: datetime) -> dict:
        """Завантаження мультитаймфреймових даних (основний 5m)"""
        timeframes = {
            '5m': {'hours_before': 120, 'hours_after': 72},  # Збільшуємо час після для довших позицій
            '15m': {'hours_before': 360, 'hours_after': 168},
            '1h': {'hours_before': 720, 'hours_after': 336}
        }

        data = {}
        logger.debug(f"Завантаження даних для {symbol}, час сигналу: {signal_time}")

        for tf, config in timeframes.items():
            start = signal_time - timedelta(hours=config['hours_before'])
            end = signal_time + timedelta(hours=config['hours_after'])

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
            else:  # 1h
                periods = {'rsi': 14, 'rsi_sma': 10, 'sma_fast': 50, 'sma_slow': 200, 'adx': 14}
                ema_fast, ema_slow = 50, 200
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

    def analyze_trend(self, multi_data: dict, signal_time: datetime) -> dict:
        """Мультитаймфреймовий аналіз тренду (оптимізований для 5m)"""
        weights = {'5m': 0.4, '15m': 0.35, '1h': 0.25}
        weighted_trend = 0
        strength_scores = []

        logger.debug(f"Аналіз тренду для часу: {signal_time}")

        for tf, weight in weights.items():
            if tf not in multi_data:
                logger.debug(f"Немає даних для {tf}")
                continue

            df = multi_data[tf]
            df['time_diff'] = (df['datetime'] - signal_time).abs()
            closest_idx = df['time_diff'].idxmin()

            if pd.isna(closest_idx):
                logger.debug(f"Не знайдено індекс для {tf}")
                continue

            candle = df.iloc[closest_idx]

            # Розрахунок трендового скору
            trend_score = 0

            # EMA тренд (основний)
            ema_trend = candle.get('trend_ema', 0)
            trend_score += ema_trend * 0.4

            # SMA тренд (підтвердження)
            sma_trend = candle.get('trend_sma', 0)
            trend_score += sma_trend * 0.3

            # ADX направленість
            adx = candle.get('adx', 0)
            if adx > (20 if tf == '5m' else 25):
                adx_dir = 1 if candle.get('adx_pos', 0) > candle.get('adx_neg', 0) else -1
                trend_score += adx_dir * 0.3

            # Для 5m додаємо momentum
            if tf == '5m':
                momentum = candle.get('momentum', 0)
                if abs(momentum) > 0.1:
                    momentum_dir = 1 if momentum > 0 else -1
                    trend_score += momentum_dir * 0.2

            trend_score = max(-1, min(1, trend_score))
            strength_scores.append(abs(trend_score))
            weighted_trend += trend_score * weight

            logger.debug(f"Тренд {tf}: {trend_score:.3f} (вага: {weight})")

        result = {
            'overall_trend': round(weighted_trend, 3),
            'trend_strength': round(np.mean(strength_scores) if strength_scores else 0, 3),
            'trend_alignment': len(strength_scores) >= 2 and np.mean(strength_scores) > 0.3
        }

        logger.debug(f"Результат трендового аналізу: {result}")
        return result

    def apply_filters(self, multi_data: dict, signal_time: datetime, direction: str, config: dict) -> tuple:
        """Застосування фільтрів входу (адаптовано для 5m)"""
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

        # 1. Трендовий фільтр
        trend = self.analyze_trend(multi_data, signal_time)
        if filter_config.get('use_trend_filter', True):
            overall = trend['overall_trend']
            aligned = trend['trend_alignment']

            if (is_long and overall > 0.15 and aligned) or \
                    (not is_long and overall < -0.15 and aligned) or \
                    abs(overall) < 0.15:
                passed.append(f"Trend({overall:.2f})")
            else:
                failed.append(f"Trend({overall:.2f})")

        # 2. ADX фільтр
        adx = candle.get('adx', 0)
        min_adx = threshold_config.get('min_adx', 18)
        if filter_config.get('use_adx_filter', True):
            if adx >= min_adx:
                passed.append(f"ADX:{adx:.0f}")
            else:
                failed.append(f"ADX:{adx:.0f}")

        # 3. Волатільність
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

        # 4. Об'єм
        vol_ratio = candle.get('volume_ratio', 1)
        min_vol_ratio = threshold_config.get('min_volume_ratio', 0.8)
        if filter_config.get('use_volume_filter', True):
            if vol_ratio >= min_vol_ratio:
                passed.append(f"Vol:{vol_ratio:.1f}x")
            else:
                failed.append(f"Vol:{vol_ratio:.1f}x")

        # 5. Momentum фільтр
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

        # Трендове коригування
        trend = self.analyze_trend(multi_data, entry_time)
        if trend['trend_alignment'] and trend['trend_strength'] > 0.4:
            sl_mult = 0.85  # Менший SL при сильному тренді
        else:
            sl_mult = 1.15  # Більший SL при слабкому тренді

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
            if curr_rsi <= short_exit_zone:  # RSI в зоні шорту
                if prev_rsi > prev_rsi_sma and curr_rsi <= curr_rsi_sma:  # Перетин вниз
                    logger.info(
                        f"Зворотний сигнал LONG->SHORT: RSI {curr_rsi:.1f} перетнув SMA {curr_rsi_sma:.1f} вниз в зоні шорту")
                    return True, f'Reverse Signal: RSI Cross Down ({curr_rsi:.1f})'
        else:
            # Для SHORT позиції: шукаємо перетин RSI вгору через SMA в зоні лонгу
            if curr_rsi >= long_exit_zone:  # RSI в зоні лонгу
                if prev_rsi < prev_rsi_sma and curr_rsi >= curr_rsi_sma:  # Перетин вгору
                    logger.info(
                        f"Зворотний сигнал SHORT->LONG: RSI {curr_rsi:.1f} перетнув SMA {curr_rsi_sma:.1f} вгору в зоні лонгу")
                    return True, f'Reverse Signal: RSI Cross Up ({curr_rsi:.1f})'

        return False, ""

    def find_entry_exit(self, multi_data: dict, signal_time: datetime, direction: str,
                        entry_rsi: float, config: dict) -> dict:
        """Основна логіка пошуку входу та виходу з зворотними сигналами"""
        if '5m' not in multi_data:
            logger.warning("Немає даних 5m для аналізу")
            return None

        # Застосування фільтрів
        filter_passed, filter_info = self.apply_filters(multi_data, signal_time, direction, config)

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

        # Розрахунок стоп-лосу
        dynamic_sl_pct = self.calc_stop_loss(multi_data, entry_time, direction, config)

        if is_long:
            sl_price = entry_price * (1 - dynamic_sl_pct)
        else:
            sl_price = entry_price * (1 + dynamic_sl_pct)

        # Параметри пошуку
        trading_params = config.get('trading_parameters', {})
        max_hours = trading_params.get('max_hold_hours', 24)
        max_time = entry_time + timedelta(hours=max_hours)

        # Трейлінг стоп
        use_trailing = trading_params.get('use_trailing_stop', True)
        best_price = entry_price
        trailing_sl = sl_price

        # Пошук виходу
        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i]
            curr_time = candle['datetime']

            # Оновлення трейлінга
            if use_trailing:
                if is_long and candle['high'] > best_price:
                    best_price = candle['high']
                    trailing_sl = best_price * (1 - dynamic_sl_pct)
                elif not is_long and candle['low'] < best_price:
                    best_price = candle['low']
                    trailing_sl = best_price * (1 + dynamic_sl_pct)

            # Перевірка таймауту
            if curr_time > max_time:
                logger.debug(f"Таймаут {max_hours}h")
                return self._create_result(entry_time, entry_price, curr_time, candle['close'],
                                           direction, f'Time Exit ({max_hours}h)', filter_info, dynamic_sl_pct)

            # Перевірка стоп-лосу
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

            # ГОЛОВНА ЗМІНА: Перевірка зворотнього RSI сигналу
            reverse_exit, reverse_reason = self.check_reverse_signal_exit(multi_data, curr_time, direction, config)
            if reverse_exit:
                logger.info(f"Зворотний сигнал виходу: {reverse_reason}")
                return self._create_result(entry_time, entry_price, curr_time, candle['close'],
                                           direction, reverse_reason, filter_info, dynamic_sl_pct)

        # Якщо не знайдено вихід - остання свічка
        last = df.iloc[-1]
        logger.debug("Досягнуто кінця даних")
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
            'entry_time_str': entry_time.strftime('%d.%m.%Y %H:%M'),  # Додаємо строкове представлення
            'exit_time': exit_time,
            'exit_time_str': exit_time.strftime('%d.%m.%Y %H:%M'),  # Додаємо строкове представлення
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_percent': pnl_pct,
            'hold_time_hours': hold_hours,
            'exit_reason': reason,
            'status': 'Profit' if pnl_pct > 0 else 'Loss',
            'filter_info': filter_info,
            'dynamic_sl_pct': dynamic_sl_pct * 100,  # Відсоток SL
            'filtered_out': False
        }

        logger.debug(f"Результат: PnL={pnl_pct:.2f}%, час={hold_hours:.1f}h, причина={reason}")
        return result

    async def analyze_signals(self, signals: list, config: dict) -> list:
        """Основний метод аналізу"""
        results = []
        signals_by_pair = {}

        # Групування по парах
        for signal in signals:
            pair = signal['pair']
            if pair not in signals_by_pair:
                signals_by_pair[pair] = []
            signals_by_pair[pair].append(signal)

        logger.info(f"Аналіз {len(signals)} сигналів для {len(signals_by_pair)} пар")

        # Обробка кожної пари
        for pair_idx, (pair, pair_signals) in enumerate(signals_by_pair.items()):
            logger.info(f"Пара {pair} ({pair_idx + 1}/{len(signals_by_pair)}) - {len(pair_signals)} сигналів")

            # Часовий діапазон
            times = [self.parse_time(s['time']) for s in pair_signals]
            min_time = min(times)

            # Завантаження даних
            multi_data = await self.fetch_multi_data(pair, min_time)
            if not multi_data or '5m' not in multi_data:
                logger.warning(f"Немає даних для {pair}")
                continue

            # Аналіз сигналів
            for signal_idx, signal in enumerate(pair_signals):
                signal_time = self.parse_time(signal['time'])
                logger.debug(
                    f"Аналіз сигналу {signal_idx + 1}/{len(pair_signals)}: {signal['direction']} в {signal_time}")

                result = self.find_entry_exit(multi_data, signal_time, signal['direction'],
                                              signal['rsi'], config)

                if result:
                    result.update({
                        **signal,
                        'signal_time': signal_time.strftime('%d.%m.%Y %H:%M')
                    })

                    results.append(result)

            await asyncio.sleep(0.05)

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
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fields)
            writer.writeheader()

            for result in results:
                row = {}
                for field in fields:
                    value = result.get(field, '')
                    if field in ['rsi', 'entry_price', 'exit_price', 'pnl_percent',
                                 'hold_time_hours', 'dynamic_sl_pct']:
                        try:
                            row[field] = round(float(value), 4) if value != '' else 0
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
        """Виведення статистики"""
        if not results:
            logger.info("Немає результатів для статистики")
            return

        traded = [r for r in results if not r.get('filtered_out', False)]
        filtered = [r for r in results if r.get('filtered_out', False)]

        print(f"\n{'=' * 70}")
        print(f"СТАТИСТИКА ТОРГІВЛІ (Зворотні RSI сигнали)")
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
        for reason, stats in exit_stats.items():
            avg_pnl = stats['pnl'] / stats['count']
            pct = stats['count'] / len(traded) * 100
            print(f"{reason}: {stats['count']} ({pct:.1f}%) - Avg PnL: {avg_pnl:.2f}%")
            if 'Reverse Signal' in reason:
                reverse_signals += stats['count']

        if reverse_signals > 0:
            print(f"\nЗВОРОТНІ СИГНАЛИ:")
            print(f"Всього зворотних виходів: {reverse_signals} ({reverse_signals / len(traded) * 100:.1f}%)")

        logger.info(f"Статистика: {len(profitable)}/{len(traded)} прибуткових ({win_rate:.1f}%), "
                    f"середній PnL: {avg_pnl:.2f}%, загальний PnL: {total_pnl:.2f}%")


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description='Аналізатор з зворотними RSI сигналами')
    parser.add_argument('input_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='results_reverse_signals.csv', help='Файл результатів')
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
    parser.add_argument('--no-filters', action='store_true', help='Вимкнути всі фільтри')
    parser.add_argument('--no-trailing', action='store_true', help='Вимкнути трейлінг стоп')

    args = parser.parse_args()

    # Завантаження конфігурації
    config = TrendSignalAnalyzer.load_config(args.config)

    # Перевизначення параметрів з командного рядка
    if args.stop_loss is not None:
        config['trading_parameters']['stop_loss'] = args.stop_loss / 100
    if args.max_hold is not None:
        config['trading_parameters']['max_hold_hours'] = args.max_hold
    if args.long_exit is not None:
        config['rsi_parameters']['long_exit_zone'] = args.long_exit
    if args.short_exit is not None:
        config['rsi_parameters']['short_exit_zone'] = args.short_exit
    if args.min_adx is not None:
        config['filter_thresholds']['min_adx'] = args.min_adx
    if args.min_vol is not None:
        config['filter_thresholds']['min_volatility'] = args.min_vol
    if args.max_vol is not None:
        config['filter_thresholds']['max_volatility'] = args.max_vol
    if args.min_vol_ratio is not None:
        config['filter_thresholds']['min_volume_ratio'] = args.min_vol_ratio
    if args.no_filters:
        config['filters']['use_filters'] = False
    if args.no_trailing:
        config['trading_parameters']['use_trailing_stop'] = False

    # Виведення інформації про конфігурацію
    trading_params = config['trading_parameters']
    rsi_params = config['rsi_parameters']

    print(f"Аналізатор сигналів з зворотними RSI сигналами")
    print(f"Файл сигналів: {args.input_file}")
    print(f"Конфігурація: {args.config}")
    print(f"Лог файл: enhanced_analyzer.log")
    print(f"Базовий SL: {trading_params['stop_loss'] * 100:.1f}% (динамічний)")
    print(f"Макс. час утримання: {trading_params['max_hold_hours']}h")
    print(f"Трейлінг стоп: {'Увімкнено' if trading_params['use_trailing_stop'] else 'Вимкнено'}")
    print(f"Зони RSI виходу: Long>{rsi_params['long_exit_zone']}, Short<{rsi_params['short_exit_zone']}")
    print(f"Фільтри: {'Увімкнено' if config['filters']['use_filters'] else 'Вимкнено'}")

    logger.info("=" * 60)
    logger.info("ЗАПУСК АНАЛІЗАТОРА З ЗВОРОТНИМИ RSI СИГНАЛАМИ")
    logger.info("=" * 60)
    logger.info(f"Файл сигналів: {args.input_file}")
    logger.info(f"Файл результатів: {args.output}")
    logger.info(f"Конфігурація: {args.config}")

    analyzer = TrendSignalAnalyzer()

    # Читання та аналіз
    signals = analyzer.read_signals_csv(args.input_file)
    if not signals:
        logger.error("Помилка читання сигналів або файл порожній")
        print("Помилка читання сигналів")
        return

    try:
        logger.info(f"Початок аналізу {len(signals)} сигналів")
        results = await analyzer.analyze_signals(signals, config)
        if results:
            analyzer.save_results(results, args.output)
            print(f"\nРезультати збережено в {args.output}")
            logger.info(f"Аналіз завершено успішно. Результати збережено в {args.output}")
        else:
            logger.warning("Немає результатів для збереження")
            print("Немає результатів")
    except Exception as e:
        logger.error(f"Критична помилка під час аналізу: {e}", exc_info=True)
        print(f"Помилка: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Аналіз перервано користувачем")
        print("\nПерервано користувачем")
    except Exception as e:
        logger.error(f"Критична помилка: {e}", exc_info=True)
        print(f"Критична помилка: {e}")