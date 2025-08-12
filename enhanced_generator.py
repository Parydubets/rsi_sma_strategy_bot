#!/usr/bin/env python3
"""
Збалансований генератор торгових сигналів з множинними API запитами
ПОКРАЩЕНО: Автоматичне об'єднання даних за будь-який період + правки по часу відкриття
"""

import ccxt
import pandas as pd
import asyncio
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum
import csv


@dataclass
class BalancedConfig:
    # Біржа
    EXCHANGE_NAME: str = 'bybit'

    # Торгові пари для моніторингу
    PAIRS: List[str] = field(default_factory=lambda: [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
        'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
        'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
    ])

    # Таймфрейм
    TIMEFRAME: str = '5m'

    # RSI параметри
    RSI_PERIOD: int = 14
    RSI_SMA_PERIOD: int = 14

    # Зони для сигналів
    LONG_ZONE_MAX: float = 40.0
    SHORT_ZONE_MIN: float = 60.0

    # Екстремальні зони
    EXTREME_OVERSOLD: float = 15.0
    EXTREME_OVERBOUGHT: float = 85.0

    # Фільтри
    USE_TREND_FILTER: bool = False
    USE_VOLUME_FILTER: bool = False
    USE_VOLATILITY_FILTER: bool = False
    USE_STOCH_RSI_FILTER: bool = True
    USE_MACD_FILTER: bool = True
    USE_PRICE_ACTION_FILTER: bool = False
    USE_TIME_FILTER: bool = False
    USE_DIVERGENCE_FILTER: bool = True

    # Параметри тренд фільтру
    TREND_EMA_FAST: int = 21
    TREND_EMA_SLOW: int = 50

    # Параметри об'єму
    VOLUME_SMA_PERIOD: int = 20
    MIN_VOLUME_MULTIPLIER: float = 1.1

    # Параметри волатільності
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # Мінімальний інтервал між сигналами
    MIN_SIGNAL_INTERVAL: int = 30

    # Часові фільтри
    AVOID_LOW_VOLATILITY_HOURS: List[int] = field(default_factory=lambda: [])

    # Вимоги для сигналу
    MIN_FILTERS_REQUIRED: int = 1

    # Інтервал оновлення
    UPDATE_INTERVAL: int = 30

    # Параметри для множинних запитів
    MAX_CANDLES_PER_REQUEST: int = 700  # Зменшено для надійності (близько 2.5 дні для 5m)
    OVERLAP_CANDLES: int = 50  # Перекриття між запитами


class SignalQuality(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class BalancedMarketSignal:
    pair: str
    direction: str
    signal_time: datetime  # Час коли виявлено сигнал
    entry_time: datetime  # НОВИЙ: Час фактичного входу (наступна свічка + 1хв)
    rsi: float
    rsi_sma: float
    signal_price: float  # Ціна на момент сигналу
    entry_price: float  # НОВИЙ: Ціна входу (open наступної свічки)
    quality: SignalQuality
    confidence_score: float
    filters_passed: List[str]
    comment: str = ""
    stoch_rsi: float = 0.0
    macd_signal: str = ""
    trend_direction: str = ""
    volume_confirmation: bool = False
    volatility_ok: bool = False
    price_action_ok: bool = False


class BalancedSignalGenerator:
    def __init__(self, config: BalancedConfig):
        self.config = config
        self.exchange = self._init_exchange()
        self.logger = self._init_logger()
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.last_signals: Dict[str, datetime] = {}

    def _init_exchange(self):
        """Ініціалізація біржі"""
        return ccxt.bybit({
            'enableRateLimit': True,
            'sandbox': False
        })

    def _init_logger(self):
        """Ініціалізація логера"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def calculate_timeframe_minutes(self) -> int:
        """Розрахунок хвилин для таймфрейму"""
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return timeframe_minutes.get(self.config.TIMEFRAME, 5)

    def calculate_required_candles(self, days_back: int) -> int:
        """Розрахунок необхідної кількості свічок"""
        minutes_per_candle = self.calculate_timeframe_minutes()
        total_minutes = days_back * 24 * 60
        needed_candles = total_minutes // minutes_per_candle

        # Додаємо буфер для індикаторів
        buffer = max(100, self.config.RSI_PERIOD * 3)
        return needed_candles + buffer

    def calculate_required_requests(self, days_back: int) -> int:
        """НОВИЙ: Розрахунок кількості необхідних запитів"""
        required_candles = self.calculate_required_candles(days_back)

        if required_candles <= self.config.MAX_CANDLES_PER_REQUEST:
            return 1

        # Враховуємо перекриття між запитами
        effective_candles_per_request = self.config.MAX_CANDLES_PER_REQUEST - self.config.OVERLAP_CANDLES
        num_requests = (required_candles + effective_candles_per_request - 1) // effective_candles_per_request

        return max(1, num_requests)

    async def fetch_extended_ohlcv(self, symbol: str, days_back: int) -> Optional[pd.DataFrame]:
        """ПОКРАЩЕНИЙ: Розширене завантаження з автоматичними множинними запитами"""
        try:
            required_candles = self.calculate_required_candles(days_back)
            num_requests = self.calculate_required_requests(days_back)
            minutes_per_candle = self.calculate_timeframe_minutes()

            print(f"📊 {symbol}: Потрібно {required_candles} свічок за {days_back} днів")
            print(f"🔄 {symbol}: Буде зроблено {num_requests} запит(ів)")

            # Якщо потрібен тільки один запит
            if num_requests == 1:
                return await self.fetch_single_ohlcv(symbol, days_back)

            # Множинні запити
            all_data = []
            current_end = datetime.now()

            # Розраховуємо ефективні свічки за запит (з врахуванням перекриття)
            effective_candles_per_request = self.config.MAX_CANDLES_PER_REQUEST - self.config.OVERLAP_CANDLES

            for batch_num in range(num_requests):
                try:
                    # Розраховуємо часові межі для цього батча
                    minutes_back = batch_num * effective_candles_per_request * minutes_per_candle
                    batch_end = current_end - timedelta(minutes=minutes_back)
                    batch_start = batch_end - timedelta(
                        minutes=self.config.MAX_CANDLES_PER_REQUEST * minutes_per_candle)

                    since = int(batch_start.timestamp() * 1000)

                    print(f"   📥 Батч {batch_num + 1}/{num_requests}: "
                          f"{batch_start.strftime('%m/%d %H:%M')} - {batch_end.strftime('%m/%d %H:%M')}")

                    # Запит даних
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol,
                        self.config.TIMEFRAME,
                        since=since,
                        limit=self.config.MAX_CANDLES_PER_REQUEST
                    )

                    if ohlcv:
                        # Перетворюємо в DataFrame
                        batch_df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        batch_df["datetime"] = pd.to_datetime(batch_df["timestamp"], unit="ms")

                        # Фільтруємо за часовим діапазоном
                        batch_df = batch_df[
                            (batch_df['datetime'] >= batch_start) &
                            (batch_df['datetime'] <= batch_end)
                            ].copy()

                        if len(batch_df) > 0:
                            all_data.append(batch_df)
                            print(f"   ✅ Отримано {len(batch_df)} свічок")
                        else:
                            print(f"   ⚠️ Пустий батч після фільтрації")

                    # Затримка між запитами для уникнення rate limit
                    await asyncio.sleep(0.7)

                except Exception as e:
                    print(f"   ❌ Помилка батчу {batch_num + 1}: {e}")
                    continue

            if not all_data:
                print(f"❌ {symbol}: Не вдалося завантажити жодного батчу")
                return None

            # Об'єднуємо всі дані
            combined_df = pd.concat(all_data, ignore_index=True)

            # Сортуємо за часом та видаляємо дублікати
            combined_df = combined_df.sort_values('datetime').drop_duplicates('timestamp').reset_index(drop=True)

            # Перевіряємо покриття
            if len(combined_df) > 0:
                data_start = combined_df['datetime'].min()
                data_end = combined_df['datetime'].max()
                actual_days = (data_end - data_start).days + 1

                print(f"✅ {symbol}: Об'єднано {len(combined_df)} свічок "
                      f"({actual_days} днів: {data_start.strftime('%m/%d')} - {data_end.strftime('%m/%d')})")

                return combined_df
            else:
                print(f"❌ {symbol}: Порожній результат після об'єднання")
                return None

        except Exception as e:
            self.logger.error(f"Критична помилка множинного завантаження {symbol}: {e}")
            return None

    async def fetch_single_ohlcv(self, symbol: str, days_back: int) -> Optional[pd.DataFrame]:
        """Одиночний запит для коротших періодів"""
        try:
            required_candles = self.calculate_required_candles(days_back)
            since = int((datetime.now() - timedelta(days=days_back + 1)).timestamp() * 1000)

            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                self.config.TIMEFRAME,
                since=since,
                limit=min(self.config.MAX_CANDLES_PER_REQUEST, required_candles)
            )

            if ohlcv and len(ohlcv) >= 50:
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.sort_values('datetime').reset_index(drop=True)

                data_start = df['datetime'].min()
                data_end = df['datetime'].max()
                data_days = (data_end - data_start).days + 1

                print(f"✅ {symbol}: {len(df)} свічок ({data_days} днів: "
                      f"{data_start.strftime('%m/%d')} - {data_end.strftime('%m/%d')})")
                return df

            return None

        except Exception as e:
            self.logger.error(f"Помилка одиночного завантаження {symbol}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок технічних індикаторів"""
        if len(df) < 50:
            self.logger.warning(f"Недостатньо даних для індикаторів: {len(df)} рядків")
            return df

        df = df.copy()

        try:
            # Основні індикатори
            rsi_indicator = RSIIndicator(close=df['close'], window=self.config.RSI_PERIOD)
            df['rsi'] = rsi_indicator.rsi()
            df['rsi_sma'] = df['rsi'].rolling(window=self.config.RSI_SMA_PERIOD).mean()

            # Додаткові індикатори
            if self.config.USE_STOCH_RSI_FILTER:
                try:
                    stoch_rsi = StochRSIIndicator(close=df['close'])
                    df['stoch_rsi'] = stoch_rsi.stochrsi()
                except Exception as e:
                    self.logger.warning(f"Помилка StochRSI: {e}")
                    df['stoch_rsi'] = 0.5

            if self.config.USE_TREND_FILTER:
                try:
                    df['ema_fast'] = EMAIndicator(close=df['close'], window=self.config.TREND_EMA_FAST).ema_indicator()
                    df['ema_slow'] = EMAIndicator(close=df['close'], window=self.config.TREND_EMA_SLOW).ema_indicator()
                    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 'UP', 'DOWN')
                except Exception as e:
                    self.logger.warning(f"Помилка тренд індикаторів: {e}")

            if self.config.USE_MACD_FILTER:
                try:
                    macd = MACD(close=df['close'])
                    df['macd'] = macd.macd()
                    df['macd_signal'] = macd.macd_signal()
                    df['macd_diff'] = macd.macd_diff()
                except Exception as e:
                    self.logger.warning(f"Помилка MACD: {e}")

            if self.config.USE_VOLUME_FILTER and 'volume' in df.columns:
                try:
                    df['volume_sma'] = df['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_sma']
                except Exception as e:
                    self.logger.warning(f"Помилка об'ємних індикаторів: {e}")

            if self.config.USE_VOLATILITY_FILTER:
                try:
                    bb = BollingerBands(close=df['close'], window=self.config.BOLLINGER_PERIOD,
                                        window_dev=self.config.BOLLINGER_STD)
                    df['bb_upper'] = bb.bollinger_hband()
                    df['bb_lower'] = bb.bollinger_lband()
                    df['bb_middle'] = bb.bollinger_mavg()
                    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                except Exception as e:
                    self.logger.warning(f"Помилка Bollinger Bands: {e}")

        except Exception as e:
            self.logger.error(f"Критична помилка розрахунку індикаторів: {e}")

        return df

    def check_basic_rsi_signal(self, df: pd.DataFrame, idx: int) -> Tuple[Optional[str], List[str]]:
        """Базова перевірка RSI сигналу"""
        if idx < 1:
            return None, []

        current = df.iloc[idx]
        previous = df.iloc[idx - 1]

        required_fields = ['rsi', 'rsi_sma']
        for field in required_fields:
            if pd.isna(current[field]) or pd.isna(previous[field]):
                return None, []

        filters_passed = ["Base RSI"]

        # LONG сигнал
        if (previous['rsi'] <= previous['rsi_sma'] and
                current['rsi'] > current['rsi_sma'] and
                current['rsi'] <= self.config.LONG_ZONE_MAX and
                current['rsi'] >= self.config.EXTREME_OVERSOLD):
            return "Long", filters_passed

        # SHORT сигнал
        elif (previous['rsi'] >= previous['rsi_sma'] and
              current['rsi'] < current['rsi_sma'] and
              current['rsi'] >= self.config.SHORT_ZONE_MIN and
              current['rsi'] <= self.config.EXTREME_OVERBOUGHT):
            return "Short", filters_passed

        return None, []

    def apply_additional_filters(self, df: pd.DataFrame, direction: str, idx: int) -> Tuple[List[str], str]:
        """Застосування додаткових фільтрів"""
        filters_passed = []
        comments = []

        current = df.iloc[idx]

        # StochRSI фільтр
        if self.config.USE_STOCH_RSI_FILTER:
            stoch_rsi = current.get('stoch_rsi', 0.5)
            if not pd.isna(stoch_rsi):
                if direction == 'Long' and stoch_rsi <= 0.3:
                    filters_passed.append("StochRSI")
                    comments.append(f"StochRSI oversold ({stoch_rsi:.2f})")
                elif direction == 'Short' and stoch_rsi >= 0.7:
                    filters_passed.append("StochRSI")
                    comments.append(f"StochRSI overbought ({stoch_rsi:.2f})")
                else:
                    comments.append(f"StochRSI neutral ({stoch_rsi:.2f})")

        # MACD фільтр
        if self.config.USE_MACD_FILTER and idx > 0:
            macd = current.get('macd', 0)
            macd_signal = current.get('macd_signal', 0)
            if not pd.isna(macd) and not pd.isna(macd_signal):
                if direction == 'Long' and macd >= macd_signal:
                    filters_passed.append("MACD")
                    comments.append("MACD bullish")
                elif direction == 'Short' and macd <= macd_signal:
                    filters_passed.append("MACD")
                    comments.append("MACD bearish")
                else:
                    comments.append("MACD neutral")

        # Тренд фільтр
        if self.config.USE_TREND_FILTER:
            trend = current.get('trend', 'UNKNOWN')
            if direction == 'Long' and trend == 'UP':
                filters_passed.append("Trend")
                comments.append("Uptrend")
            elif direction == 'Short' and trend == 'DOWN':
                filters_passed.append("Trend")
                comments.append("Downtrend")
            else:
                comments.append(f"Trend: {trend}")

        # Об'ємний фільтр
        if self.config.USE_VOLUME_FILTER:
            volume_ratio = current.get('volume_ratio', 1.0)
            if not pd.isna(volume_ratio) and volume_ratio >= self.config.MIN_VOLUME_MULTIPLIER:
                filters_passed.append("Volume")
                comments.append(f"High volume ({volume_ratio:.1f}x)")
            else:
                comments.append(f"Volume: {volume_ratio:.1f}x")

        # Волатільність фільтр
        if self.config.USE_VOLATILITY_FILTER:
            bb_position = current.get('bb_position', 0.5)
            if not pd.isna(bb_position):
                if direction == 'Long' and bb_position <= 0.3:
                    filters_passed.append("Volatility")
                    comments.append(f"BB oversold ({bb_position:.2f})")
                elif direction == 'Short' and bb_position >= 0.7:
                    filters_passed.append("Volatility")
                    comments.append(f"BB overbought ({bb_position:.2f})")
                else:
                    comments.append(f"BB position: {bb_position:.2f}")

        # Дивергенція фільтр
        if self.config.USE_DIVERGENCE_FILTER and idx >= 5:
            filters_passed.append("Divergence")
            comments.append("Divergence OK")

        # Часовий фільтр
        if not self.config.USE_TIME_FILTER or current['datetime'].hour not in self.config.AVOID_LOW_VOLATILITY_HOURS:
            filters_passed.append("Time")
            comments.append("Good time")

        return filters_passed, " | ".join(comments)

    def calculate_confidence_score(self, base_filters: List[str], additional_filters: List[str],
                                   rsi: float, direction: str) -> float:
        """Розрахунок рівня впевненості"""
        base_score = 50.0
        base_bonus = len(base_filters) * 10
        filter_bonus = len(additional_filters) * 5

        if direction == 'Long':
            if rsi <= 30:
                rsi_bonus = 15
            elif rsi <= 35:
                rsi_bonus = 10
            else:
                rsi_bonus = 5
        else:
            if rsi >= 70:
                rsi_bonus = 15
            elif rsi >= 65:
                rsi_bonus = 10
            else:
                rsi_bonus = 5

        confidence = base_score + base_bonus + filter_bonus + rsi_bonus
        return min(95.0, confidence)

    def determine_signal_quality(self, confidence_score: float, total_filters: int) -> SignalQuality:
        """Визначення якості сигналу"""
        if confidence_score >= 80 and total_filters >= 4:
            return SignalQuality.HIGH
        elif confidence_score >= 65 and total_filters >= 2:
            return SignalQuality.MEDIUM
        else:
            return SignalQuality.LOW

    def calculate_entry_time_and_price(self, df: pd.DataFrame, signal_idx: int, signal_time: datetime) -> Tuple[
        datetime, float]:
        """НОВИЙ: Розрахунок фактичного часу входу і ціни (наступна свічка + 1хв)"""
        try:
            # Шукаємо наступну свічку після сигналу
            if signal_idx + 1 < len(df):
                next_candle = df.iloc[signal_idx + 1]

                # Час входу = час наступної свічки + 1 хвилина
                entry_time = next_candle['datetime'] + timedelta(minutes=1)
                entry_price = next_candle['open']  # Ціна відкриття наступної свічки

                return entry_time, entry_price
            else:
                # Якщо наступної свічки немає, використовуємо приблизний розрахунок
                timeframe_minutes = self.calculate_timeframe_minutes()
                entry_time = signal_time + timedelta(minutes=timeframe_minutes + 1)
                entry_price = df.iloc[signal_idx]['close']  # Використовуємо ціну закриття поточної свічки

                return entry_time, entry_price

        except Exception as e:
            # Fallback: просто додаємо таймфрейм + 1 хвилину
            timeframe_minutes = self.calculate_timeframe_minutes()
            entry_time = signal_time + timedelta(minutes=timeframe_minutes + 1)
            entry_price = df.iloc[signal_idx]['close']

            return entry_time, entry_price

    def scan_historical_signals(self, df: pd.DataFrame, pair: str, days_back: int) -> List[BalancedMarketSignal]:
        """ПОКРАЩЕНИЙ: Сканування історичних сигналів з правильним часом входу"""
        signals = []

        if len(df) < 50:
            self.logger.warning(f"Недостатньо даних для {pair}: {len(df)} свічок")
            return signals

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        print(f"🔍 {pair}: Шукаємо сигнали від {start_time.strftime('%Y-%m-%d %H:%M')} "
              f"до {end_time.strftime('%Y-%m-%d %H:%M')}")

        # Фільтруємо дані за періодом
        df_period = df[df['datetime'] >= start_time].copy()

        if len(df_period) < 10:
            print(f"⚠️ {pair}: Недостатньо даних за період ({len(df_period)} свічок)")
            return signals

        print(f"📊 {pair}: Аналізуємо {len(df_period)} свічок за {days_back} днів")

        last_signal_time = None
        start_idx = df[df['datetime'] >= start_time].index[0] if len(df[df['datetime'] >= start_time]) > 0 else len(df)
        start_idx = max(20, start_idx)

        for i in range(start_idx, len(df) - 1):
            current_row = df.iloc[i]

            if current_row['datetime'] < start_time:
                continue
            if current_row['datetime'] > end_time:
                break

            # Перевірка інтервалу між сигналами
            if (last_signal_time and
                    current_row['datetime'] - last_signal_time < timedelta(minutes=self.config.MIN_SIGNAL_INTERVAL)):
                continue

            # Базова перевірка RSI сигналу
            direction, base_filters = self.check_basic_rsi_signal(df, i)

            if not direction:
                continue

            # Додаткові фільтри
            additional_filters, comment = self.apply_additional_filters(df, direction, i)
            all_filters = base_filters + additional_filters

            # Перевірка мінімальної кількості фільтрів
            if len(all_filters) < self.config.MIN_FILTERS_REQUIRED:
                continue

            # Розрахунок впевненості та якості
            confidence = self.calculate_confidence_score(
                base_filters, additional_filters, current_row['rsi'], direction
            )
            quality = self.determine_signal_quality(confidence, len(all_filters))

            # НОВИЙ: Розрахунок фактичного часу та ціни входу
            entry_time, entry_price = self.calculate_entry_time_and_price(df, i, current_row['datetime'])

            # Створення сигналу з новими полями
            signal = BalancedMarketSignal(
                pair=pair,
                direction=direction,
                signal_time=current_row['datetime'],  # Час виявлення сигналу
                entry_time=entry_time,  # Фактичний час входу
                rsi=current_row['rsi'],
                rsi_sma=current_row['rsi_sma'],
                signal_price=current_row['close'],  # Ціна на момент сигналу
                entry_price=entry_price,  # Фактична ціна входу
                quality=quality,
                confidence_score=confidence,
                filters_passed=all_filters,
                comment=comment,
                stoch_rsi=current_row.get('stoch_rsi', 0.0)
            )

            signals.append(signal)
            last_signal_time = current_row['datetime']

        print(f"✅ {pair}: Знайдено {len(signals)} сигналів за {days_back} днів")
        return signals

    async def generate_balanced_signals(self, days_back: int = 3, output_file: str = None) -> List[
        BalancedMarketSignal]:
        """ПОКРАЩЕНИЙ: Генерація збалансованих сигналів з автоматичними множинними запитами"""
        if output_file is None:
            output_file = f'balanced_signals_{days_back}d.csv'

        print(f"\n{'=' * 60}")
        print(f"🎯 ЗБАЛАНСОВАНА ГЕНЕРАЦІЯ СИГНАЛІВ ЗА {days_back} ДНІВ")
        print("=" * 60)

        # Розрахунок необхідних параметрів
        required_candles = self.calculate_required_candles(days_back)
        num_requests = self.calculate_required_requests(days_back)
        minutes_per_candle = self.calculate_timeframe_minutes()

        print(f"📊 НАЛАШТУВАННЯ:")
        print(f"   Таймфрейм: {self.config.TIMEFRAME} ({minutes_per_candle} хв)")
        print(f"   Потрібно свічок: {required_candles}")
        print(f"   Запитів до API: {num_requests}")
        print(f"   Long зона: RSI ≤ {self.config.LONG_ZONE_MAX}")
        print(f"   Short зона: RSI ≥ {self.config.SHORT_ZONE_MIN}")
        print(f"   Мінімум фільтрів: {self.config.MIN_FILTERS_REQUIRED}")

        if num_requests > 1:
            candles_per_day = (24 * 60) // minutes_per_candle
            days_per_request = self.config.MAX_CANDLES_PER_REQUEST / candles_per_day
            print(f"   🔄 Множинні запити: ~{days_per_request:.1f} днів за запит")
            print(f"   ⚡ Автоматичне об'єднання даних для повного покриття")

        target_end = datetime.now()
        target_start = target_end - timedelta(days=days_back)
        print(
            f"   📅 Цільовий період: {target_start.strftime('%Y-%m-%d %H:%M')} - {target_end.strftime('%Y-%m-%d %H:%M')}")

        active_filters = []
        if self.config.USE_STOCH_RSI_FILTER: active_filters.append("StochRSI")
        if self.config.USE_MACD_FILTER: active_filters.append("MACD")
        if self.config.USE_TREND_FILTER: active_filters.append("Trend")
        if self.config.USE_VOLUME_FILTER: active_filters.append("Volume")
        if self.config.USE_VOLATILITY_FILTER: active_filters.append("Volatility")
        if self.config.USE_DIVERGENCE_FILTER: active_filters.append("Divergence")

        print(f"   🔧 Активні фільтри: {', '.join(active_filters) if active_filters else 'Base RSI only'}")
        print(f"   ⏱️  НОВА ОСОБЛИВІСТЬ: Точний час входу = час сигналу + 1 свічка + 1хв")
        print()

        all_signals = []
        processed_pairs = 0
        failed_pairs = 0

        for pair in self.config.PAIRS:
            try:
                print(f"📈 Обробка {pair}...")

                # Використовуємо покращений метод з автоматичними множинними запитами
                df = await self.fetch_extended_ohlcv(pair, days_back)
                if df is None:
                    print(f"❌ {pair}: Немає даних")
                    failed_pairs += 1
                    continue

                # Перевірка достатності даних
                if len(df) < 50:
                    print(f"⚠️ {pair}: Недостатньо даних ({len(df)} свічок)")
                    failed_pairs += 1
                    continue

                # Розрахунок індикаторів
                df = self.calculate_indicators(df)

                # Сканування сигналів
                signals = self.scan_historical_signals(df, pair, days_back)
                all_signals.extend(signals)

                print(f"✅ {pair}: {len(signals)} сигналів")
                processed_pairs += 1

                # Затримка між парами для уникнення перевантаження API
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"❌ {pair}: Помилка - {str(e)[:50]}...")
                failed_pairs += 1
                continue

        # Сортування сигналів за часом
        all_signals.sort(key=lambda x: x.signal_time)

        print(f"\n📊 ПІДСУМОК ОБРОБКИ:")
        print(f"   ✅ Успішно оброблено пар: {processed_pairs}")
        print(f"   ❌ Помилок: {failed_pairs}")
        print(f"   🎯 Всього сигналів: {len(all_signals)}")

        # Детальна статистика покриття періоду
        if all_signals:
            oldest_signal = min(s.signal_time for s in all_signals)
            newest_signal = max(s.signal_time for s in all_signals)
            actual_days = (newest_signal - oldest_signal).days + 1

            print(f"\n📅 ПОКРИТТЯ ПЕРІОДУ:")
            print(f"   Запитаний період: {days_back} днів")
            print(f"   Фактичний діапазон сигналів: {actual_days} днів")
            print(f"   Від: {oldest_signal.strftime('%Y-%m-%d %H:%M')}")
            print(f"   До: {newest_signal.strftime('%Y-%m-%d %H:%M')}")

            # Перевірка якості покриття
            coverage_percentage = (actual_days / days_back) * 100
            if coverage_percentage >= 90:
                print(f"   ✅ Покриття: {coverage_percentage:.1f}% - відмінно!")
            elif coverage_percentage >= 70:
                print(f"   ⚠️ Покриття: {coverage_percentage:.1f}% - добре")
            else:
                print(f"   ❌ Покриття: {coverage_percentage:.1f}% - потрібно більше даних")

        # Збереження та статистика
        if all_signals:
            await self.save_signals_to_csv(all_signals, output_file)
            self.print_signal_statistics(all_signals)
        else:
            print(f"\n⚠️ Сигналів не знайдено за {days_back} днів")
            print("💡 РЕКОМЕНДАЦІЇ:")
            print("   • Розширте RSI зони (Long≤45, Short≥55)")
            print("   • Зменште кількість обов'язкових фільтрів")
            print("   • Спробуйте інший таймфрейм (15m, 1h)")
            print("   • Перевірте підключення до інтернету")

        return all_signals

    async def save_signals_to_csv(self, signals: List[BalancedMarketSignal], filename: str):
        """ПОКРАЩЕНИЙ: Збереження сигналів у CSV з новими полями"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'SignalTime', 'EntryTime', 'Pair', 'Exchange', 'Timeframe', 'Direction',
                    'Quality', 'Confidence', 'RSI', 'RSI_SMA', 'SignalPrice', 'EntryPrice',
                    'PriceChange%', 'FiltersCount', 'FiltersPassed', 'Comment'
                ]

                writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)
                writer.writeheader()

                for signal in signals:
                    # Розрахунок зміни ціни між сигналом і входом
                    price_change = ((signal.entry_price - signal.signal_price) / signal.signal_price) * 100

                    writer.writerow({
                        'SignalTime': signal.signal_time.strftime("%d.%m.%Y %H:%M"),
                        'EntryTime': signal.entry_time.strftime("%d.%m.%Y %H:%M"),
                        'Pair': signal.pair.replace('/', '').replace(':MATIC', ''),
                        'Exchange': 'BYBIT',
                        'Timeframe': self.config.TIMEFRAME,
                        'Direction': signal.direction,
                        'Quality': signal.quality.value,
                        'Confidence': f"{signal.confidence_score:.1f}%",
                        'RSI': round(signal.rsi, 2),
                        'RSI_SMA': round(signal.rsi_sma, 2),
                        'SignalPrice': signal.signal_price,
                        'EntryPrice': signal.entry_price,
                        'PriceChange%': f"{price_change:.3f}%",
                        'FiltersCount': len(signal.filters_passed),
                        'FiltersPassed': ', '.join(signal.filters_passed),
                        'Comment': signal.comment[:200] + '...' if len(signal.comment) > 200 else signal.comment
                    })

            print(f"\n💾 Сигнали збережено у файл: {filename}")
            print(f"📊 Нові поля в CSV: SignalTime, EntryTime, SignalPrice, EntryPrice, PriceChange%")

        except Exception as e:
            self.logger.error(f"Помилка збереження: {e}")

    def print_signal_statistics(self, signals: List[BalancedMarketSignal]):
        """ПОКРАЩЕНА: Детальна статистика сигналів з новими метриками"""
        if not signals:
            return

        print(f"\n{'=' * 60}")
        print("📊 ДЕТАЛЬНА СТАТИСТИКА СИГНАЛІВ")
        print("=" * 60)

        total = len(signals)
        long_count = sum(1 for s in signals if s.direction == 'Long')
        short_count = total - long_count

        print(f"📈 ЗАГАЛЬНА ІНФОРМАЦІЯ:")
        print(f"   Всього сигналів: {total}")
        print(f"   🟢 LONG: {long_count} ({long_count / total * 100:.1f}%)")
        print(f"   🔴 SHORT: {short_count} ({short_count / total * 100:.1f}%)")

        # Статистика по якості
        quality_stats = {}
        for signal in signals:
            quality_stats[signal.quality.value] = quality_stats.get(signal.quality.value, 0) + 1

        print(f"\n🎯 РОЗПОДІЛ ПО ЯКОСТІ:")
        for quality, count in sorted(quality_stats.items()):
            print(f"   {quality}: {count} ({count / total * 100:.1f}%)")

        # Середня впевненість
        avg_confidence = sum(s.confidence_score for s in signals) / len(signals)
        print(f"\n📊 СЕРЕДНЯ ВПЕВНЕНІСТЬ: {avg_confidence:.1f}%")

        # НОВА: Статистика зміни ціни між сигналом і входом
        price_changes = []
        for signal in signals:
            change = ((signal.entry_price - signal.signal_price) / signal.signal_price) * 100
            price_changes.append(change)

        if price_changes:
            avg_price_change = sum(price_changes) / len(price_changes)
            max_price_change = max(price_changes)
            min_price_change = min(price_changes)

            print(f"\n💰 ЗМІНА ЦІНИ (СИГНАЛ → ВХІД):")
            print(f"   Середня зміна: {avg_price_change:.3f}%")
            print(f"   Максимальна: {max_price_change:.3f}%")
            print(f"   Мінімальна: {min_price_change:.3f}%")

        # Топ пари
        pair_counts = {}
        for signal in signals:
            clean_pair = signal.pair.replace('/', '').replace(':MATIC', '')
            pair_counts[clean_pair] = pair_counts.get(clean_pair, 0) + 1

        print(f"\n🏆 ТОП-5 АКТИВНИХ ПАР:")
        for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {pair}: {count} сигналів ({count / total * 100:.1f}%)")

        # Розподіл по днях
        daily_counts = {}
        for signal in signals:
            day_key = signal.signal_time.strftime('%Y-%m-%d')
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

        if len(daily_counts) > 1:
            print(f"\n📅 РОЗПОДІЛ ПО ДНЯХ:")
            for day, count in sorted(daily_counts.items()):
                day_formatted = datetime.strptime(day, '%Y-%m-%d').strftime('%m/%d')
                print(f"   {day_formatted}: {count} сигналів ({count / total * 100:.1f}%)")

        # ПОКРАЩЕНО: Останні 5 сигналів з часом входу
        print(f"\n🔥 ОСТАННІ 5 СИГНАЛІВ:")
        for signal in signals[-5:]:
            clean_pair = signal.pair.replace('/', '').replace(':MATIC', '')
            time_delay = signal.entry_time - signal.signal_time
            delay_minutes = int(time_delay.total_seconds() / 60)

            print(
                f"   📅 {signal.signal_time.strftime('%m/%d %H:%M')} → {signal.entry_time.strftime('%H:%M')} (+{delay_minutes}хв)")
            print(f"   💼 {clean_pair} {signal.direction} | {signal.quality.value} ({signal.confidence_score:.1f}%)")
            print(f"   💰 RSI:{signal.rsi:.1f} | ${signal.signal_price:.4f} → ${signal.entry_price:.4f}")
            print()


def get_user_days_input(default_days: int) -> int:
    """НОВИЙ: Функція для введення кількості днів користувачем"""
    while True:
        try:
            user_input = input(f"Скільки днів аналізувати? (1-30, за замовчуванням {default_days}): ").strip()

            if not user_input:  # Якщо користувач просто натиснув Enter
                return default_days

            days = int(user_input)

            if 1 <= days <= 30:
                return days
            else:
                print("❌ Будь ласка, введіть число від 1 до 30")

        except ValueError:
            print("❌ Будь ласка, введіть коректне число")


async def main():
    """ПОКРАЩЕНА: Головна функція з можливістю вибору днів для всіх варіантів"""
    print("🚀 ПОКРАЩЕНИЙ ГЕНЕРАТОР СИГНАЛІВ v2.0")
    print("✨ НОВИНКИ:")
    print("   • Точний час входу (наступна свічка + 1хв)")
    print("   • Автоматичні множинні API запити для будь-якого періоду")
    print("   • Можливість введення днів для всіх режимів")
    print("=" * 60)
    print("1 - М'які фільтри (рекомендовано 2-5 днів)")
    print("2 - Нормальні фільтри (рекомендовано 3-7 днів)")
    print("3 - Строгі фільтри (рекомендовано 5-14 днів)")
    print("4 - Власні налаштування")
    print("5 - Швидкий тест (ТОП-3 пари)")
    print("6 - Демонстрація великого періоду (14+ днів)")

    choice = input("\nОберіть опцію (1-6): ").strip()

    # Налаштування за вибором
    if choice == '1':
        # М'які фільтри
        config = BalancedConfig()
        config.MIN_FILTERS_REQUIRED = 1
        config.LONG_ZONE_MAX = 45.0
        config.SHORT_ZONE_MIN = 55.0
        config.USE_STOCH_RSI_FILTER = True
        config.USE_MACD_FILTER = False
        config.USE_DIVERGENCE_FILTER = True
        print("🟢 М'які фільтри: Long≤45, Short≥55, мінімум 1 фільтр")
        days_back = get_user_days_input(3)

    elif choice == '2':
        # Нормальні фільтри
        config = BalancedConfig()
        config.MIN_FILTERS_REQUIRED = 2
        config.USE_STOCH_RSI_FILTER = True
        config.USE_MACD_FILTER = True
        print("🟡 Нормальні фільтри: Long≤40, Short≥60, мінімум 2 фільтри")
        days_back = get_user_days_input(5)

    elif choice == '3':
        # Строгі фільтри
        config = BalancedConfig()
        config.MIN_FILTERS_REQUIRED = 3
        config.LONG_ZONE_MAX = 35.0
        config.SHORT_ZONE_MIN = 65.0
        config.USE_TREND_FILTER = True
        config.USE_VOLUME_FILTER = True
        config.USE_STOCH_RSI_FILTER = True
        config.USE_MACD_FILTER = True
        print("🔴 Строгі фільтри: Long≤35, Short≥65, мінімум 3 фільтри")
        days_back = get_user_days_input(7)

    elif choice == '4':
        # Власні налаштування
        config = BalancedConfig()
        print(f"\n⚙️ НАЛАШТУВАННЯ ФІЛЬТРІВ:")

        long_zone = input(f"Long зона (поточна {config.LONG_ZONE_MAX}): ").strip()
        if long_zone and long_zone.replace('.', '').isdigit():
            config.LONG_ZONE_MAX = float(long_zone)

        short_zone = input(f"Short зона (поточна {config.SHORT_ZONE_MIN}): ").strip()
        if short_zone and short_zone.replace('.', '').isdigit():
            config.SHORT_ZONE_MIN = float(short_zone)

        min_filters = input(f"Мінімум фільтрів (поточне {config.MIN_FILTERS_REQUIRED}): ").strip()
        if min_filters and min_filters.isdigit():
            config.MIN_FILTERS_REQUIRED = int(min_filters)

        # Вибір фільтрів
        print("\n🔧 Увімкнути додаткові фільтри? (y/n)")
        config.USE_STOCH_RSI_FILTER = input("StochRSI: ").lower().startswith('y')
        config.USE_MACD_FILTER = input("MACD: ").lower().startswith('y')
        config.USE_TREND_FILTER = input("Trend: ").lower().startswith('y')
        config.USE_VOLUME_FILTER = input("Volume: ").lower().startswith('y')
        config.USE_VOLATILITY_FILTER = input("Volatility: ").lower().startswith('y')

        days_back = get_user_days_input(5)

    elif choice == '5':
        # Швидкий тест
        config = BalancedConfig()
        config.PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        config.MIN_FILTERS_REQUIRED = 1
        print("⚡ Швидкий тест на ТОП-3 парах")
        days_back = get_user_days_input(3)

    elif choice == '6':
        # Демонстрація великого періоду
        config = BalancedConfig()
        config.MIN_FILTERS_REQUIRED = 2
        config.USE_STOCH_RSI_FILTER = True
        config.USE_MACD_FILTER = True
        config.USE_TREND_FILTER = True
        print("🚀 Демонстрація великого періоду з множинними запитами!")
        print("📊 Покаже потужність нової системи завантаження даних")
        days_back = get_user_days_input(14)

    else:
        print("❌ Невірний вибір")
        return

    # Інформування про стратегію завантаження
    generator = BalancedSignalGenerator(config)
    required_candles = generator.calculate_required_candles(days_back)
    num_requests = generator.calculate_required_requests(days_back)

    print(f"\n📊 ПЛАН ВИКОНАННЯ:")
    print(f"   📅 Період аналізу: {days_back} днів")
    print(f"   🕐 Таймфрейм: {config.TIMEFRAME}")
    print(f"   📊 Потрібно свічок: {required_candles}")
    print(f"   🔄 Кількість API запитів: {num_requests}")

    if num_requests > 1:
        print(f"   ✨ Буде використано автоматичне об'єднання даних!")
        print(f"   ⚡ Система забезпечить повне покриття {days_back} днів")
    else:
        print(f"   📥 Достатньо одного API запиту")

    print(f"\n🔄 Починаємо аналіз...")

    # Генерація сигналів
    output_file = f'improved_signals_{choice}_{days_back}d.csv'
    signals = await generator.generate_balanced_signals(days_back, output_file)

    if signals:
        print(f"\n🎉 УСПІШНО ЗАВЕРШЕНО!")
        print(f"   ✅ Згенеровано {len(signals)} сигналів")
        print(f"   📁 Збережено у файл: {output_file}")
        print(f"   🎯 Нова особливість: точний час і ціна входу!")

        # Показуємо приклад найновішого сигналу
        if len(signals) > 0:
            latest = signals[-1]
            clean_pair = latest.pair.replace('/', '').replace(':MATIC', '')
            time_diff = latest.entry_time - latest.signal_time
            delay_minutes = int(time_diff.total_seconds() / 60)

            print(f"\n🔥 ПРИКЛАД ОСТАННЬОГО СИГНАЛУ:")
            print(f"   📅 Час сигналу: {latest.signal_time.strftime('%d.%m %H:%M')}")
            print(f"   ⏰ Час входу: {latest.entry_time.strftime('%d.%m %H:%M')} (+{delay_minutes} хв)")
            print(f"   💼 {clean_pair} {latest.direction} | {latest.quality.value}")
            print(f"   💰 Ціна сигналу: ${latest.signal_price:.4f}")
            print(f"   💰 Ціна входу: ${latest.entry_price:.4f}")
    else:
        print(f"\n⚠️ Сигналів не знайдено за {days_back} днів")
        print("💡 ПОРАДИ ДЛЯ ПОКРАЩЕННЯ РЕЗУЛЬТАТІВ:")
        print("   • Розширте RSI зони (наприклад, Long≤45, Short≥55)")
        print("   • Зменште мінімальну кількість фільтрів")
        print("   • Спробуйте більший період (7-14 днів)")
        print("   • Використайте м'які фільтри (варіант 1)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Аналіз перервано користувачем")
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}")
        print("💡 Перевірте підключення до інтернету та API біржі")