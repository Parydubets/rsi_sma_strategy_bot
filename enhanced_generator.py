#!/usr/bin/env python3
"""
Покращений генератор торгових сигналів v2.2
ОНОВЛЕНА ЛОГІКА:
- Повна інформація про всі перевірки
- Делей тільки для підтверджених сигналів
"""

import ccxt
import pandas as pd
import asyncio
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum
import csv


@dataclass
class AdvancedConfig:
    # Біржа
    EXCHANGE_NAME: str = 'bybit'

    # Торгові пари
    PAIRS: List[str] = field(default_factory=lambda: [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
        'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
        'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
    ])

    # ТАЙМФРЕЙМИ (налаштовувані)
    PRIMARY_TIMEFRAME: str = '1m'  # Основний для входу
    CONFIRMATION_TIMEFRAME: str = '5m'  # Підтвердження

    # ОНОВЛЕНІ ПАРАМЕТРИ СТРАТЕГІЇ
    RSI_PERIOD: int = 14
    RSI_SMA_PERIOD: int = 14
    MIN_DIFF: float = 2.0
    TREND_CANDLES: int = 3
    DELAY_CANDLES: int = 5
    LONG_ENTRY_MAX_RSI: float = 40.0
    SHORT_ENTRY_MIN_RSI: float = 60.0
    OVERBOUGHT_LEVEL: float = 70.0
    OVERSOLD_LEVEL: float = 30.0
    PREOVERSOLD_LEVEL: float = 33.0  # НОВИЙ параметр для закриття SHORT
    PREBOUGHT_LEVEL: float = 67.0  # НОВИЙ параметр для закриття LONG
    DIFF_SMOOTHING_PERIODS: int = 3

    # Болінгер бенди (опціонально)
    USE_BOLLINGER: bool = False
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # Інші параметри
    UPDATE_INTERVAL: int = 30
    MAX_CANDLES_PER_REQUEST: int = 700
    OVERLAP_CANDLES: int = 50


class SignalQuality(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SignalStatus(Enum):
    OPEN = "open"
    SKIP = "skip"
    CLOSED = "closed"


@dataclass
class CheckResult:
    """Результат перевірки умови"""
    name: str
    passed: bool
    value: str
    description: str


@dataclass
class AdvancedMarketSignal:
    pair: str
    direction: str
    signal_time: datetime
    entry_time: datetime

    # RSI дані 1m
    rsi_1m: float
    rsi_sma_1m: float
    rsi_diff_1m: float

    # RSI дані 5m
    rsi_5m: float
    rsi_sma_5m: float
    rsi_diff_5m: float
    avg_diff_5m: float

    signal_price: float
    entry_price: float
    quality: SignalQuality
    confidence_score: float
    status: SignalStatus

    # Детальна інформація про перевірки
    primary_checks_passed: List[CheckResult]
    primary_checks_failed: List[CheckResult]
    confirmation_checks_passed: List[CheckResult]
    confirmation_checks_failed: List[CheckResult]

    skip_reason: str = ""
    comment: str = ""

    # Історія останніх позицій
    last_position_delay: int = 0


class AdvancedSignalGenerator:
    def __init__(self, config: AdvancedConfig):
        self.config = config
        self.exchange = self._init_exchange()
        self.logger = self._init_logger()

        # Дані по таймфреймам
        self.market_data_1m: Dict[str, pd.DataFrame] = {}
        self.market_data_5m: Dict[str, pd.DataFrame] = {}

        # Історія позицій для делею - тільки підтверджені сигнали!
        self.last_confirmed_positions: Dict[str, List[datetime]] = {}

        # Відкладені сигнали для повторної перевірки SMA
        self.delayed_signals: Dict[str, Tuple[datetime, str, int, pd.Series, pd.Series]] = {}

    def _init_exchange(self):
        return ccxt.bybit({
            'enableRateLimit': True,
            'sandbox': False
        })

    def _init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def calculate_timeframe_minutes(self, timeframe: str) -> int:
        """Розрахунок хвилин для таймфрейму"""
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        return timeframe_minutes.get(timeframe, 1)

    async def fetch_dual_timeframe_data(self, symbol: str, days_back: int) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Завантаження даних для двох таймфреймів"""
        try:
            print(
                f"📊 {symbol}: Завантажуємо дані для {self.config.PRIMARY_TIMEFRAME} та {self.config.CONFIRMATION_TIMEFRAME}...")

            # Завантажуємо дані основного таймфрейму
            df_primary = await self.fetch_ohlcv_for_timeframe(symbol, self.config.PRIMARY_TIMEFRAME, days_back)

            # Завантажуємо дані таймфрейму підтвердження
            df_confirmation = await self.fetch_ohlcv_for_timeframe(symbol, self.config.CONFIRMATION_TIMEFRAME,
                                                                   days_back)

            if df_primary is None or df_confirmation is None:
                print(f"❌ {symbol}: Не вдалося завантажити дані для одного з таймфреймів")
                return None, None

            print(
                f"✅ {symbol}: {self.config.PRIMARY_TIMEFRAME}={len(df_primary)} свічок, {self.config.CONFIRMATION_TIMEFRAME}={len(df_confirmation)} свічок")
            return df_primary, df_confirmation

        except Exception as e:
            self.logger.error(f"Помилка завантаження даних {symbol}: {e}")
            return None, None

    async def fetch_ohlcv_for_timeframe(self, symbol: str, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
        """Загрузка OHLCV для конкретного таймфрейма"""
        try:
            minutes_per_candle = self.calculate_timeframe_minutes(timeframe)
            required_candles = (days_back * 24 * 60) // minutes_per_candle + 100

            # Если нужно больше одного запроса
            if required_candles > self.config.MAX_CANDLES_PER_REQUEST:
                return await self.fetch_extended_ohlcv_timeframe(symbol, timeframe, days_back)
            else:
                return await self.fetch_single_ohlcv_timeframe(symbol, timeframe, days_back)

        except Exception as e:
            self.logger.error(f"Помилка завантаження {symbol} {timeframe}: {e}")
            return None

    async def fetch_single_ohlcv_timeframe(self, symbol: str, timeframe: str, days_back: int) -> Optional[pd.DataFrame]:
        """Одиничний запит для таймфрейму"""
        try:
            since = int((datetime.now() - timedelta(days=days_back + 1)).timestamp() * 1000)

            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=since,
                limit=self.config.MAX_CANDLES_PER_REQUEST
            )

            if ohlcv and len(ohlcv) >= 50:
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.sort_values('datetime').reset_index(drop=True)
                return df

            return None

        except Exception as e:
            self.logger.error(f"Помилка одиничиного завантаження {symbol} {timeframe}: {e}")
            return None

    async def fetch_extended_ohlcv_timeframe(self, symbol: str, timeframe: str, days_back: int) -> Optional[
        pd.DataFrame]:
        """Множинні запити для великого періоду"""
        try:
            minutes_per_candle = self.calculate_timeframe_minutes(timeframe)
            required_candles = (days_back * 24 * 60) // minutes_per_candle + 100

            effective_candles_per_request = self.config.MAX_CANDLES_PER_REQUEST - self.config.OVERLAP_CANDLES
            num_requests = (required_candles + effective_candles_per_request - 1) // effective_candles_per_request

            all_data = []
            current_end = datetime.now()

            for batch_num in range(num_requests):
                try:
                    minutes_back = batch_num * effective_candles_per_request * minutes_per_candle
                    batch_end = current_end - timedelta(minutes=minutes_back)
                    batch_start = batch_end - timedelta(
                        minutes=self.config.MAX_CANDLES_PER_REQUEST * minutes_per_candle)

                    since = int(batch_start.timestamp() * 1000)

                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, timeframe, since=since,
                        limit=self.config.MAX_CANDLES_PER_REQUEST
                    )

                    if ohlcv:
                        batch_df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        batch_df["datetime"] = pd.to_datetime(batch_df["timestamp"], unit="ms")

                        batch_df = batch_df[
                            (batch_df['datetime'] >= batch_start) &
                            (batch_df['datetime'] <= batch_end)
                            ].copy()

                        if len(batch_df) > 0:
                            all_data.append(batch_df)

                    await asyncio.sleep(0.7)

                except Exception as e:
                    print(f"   ❌ Помилка батча {batch_num + 1}: {e}")
                    continue

            if not all_data:
                return None

            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('datetime').drop_duplicates('timestamp').reset_index(drop=True)

            return combined_df

        except Exception as e:
            self.logger.error(f"Критична помилка множинного завантаження {symbol} {timeframe}: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок індикаторів для нової стратегії"""
        if len(df) < 50:
            return df

        df = df.copy()

        try:
            # Основні RSI індикатори
            rsi_indicator = RSIIndicator(close=df['close'], window=self.config.RSI_PERIOD)
            df['rsi'] = rsi_indicator.rsi()

            # RSI SMA (зглажений RSI)
            df['rsi_sma'] = df['rsi'].rolling(window=self.config.RSI_SMA_PERIOD).mean()

            # Різниця RSI - RSI_SMA
            df['rsi_diff'] = df['rsi'] - df['rsi_sma']

            # Середня різниця з DIFF_SMOOTHING_PERIODS
            df['avg_diff'] = df['rsi_diff'].rolling(window=self.config.DIFF_SMOOTHING_PERIODS).mean()

            # Напрямок тренду RSI SMA
            df['rsi_sma_trend'] = df['rsi_sma'].diff()
            df['rsi_sma_trend_direction'] = np.where(df['rsi_sma_trend'] > 0, 1,
                                                     np.where(df['rsi_sma_trend'] < 0, -1, 0))

            # Підрахунок периодів тренду (для TREND_CANDLES перевірки)
            df['trend_periods_up'] = 0
            df['trend_periods_down'] = 0

            for i in range(self.config.TREND_CANDLES, len(df)):
                # Перевіряєм останні TREND_CANDLES періодів
                recent_trends = df['rsi_sma_trend_direction'].iloc[i - self.config.TREND_CANDLES + 1:i + 1]

                if all(trend >= 0 for trend in recent_trends):  # Висхідний або нейтральный тренд
                    df.loc[i, 'trend_periods_up'] = self.config.TREND_CANDLES

                if all(trend <= 0 for trend in recent_trends):  # Нисхідниий чи нейтральный тренд
                    df.loc[i, 'trend_periods_down'] = self.config.TREND_CANDLES

            # Болінджер бенди (опціонально)
            if self.config.USE_BOLLINGER:
                bb = BollingerBands(close=df['close'], window=self.config.BOLLINGER_PERIOD,
                                    window_dev=self.config.BOLLINGER_STD)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_middle'] = bb.bollinger_mavg()

        except Exception as e:
            self.logger.error(f"Помилка розрахунку індикаторів: {e}")

        return df

    def check_advanced_rsi_signal(self, df_primary: pd.DataFrame, df_confirmation: pd.DataFrame,
                                  idx_primary: int, symbol: str) -> Tuple[
        Optional[str], List[CheckResult], List[CheckResult], List[CheckResult], List[CheckResult], SignalStatus, str]:
        """
        ВИПРАВЛЕНА ЛОГІКА: Перевірка сигналів з детальною інформацією про всі перевірки
        ТЕПЕР ВСІ УМОВИ МАЮТЬ БУТИ ВИКОНАНІ ДЛЯ ПІДТВЕРДЖЕННЯ СИГНАЛУ
        """
        if idx_primary < 1:
            return None, [], [], [], [], SignalStatus.SKIP, "Insufficient data"

        current_primary = df_primary.iloc[idx_primary]
        previous_primary = df_primary.iloc[idx_primary - 1]

        # Знаходимо відповідну свічку на таймреймі для підтвердження
        current_time = current_primary['datetime']
        df_confirmation_filtered = df_confirmation[df_confirmation['datetime'] <= current_time]

        if len(df_confirmation_filtered) < self.config.TREND_CANDLES + 1:
            return None, [], [], [], [], SignalStatus.SKIP, "Insufficient confirmation data"

        current_confirmation = df_confirmation_filtered.iloc[-1]

        # Перевіряєм наявність необхідних полів
        required_fields_primary = ['rsi', 'rsi_sma', 'rsi_diff']
        required_fields_confirmation = ['rsi', 'rsi_sma', 'rsi_diff']

        for field in required_fields_primary:
            if pd.isna(current_primary.get(field)) or pd.isna(previous_primary.get(field)):
                return None, [], [], [], [], SignalStatus.SKIP, f"Missing {field} on primary timeframe"

        for field in required_fields_confirmation:
            if pd.isna(current_confirmation.get(field)):
                return None, [], [], [], [], SignalStatus.SKIP, f"Missing {field} on confirmation timeframe"

        # === ПЕРЕВІРКА LONG СИГНАЛУ ===
        long_primary_passed, long_primary_failed = self.check_long_conditions_primary_detailed(
            current_primary, previous_primary, symbol
        )

        # ВИПРАВЛЕНО: перевіряємо що ВСІ умови первинного таймфрейму виконані
        if len(long_primary_passed) > 0 and len(long_primary_failed) == 0:  # ВСІ умови виконані
            long_confirmation_passed, long_confirmation_failed, skip_reason = self.check_long_conditions_confirmation_detailed(
                df_confirmation_filtered, current_confirmation
            )

            # ВИПРАВЛЕНО: перевіряємо що ВСІ умови підтвердження виконані
            if len(long_confirmation_passed) > 0 and len(long_confirmation_failed) == 0:  # ВСІ умови виконані
                return ("Long", long_primary_passed, long_primary_failed,
                        long_confirmation_passed, long_confirmation_failed, SignalStatus.OPEN, "")
            else:
                return ("Long", long_primary_passed, long_primary_failed,
                        long_confirmation_passed, long_confirmation_failed, SignalStatus.SKIP,
                        skip_reason if skip_reason else "Не всі умови підтвердження виконані")

        # === ПЕРЕВІРКА SHORT СИГНАЛУ ===
        short_primary_passed, short_primary_failed = self.check_short_conditions_primary_detailed(
            current_primary, previous_primary, symbol
        )

        # ВИПРАВЛЕНО: перевіряємо що ВСІ умови первинного таймфрейму виконані
        if len(short_primary_passed) > 0 and len(short_primary_failed) == 0:  # ВСІ умови виконані
            short_confirmation_passed, short_confirmation_failed, skip_reason = self.check_short_conditions_confirmation_detailed(
                df_confirmation_filtered, current_confirmation
            )

            # ВИПРАВЛЕНО: перевіряємо що ВСІ умови підтвердження виконані
            if len(short_confirmation_passed) > 0 and len(short_confirmation_failed) == 0:  # ВСІ умови виконані
                return ("Short", short_primary_passed, short_primary_failed,
                        short_confirmation_passed, short_confirmation_failed, SignalStatus.OPEN, "")
            else:
                return ("Short", short_primary_passed, short_primary_failed,
                        short_confirmation_passed, short_confirmation_failed, SignalStatus.SKIP,
                        skip_reason if skip_reason else "Не всі умови підтвердження виконані")

    def check_long_conditions_primary_detailed(self, current_primary: pd.Series, previous_primary: pd.Series,
                                               symbol: str) -> \
            Tuple[List[CheckResult], List[CheckResult]]:
        """Детальна перевірка умов LONG на первинному таймфреймі"""
        passed = []
        failed = []

        # 1. Зглажений RSI перетинає RSI SMA знизу вверх
        rsi_cross_up = (previous_primary['rsi'] <= previous_primary['rsi_sma'] and
                        current_primary['rsi'] > current_primary['rsi_sma'])

        check = CheckResult(
            name="RSI Cross SMA Up",
            passed=rsi_cross_up,
            value=f"Prev: RSI={previous_primary['rsi']:.2f} vs SMA={previous_primary['rsi_sma']:.2f}, Curr: RSI={current_primary['rsi']:.2f} vs SMA={current_primary['rsi_sma']:.2f}",
            description="RSI перетинає SMA знизу вверх"
        )

        if rsi_cross_up:
            passed.append(check)
        else:
            failed.append(check)
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 2. Зглажений RSI < LONG_ENTRY_MAX_RSI
        rsi_in_zone = previous_primary['rsi_sma'] < self.config.LONG_ENTRY_MAX_RSI

        check = CheckResult(
            name="RSI Entry Zone",
            passed=rsi_in_zone,
            value=f"RSI_SMA={previous_primary['rsi_sma']:.2f} < {self.config.LONG_ENTRY_MAX_RSI}",
            description=f"RSI нижче рівня входу {self.config.LONG_ENTRY_MAX_RSI}"
        )

        if rsi_in_zone:
            passed.append(check)
        else:
            failed.append(check)
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 3. Делей: остання ПІДТВЕРДЖЕНА LONG позиція закрита мінімум DELAY_CANDLES тому
        delay_ok = self.check_confirmed_position_delay(symbol, "Long", current_primary['datetime'])

        check = CheckResult(
            name="Position Delay",
            passed=delay_ok,
            value=f"Delay >= {self.config.DELAY_CANDLES} candles",
            description=f"Мінімум {self.config.DELAY_CANDLES} свічок після останньої підтвердженої Long позиції"
        )

        if delay_ok:
            passed.append(check)
        else:
            failed.append(check)

        return passed, failed

    def check_long_conditions_confirmation_detailed(self, df_confirmation: pd.DataFrame,
                                                    current_confirmation: pd.Series) -> \
            Tuple[List[CheckResult], List[CheckResult], str]:
        """Детальна перевірка підтвердження LONG на таймфреймі підтвердження"""
        passed = []
        failed = []
        skip_reason = ""

        # 1. RSI останньої закритої свічки менший за SMA на MIN_DIFF
        rsi_sma_diff = current_confirmation['rsi_sma'] - current_confirmation['rsi']
        diff_ok = rsi_sma_diff >= self.config.MIN_DIFF

        check = CheckResult(
            name="SMA-RSI Difference",
            passed=diff_ok,
            value=f"SMA-RSI={rsi_sma_diff:.2f} >= {self.config.MIN_DIFF}",
            description=f"Різниця SMA-RSI достатня для входу"
        )

        if diff_ok:
            passed.append(check)
        else:
            failed.append(check)
            skip_reason = f"SMA-RSI різниця {rsi_sma_diff:.2f} < {self.config.MIN_DIFF}"
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 2. Якщо RSI > 50 - сигнал пропускається
        rsi_below_50 = current_confirmation['rsi'] <= 50

        check = CheckResult(
            name="RSI Below 50",
            passed=rsi_below_50,
            value=f"RSI={current_confirmation['rsi']:.2f} <= 50",
            description="RSI нижче нейтрального рівня 50"
        )

        if rsi_below_50:
            passed.append(check)
        else:
            failed.append(check)
            if not skip_reason:  # якщо ще не встановлена причина
                skip_reason = f"RSI {current_confirmation['rsi']:.2f} > 50"
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 3. Перевірка що SMA НЕ СПАДАЄ (критична для LONG)
        if len(df_confirmation) >= 2:
            previous_confirmation = df_confirmation.iloc[-2]
            sma_not_falling = current_confirmation['rsi_sma'] >= previous_confirmation['rsi_sma']

            # Тимчасове вимкнення фільтра
            sma_not_falling = True

            check = CheckResult(
                name="SMA Not Falling",
                passed=sma_not_falling,
                value=f"SMA: {previous_confirmation['rsi_sma']:.2f} -> {current_confirmation['rsi_sma']:.2f}",
                description="SMA не спадає"
            )

            if sma_not_falling:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = "SMA спадає - не підходить для LONG"
                # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 4. Перевірка тренду SMA за TREND_CANDLES періодів
        if len(df_confirmation) >= self.config.TREND_CANDLES:
            recent_sma_changes = []
            for i in range(self.config.TREND_CANDLES):
                idx = -(i + 1)
                if abs(idx) <= len(df_confirmation):
                    if abs(idx) < len(df_confirmation):
                        current_sma = df_confirmation.iloc[idx]['rsi_sma']
                        prev_sma = df_confirmation.iloc[idx - 1]['rsi_sma'] if abs(idx - 1) <= len(
                            df_confirmation) else current_sma
                        recent_sma_changes.append(current_sma - prev_sma)

            no_uptrend = (recent_sma_changes and all(change > 0 for change in recent_sma_changes))
            no_uptrend = True # Тимчасвоо вимкнуто
            check = CheckResult(
                name="No SMA Uptrend",
                passed=no_uptrend,
                value=f"SMA changes: {[f'{change:.3f}' for change in recent_sma_changes]}",
                description=f"Немає стійкого SMA тренду вгору за {self.config.TREND_CANDLES} періодів"
            )

            if no_uptrend:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = f"SMA тренд вгору за {self.config.TREND_CANDLES} періодів"

        # Повертаємо результати всіх перевірок
        return passed, failed, skip_reason

    def scan_dual_timeframe_signals(self, df_primary: pd.DataFrame, df_confirmation: pd.DataFrame,
                                    pair: str, days_back: int) -> List[AdvancedMarketSignal]:
        """Сканирование сигналов по двухтаймфреймовой стратегии с детальною інформацією"""
        signals = []

        if len(df_primary) < 50 or len(df_confirmation) < 10:
            print(
                f"⚠️ {pair}: Недостатньо даних ({self.config.PRIMARY_TIMEFRAME}: {len(df_primary)}, {self.config.CONFIRMATION_TIMEFRAME}: {len(df_confirmation)})")
            return signals

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        print(f"🔍 {pair}: Шукаєм сигнали з {start_time.strftime('%Y-%m-%d %H:%M')}")

        # Фильтруем данные по периоду
        df_primary_period = df_primary[df_primary['datetime'] >= start_time].copy()

        if len(df_primary_period) < 10:
            print(f"⚠️ {pair}: Недостатньо даних за період")
            return signals

        print(f"📊 {pair}: Аналізуємо {len(df_primary_period)} свічей {self.config.PRIMARY_TIMEFRAME}")

        # Находим начальный индекс
        start_idx = df_primary[df_primary['datetime'] >= start_time].index[0] if len(
            df_primary[df_primary['datetime'] >= start_time]) > 0 else len(df_primary)
        start_idx = max(20, start_idx)

        for i in range(start_idx, len(df_primary) - 1):
            current_row = df_primary.iloc[i]

            if current_row['datetime'] < start_time or current_row['datetime'] > end_time:
                continue

            try:
                # Проверка сигнала по новой детальной логике с обработкой ошибок
                signal_result = self.check_advanced_rsi_signal(df_primary, df_confirmation, i, pair)

                # ВИПРАВЛЕНО: Перевіряємо що результат не None і має правильну структуру
                if signal_result is None or len(signal_result) != 7:
                    continue

                direction, primary_passed, primary_failed, confirmation_passed, confirmation_failed, status, skip_reason = signal_result

                if not direction:
                    continue

                # Находим соответствующие данные на таймфрейме подтверждения
                current_time = current_row['datetime']
                df_confirmation_filtered = df_confirmation[df_confirmation['datetime'] <= current_time]
                if len(df_confirmation_filtered) == 0:
                    continue

                current_confirmation = df_confirmation_filtered.iloc[-1]

                # Расчет уверенности и качества
                total_passed_checks = len(primary_passed) + len(confirmation_passed)
                total_failed_checks = len(primary_failed) + len(confirmation_failed)

                base_confidence = 50.0 if status == SignalStatus.SKIP else 70.0
                confidence = min(95.0, base_confidence + total_passed_checks * 8.0 - total_failed_checks * 2.0)

                if status == SignalStatus.OPEN:
                    if confidence >= 80 and total_passed_checks >= 5:
                        quality = SignalQuality.HIGH
                    elif confidence >= 65 and total_passed_checks >= 3:
                        quality = SignalQuality.MEDIUM
                    else:
                        quality = SignalQuality.LOW
                else:
                    quality = SignalQuality.LOW

                # Расчет времени и цены входа
                entry_time, entry_price = self.calculate_entry_time_and_price_advanced(
                    df_primary, i, current_row['datetime']
                )

                # Регистрируем ТОЛЬКО ПОДТВЕРЖДЕННЫЕ позиции для делея
                if status == SignalStatus.OPEN:
                    self.register_confirmed_position(pair, direction, current_row['datetime'])

                # Создание сигнала с детальной информацией
                signal = AdvancedMarketSignal(
                    pair=pair,
                    direction=direction,
                    signal_time=current_row['datetime'],
                    entry_time=entry_time,

                    # RSI данные основного таймфрейма
                    rsi_1m=current_row.get('rsi', 0),
                    rsi_sma_1m=current_row.get('rsi_sma', 0),
                    rsi_diff_1m=current_row.get('rsi_diff', 0),

                    # RSI данные таймфрейма подтверждения
                    rsi_5m=current_confirmation.get('rsi', 0),
                    rsi_sma_5m=current_confirmation.get('rsi_sma', 0),
                    rsi_diff_5m=current_confirmation.get('rsi_diff', 0),
                    avg_diff_5m=current_confirmation.get('avg_diff', 0),

                    signal_price=current_row['close'],
                    entry_price=entry_price,
                    quality=quality,
                    confidence_score=confidence,
                    status=status,

                    # Детальна інформація про перевірки
                    primary_checks_passed=primary_passed,
                    primary_checks_failed=primary_failed,
                    confirmation_checks_passed=confirmation_passed,
                    confirmation_checks_failed=confirmation_failed,

                    skip_reason=skip_reason,
                    comment=f"{self.config.PRIMARY_TIMEFRAME} cross, {self.config.CONFIRMATION_TIMEFRAME} {'confirmed' if status == SignalStatus.OPEN else 'rejected'}"
                )

                signals.append(signal)

                # Виводимо детальну інформацію про кожен сигнал
                self.print_detailed_signal_info(signal)

            except Exception as e:
                # ВИПРАВЛЕНО: Додаємо обробку помилок для кожної ітерації
                self.logger.error(f"Помилка обробки свічки {i} для {pair}: {e}")
                continue

        return signals

    def check_short_conditions_confirmation_detailed(self, df_confirmation: pd.DataFrame,
                                                     current_confirmation: pd.Series) -> \
            Tuple[List[CheckResult], List[CheckResult], str]:
        """Детальна перевірка підтвердження SHORT на таймфреймі підтвердження"""
        passed = []
        failed = []
        skip_reason = ""

        # 1. RSI останньої закритої свічки більший за SMA на MIN_DIFF
        rsi_sma_diff = current_confirmation['rsi'] - current_confirmation['rsi_sma']
        diff_ok = rsi_sma_diff >= self.config.MIN_DIFF

        check = CheckResult(
            name="RSI-SMA Difference",
            passed=diff_ok,
            value=f"RSI-SMA={rsi_sma_diff:.2f} >= {self.config.MIN_DIFF}",
            description=f"The RSI-SMA difference isn`t enough for the entry"
        )

        if diff_ok:
            passed.append(check)
        else:
            failed.append(check)
            skip_reason = f"RSI-SMA різниця {rsi_sma_diff:.2f} < {self.config.MIN_DIFF}"
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 2. RSI > 50
        rsi_above_50 = current_confirmation['rsi'] >= 50

        check = CheckResult(
            name="RSI Above 50",
            passed=rsi_above_50,
            value=f"RSI={current_confirmation['rsi']:.2f} >= 50",
            description="RSI вище нейтрального рівня 50"
        )

        if rsi_above_50:
            passed.append(check)
        else:
            failed.append(check)
            if not skip_reason:
                skip_reason = f"RSI {current_confirmation['rsi']:.2f} < 50"
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 3. Перевірка що SMA НЕ ЗРОСТАЄ (критична для SHORT)
        if len(df_confirmation) >= 2:
            previous_confirmation = df_confirmation.iloc[-2]
            sma_not_increasing = current_confirmation['rsi_sma'] <= previous_confirmation['rsi_sma']

            # Тимчасове вимкнення фільтра
            sma_not_increasing = True

            check = CheckResult(
                name="SMA Not Increasing",
                passed=sma_not_increasing,
                value=f"SMA: {previous_confirmation['rsi_sma']:.2f} -> {current_confirmation['rsi_sma']:.2f}",
                description="SMA не зростає"
            )

            if sma_not_increasing:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = "SMA зростає - не підходить для SHORT"
                # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 4. Перевірка тренду SMA за TREND_CANDLES періодів
        if len(df_confirmation) >= self.config.TREND_CANDLES:
            recent_sma_changes = []
            for i in range(self.config.TREND_CANDLES):
                idx = -(i + 1)
                if abs(idx) <= len(df_confirmation):
                    if abs(idx) < len(df_confirmation):
                        current_sma = df_confirmation.iloc[idx]['rsi_sma']
                        prev_sma = df_confirmation.iloc[idx - 1]['rsi_sma'] if abs(idx - 1) <= len(
                            df_confirmation) else current_sma
                        recent_sma_changes.append(current_sma - prev_sma)

            no_downtrend =  (recent_sma_changes and all(change < 0 for change in recent_sma_changes))

            no_downtrend = True # Тимчасово вимкнуто
            check = CheckResult(
                name="No SMA Downtrend",
                passed=no_downtrend,
                value=f"SMA changes: {[f'{change:.3f}' for change in recent_sma_changes]}",
                description=f"Немає стійкого SMA тренду вниз за {self.config.TREND_CANDLES} періодів"
            )

            if no_downtrend:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = f"SMA тренд вниз за {self.config.TREND_CANDLES} періодів"

        # Повертаємо результати всіх перевірок
        return passed, failed, skip_reason

    def check_short_conditions_primary_detailed(self, current_primary: pd.Series, previous_primary: pd.Series,
                                                symbol: str) -> \
            Tuple[List[CheckResult], List[CheckResult]]:
        """Детальна перевірка умов SHORT на первинному таймфреймі"""
        passed = []
        failed = []

        # 1. Зглажений RSI перетинає RSI SMA зверху вниз
        rsi_cross_down = (previous_primary['rsi'] >= previous_primary['rsi_sma'] and
                          current_primary['rsi'] < current_primary['rsi_sma'])

        check = CheckResult(
            name="RSI Cross SMA Down",
            passed=rsi_cross_down,
            value=f"Prev: RSI={previous_primary['rsi']:.2f} vs SMA={previous_primary['rsi_sma']:.2f}, Curr: RSI={current_primary['rsi']:.2f} vs SMA={current_primary['rsi_sma']:.2f}",
            description="RSI перетинає SMA зверху вниз"
        )

        if rsi_cross_down:
            passed.append(check)
        else:
            failed.append(check)
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 2. Зглажений RSI > SHORT_ENTRY_MIN_RSI
        rsi_in_zone = previous_primary['rsi_sma'] > self.config.SHORT_ENTRY_MIN_RSI

        check = CheckResult(
            name="RSI Entry Zone",
            passed=rsi_in_zone,
            value=f"RSI_SMA={previous_primary['rsi_sma']:.2f} > {self.config.SHORT_ENTRY_MIN_RSI}",
            description=f"RSI вище рівня входу {self.config.SHORT_ENTRY_MIN_RSI}"
        )

        if rsi_in_zone:
            passed.append(check)
        else:
            failed.append(check)
            # ВИПРАВЛЕНО: НЕ повертаємо тут, а продовжуємо перевірку всіх умов

        # 3. Делей: остання ПІДТВЕРДЖЕНА SHORT позиція закрита мінімум DELAY_CANDLES тому
        delay_ok = self.check_confirmed_position_delay(symbol, "Short", current_primary['datetime'])

        check = CheckResult(
            name="Position Delay",
            passed=delay_ok,
            value=f"Delay >= {self.config.DELAY_CANDLES} candles",
            description=f"Мінімум {self.config.DELAY_CANDLES} свічок після останньої підтвердженої Short позиції"
        )

        if delay_ok:
            passed.append(check)
        else:
            failed.append(check)

        return passed, failed

    def check_confirmed_position_delay(self, symbol: str, direction: str, current_time: datetime) -> bool:
        """Перевірка делею між ПІДТВЕРДЖЕНИМИ позиціями"""
        if symbol not in self.last_confirmed_positions:
            self.last_confirmed_positions[symbol] = []
            return True

        # Фільтруємо позиції по напрямку та часу
        position_key = f"{symbol}_{direction}"
        recent_positions = [
            pos_time for pos_time in self.last_confirmed_positions.get(position_key, [])
            if (current_time - pos_time).total_seconds() / 60 < self.config.DELAY_CANDLES
        ]

        return len(recent_positions) == 0

    def register_confirmed_position(self, symbol: str, direction: str, position_time: datetime):
        """Реєстрація нової ПІДТВЕРДЖЕНОЇ позиції для відстеження делею"""
        position_key = f"{symbol}_{direction}"

        if position_key not in self.last_confirmed_positions:
            self.last_confirmed_positions[position_key] = []

        self.last_confirmed_positions[position_key].append(position_time)

        # Очищуємо старі позиції (старіші за DELAY_CANDLES * 2)
        cutoff_time = position_time - timedelta(minutes=self.config.DELAY_CANDLES * 2)
        self.last_confirmed_positions[position_key] = [
            pos_time for pos_time in self.last_confirmed_positions[position_key]
            if pos_time > cutoff_time
        ]

    def check_exit_conditions(self, df_confirmation: pd.DataFrame, direction: str, current_confirmation: pd.Series) -> \
            Tuple[bool, str]:
        """Перевірка умов закриття позиції"""
        if direction == "Long":
            # Закриття за умови досягнення RSI зони PREBOUGHT_LEVEL на таймфреймі підтвердження
            if current_confirmation['rsi'] >= self.config.PREBOUGHT_LEVEL:
                return True, f"RSI >= {self.config.PREBOUGHT_LEVEL}"

            # Закриття за умови перетину rsi sma вниз на таймфреймі підтвердження
            if len(df_confirmation) >= 2:
                previous_confirmation = df_confirmation.iloc[-2]
                if (previous_confirmation['rsi'] >= previous_confirmation['rsi_sma'] and
                        current_confirmation['rsi'] < current_confirmation['rsi_sma']):
                    return True, "RSI перетнув SMA вниз"

        elif direction == "Short":
            # Закриття за умови досягнення RSI зони PREOVERSOLD_LEVEL на таймфреймі підтвердження
            if current_confirmation['rsi'] <= self.config.PREOVERSOLD_LEVEL:
                return True, f"RSI <= {self.config.PREOVERSOLD_LEVEL}"

            # Закриття за умови перетину rsi sma вгору на таймфреймі підтвердження
            if len(df_confirmation) >= 2:
                previous_confirmation = df_confirmation.iloc[-2]
                if (previous_confirmation['rsi'] <= previous_confirmation['rsi_sma'] and
                        current_confirmation['rsi'] > current_confirmation['rsi_sma']):
                    return True, "RSI перетнув SMA вгору"

        return False, ""

    def calculate_entry_time_and_price_advanced(self, df_primary: pd.DataFrame, signal_idx: int,
                                                signal_time: datetime) -> Tuple[datetime, float]:
        """Розрахунок часу та ціни входу для нової стратегії"""
        try:
            if signal_idx + 1 < len(df_primary):
                next_candle = df_primary.iloc[signal_idx + 1]
                primary_minutes = self.calculate_timeframe_minutes(self.config.PRIMARY_TIMEFRAME)
                entry_time = next_candle['datetime'] + timedelta(minutes=primary_minutes)
                entry_price = next_candle['open']
                return entry_time, entry_price
            else:
                primary_minutes = self.calculate_timeframe_minutes(self.config.PRIMARY_TIMEFRAME)
                entry_time = signal_time + timedelta(minutes=primary_minutes * 2)
                entry_price = df_primary.iloc[signal_idx]['close']
                return entry_time, entry_price

        except Exception:
            primary_minutes = self.calculate_timeframe_minutes(self.config.PRIMARY_TIMEFRAME)
            entry_time = signal_time + timedelta(minutes=primary_minutes * 2)
            entry_price = df_primary.iloc[signal_idx]['close']
            return entry_time, entry_price

    def print_detailed_signal_info(self, signal: AdvancedMarketSignal):
        """Виведення детальної інформації про сигнал"""
        status_emoji = "✅" if signal.status == SignalStatus.OPEN else "⏭️"

        print(f"\n{status_emoji} {signal.pair} {signal.direction} - {signal.signal_time.strftime('%Y-%m-%d %H:%M')}")
        print(
            f"   Статус: {signal.status.value} | Впевненість: {signal.confidence_score:.1f}% | Якість: {signal.quality.value}")

        if signal.status == SignalStatus.SKIP:
            print(f"   💡 Причина пропуску: {signal.skip_reason}")

        # Перевірки первинного таймфрейму
        if signal.primary_checks_passed or signal.primary_checks_failed:
            print(f"   📊 Первинний таймфрейм ({self.config.PRIMARY_TIMEFRAME}):")

            if signal.primary_checks_passed:
                print(f"      ✅ Успішні перевірки:")
                for check in signal.primary_checks_passed:
                    print(f"         • {check.name}: {check.value}")

            if signal.primary_checks_failed:
                print(f"      ❌ Невдалі перевірки:")
                for check in signal.primary_checks_failed:
                    print(f"         • {check.name}: {check.value}")

        # Перевірки таймфрейму підтвердження
        if signal.confirmation_checks_passed or signal.confirmation_checks_failed:
            print(f"   🔍 Таймфрейм підтвердження ({self.config.CONFIRMATION_TIMEFRAME}):")

            if signal.confirmation_checks_passed:
                print(f"      ✅ Успішні перевірки:")
                for check in signal.confirmation_checks_passed:
                    print(f"         • {check.name}: {check.value}")

            if signal.confirmation_checks_failed:
                print(f"      ❌ Невдалі перевірки:")
                for check in signal.confirmation_checks_failed:
                    print(f"         • {check.name}: {check.value}")

        # RSI дані
        print(f"   📈 RSI дані:")
        print(
            f"      {self.config.PRIMARY_TIMEFRAME}: RSI={signal.rsi_1m:.2f}, SMA={signal.rsi_sma_1m:.2f}, Diff={signal.rsi_diff_1m:.2f}")
        print(
            f"      {self.config.CONFIRMATION_TIMEFRAME}: RSI={signal.rsi_5m:.2f}, SMA={signal.rsi_sma_5m:.2f}, Diff={signal.rsi_diff_5m:.2f}")


    async def analyze_pair_advanced(self, pair: str, days_back: int = 7) -> List[AdvancedMarketSignal]:
        """Аналіз пари з новою стратегією"""
        try:
            print(f"\n🔍 Аналізуємо {pair}...")

            # Завантажуємо дані для двох таймфреймів
            df_primary, df_confirmation = await self.fetch_dual_timeframe_data(pair, days_back)

            if df_primary is None or df_confirmation is None:
                print(f"❌ {pair}: Не вдалося завантажити дані")
                return []

            # Обчислюємо індикатори для обох таймфреймів
            df_primary = self.calculate_advanced_indicators(df_primary)
            df_confirmation = self.calculate_advanced_indicators(df_confirmation)

            # Зберігаємо дані
            self.market_data_1m[pair] = df_primary
            self.market_data_5m[pair] = df_confirmation

            # Сканируем сигналы
            signals = self.scan_dual_timeframe_signals(df_primary, df_confirmation, pair, days_back)

            print(f"\n📊 {pair}: Знайдено {len(signals)} сигналів")
            open_signals = [s for s in signals if s.status == SignalStatus.OPEN]
            skip_signals = [s for s in signals if s.status == SignalStatus.SKIP]
            print(f"   ✅ Підтверджених: {len(open_signals)}")
            print(f"   ⏭️ Пропущених: {len(skip_signals)}")

            return signals

        except Exception as e:
            self.logger.error(f"Помилка аналізу {pair}: {e}")
            return []

    def save_signals_to_csv(self, all_signals: List[AdvancedMarketSignal], filename: str = None):
        """Збереження сигналів у CSV з детальною інформацією"""
        if not all_signals:
            print("📄 Немає сигналів для збереження")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_signals_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=";")

                # Заголовки
                headers = [
                    'Pair', 'Direction', 'Status', 'Signal_Time', 'Entry_Time',
                    f'RSI_{self.config.PRIMARY_TIMEFRAME}', f'RSI_SMA_{self.config.PRIMARY_TIMEFRAME}',
                    f'RSI_Diff_{self.config.PRIMARY_TIMEFRAME}',
                    f'RSI_{self.config.CONFIRMATION_TIMEFRAME}', f'RSI_SMA_{self.config.CONFIRMATION_TIMEFRAME}',
                    f'RSI_Diff_{self.config.CONFIRMATION_TIMEFRAME}', f'Avg_Diff_{self.config.CONFIRMATION_TIMEFRAME}',
                    'Signal_Price', 'Entry_Price', 'Quality', 'Confidence',
                    f'Primary_Passed_{self.config.PRIMARY_TIMEFRAME}',
                    f'Primary_Failed_{self.config.PRIMARY_TIMEFRAME}',
                    f'Confirmation_Passed_{self.config.CONFIRMATION_TIMEFRAME}',
                    f'Confirmation_Failed_{self.config.CONFIRMATION_TIMEFRAME}',
                    'Skip_Reason', 'Comment'
                ]
                writer.writerow(headers)

                # Дані
                for signal in all_signals:
                    # Форматування перевірок
                    primary_passed_str = '; '.join([f"{c.name}: {c.value}" for c in signal.primary_checks_passed])
                    primary_failed_str = '; '.join([f"{c.name}: {c.value}" for c in signal.primary_checks_failed])
                    confirmation_passed_str = '; '.join(
                        [f"{c.name}: {c.value}" for c in signal.confirmation_checks_passed])
                    confirmation_failed_str = '; '.join(
                        [f"{c.name}: {c.value}" for c in signal.confirmation_checks_failed])

                    row = [
                        signal.pair,
                        signal.direction,
                        signal.status.value,
                        signal.signal_time.strftime('%Y-%m-%d %H:%M:%S'),
                        signal.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                        f"{str(signal.rsi_1m).replace(".", ",")}",
                        f"{str(signal.rsi_sma_1m).replace(".", ",")}",
                        f"{str(signal.rsi_diff_1m).replace(".", ",")}",
                        f"{str(signal.rsi_5m).replace(".", ",")}",
                        f"{str(signal.rsi_sma_5m).replace(".", ",")}",
                        f"{str(signal.rsi_diff_5m).replace(".", ",")}",
                        f"{str(signal.avg_diff_5m).replace(".", ",")}",
                        f"{str(signal.signal_price).replace(".", ",")}",
                        f"{str(signal.entry_price).replace(".", ",")}",
                        signal.quality.value,
                        f"{signal.confidence_score:.1f}%",
                        primary_passed_str,
                        primary_failed_str,
                        confirmation_passed_str,
                        confirmation_failed_str,
                        signal.skip_reason,
                        signal.comment
                    ]
                    writer.writerow(row)

            print(f"📄 Детальні сигнали збережено у {filename}")

            # Статистика
            open_signals = [s for s in all_signals if s.status == SignalStatus.OPEN]
            skip_signals = [s for s in all_signals if s.status == SignalStatus.SKIP]

            print(f"📈 Загальна статистика:")
            print(f"   Всього сигналів: {len(all_signals)}")
            print(f"   Підтверджених позицій: {len(open_signals)}")
            print(f"   Пропущених сигналів: {len(skip_signals)}")

            if open_signals:
                long_signals = [s for s in open_signals if s.direction == 'Long']
                short_signals = [s for s in open_signals if s.direction == 'Short']
                print(f"   Long: {len(long_signals)}, Short: {len(short_signals)}")

                avg_confidence = sum(s.confidence_score for s in open_signals) / len(open_signals)
                print(f"   Середня впевненість: {avg_confidence:.1f}%")

        except Exception as e:
            self.logger.error(f"Помилка збереження у CSV: {e}")

    async def run_advanced_analysis(self, days_back: int = 7):
        """Запуск поглибленого аналізу для всіх пар"""
        print(
            f"🚀 Запуск аналізу за покращеною стратегією ({self.config.PRIMARY_TIMEFRAME}/{self.config.CONFIRMATION_TIMEFRAME})")
        print(f"📅 Період: {days_back} днів")
        print(f"💰 Пари: {len(self.config.PAIRS)}")
        print(f"🔧 Делей працює тільки для підтверджених сигналів!")

        all_signals = []

        for pair in self.config.PAIRS:
            try:
                signals = await self.analyze_pair_advanced(pair, days_back)
                all_signals.extend(signals)

                # Пауза між запитами
                await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Помилка аналізу {pair}: {e}")
                continue

        # Зберігаємо результати
        if all_signals:
            self.save_signals_to_csv(all_signals)

        print(f"\n✅ Аналіз завершено. Знайдено {len(all_signals)} сигналів")
        print(f"📊 Підтверджених: {len([s for s in all_signals if s.status == SignalStatus.OPEN])}")
        print(f"⏭️ Пропущених: {len([s for s in all_signals if s.status == SignalStatus.SKIP])}")

        return all_signals


# Головна функція запуску
async def main():
    """Головна функція"""
    # Створюємо конфігурацію
    config = AdvancedConfig(
        # Налаштування таймфреймів
        PRIMARY_TIMEFRAME='1m',
        CONFIRMATION_TIMEFRAME='5m',

        # Налаштування параметрів стратегії
        MIN_DIFF=2.0,
        TREND_CANDLES=3,
        DELAY_CANDLES=5,
        LONG_ENTRY_MAX_RSI=40.0,
        SHORT_ENTRY_MIN_RSI=60.0,
        PREOVERSOLD_LEVEL=33.0,
        PREBOUGHT_LEVEL=67.0,

        # Торгові пари
        PAIRS=[
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
            'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
            'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
        ]
    )

    # Створюємо генератор сигналів
    generator = AdvancedSignalGenerator(config)

    # Запускаємо аналіз
    try:
        signals = await generator.run_advanced_analysis(days_back=7)

        # Додатковий аналіз результатів
        if signals:
            print(f"\n📊 Детальна статистика:")

            open_signals = [s for s in signals if s.status == SignalStatus.OPEN]
            skip_signals = [s for s in signals if s.status == SignalStatus.SKIP]

            if open_signals:
                print(f"🟢 Підтверджені позиції ({len(open_signals)}):")
                for signal in open_signals[-5:]:  # Показуємо останні 5
                    total_checks = len(signal.primary_checks_passed) + len(signal.confirmation_checks_passed)
                    print(f"   {signal.pair} {signal.direction} {signal.signal_time.strftime('%m-%d %H:%M')} "
                          f"Впевненість: {signal.confidence_score:.1f}% Перевірок пройдено: {total_checks}")

            if skip_signals:
                print(f"🟡 Пропущені сигнали ({len(skip_signals)}):")
                skip_reasons = {}
                for signal in skip_signals:
                    reason = signal.skip_reason
                    if reason in skip_reasons:
                        skip_reasons[reason] += 1
                    else:
                        skip_reasons[reason] = 1

                for reason, count in sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"   {reason}: {count} разів")

    except KeyboardInterrupt:
        print("\n⏹️ Аналіз перервано користувачем")
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}")


if __name__ == "__main__":
    asyncio.run(main())