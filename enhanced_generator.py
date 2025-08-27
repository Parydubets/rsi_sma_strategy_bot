#!/usr/bin/env python3
"""
Покращений генератор торгових сигналів v2.3 (ЗАВЕРШЕНИЙ)
НОВІ МОЖЛИВОСТІ:
- Перевірки об'єму та волатільності
- Покращена система скорингу
- Вибір періоду для сигналів
- Об'єднана система confidence та scoring
"""

import ccxt
import pandas as pd
import asyncio
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
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
    PRIMARY_TIMEFRAME: str = '5m'  # Основний для входу
    CONFIRMATION_TIMEFRAME: str = '15m'  # Підтвердження

    # НОВИЙ: Період для сигналів (можна вибирати)
    SIGNAL_PERIOD_HOURS: int = 24  # За скільки годин шукати сигнали (можна змінити на будь-який період)
    SIGNAL_PERIOD_DAYS: int = 0  # Додатково дні (для більших періодів)

    # ОНОВЛЕНІ ПАРАМЕТРИ СТРАТЕГІЇ
    RSI_PERIOD: int = 14
    RSI_SMA_PERIOD: int = 14
    MIN_DIFF: float = 2.0
    TREND_CANDLES: int = 3
    DELAY_CANDLES: int = 12
    LONG_ENTRY_MAX_RSI: float = 40.0
    SHORT_ENTRY_MIN_RSI: float = 60.0
    OVERBOUGHT_LEVEL: float = 70.0
    OVERSOLD_LEVEL: float = 30.0
    PREOVERSOLD_LEVEL: float = 33.0
    PREBOUGHT_LEVEL: float = 67.0
    DIFF_SMOOTHING_PERIODS: int = 3

    # НОВІ ПАРАМЕТРИ ОБ'ЄМУ
    USE_VOLUME_CHECKS: bool = True
    VOLUME_SMA_PERIOD: int = 20
    MIN_VOLUME_MULTIPLIER: float = 1.15  # Мінімальний множник об'єму відносно середнього
    HIGH_VOLUME_MULTIPLIER: float = 1.8  # Високий об'єм для бонусних балів

    # НОВІ ПАРАМЕТРИ ВОЛАТІЛЬНОСТІ
    USE_VOLATILITY_CHECKS: bool = True
    ATR_PERIOD: int = 14
    MIN_VOLATILITY_MULTIPLIER: float = 0.8  # Мінімальна волатільність
    HIGH_VOLATILITY_MULTIPLIER: float = 1.5  # Висока волатільність для бонусних балів

    # СИСТЕМА СКОРИНГУ
    SCORING_SYSTEM: Dict[str, float] = field(default_factory=lambda: {
        # Базові бали за перевірки
        'primary_check_passed': 15.0,
        'confirmation_check_passed': 20.0,
        'primary_check_failed': -10.0,
        'confirmation_check_failed': -15.0,

        # Бонуси за якість сигналу
        'perfect_rsi_cross': 10.0,  # Ідеальний перетин RSI/SMA
        'strong_diff': 8.0,  # Сильна різниця RSI-SMA
        'trend_alignment': 6.0,  # Тренд в правильному напрямку

        # Бонуси за об'єм
        'normal_volume': 3.0,  # Нормальний об'єм
        'high_volume': 8.0,  # Високий об'єм
        'very_high_volume': 12.0,  # Дуже високий об'єм

        # Бонуси за волатільність
        'normal_volatility': 2.0,  # Нормальна волатільність
        'high_volatility': 6.0,  # Висока волатільність
        'optimal_volatility': 10.0,  # Оптимальна волатільність

        # Штрафи
        'low_volume': -5.0,  # Низький об'єм
        'low_volatility': -3.0,  # Низька волатільність
        'weak_signal': -4.0,  # Слабкий сигнал

        # Базовий скор
        'base_score': 50.0
    })

    # Болінгер бенди (опціонально)
    USE_BOLLINGER: bool = False
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0

    # Інші параметри
    UPDATE_INTERVAL: int = 30
    MAX_CANDLES_PER_REQUEST: int = 700
    OVERLAP_CANDLES: int = 50


class SignalQuality(Enum):
    EXCELLENT = "Excellent"  # 85-100 балів
    HIGH = "High"  # 70-84 балів
    MEDIUM = "Medium"  # 55-69 балів
    LOW = "Low"  # 40-54 балів
    POOR = "Poor"  # < 40 балів


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
    score_impact: float = 0.0  # Вплив на загальний скор


@dataclass
class VolumeAnalysis:
    """Аналіз об'єму"""
    current_volume: float
    avg_volume: float
    volume_ratio: float
    is_above_average: bool
    volume_score: float
    volume_quality: str


@dataclass
class VolatilityAnalysis:
    """Аналіз волатільності"""
    current_atr: float
    avg_atr: float
    volatility_ratio: float
    is_adequate: bool
    volatility_score: float
    volatility_quality: str


@dataclass
class SignalScore:
    """Детальний скор сигналу"""
    base_score: float
    technical_score: float
    volume_score: float
    volatility_score: float
    bonus_score: float
    penalty_score: float
    total_score: float
    max_possible_score: float
    score_percentage: float


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
    confidence_score: float  # Тепер синхронізований з total_score
    status: SignalStatus

    # НОВІ: Аналіз об'єму та волатільності
    volume_analysis: VolumeAnalysis
    volatility_analysis: VolatilityAnalysis
    signal_score: SignalScore

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

    def get_signal_period_days(self) -> int:
        """Розрахунок загального періоду в днях для сигналів"""
        total_hours = self.config.SIGNAL_PERIOD_HOURS + (self.config.SIGNAL_PERIOD_DAYS * 24)
        return max(1, int(total_hours / 24)) + 1  # +1 для запасу даних

    def get_signal_period_timedelta(self) -> timedelta:
        """Повертає timedelta для періоду сигналів"""
        return timedelta(
            days=self.config.SIGNAL_PERIOD_DAYS,
            hours=self.config.SIGNAL_PERIOD_HOURS
        )

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
        """РОЗШИРЕНИЙ розрахунок індикаторів з об'ємом та волатільністю"""
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

            # НОВІ: Індикатори об'єму
            if self.config.USE_VOLUME_CHECKS:
                # Середній об'єм - використовуємо простий rolling mean
                df['volume_sma'] = df['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()

                # Відношення поточного об'єму до середнього
                df['volume_ratio'] = df['volume'] / df['volume_sma']

                # Категорії об'єму
                df['volume_category'] = np.where(
                    df['volume_ratio'] >= self.config.HIGH_VOLUME_MULTIPLIER, 'high',
                    np.where(df['volume_ratio'] >= self.config.MIN_VOLUME_MULTIPLIER, 'normal', 'low')
                )

            # НОВІ: Індикатори волатільності
            if self.config.USE_VOLATILITY_CHECKS:
                # Average True Range
                atr_indicator = AverageTrueRange(
                    high=df['high'], low=df['low'], close=df['close'],
                    window=self.config.ATR_PERIOD
                )
                df['atr'] = atr_indicator.average_true_range()

                # Середній ATR для порівняння
                df['atr_sma'] = df['atr'].rolling(window=self.config.ATR_PERIOD).mean()

                # Відношення поточної волатільності до середньої
                df['volatility_ratio'] = df['atr'] / df['atr_sma']

                # Категорії волатільності
                df['volatility_category'] = np.where(
                    df['volatility_ratio'] >= self.config.HIGH_VOLATILITY_MULTIPLIER, 'high',
                    np.where(df['volatility_ratio'] >= self.config.MIN_VOLATILITY_MULTIPLIER, 'normal', 'low')
                )

            # Підрахунок периодів тренду (для TREND_CANDLES перевірки)
            df['trend_periods_up'] = 0
            df['trend_periods_down'] = 0

            for i in range(self.config.TREND_CANDLES, len(df)):
                # Перевіряєм останні TREND_CANDLES періодів
                recent_trends = df['rsi_sma_trend_direction'].iloc[i - self.config.TREND_CANDLES + 1:i + 1]

                if all(trend >= 0 for trend in recent_trends):  # Висхідний або нейтральний тренд
                    df.loc[i, 'trend_periods_up'] = self.config.TREND_CANDLES

                if all(trend <= 0 for trend in recent_trends):  # Нисхідний чи нейтральний тренд
                    df.loc[i, 'trend_periods_down'] = self.config.TREND_CANDLES

            # Болінгер бенди (опціонально)
            if self.config.USE_BOLLINGER:
                bb = BollingerBands(close=df['close'], window=self.config.BOLLINGER_PERIOD,
                                    window_dev=self.config.BOLLINGER_STD)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_middle'] = bb.bollinger_mavg()

        except Exception as e:
            self.logger.error(f"Помилка розрахунку індикаторів: {e}")

        return df

    def analyze_volume(self, current_row: pd.Series) -> VolumeAnalysis:
        """Аналіз об'єму для поточної свічки"""
        try:
            if not self.config.USE_VOLUME_CHECKS:
                return VolumeAnalysis(0, 0, 1.0, True, 0, "disabled")

            current_volume = current_row.get('volume', 0)
            avg_volume = current_row.get('volume_sma', current_volume)

            if avg_volume == 0:
                volume_ratio = 1.0
            else:
                volume_ratio = current_volume / avg_volume

            is_above_average = volume_ratio >= self.config.MIN_VOLUME_MULTIPLIER

            # Розрахунок скору об'єму
            if volume_ratio >= self.config.HIGH_VOLUME_MULTIPLIER * 1.5:
                volume_score = self.config.SCORING_SYSTEM['very_high_volume']
                volume_quality = "Very High"
            elif volume_ratio >= self.config.HIGH_VOLUME_MULTIPLIER:
                volume_score = self.config.SCORING_SYSTEM['high_volume']
                volume_quality = "High"
            elif volume_ratio >= self.config.MIN_VOLUME_MULTIPLIER:
                volume_score = self.config.SCORING_SYSTEM['normal_volume']
                volume_quality = "Normal"
            else:
                volume_score = self.config.SCORING_SYSTEM['low_volume']
                volume_quality = "Low"

            return VolumeAnalysis(
                current_volume=current_volume,
                avg_volume=avg_volume,
                volume_ratio=volume_ratio,
                is_above_average=is_above_average,
                volume_score=volume_score,
                volume_quality=volume_quality
            )

        except Exception as e:
            self.logger.error(f"Помилка аналізу об'єму: {e}")
            return VolumeAnalysis(0, 0, 1.0, True, 0, "error")

    def analyze_volatility(self, current_row: pd.Series) -> VolatilityAnalysis:
        """Аналіз волатільності для поточної свічки"""
        try:
            if not self.config.USE_VOLATILITY_CHECKS:
                return VolatilityAnalysis(0, 0, 1.0, True, 0, "disabled")

            current_atr = current_row.get('atr', 0)
            avg_atr = current_row.get('atr_sma', current_atr)

            if avg_atr == 0:
                volatility_ratio = 1.0
            else:
                volatility_ratio = current_atr / avg_atr

            is_adequate = volatility_ratio >= self.config.MIN_VOLATILITY_MULTIPLIER

            # Розрахунок скору волатільності
            if self.config.MIN_VOLATILITY_MULTIPLIER <= volatility_ratio <= self.config.HIGH_VOLATILITY_MULTIPLIER:
                volatility_score = self.config.SCORING_SYSTEM['optimal_volatility']
                volatility_quality = "Optimal"
            elif volatility_ratio >= self.config.HIGH_VOLATILITY_MULTIPLIER:
                volatility_score = self.config.SCORING_SYSTEM['high_volatility']
                volatility_quality = "High"
            elif volatility_ratio >= self.config.MIN_VOLATILITY_MULTIPLIER:
                volatility_score = self.config.SCORING_SYSTEM['normal_volatility']
                volatility_quality = "Normal"
            else:
                volatility_score = self.config.SCORING_SYSTEM['low_volatility']
                volatility_quality = "Low"

            return VolatilityAnalysis(
                current_atr=current_atr,
                avg_atr=avg_atr,
                volatility_ratio=volatility_ratio,
                is_adequate=is_adequate,
                volatility_score=volatility_score,
                volatility_quality=volatility_quality
            )

        except Exception as e:
            self.logger.error(f"Помилка аналізу волатільності: {e}")
            return VolatilityAnalysis(0, 0, 1.0, True, 0, "error")

    def calculate_signal_score(self,
                               primary_passed: List[CheckResult],
                               primary_failed: List[CheckResult],
                               confirmation_passed: List[CheckResult],
                               confirmation_failed: List[CheckResult],
                               volume_analysis: VolumeAnalysis,
                               volatility_analysis: VolatilityAnalysis,
                               current_primary: pd.Series,
                               current_confirmation: pd.Series,
                               direction: str) -> SignalScore:
        """Розрахунок детального скору сигналу"""
        try:
            # Базовий скор
            base_score = self.config.SCORING_SYSTEM['base_score']

            # Технічний скор
            technical_score = 0.0

            # Бали за успішні перевірки
            for check in primary_passed:
                technical_score += self.config.SCORING_SYSTEM['primary_check_passed']

            for check in confirmation_passed:
                technical_score += self.config.SCORING_SYSTEM['confirmation_check_passed']

            # Штрафи за невдалі перевірки
            for check in primary_failed:
                technical_score += self.config.SCORING_SYSTEM['primary_check_failed']

            for check in confirmation_failed:
                technical_score += self.config.SCORING_SYSTEM['confirmation_check_failed']

            # Бонуси за якість сигналу
            bonus_score = 0.0

            # Бонус за ідеальний перетин RSI/SMA
            rsi_cross_quality = abs(current_primary.get('rsi', 0) - current_primary.get('rsi_sma', 0))
            if rsi_cross_quality >= 3.0:
                bonus_score += self.config.SCORING_SYSTEM['perfect_rsi_cross']

            # Бонус за сильну різницю на підтвердженні
            confirmation_diff = abs(current_confirmation.get('rsi_diff', 0))
            if confirmation_diff >= self.config.MIN_DIFF * 1.5:
                bonus_score += self.config.SCORING_SYSTEM['strong_diff']

            # Бонус за трендове вирівнювання
            rsi_trend = current_confirmation.get('rsi_sma_trend', 0)
            if direction == "Long" and rsi_trend >= 0:
                bonus_score += self.config.SCORING_SYSTEM['trend_alignment']
            elif direction == "Short" and rsi_trend <= 0:
                bonus_score += self.config.SCORING_SYSTEM['trend_alignment']

            # Скор об'єму та волатільності
            volume_score = volume_analysis.volume_score
            volatility_score = volatility_analysis.volatility_score

            # Штрафи
            penalty_score = 0.0
            if not volume_analysis.is_above_average:
                penalty_score += self.config.SCORING_SYSTEM['weak_signal']
            if not volatility_analysis.is_adequate:
                penalty_score += self.config.SCORING_SYSTEM['weak_signal']

            # Загальний скор
            total_score = (base_score + technical_score + bonus_score +
                           volume_score + volatility_score + penalty_score)

            # Максимально можливий скор (для розрахунку відсотка)
            max_possible_score = (base_score +
                                  len(primary_passed) * self.config.SCORING_SYSTEM['primary_check_passed'] +
                                  len(confirmation_passed) * self.config.SCORING_SYSTEM['confirmation_check_passed'] +
                                  self.config.SCORING_SYSTEM['perfect_rsi_cross'] +
                                  self.config.SCORING_SYSTEM['strong_diff'] +
                                  self.config.SCORING_SYSTEM['trend_alignment'] +
                                  self.config.SCORING_SYSTEM['very_high_volume'] +
                                  self.config.SCORING_SYSTEM['optimal_volatility'])

            # Відсоток від максимального скору
            score_percentage = min(100.0, max(0.0, (total_score / max_possible_score) * 100.0))

            return SignalScore(
                base_score=base_score,
                technical_score=technical_score,
                volume_score=volume_score,
                volatility_score=volatility_score,
                bonus_score=bonus_score,
                penalty_score=penalty_score,
                total_score=total_score,
                max_possible_score=max_possible_score,
                score_percentage=score_percentage
            )

        except Exception as e:
            self.logger.error(f"Помилка розрахунку скору: {e}")
            # Повертаємо мінімальний скор у разі помилки
            return SignalScore(50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 100.0, 50.0)

    def determine_quality_from_score(self, total_score: float) -> SignalQuality:
        """Визначення якості сигналу на основі скору"""
        if total_score >= 85:
            return SignalQuality.EXCELLENT
        elif total_score >= 70:
            return SignalQuality.HIGH
        elif total_score >= 55:
            return SignalQuality.MEDIUM
        elif total_score >= 40:
            return SignalQuality.LOW
        else:
            return SignalQuality.POOR

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

        # Знаходимо відповідну свічку на таймфреймі для підтвердження
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

        # Якщо жоден з сигналів не підтвердився
        return None, [], [], [], [], SignalStatus.SKIP, "No valid signal conditions"

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
            description="RSI перетинає SMA знизу вверх",
            score_impact=15.0 if rsi_cross_up else -10.0
        )

        if rsi_cross_up:
            passed.append(check)
        else:
            failed.append(check)

        # 2. Зглажений RSI < LONG_ENTRY_MAX_RSI
        rsi_in_zone = previous_primary['rsi_sma'] < self.config.LONG_ENTRY_MAX_RSI

        check = CheckResult(
            name="RSI Entry Zone",
            passed=rsi_in_zone,
            value=f"RSI_SMA={previous_primary['rsi_sma']:.2f} < {self.config.LONG_ENTRY_MAX_RSI}",
            description=f"RSI нижче рівня входу {self.config.LONG_ENTRY_MAX_RSI}",
            score_impact=15.0 if rsi_in_zone else -10.0
        )

        if rsi_in_zone:
            passed.append(check)
        else:
            failed.append(check)

        # 3. Делей: остання ПІДТВЕРДЖЕНА LONG позиція закрита мінімум DELAY_CANDLES тому
        delay_ok = self.check_confirmed_position_delay(symbol, "Long", current_primary['datetime'])

        check = CheckResult(
            name="Position Delay",
            passed=delay_ok,
            value=f"Delay >= {self.config.DELAY_CANDLES} candles",
            description=f"Мінімум {self.config.DELAY_CANDLES} свічок після останньої підтвердженої Long позиції",
            score_impact=15.0 if delay_ok else -10.0
        )

        if delay_ok:
            passed.append(check)
        else:
            failed.append(check)

        return passed, failed

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
            description="RSI перетинає SMA зверху вниз",
            score_impact=15.0 if rsi_cross_down else -10.0
        )

        if rsi_cross_down:
            passed.append(check)
        else:
            failed.append(check)

        # 2. Зглажений RSI > SHORT_ENTRY_MIN_RSI
        rsi_in_zone = previous_primary['rsi_sma'] > self.config.SHORT_ENTRY_MIN_RSI

        check = CheckResult(
            name="RSI Entry Zone",
            passed=rsi_in_zone,
            value=f"RSI_SMA={previous_primary['rsi_sma']:.2f} > {self.config.SHORT_ENTRY_MIN_RSI}",
            description=f"RSI вище рівня входу {self.config.SHORT_ENTRY_MIN_RSI}",
            score_impact=15.0 if rsi_in_zone else -10.0
        )

        if rsi_in_zone:
            passed.append(check)
        else:
            failed.append(check)

        # 3. Делей: остання ПІДТВЕРДЖЕНА SHORT позиція закрита мінімум DELAY_CANDLES тому
        delay_ok = self.check_confirmed_position_delay(symbol, "Short", current_primary['datetime'])

        check = CheckResult(
            name="Position Delay",
            passed=delay_ok,
            value=f"Delay >= {self.config.DELAY_CANDLES} candles",
            description=f"Мінімум {self.config.DELAY_CANDLES} свічок після останньої підтвердженої Short позиції",
            score_impact=15.0 if delay_ok else -10.0
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
            description=f"Різниця SMA-RSI достатня для входу",
            score_impact=20.0 if diff_ok else -15.0
        )

        if diff_ok:
            passed.append(check)
        else:
            failed.append(check)
            skip_reason = f"SMA-RSI diff {rsi_sma_diff:.2f} < {self.config.MIN_DIFF}"

        # 2. Якщо RSI > 50 - сигнал пропускається
        rsi_below_50 = current_confirmation['rsi'] <= 50

        check = CheckResult(
            name="RSI Below 50",
            passed=rsi_below_50,
            value=f"RSI={current_confirmation['rsi']:.2f} <= 50",
            description="RSI нижче нейтрального рівня 50",
            score_impact=20.0 if rsi_below_50 else -15.0
        )

        if rsi_below_50:
            passed.append(check)
        else:
            failed.append(check)
            if not skip_reason:
                skip_reason = f"RSI {current_confirmation['rsi']:.2f} > 50"

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
                description="SMA не спадає",
                score_impact=20.0 if sma_not_falling else -15.0
            )

            if sma_not_falling:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = "SMA спадає - не підходить для LONG"

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

            no_uptrend = not (recent_sma_changes and all(change > 0 for change in recent_sma_changes))
            no_uptrend = True  # Тимчасово вимкнуто
            check = CheckResult(
                name="No SMA Uptrend",
                passed=no_uptrend,
                value=f"SMA changes: {[f'{change:.3f}' for change in recent_sma_changes]}",
                description=f"Немає стійкого SMA тренду вгору за {self.config.TREND_CANDLES} періодів",
                score_impact=20.0 if no_uptrend else -15.0
            )

            if no_uptrend:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = f"SMA тренд вгору за {self.config.TREND_CANDLES} періодів"

        return passed, failed, skip_reason

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
            description=f"Різниця RSI-SMA достатня для входу",
            score_impact=20.0 if diff_ok else -15.0
        )

        if diff_ok:
            passed.append(check)
        else:
            failed.append(check)
            skip_reason = f"RSI-SMA різниця {rsi_sma_diff:.2f} < {self.config.MIN_DIFF}"

        # 2. RSI > 50
        rsi_above_50 = current_confirmation['rsi'] >= 50

        check = CheckResult(
            name="RSI Above 50",
            passed=rsi_above_50,
            value=f"RSI={current_confirmation['rsi']:.2f} >= 50",
            description="RSI вище нейтрального рівня 50",
            score_impact=20.0 if rsi_above_50 else -15.0
        )

        if rsi_above_50:
            passed.append(check)
        else:
            failed.append(check)
            if not skip_reason:
                skip_reason = f"RSI {current_confirmation['rsi']:.2f} < 50"

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
                description="SMA не зростає",
                score_impact=20.0 if sma_not_increasing else -15.0
            )

            if sma_not_increasing:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = "SMA зростає - не підходить для SHORT"

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

            no_downtrend = not (recent_sma_changes and all(change < 0 for change in recent_sma_changes))
            no_downtrend = True  # Тимчасово вимкнуто
            check = CheckResult(
                name="No SMA Downtrend",
                passed=no_downtrend,
                value=f"SMA changes: {[f'{change:.3f}' for change in recent_sma_changes]}",
                description=f"Немає стійкого SMA тренду вниз за {self.config.TREND_CANDLES} періодів",
                score_impact=20.0 if no_downtrend else -15.0
            )

            if no_downtrend:
                passed.append(check)
            else:
                failed.append(check)
                if not skip_reason:
                    skip_reason = f"SMA тренд вниз за {self.config.TREND_CANDLES} періодів"

        return passed, failed, skip_reason

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

    def scan_dual_timeframe_signals(self, df_primary: pd.DataFrame, df_confirmation: pd.DataFrame,
                                    pair: str) -> List[AdvancedMarketSignal]:
        """ОНОВЛЕНЕ сканування сигналів з гнучким періодом та покращеним скорингом"""
        signals = []

        if len(df_primary) < 50 or len(df_confirmation) < 10:
            print(
                f"⚠️ {pair}: Недостатньо даних ({self.config.PRIMARY_TIMEFRAME}: {len(df_primary)}, {self.config.CONFIRMATION_TIMEFRAME}: {len(df_confirmation)})")
            return signals

        # НОВИЙ: Використовуємо гнучкий період для сигналів
        end_time = datetime.now()
        start_time = end_time - self.get_signal_period_timedelta()

        print(f"🔍 {pair}: Шукаємо сигнали з {start_time.strftime('%Y-%m-%d %H:%M')} "
              f"(період: {self.config.SIGNAL_PERIOD_DAYS}д {self.config.SIGNAL_PERIOD_HOURS}г)")

        # Фільтруємо дані по періоду
        df_primary_period = df_primary[df_primary['datetime'] >= start_time].copy()

        if len(df_primary_period) < 10:
            print(f"⚠️ {pair}: Недостатньо даних за період")
            return signals

        print(f"📊 {pair}: Аналізуємо {len(df_primary_period)} свічей {self.config.PRIMARY_TIMEFRAME}")

        # Знаходимо початковий індекс
        start_idx = df_primary[df_primary['datetime'] >= start_time].index[0] if len(
            df_primary[df_primary['datetime'] >= start_time]) > 0 else len(df_primary)
        start_idx = max(20, start_idx)

        for i in range(start_idx, len(df_primary) - 1):
            current_row = df_primary.iloc[i]

            if current_row['datetime'] < start_time or current_row['datetime'] > end_time:
                continue

            try:
                # Проверка сигнала по новой детальной логике
                signal_result = self.check_advanced_rsi_signal(df_primary, df_confirmation, i, pair)

                if signal_result is None or len(signal_result) != 7:
                    continue

                direction, primary_passed, primary_failed, confirmation_passed, confirmation_failed, status, skip_reason = signal_result

                if not direction:
                    continue

                # Знаходимо відповідні дані на таймфреймі підтвердження
                current_time = current_row['datetime']
                df_confirmation_filtered = df_confirmation[df_confirmation['datetime'] <= current_time]
                if len(df_confirmation_filtered) == 0:
                    continue

                current_confirmation = df_confirmation_filtered.iloc[-1]

                # НОВИЙ: Аналіз об'єму та волатільності
                volume_analysis = self.analyze_volume(current_confirmation)
                volatility_analysis = self.analyze_volatility(current_confirmation)

                # НОВИЙ: Розрахунок детального скору
                signal_score = self.calculate_signal_score(
                    primary_passed, primary_failed,
                    confirmation_passed, confirmation_failed,
                    volume_analysis, volatility_analysis,
                    current_row, current_confirmation, direction
                )

                # Визначення якості на основі скору
                quality = self.determine_quality_from_score(signal_score.total_score)

                # ОНОВЛЕНО: confidence_score тепер синхронізований з total_score
                confidence = min(100.0, max(0.0, signal_score.total_score))

                # Розрахунок часу та ціни входу
                entry_time, entry_price = self.calculate_entry_time_and_price_advanced(
                    df_primary, i, current_row['datetime']
                )

                # Реєструємо ТІЛЬКИ ПІДТВЕРДЖЕНІ позиції для делею
                if status == SignalStatus.OPEN:
                    self.register_confirmed_position(pair, direction, current_row['datetime'])

                # Створення сигналу з НОВИМИ даними
                signal = AdvancedMarketSignal(
                    pair=pair,
                    direction=direction,
                    signal_time=current_row['datetime'],
                    entry_time=entry_time,

                    # RSI дані основного таймфрейму
                    rsi_1m=current_row.get('rsi', 0),
                    rsi_sma_1m=current_row.get('rsi_sma', 0),
                    rsi_diff_1m=current_row.get('rsi_diff', 0),

                    # RSI дані таймфрейму підтвердження
                    rsi_5m=current_confirmation.get('rsi', 0),
                    rsi_sma_5m=current_confirmation.get('rsi_sma', 0),
                    rsi_diff_5m=current_confirmation.get('rsi_diff', 0),
                    avg_diff_5m=current_confirmation.get('avg_diff', 0),

                    signal_price=current_row['close'],
                    entry_price=entry_price,
                    quality=quality,
                    confidence_score=confidence,
                    status=status,

                    # НОВІ: Аналіз об'єму та волатільності
                    volume_analysis=volume_analysis,
                    volatility_analysis=volatility_analysis,
                    signal_score=signal_score,

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
                # Додаємо обробку помилок для кожної ітерації
                self.logger.error(f"Помилка обробки свічки {i} для {pair}: {e}")
                continue

        return signals

    def print_detailed_signal_info(self, signal: AdvancedMarketSignal):
        """ОНОВЛЕНИЙ виведення детальної інформації про сигнал з новими даними"""
        status_emoji = "✅" if signal.status == SignalStatus.OPEN else "⏭️"

        print(f"\n{status_emoji} {signal.pair} {signal.direction} - {signal.signal_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Статус: {signal.status.value} | Скор: {signal.signal_score.total_score:.1f} | "
              f"Впевненість: {signal.confidence_score:.1f}% | Якість: {signal.quality.value}")

        if signal.status == SignalStatus.SKIP:
            print(f"   💡 Причина пропуску: {signal.skip_reason}")

        # НОВИЙ: Інформація про об'єм та волатільність
        if signal.volume_analysis.volume_quality != "disabled":
            print(f"   📊 Об'єм: {signal.volume_analysis.volume_quality} "
                  f"(ratio: {signal.volume_analysis.volume_ratio:.2f}, score: {signal.volume_analysis.volume_score:+.1f})")

        if signal.volatility_analysis.volatility_quality != "disabled":
            print(f"   📈 Волатільність: {signal.volatility_analysis.volatility_quality} "
                  f"(ratio: {signal.volatility_analysis.volatility_ratio:.2f}, score: {signal.volatility_analysis.volatility_score:+.1f})")

        # НОВИЙ: Детальний розклад скору
        print(f"   🎯 Скор: Base={signal.signal_score.base_score:.1f} | "
              f"Technical={signal.signal_score.technical_score:+.1f} | "
              f"Volume={signal.signal_score.volume_score:+.1f} | "
              f"Volatility={signal.signal_score.volatility_score:+.1f} | "
              f"Bonus={signal.signal_score.bonus_score:+.1f} | "
              f"Penalty={signal.signal_score.penalty_score:+.1f}")

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

    async def analyze_pair_advanced(self, pair: str) -> List[AdvancedMarketSignal]:
        """ОНОВЛЕНИЙ аналіз пари з новою системою скорингу"""
        try:
            print(f"\n🔍 Аналізуємо {pair}...")

            # Розраховуємо необхідні дні для завантаження даних (більше ніж період сигналів для індикаторів)
            days_back = max(self.get_signal_period_days(), 7)

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

            # Скануємо сигнали
            signals = self.scan_dual_timeframe_signals(df_primary, df_confirmation, pair)

            print(f"\n📊 {pair}: Знайдено {len(signals)} сигналів")
            open_signals = [s for s in signals if s.status == SignalStatus.OPEN]
            skip_signals = [s for s in signals if s.status == SignalStatus.SKIP]
            print(f"   ✅ Підтверджених: {len(open_signals)}")
            print(f"   ⏭️ Пропущених: {len(skip_signals)}")

            if open_signals:
                avg_score = sum(s.signal_score.total_score for s in open_signals) / len(open_signals)
                print(f"   🎯 Середній скор підтверджених: {avg_score:.1f}")

            return signals

        except Exception as e:
            self.logger.error(f"Помилка аналізу {pair}: {e}")
            return []

    def save_signals_to_csv(self, all_signals: List[AdvancedMarketSignal], filename: str = None):
        """ОНОВЛЕНЕ збереження сигналів у CSV з новими даними"""
        if not all_signals:
            print("📄 Немає сигналів для збереження")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_signals_v23_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=";")

                # ОНОВЛЕНІ заголовки з новими полями
                headers = [
                    'Pair', 'Direction', 'Status', 'Signal_Time', 'Entry_Time',
                    f'RSI_{self.config.PRIMARY_TIMEFRAME}', f'RSI_SMA_{self.config.PRIMARY_TIMEFRAME}',
                    f'RSI_Diff_{self.config.PRIMARY_TIMEFRAME}',
                    f'RSI_{self.config.CONFIRMATION_TIMEFRAME}', f'RSI_SMA_{self.config.CONFIRMATION_TIMEFRAME}',
                    f'RSI_Diff_{self.config.CONFIRMATION_TIMEFRAME}', f'Avg_Diff_{self.config.CONFIRMATION_TIMEFRAME}',
                    'Signal_Price', 'Entry_Price', 'Quality', 'Confidence',

                    # НОВІ колонки для скорингу
                    'Total_Score', 'Base_Score', 'Technical_Score', 'Volume_Score',
                    'Volatility_Score', 'Bonus_Score', 'Penalty_Score', 'Score_Percentage',

                    # НОВІ колонки для об'єму
                    'Volume_Quality', 'Volume_Ratio', 'Current_Volume', 'Avg_Volume',

                    # НОВІ колонки для волатільності
                    'Volatility_Quality', 'Volatility_Ratio', 'Current_ATR', 'Avg_ATR',

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
                        f"{str(signal.rsi_1m).replace('.', ',')}",
                        f"{str(signal.rsi_sma_1m).replace('.', ',')}",
                        f"{str(signal.rsi_diff_1m).replace('.', ',')}",
                        f"{str(signal.rsi_5m).replace('.', ',')}",
                        f"{str(signal.rsi_sma_5m).replace('.', ',')}",
                        f"{str(signal.rsi_diff_5m).replace('.', ',')}",
                        f"{str(signal.avg_diff_5m).replace('.', ',')}",
                        f"{str(signal.signal_price).replace('.', ',')}",
                        f"{str(signal.entry_price).replace('.', ',')}",
                        signal.quality.value,
                        f"{signal.confidence_score:.1f}%",

                        # НОВІ дані скорингу
                        f"{str(signal.signal_score.total_score).replace('.', ',')}",
                        f"{str(signal.signal_score.base_score).replace('.', ',')}",
                        f"{str(signal.signal_score.technical_score).replace('.', ',')}",
                        f"{str(signal.signal_score.volume_score).replace('.', ',')}",
                        f"{str(signal.signal_score.volatility_score).replace('.', ',')}",
                        f"{str(signal.signal_score.bonus_score).replace('.', ',')}",
                        f"{str(signal.signal_score.penalty_score).replace('.', ',')}",
                        f"{str(signal.signal_score.score_percentage).replace('.', ',')}",

                        # НОВІ дані об'єму
                        signal.volume_analysis.volume_quality,
                        f"{str(signal.volume_analysis.volume_ratio).replace('.', ',')}",
                        f"{str(signal.volume_analysis.current_volume).replace('.', ',')}",
                        f"{str(signal.volume_analysis.avg_volume).replace('.', ',')}",

                        # НОВІ дані волатільності
                        signal.volatility_analysis.volatility_quality,
                        f"{str(signal.volatility_analysis.volatility_ratio).replace('.', ',')}",
                        f"{str(signal.volatility_analysis.current_atr).replace('.', ',')}",
                        f"{str(signal.volatility_analysis.avg_atr).replace('.', ',')}",

                        primary_passed_str,
                        primary_failed_str,
                        confirmation_passed_str,
                        confirmation_failed_str,
                        signal.skip_reason,
                        signal.comment
                    ]
                    writer.writerow(row)

            print(f"📄 Розширені сигнали збережено у {filename}")

            # ОНОВЛЕНА статистика
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

                avg_score = sum(s.signal_score.total_score for s in open_signals) / len(open_signals)
                avg_confidence = sum(s.confidence_score for s in open_signals) / len(open_signals)
                print(f"   Середній скор: {avg_score:.1f}")
                print(f"   Середня впевненість: {avg_confidence:.1f}%")

                # Розподіл по якості
                quality_counts = {}
                for signal in open_signals:
                    quality = signal.quality.value
                    quality_counts[quality] = quality_counts.get(quality, 0) + 1

                print(f"   Розподіл по якості: {dict(quality_counts)}")

        except Exception as e:
            self.logger.error(f"Помилка збереження у CSV: {e}")

    async def run_advanced_analysis(self):
        """ОНОВЛЕНИЙ запуск аналізу з новими можливостями"""
        print(f"🚀 Запуск покращеного аналізу v2.3")
        print(f"📊 Таймфрейми: {self.config.PRIMARY_TIMEFRAME}/{self.config.CONFIRMATION_TIMEFRAME}")
        print(f"⏰ Період сигналів: {self.config.SIGNAL_PERIOD_DAYS} днів {self.config.SIGNAL_PERIOD_HOURS} годин")
        print(f"💰 Пари: {len(self.config.PAIRS)}")
        print(f"📈 Об'єм: {'Увімкнено' if self.config.USE_VOLUME_CHECKS else 'Вимкнено'}")
        print(f"📊 Волатільність: {'Увімкнено' if self.config.USE_VOLATILITY_CHECKS else 'Вимкнено'}")
        print(f"🔧 Делей працює тільки для підтверджених сигналів!")

        all_signals = []

        for pair in self.config.PAIRS:
            try:
                signals = await self.analyze_pair_advanced(pair)
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

        open_signals = [s for s in all_signals if s.status == SignalStatus.OPEN]
        skip_signals = [s for s in all_signals if s.status == SignalStatus.SKIP]

        print(f"📊 Підтверджених: {len(open_signals)}")
        print(f"⏭️ Пропущених: {len(skip_signals)}")

        if open_signals:
            excellent_signals = [s for s in open_signals if s.quality == SignalQuality.EXCELLENT]
            high_signals = [s for s in open_signals if s.quality == SignalQuality.HIGH]

            print(f"🌟 Відмінних: {len(excellent_signals)}")
            print(f"⭐ Високої якості: {len(high_signals)}")

            avg_score = sum(s.signal_score.total_score for s in open_signals) / len(open_signals)
            print(f"🎯 Середній скор: {avg_score:.1f}")

        return all_signals


# Головна функція запуску
async def main():
    """ОНОВЛЕНА головна функція з новими налаштуваннями"""
    # Створюємо конфігурацію з НОВИМИ параметрами
    config = AdvancedConfig(
        # Налаштування таймфреймів
        PRIMARY_TIMEFRAME='5m',
        CONFIRMATION_TIMEFRAME='15m',

        # НОВИЙ: Гнучкий період для сигналів
        SIGNAL_PERIOD_HOURS=0,  # Можна змінити на будь-який період!
        SIGNAL_PERIOD_DAYS=7,  # Додатково дні (наприклад, 3 дні + 12 годин)

        # Налаштування параметрів стратегії
        MIN_DIFF=2.0,
        TREND_CANDLES=3,
        DELAY_CANDLES=12,
        LONG_ENTRY_MAX_RSI=40.0,
        SHORT_ENTRY_MIN_RSI=60.0,
        PREOVERSOLD_LEVEL=33.0,
        PREBOUGHT_LEVEL=67.0,

        # НОВІ: Параметри об'єму
        USE_VOLUME_CHECKS=True,
        VOLUME_SMA_PERIOD=20,
        MIN_VOLUME_MULTIPLIER=1.15,
        HIGH_VOLUME_MULTIPLIER=1.8,

        # НОВІ: Параметри волатільності
        USE_VOLATILITY_CHECKS=True,
        ATR_PERIOD=14,
        MIN_VOLATILITY_MULTIPLIER=0.7,
        HIGH_VOLATILITY_MULTIPLIER=1.3,

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
        signals = await generator.run_advanced_analysis()

        # Додатковий аналіз результатів
        if signals:
            print(f"\n📊 Детальна статистика:")

            open_signals = [s for s in signals if s.status == SignalStatus.OPEN]
            skip_signals = [s for s in signals if s.status == SignalStatus.SKIP]

            if open_signals:
                print(f"🟢 Підтверджені позиції ({len(open_signals)}):")
                # Сортуємо по скору та показуємо топ 5
                sorted_signals = sorted(open_signals, key=lambda x: x.signal_score.total_score, reverse=True)
                for signal in sorted_signals[:5]:
                    print(f"   {signal.pair} {signal.direction} {signal.signal_time.strftime('%m-%d %H:%M')} "
                          f"Скор: {signal.signal_score.total_score:.1f} Якість: {signal.quality.value}")

            if skip_signals:
                print(f"🟡 Топ причини пропусків:")
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