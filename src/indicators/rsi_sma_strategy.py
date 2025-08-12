"""
RSI-SMA Trading Strategy Implementation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging


@dataclass
class StrategyConfig:
    rsi_period: int = 14
    rsi_sma_period: int = 14
    min_diff: float = 2.0
    trend_candles: int = 3
    delay_candles: int = 5
    long_entry_max_rsi: int = 40
    short_entry_min_rsi: int = 60
    overbought_level: int = 70
    oversold_level: int = 30
    diff_smoothing_periods: int = 3


@dataclass
class Signal:
    timestamp: datetime
    exchange: str
    symbol: str
    direction: str  # LONG or SHORT
    action: str  # OPEN or CLOSE
    rsi_1m: float
    sma_1m: float
    rsi_5m: float
    sma_5m: float
    price: float
    volume: float
    diff: float
    trend_confirmed: bool
    delay_ok: bool
    comment: str


class RSISMAStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Збереження історії закриття позицій для делею
        self.position_history: Dict[str, datetime] = {}

        # Кеш для зберігання останніх даних
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Розрахунок RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Розрахунок Simple Moving Average"""
        return data.rolling(window=period).mean()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок всіх індикаторів"""
        df = df.copy()

        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.config.rsi_period)

        # RSI SMA (згладжений RSI)
        df['rsi_sma'] = self.calculate_sma(df['rsi'], self.config.rsi_sma_period)

        # Різниця між RSI і RSI SMA
        df['rsi_diff'] = df['rsi'] - df['rsi_sma']

        # Середня різниця за N періодів
        df['avg_diff'] = df['rsi_diff'].rolling(window=self.config.diff_smoothing_periods).mean().abs()

        return df

    def check_trend(self, sma_values: pd.Series, direction: str) -> bool:
        """Перевірка тренду SMA за N періодів"""
        if len(sma_values) < self.config.trend_candles:
            return False

        recent_values = sma_values.tail(self.config.trend_candles).values

        if direction == 'LONG':
            # Перевіряємо зростаючий тренд
            return all(recent_values[i] <= recent_values[i + 1] for i in range(len(recent_values) - 1))
        elif direction == 'SHORT':
            # Перевіряємо спадаючий тренд
            return all(recent_values[i] >= recent_values[i + 1] for i in range(len(recent_values) - 1))

        return False

    def check_crossover(self, rsi: pd.Series, sma: pd.Series, direction: str) -> bool:
        """Перевірка перетину RSI і RSI SMA"""
        if len(rsi) < 2 or len(sma) < 2:
            return False

        current_rsi, prev_rsi = rsi.iloc[-1], rsi.iloc[-2]
        current_sma, prev_sma = sma.iloc[-1], sma.iloc[-2]

        if direction == 'LONG':
            # RSI перетинає SMA знизу вгору
            return prev_rsi <= prev_sma and current_rsi > current_sma
        elif direction == 'SHORT':
            # RSI перетинає SMA зверху вниз
            return prev_rsi >= prev_sma and current_rsi < current_sma
        elif direction == 'CLOSE_LONG':
            # RSI перетинає SMA зверху вниз (закриття лонгу)
            return prev_rsi >= prev_sma and current_rsi < current_sma
        elif direction == 'CLOSE_SHORT':
            # RSI перетинає SMA знизу вгору (закриття шорту)
            return prev_rsi <= prev_sma and current_rsi > current_sma

        return False

    def check_delay(self, exchange: str, symbol: str, direction: str) -> bool:
        """Перевірка делею між позиціями"""
        key = f"{exchange}_{symbol}_{direction}"

        if key not in self.position_history:
            return True

        last_close = self.position_history[key]
        time_diff = datetime.now() - last_close

        # Перевіряємо чи пройшло достатньо часу (delay_candles * 1 хвилина)
        required_delay = timedelta(minutes=self.config.delay_candles)
        return time_diff >= required_delay

    def update_position_history(self, exchange: str, symbol: str, direction: str):
        """Оновлення історії закриття позицій"""
        key = f"{exchange}_{symbol}_{direction}"
        self.position_history[key] = datetime.now()

    def check_5m_confirmation(self, df_5m: pd.DataFrame, direction: str) -> Tuple[bool, str]:
        """Підтвердження сигналу на 5-хвилинному таймфреймі"""
        if len(df_5m) < 2:
            return False, "Insufficient 5m data"

        df_5m = self.calculate_indicators(df_5m)
        current_rsi = df_5m['rsi'].iloc[-1]
        current_sma = df_5m['rsi_sma'].iloc[-1]
        prev_sma = df_5m['rsi_sma'].iloc[-2] if len(df_5m) >= 2 else current_sma

        if direction == 'LONG':
            # Для лонгу: RSI >= SMA і немає різкого падіння SMA
            rsi_condition = current_rsi >= current_sma
            sma_stable = current_sma >= prev_sma * 0.98  # не більше 2% падіння

            if not rsi_condition:
                return False, "5m RSI below SMA"
            if not sma_stable:
                return False, "5m SMA falling sharply"

            return True, "5m confirmation OK"

        elif direction == 'SHORT':
            # Для шорту: RSI <= SMA і немає різкого зростання SMA
            rsi_condition = current_rsi <= current_sma
            sma_stable = current_sma <= prev_sma * 1.02  # не більше 2% зростання

            if not rsi_condition:
                return False, "5m RSI above SMA"
            if not sma_stable:
                return False, "5m SMA rising sharply"

            return True, "5m confirmation OK"

        return False, "Unknown direction"

    def analyze_entry_signals(self, exchange: str, symbol: str,
                              df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> List[Signal]:
        """Аналіз сигналів для входу в позицію"""
        signals = []

        if len(df_1m) < max(self.config.rsi_period, self.config.rsi_sma_period) + self.config.trend_candles:
            return signals

        # Розрахунок індикаторів для 1m
        df_1m = self.calculate_indicators(df_1m)
        df_5m = self.calculate_indicators(df_5m) if len(df_5m) > 0 else pd.DataFrame()

        current_time = datetime.now()
        current_rsi = df_1m['rsi'].iloc[-1]
        current_sma = df_1m['rsi_sma'].iloc[-1]
        current_diff = df_1m['avg_diff'].iloc[-1]
        current_price = df_1m['close'].iloc[-1]
        current_volume = df_1m['volume'].iloc[-1]

        # Перевірка LONG сигналу
        if (self.check_crossover(df_1m['rsi'], df_1m['rsi_sma'], 'LONG') and
                current_rsi < self.config.long_entry_max_rsi and
                current_diff >= self.config.min_diff and
                self.check_trend(df_1m['rsi_sma'], 'LONG') and
                self.check_delay(exchange, symbol, 'LONG')):

            # Підтвердження на 5m
            if len(df_5m) > 0:
                confirmation_ok, comment = self.check_5m_confirmation(df_5m, 'LONG')
                if confirmation_ok:
                    rsi_5m = df_5m['rsi'].iloc[-1] if len(df_5m) > 0 else 0
                    sma_5m = df_5m['rsi_sma'].iloc[-1] if len(df_5m) > 0 else 0

                    signal = Signal(
                        timestamp=current_time,
                        exchange=exchange,
                        symbol=symbol,
                        direction='LONG',
                        action='OPEN',
                        rsi_1m=current_rsi,
                        sma_1m=current_sma,
                        rsi_5m=rsi_5m,
                        sma_5m=sma_5m,
                        price=current_price,
                        volume=current_volume,
                        diff=current_diff,
                        trend_confirmed=True,
                        delay_ok=True,
                        comment=comment
                    )
                    signals.append(signal)

        # Перевірка SHORT сигналу
        if (self.check_crossover(df_1m['rsi'], df_1m['rsi_sma'], 'SHORT') and
                current_rsi > self.config.short_entry_min_rsi and
                current_diff >= self.config.min_diff and
                self.check_trend(df_1m['rsi_sma'], 'SHORT') and
                self.check_delay(exchange, symbol, 'SHORT')):

            # Підтвердження на 5m
            if len(df_5m) > 0:
                confirmation_ok, comment = self.check_5m_confirmation(df_5m, 'SHORT')
                if confirmation_ok:
                    rsi_5m = df_5m['rsi'].iloc[-1] if len(df_5m) > 0 else 0
                    sma_5m = df_5m['rsi_sma'].iloc[-1] if len(df_5m) > 0 else 0

                    signal = Signal(
                        timestamp=current_time,
                        exchange=exchange,
                        symbol=symbol,
                        direction='SHORT',
                        action='OPEN',
                        rsi_1m=current_rsi,
                        sma_1m=current_sma,
                        rsi_5m=rsi_5m,
                        sma_5m=sma_5m,
                        price=current_price,
                        volume=current_volume,
                        diff=current_diff,
                        trend_confirmed=True,
                        delay_ok=True,
                        comment=comment
                    )
                    signals.append(signal)

        return signals

    def analyze_exit_signals(self, exchange: str, symbol: str,
                             df_1m: pd.DataFrame, df_5m: pd.DataFrame,
                             open_positions: List[Dict]) -> List[Signal]:
        """Аналіз сигналів для виходу з позиції"""
        signals = []

        if len(df_1m) < 2:
            return signals

        df_1m = self.calculate_indicators(df_1m)
        df_5m = self.calculate_indicators(df_5m) if len(df_5m) > 0 else pd.DataFrame()

        current_time = datetime.now()
        current_rsi_1m = df_1m['rsi'].iloc[-1]
        current_sma_1m = df_1m['rsi_sma'].iloc[-1]
        current_price = df_1m['close'].iloc[-1]
        current_volume = df_1m['volume'].iloc[-1]

        rsi_5m = df_5m['rsi'].iloc[-1] if len(df_5m) > 0 else 0
        sma_5m = df_5m['rsi_sma'].iloc[-1] if len(df_5m) > 0 else 0

        for position in open_positions:
            direction = position.get('side', '').upper()

            # Перевірка негайного закриття LONG позицій
            if direction == 'LONG':
                immediate_close = False
                comment = ""

                # Crossover down на 5m або RSI > overbought на 5m
                if len(df_5m) >= 2:
                    if self.check_crossover(df_5m['rsi'], df_5m['rsi_sma'], 'CLOSE_LONG'):
                        immediate_close = True
                        comment = "5m crossover down"
                    elif rsi_5m > self.config.overbought_level:
                        immediate_close = True
                        comment = f"5m RSI overbought ({rsi_5m:.1f})"

                # Звичайне закриття - crossover на 1m
                if not immediate_close and self.check_crossover(df_1m['rsi'], df_1m['rsi_sma'], 'CLOSE_LONG'):
                    immediate_close = True
                    comment = "1m crossover down"

                if immediate_close:
                    signal = Signal(
                        timestamp=current_time,
                        exchange=exchange,
                        symbol=symbol,
                        direction='LONG',
                        action='CLOSE',
                        rsi_1m=current_rsi_1m,
                        sma_1m=current_sma_1m,
                        rsi_5m=rsi_5m,
                        sma_5m=sma_5m,
                        price=current_price,
                        volume=current_volume,
                        diff=0,
                        trend_confirmed=False,
                        delay_ok=False,
                        comment=comment
                    )
                    signals.append(signal)

            # Перевірка негайного закриття SHORT позицій
            elif direction == 'SHORT':
                immediate_close = False
                comment = ""

                # Crossover up на 5m або RSI < oversold на 5m
                if len(df_5m) >= 2:
                    if self.check_crossover(df_5m['rsi'], df_5m['rsi_sma'], 'CLOSE_SHORT'):
                        immediate_close = True
                        comment = "5m crossover up"
                    elif rsi_5m < self.config.oversold_level:
                        immediate_close = True
                        comment = f"5m RSI oversold ({rsi_5m:.1f})"

                # Звичайне закриття - crossover на 1m
                if not immediate_close and self.check_crossover(df_1m['rsi'], df_1m['rsi_sma'], 'CLOSE_SHORT'):
                    immediate_close = True
                    comment = "1m crossover up"

                if immediate_close:
                    signal = Signal(
                        timestamp=current_time,
                        exchange=exchange,
                        symbol=symbol,
                        direction='SHORT',
                        action='CLOSE',
                        rsi_1m=current_rsi_1m,
                        sma_1m=current_sma_1m,
                        rsi_5m=rsi_5m,
                        sma_5m=sma_5m,
                        price=current_price,
                        volume=current_volume,
                        diff=0,
                        trend_confirmed=False,
                        delay_ok=False,
                        comment=comment
                    )
                    signals.append(signal)

        return signals

    def process_signals(self, exchange: str, symbol: str,
                        df_1m: pd.DataFrame, df_5m: pd.DataFrame,
                        open_positions: List[Dict] = None) -> List[Signal]:
        """Основний метод для обробки сигналів"""
        all_signals = []

        if open_positions is None:
            open_positions = []

        # Аналіз сигналів входу
        entry_signals = self.analyze_entry_signals(exchange, symbol, df_1m, df_5m)
        all_signals.extend(entry_signals)

        # Аналіз сигналів виходу
        if open_positions:
            exit_signals = self.analyze_exit_signals(exchange, symbol, df_1m, df_5m, open_positions)
            all_signals.extend(exit_signals)

        return all_signals