"""
RSI-SMA Trading Strategy Implementation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class RSISMAStrategy:
    """Імплементація торгової стратегії RSI-SMA"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Параметри індикаторів
        indicators_config = config.get('indicators', {})
        self.rsi_period = indicators_config.get('rsi_period', 14)
        self.rsi_sma_period = indicators_config.get('rsi_sma_period', 14)
        self.bollinger_stddev = indicators_config.get('bollinger_stddev', 2)

        # Параметри стратегії
        params = config.get('parameters', {})
        self.min_diff = params.get('min_diff', 2.0)
        self.trend_candles = params.get('trend_candles', 3)
        self.delay_candles = params.get('delay_candles', 5)
        self.long_entry_max_rsi = params.get('long_entry_max_rsi', 40)
        self.short_entry_min_rsi = params.get('short_entry_min_rsi', 60)
        self.overbought_level = params.get('overbought_level', 70)
        self.oversold_level = params.get('oversold_level', 30)
        self.diff_smoothing_periods = params.get('diff_smoothing_periods', 3)

        # Кеш для попередніх розрахунків
        self.cache = {}

        self.logger.info(f"Стратегія RSI-SMA ініціалізована з параметрами: "
                         f"RSI={self.rsi_period}, SMA={self.rsi_sma_period}")

    def calculate_indicators(self, klines: List[Dict], timeframe: str) -> Optional[Dict]:
        """Розрахунок всіх індикаторів для даного таймфрейму"""
        try:
            if len(klines) < max(self.rsi_period, self.rsi_sma_period) + 20:
                self.logger.warning(f"Недостатньо даних для розрахунку індикаторів на {timeframe}")
                return None

            # Конвертація в pandas DataFrame
            df = pd.DataFrame(klines)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Розрахунок RSI
            rsi = self._calculate_rsi(df['close'], self.rsi_period)

            # Розрахунок SMA для RSI
            rsi_sma = self._calculate_sma(rsi, self.rsi_sma_period)

            # Розрахунок різниці між RSI і SMA
            diff = rsi - rsi_sma

            # Згладжування різниці
            smoothed_diff = self._calculate_sma(diff, self.diff_smoothing_periods)

            # Додаткові індикатори для фільтрації
            bollinger_bands = self._calculate_bollinger_bands(
                df['close'], period=20, std_dev=self.bollinger_stddev
            )

            return {
                'rsi': rsi.values,
                'rsi_sma': rsi_sma.values,
                'diff': diff.values,
                'smoothed_diff': smoothed_diff.values,
                'bollinger_upper': bollinger_bands['upper'].values,
                'bollinger_lower': bollinger_bands['lower'].values,
                'bollinger_middle': bollinger_bands['middle'].values,
                'close_prices': df['close'].values,
                'volumes': df['volume'].values,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Помилка розрахунку індикаторів для {timeframe}: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Розрахунок RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Розрахунок Simple Moving Average"""
        return data.rolling(window=period).mean()

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20,
                                   std_dev: float = 2) -> Dict[str, pd.Series]:
        """Розрахунок Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return {
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev),
            'middle': sma
        }

    def check_long_conditions(self, indicators_1m: Dict, indicators_5m: Dict,
                              exchange: str, pair: str) -> Optional[Dict]:
        """Перевірка умов для входу в LONG позицію"""
        try:
            rsi_1m = indicators_1m['rsi']
            sma_1m = indicators_1m['rsi_sma']
            diff_1m = indicators_1m['smoothed_diff']

            rsi_5m = indicators_5m['rsi']
            sma_5m = indicators_5m['rsi_sma']

            # Перевірка достатності даних
            if len(rsi_1m) < self.trend_candles + 2 or len(rsi_5m) < self.trend_candles + 2:
                return None

            # 1. Перевірка crossover знизу вгору на 1m
            current_rsi = rsi_1m[-1]
            current_sma = sma_1m[-1]
            prev_rsi = rsi_1m[-2]
            prev_sma = sma_1m[-2]

            crossover_up = prev_rsi <= prev_sma and current_rsi > current_sma

            if not crossover_up:
                return None

            # 2. Перевірка RSI < LONG_ENTRY_MAX_RSI
            if current_rsi >= self.long_entry_max_rsi:
                return None

            # 3. Перевірка мінімальної різниці
            avg_diff = np.mean(diff_1m[-self.diff_smoothing_periods:])
            if avg_diff < self.min_diff:
                return None

            # 4. Перевірка тренду SMA вгору
            sma_trend_up = self._check_uptrend(sma_1m, self.trend_candles)
            if not sma_trend_up:
                return None

            # 5. Підтвердження на 5m таймфрейм
            confirmation_5m = self._check_long_confirmation_5m(indicators_5m)
            if not confirmation_5m:
                return None

            # Розрахунок рівня впевненості
            confidence = self._calculate_long_confidence(
                indicators_1m, indicators_5m, avg_diff
            )

            return {
                'signal': True,
                'confidence': confidence,
                'entry_price': indicators_1m['close_prices'][-1],
                'rsi_1m': current_rsi,
                'sma_1m': current_sma,
                'rsi_5m': rsi_5m[-1],
                'sma_5m': sma_5m[-1],
                'avg_diff': avg_diff
            }

        except Exception as e:
            self.logger.error(f"Помилка перевірки LONG умов для {exchange}-{pair}: {e}")
            return None

    def check_short_conditions(self, indicators_1m: Dict, indicators_5m: Dict,
                               exchange: str, pair: str) -> Optional[Dict]:
        """Перевірка умов для входу в SHORT позицію"""
        try:
            rsi_1m = indicators_1m['rsi']
            sma_1m = indicators_1m['rsi_sma']
            diff_1m = indicators_1m['smoothed_diff']

            rsi_5m = indicators_5m['rsi']
            sma_5m = indicators_5m['rsi_sma']

            # Перевірка достатності даних
            if len(rsi_1m) < self.trend_candles + 2 or len(rsi_5m) < self.trend_candles + 2:
                return None

            # 1. Перевірка crossover зверху вниз на 1m
            current_rsi = rsi_1m[-1]
            current_sma = sma_1m[-1]
            prev_rsi = rsi_1m[-2]
            prev_sma = sma_1m[-2]

            crossover_down = prev_rsi >= prev_sma and current_rsi < current_sma

            if not crossover_down:
                return None

            # 2. Перевірка RSI > SHORT_ENTRY_MIN_RSI
            if current_rsi <= self.short_entry_min_rsi:
                return None

            # 3. Перевірка мінімальної різниці
            avg_diff = np.mean(np.abs(diff_1m[-self.diff_smoothing_periods:]))
            if avg_diff < self.min_diff:
                return None

            # 4. Перевірка тренду SMA вниз
            sma_trend_down = self._check_downtrend(sma_1m, self.trend_candles)
            if not sma_trend_down:
                return None

            # 5. Підтвердження на 5m таймфрейм
            confirmation_5m = self._check_short_confirmation_5m(indicators_5m)
            if not confirmation_5m:
                return None

            # Розрахунок рівня впевненості
            confidence = self._calculate_short_confidence(
                indicators_1m, indicators_5m, avg_diff
            )

            return {
                'signal': True,
                'confidence': confidence,
                'entry_price': indicators_1m['close_prices'][-1],
                'rsi_1m': current_rsi,
                'sma_1m': current_sma,
                'rsi_5m': rsi_5m[-1],
                'sma_5m': sma_5m[-1],
                'avg_diff': avg_diff
            }

        except Exception as e:
            self.logger.error(f"Помилка перевірки SHORT умов для {exchange}-{pair}: {e}")
            return None

    def _check_uptrend(self, values: np.ndarray, periods: int) -> bool:
        """Перевірка висхідного тренду"""
        if len(values) < periods + 1:
            return False

        recent_values = values[-periods - 1:]

        # Перевірка, що кожне наступне значення більше попереднього
        for i in range(1, len(recent_values)):
            if recent_values[i] <= recent_values[i - 1]:
                return False

        return True

    def _check_downtrend(self, values: np.ndarray, periods: int) -> bool:
        """Перевірка спадного тренду"""
        if len(values) < periods + 1:
            return False

        recent_values = values[-periods - 1:]

        # Перевірка, що кожне наступне значення менше попереднього
        for i in range(1, len(recent_values)):
            if recent_values[i] >= recent_values[i - 1]:
                return False

        return True

    def _check_long_confirmation_5m(self, indicators_5m: Dict) -> bool:
        """Підтвердження LONG сигналу на 5m таймфрейм"""
        try:
            rsi_5m = indicators_5m['rsi']
            sma_5m = indicators_5m['rsi_sma']

            if len(rsi_5m) < 3 or len(sma_5m) < 3:
                return False

            # RSI >= SMA на 5m
            current_condition = rsi_5m[-1] >= sma_5m[-1]

            # Перевірка відсутності різкого падіння SMA
            sma_change = (sma_5m[-1] - sma_5m[-3]) / sma_5m[-3] * 100
            no_sharp_decline = sma_change > -2.0  # Не більше 2% падіння

            return current_condition and no_sharp_decline

        except Exception as e:
            self.logger.error(f"Помилка підтвердження LONG на 5m: {e}")
            return False

    def _check_short_confirmation_5m(self, indicators_5m: Dict) -> bool:
        """Підтвердження SHORT сигналу на 5m таймфрейм"""
        try:
            rsi_5m = indicators_5m['rsi']
            sma_5m = indicators_5m['rsi_sma']

            if len(rsi_5m) < 3 or len(sma_5m) < 3:
                return False

            # RSI <= SMA на 5m
            current_condition = rsi_5m[-1] <= sma_5m[-1]

            # Перевірка відсутності різкого зростання SMA
            sma_change = (sma_5m[-1] - sma_5m[-3]) / sma_5m[-3] * 100
            no_sharp_rise = sma_change < 2.0  # Не більше 2% зростання

            return current_condition and no_sharp_rise

        except Exception as e:
            self.logger.error(f"Помилка підтвердження SHORT на 5m: {e}")
            return False

    def _calculate_long_confidence(self, indicators_1m: Dict, indicators_5m: Dict,
                                   avg_diff: float) -> float:
        """Розрахунок рівня впевненості для LONG сигналу"""
        try:
            confidence = 0.5  # Базовий рівень

            # Бонус за силу різниці
            diff_bonus = min(avg_diff / 10.0, 0.3)
            confidence += diff_bonus

            # Бонус за RSI рівень (чим нижче, тим краще для LONG)
            rsi_current = indicators_1m['rsi'][-1]
            if rsi_current < 30:
                confidence += 0.2
            elif rsi_current < 35:
                confidence += 0.1

            # Бонус за підтвердження на 5m
            rsi_5m = indicators_5m['rsi'][-1]
            sma_5m = indicators_5m['rsi_sma'][-1]
            if rsi_5m > sma_5m:
                confidence += 0.1

            # Обмеження в межах [0, 1]
            return min(max(confidence, 0.0), 1.0)

        except Exception:
            return 0.5

    def _calculate_short_confidence(self, indicators_1m: Dict, indicators_5m: Dict,
                                    avg_diff: float) -> float:
        """Розрахунок рівня впевненості для SHORT сигналу"""
        try:
            confidence = 0.5  # Базовий рівень

            # Бонус за силу різниці
            diff_bonus = min(avg_diff / 10.0, 0.3)
            confidence += diff_bonus

            # Бонус за RSI рівень (чим вище, тим краще для SHORT)
            rsi_current = indicators_1m['rsi'][-1]
            if rsi_current > 70:
                confidence += 0.2
            elif rsi_current > 65:
                confidence += 0.1

            # Бонус за підтвердження на 5m
            rsi_5m = indicators_5m['rsi'][-1]
            sma_5m = indicators_5m['rsi_sma'][-1]
            if rsi_5m < sma_5m:
                confidence += 0.1

            # Обмеження в межах [0, 1]
            return min(max(confidence, 0.0), 1.0)

        except Exception:
            return 0.5

    def check_exit_conditions(self, indicators_1m: Dict, indicators_5m: Dict,
                              position_direction: str) -> Dict:
        """Перевірка умов виходу з позиції"""
        try:
            exit_signals = {
                'immediate_exit': False,
                'normal_exit': False,
                'reason': None
            }

            rsi_1m = indicators_1m['rsi'][-1]
            sma_1m = indicators_1m['rsi_sma'][-1]
            rsi_5m = indicators_5m['rsi'][-1]
            sma_5m = indicators_5m['rsi_sma'][-1]

            if position_direction.upper() == 'LONG':
                # Негайний вихід з LONG
                # 1. Crossover down на 5m
                if len(indicators_5m['rsi']) >= 2 and len(indicators_5m['rsi_sma']) >= 2:
                    prev_rsi_5m = indicators_5m['rsi'][-2]
                    prev_sma_5m = indicators_5m['rsi_sma'][-2]

                    crossover_down_5m = (prev_rsi_5m >= prev_sma_5m and rsi_5m < sma_5m)
                    if crossover_down_5m:
                        exit_signals['immediate_exit'] = True
                        exit_signals['reason'] = 'Crossover down на 5m'
                        return exit_signals

                # 2. RSI > overbought на 5m
                if rsi_5m > self.overbought_level:
                    exit_signals['immediate_exit'] = True
                    exit_signals['reason'] = f'RSI > {self.overbought_level} на 5m'
                    return exit_signals

                # Звичайний вихід з LONG
                # Crossover down на 1m
                if len(indicators_1m['rsi']) >= 2 and len(indicators_1m['rsi_sma']) >= 2:
                    prev_rsi_1m = indicators_1m['rsi'][-2]
                    prev_sma_1m = indicators_1m['rsi_sma'][-2]

                    crossover_down_1m = (prev_rsi_1m >= prev_sma_1m and rsi_1m < sma_1m)
                    if crossover_down_1m:
                        exit_signals['normal_exit'] = True
                        exit_signals['reason'] = 'Crossover down на 1m'

            elif position_direction.upper() == 'SHORT':
                # Негайний вихід з SHORT
                # 1. Crossover up на 5m
                if len(indicators_5m['rsi']) >= 2 and len(indicators_5m['rsi_sma']) >= 2:
                    prev_rsi_5m = indicators_5m['rsi'][-2]
                    prev_sma_5m = indicators_5m['rsi_sma'][-2]

                    crossover_up_5m = (prev_rsi_5m <= prev_sma_5m and rsi_5m > sma_5m)
                    if crossover_up_5m:
                        exit_signals['immediate_exit'] = True
                        exit_signals['reason'] = 'Crossover up на 5m'
                        return exit_signals

                # 2. RSI < oversold на 5m
                if rsi_5m < self.oversold_level:
                    exit_signals['immediate_exit'] = True
                    exit_signals['reason'] = f'RSI < {self.oversold_level} на 5m'
                    return exit_signals

                # Звичайний вихід з SHORT
                # Crossover up на 1m
                if len(indicators_1m['rsi']) >= 2 and len(indicators_1m['rsi_sma']) >= 2:
                    prev_rsi_1m = indicators_1m['rsi'][-2]
                    prev_sma_1m = indicators_1m['rsi_sma'][-2]

                    crossover_up_1m = (prev_rsi_1m <= prev_sma_1m and rsi_1m > sma_1m)
                    if crossover_up_1m:
                        exit_signals['normal_exit'] = True
                        exit_signals['reason'] = 'Crossover up на 1m'

            return exit_signals

        except Exception as e:
            self.logger.error(f"Помилка перевірки умов виходу: {e}")
            return {'immediate_exit': False, 'normal_exit': False, 'reason': None}

    def get_strategy_info(self) -> Dict:
        """Отримання інформації про стратегію"""
        return {
            'name': 'RSI-SMA Strategy',
            'version': '1.0',
            'parameters': {
                'rsi_period': self.rsi_period,
                'rsi_sma_period': self.rsi_sma_period,
                'min_diff': self.min_diff,
                'trend_candles': self.trend_candles,
                'long_entry_max_rsi': self.long_entry_max_rsi,
                'short_entry_min_rsi': self.short_entry_min_rsi,
                'overbought_level': self.overbought_level,
                'oversold_level': self.oversold_level,
                'diff_smoothing_periods': self.diff_smoothing_periods
            }
        }