#!/usr/bin/env python3
"""
Покращений аналізатор торгових сигналів з реальними ринковими даними
Включає додаткові фільтри, динамічні параметри та покращену логіку виходу
"""

import ccxt
import pandas as pd
import asyncio
import csv
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
import logging
from typing import Dict, List, Optional, Tuple
import argparse
import os
import numpy as np

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedSignalAnalyzer:
    """Покращений аналізатор торгових сигналів з додатковими фільтрами"""

    def __init__(self):
        self.exchange = self._init_exchange()
        self.results = []

    def _init_exchange(self):
        """Ініціалізація біржі"""
        try:
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'sandbox': False
            })
            return exchange
        except Exception as e:
            logger.error(f"Помилка ініціалізації біржі: {e}")
            return None

    def read_signals_csv(self, filename: str) -> List[Dict]:
        """Читання CSV файлу з сигналами"""
        signals = []

        try:
            encodings = ['utf-8', 'cp1251', 'windows-1251']
            df = None

            for encoding in encodings:
                try:
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(filename, encoding=encoding, sep=sep)
                            if len(df.columns) > 3:
                                logger.info(f"Файл прочитано з кодуванням {encoding} та роздільником '{sep}'")
                                break
                        except:
                            continue
                    if df is not None and len(df.columns) > 3:
                        break
                except Exception as e:
                    continue

            if df is None or len(df.columns) <= 3:
                raise ValueError("Не вдалося правильно прочитати файл")

            logger.info(f"Знайдено колонки: {list(df.columns)}")

            # Мапінг колонок
            column_mapping = {
                'Time': ['Time', 'Час', 'DateTime', 'Timestamp'],
                'Pair': ['Pair', 'Symbol', 'Пара', 'Символ'],
                'Exchange': ['Exchange', 'Біржа'],
                'Direction': ['Direction', 'Напрямок', 'Type'],
                'RSI': ['RSI'],
                'SMA': ['SMA', 'RSI_SMA'],
                'Comment': ['Comment', 'Коментар', 'Note']
            }

            mapped_columns = {}
            for target_col, possible_names in column_mapping.items():
                for col in df.columns:
                    if any(name.lower() in col.lower() for name in possible_names):
                        mapped_columns[target_col] = col
                        break

            logger.info(f"Мапінг колонок: {mapped_columns}")

            # Обробляємо кожен рядок
            for idx, row in df.iterrows():
                try:
                    signal = {
                        'time': row.get(mapped_columns.get('Time', ''), '').strip(),
                        'pair': row.get(mapped_columns.get('Pair', ''), '').strip().replace('/', ''),
                        'exchange': row.get(mapped_columns.get('Exchange', ''), 'BYBIT').strip(),
                        'direction': row.get(mapped_columns.get('Direction', ''), '').strip(),
                        'rsi': self._safe_float(row.get(mapped_columns.get('RSI', ''), 0)),
                        'sma': self._safe_float(row.get(mapped_columns.get('SMA', ''), 0)),
                        'comment': row.get(mapped_columns.get('Comment', ''), '').strip()
                    }

                    if (signal['pair'] and signal['direction'] and
                            signal['rsi'] > 0 and signal['time']):
                        signals.append(signal)

                except Exception as e:
                    logger.warning(f"Помилка обробки рядка {idx}: {e}")
                    continue

            logger.info(f"Успішно завантажено {len(signals)} сигналів")
            return signals

        except Exception as e:
            logger.error(f"Помилка читання файлу {filename}: {e}")
            return []

    def _safe_float(self, value) -> float:
        """Безпечне перетворення в float"""
        try:
            if isinstance(value, str):
                value = value.replace('Сер', '').replace(',', '.')
            return float(value)
        except:
            return 0.0

    def parse_signal_time(self, time_str: str) -> datetime:
        """Парсинг часу сигналу"""
        formats = [
            '%d.%m.%Y %H:%M',
            '%Y-%m-%d %H:%M',
            '%d/%m/%Y %H:%M',
            '%d.%m.%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt)
            except ValueError:
                continue

        logger.warning(f"Не вдалося розпарсити час: {time_str}")
        return datetime.now()

    async def fetch_market_data(self, symbol: str, timeframe: str,
                                start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Завантаження ринкових даних з біржі"""
        try:
            if not self.exchange:
                raise Exception("Біржа не ініціалізована")

            api_symbol = f"{symbol}/USDT" if not symbol.endswith('USDT') else f"{symbol[:-4]}/USDT"
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)

            all_data = []
            current_since = since

            while current_since < until:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        api_symbol,
                        timeframe,
                        since=current_since,
                        limit=1000
                    )

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1

                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Помилка завантаження даних для {api_symbol}: {e}")
                    break

            if not all_data:
                return pd.DataFrame()

            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            # Розрахунок всіх індикаторів
            df = self.calculate_enhanced_indicators(df)

            logger.info(f"Завантажено {len(df)} свічок для {api_symbol}")
            return df

        except Exception as e:
            logger.error(f"Помилка завантаження даних для {symbol}: {e}")
            return pd.DataFrame()

    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок розширеного набору технічних індикаторів"""
        if len(df) < 50:
            return df

        try:
            # Основні індикатори
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            df['rsi'] = rsi_indicator.rsi()
            df['rsi_sma'] = df['rsi'].rolling(window=14).mean()

            # Додаткові індикатори для фільтрації

            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Stochastic
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # Bollinger Bands
            bb = BollingerBands(close=df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

            # Moving Averages
            df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()

            # Average True Range (волатильність)
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
            df['atr'] = atr.average_true_range()
            df['atr_pct'] = (df['atr'] / df['close']) * 100

            # Volume indicators - замінюємо VolumeSMAIndicator на простий rolling mean
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Додаткові фільтри
            df['price_vs_ema20'] = (df['close'] / df['ema_20'] - 1) * 100
            df['price_vs_ema50'] = (df['close'] / df['ema_50'] - 1) * 100

            # Momentum
            df['price_change_5'] = df['close'].pct_change(5) * 100  # 5-періодна зміна ціни

        except Exception as e:
            logger.warning(f"Помилка розрахунку індикаторів: {e}")

        return df

    def apply_signal_filters(self, df: pd.DataFrame, signal_idx: int,
                             direction: str, config: Dict) -> Tuple[bool, str]:
        """Застосування додаткових фільтрів для покращення якості сигналів"""
        if signal_idx >= len(df):
            return False, "Index out of range"

        candle = df.iloc[signal_idx]
        is_long = direction.lower() == 'long'

        filters_passed = []
        filters_failed = []

        # 1. Фільтр тренду (EMA)
        if not pd.isna(candle.get('ema_20', np.nan)) and not pd.isna(candle.get('ema_50', np.nan)):
            ema_trend_bullish = candle['ema_20'] > candle['ema_50']

            if config.get('use_trend_filter', True):
                if is_long and ema_trend_bullish:
                    filters_passed.append("Trend↗")
                elif not is_long and not ema_trend_bullish:
                    filters_passed.append("Trend↘")
                else:
                    filters_failed.append(f"Trend{'↗' if ema_trend_bullish else '↘'}")

        # 2. Фільтр волатільності (ATR)
        atr_pct = candle.get('atr_pct', 0)
        min_volatility = config.get('min_volatility', 0.5)  # Мінімум 0.5% волатільності
        max_volatility = config.get('max_volatility', 8.0)  # Максимум 8% волатільності

        if config.get('use_volatility_filter', True):
            if min_volatility <= atr_pct <= max_volatility:
                filters_passed.append(f"Vol:{atr_pct:.1f}%")
            else:
                filters_failed.append(f"Vol:{atr_pct:.1f}%")

        # 3. Фільтр об'єму
        volume_ratio = candle.get('volume_ratio', 1)
        min_volume_ratio = config.get('min_volume_ratio', 0.8)  # Мінімум 80% від середнього об'єму

        if config.get('use_volume_filter', True):
            if volume_ratio >= min_volume_ratio:
                filters_passed.append(f"Vol:{volume_ratio:.1f}x")
            else:
                filters_failed.append(f"Vol:{volume_ratio:.1f}x")

        # 4. Фільтр MACD підтвердження
        macd_diff = candle.get('macd_diff', 0)
        if config.get('use_macd_filter', True) and not pd.isna(macd_diff):
            macd_bullish = macd_diff > 0
            if (is_long and macd_bullish) or (not is_long and not macd_bullish):
                filters_passed.append("MACD✓")
            else:
                filters_failed.append("MACD✗")

        # 5. Фільтр Stochastic
        stoch_k = candle.get('stoch_k', 50)
        if config.get('use_stochastic_filter', True) and not pd.isna(stoch_k):
            if is_long and stoch_k < 30:  # Stochastic в oversold для лонга
                filters_passed.append("Stoch✓")
            elif not is_long and stoch_k > 70:  # Stochastic в overbought для шорта
                filters_passed.append("Stoch✓")
            else:
                filters_failed.append("Stoch✗")

        # 6. Фільтр Bollinger Bands
        if not pd.isna(candle.get('bb_lower', np.nan)) and not pd.isna(candle.get('bb_upper', np.nan)):
            close_price = candle['close']
            bb_position = ""

            if close_price <= candle['bb_lower']:
                bb_position = "Lower"
                if is_long:
                    filters_passed.append("BB✓")
                else:
                    filters_failed.append("BB✗")
            elif close_price >= candle['bb_upper']:
                bb_position = "Upper"
                if not is_long:
                    filters_passed.append("BB✓")
                else:
                    filters_failed.append("BB✗")

        # 7. Фільтр часу (уникнення низьколіквідних періодів)
        # Потрібно отримати час з даних свічки, а не з сигналу
        signal_hour = candle.get('datetime', datetime.now()).hour if hasattr(candle.get('datetime', None), 'hour') else 12
        low_liquidity_hours = config.get('avoid_hours', [23, 0, 1, 2, 3, 4, 5])  # UTC години з низькою ліквідністю

        if config.get('use_time_filter', True):
            if signal_hour not in low_liquidity_hours:
                filters_passed.append("Time✓")
            else:
                filters_failed.append("Time✗")

        # Визначаємо чи пройшов сигнал фільтрацію
        min_filters = config.get('min_filters_required', 3)
        passed = len(filters_passed) >= min_filters

        filter_summary = f"Pass:{','.join(filters_passed)} Fail:{','.join(filters_failed)}"

        return passed, filter_summary

    def calculate_dynamic_exits(self, df: pd.DataFrame, entry_idx: int,
                                direction: str, config: Dict) -> Dict:
        """Розрахунок динамічних рівнів виходу на основі волатільності та ринкових умов"""
        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        atr_pct = entry_candle.get('atr_pct', 2.0)

        is_long = direction.lower() == 'long'

        # Базові параметри
        base_stop_loss = config.get('stop_loss', 0.03)
        base_take_profit = config.get('take_profit', 0.05)

        # Динамічне коригування на основі волатільності
        volatility_multiplier = min(max(atr_pct / 2.0, 0.5), 2.5)  # 0.5x до 2.5x від базових значень

        dynamic_stop_loss = base_stop_loss * volatility_multiplier
        dynamic_take_profit = base_take_profit * volatility_multiplier

        # Коригування на основі RSI сили
        rsi = entry_candle.get('rsi', 50)
        if is_long:
            # Для лонга: чим нижче RSI, тим більший потенціал росту
            rsi_strength = max(0.5, (40 - rsi) / 20) if rsi < 40 else 0.5
        else:
            # Для шорта: чим вище RSI, тим більший потенціал падіння
            rsi_strength = max(0.5, (rsi - 60) / 20) if rsi > 60 else 0.5

        # Застосовуємо RSI коригування до take profit
        dynamic_take_profit *= (1 + rsi_strength)

        # Розрахунок фінальних рівнів
        if is_long:
            stop_loss_price = entry_price * (1 - dynamic_stop_loss)
            take_profit_price = entry_price * (1 + dynamic_take_profit)
        else:
            stop_loss_price = entry_price * (1 + dynamic_stop_loss)
            take_profit_price = entry_price * (1 - dynamic_take_profit)

        return {
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'dynamic_stop_loss_pct': dynamic_stop_loss * 100,
            'dynamic_take_profit_pct': dynamic_take_profit * 100,
            'volatility_multiplier': volatility_multiplier,
            'rsi_strength': rsi_strength
        }

    def check_enhanced_exit_conditions(self, df: pd.DataFrame, current_idx: int,
                                       direction: str, config: Dict) -> Tuple[bool, str]:
        """Покращена логіка виходу з додатковими умовами"""
        if current_idx < 1:
            return False, ""

        current_candle = df.iloc[current_idx]
        prev_candle = df.iloc[current_idx - 1]

        current_rsi = current_candle.get('rsi', 0)
        current_rsi_sma = current_candle.get('rsi_sma', 0)
        prev_rsi = prev_candle.get('rsi', 0)
        prev_rsi_sma = prev_candle.get('rsi_sma', 0)

        if pd.isna(current_rsi) or pd.isna(current_rsi_sma) or pd.isna(prev_rsi) or pd.isna(prev_rsi_sma):
            return False, ""

        is_long = direction.lower() == 'long'

        # Базові зони виходу
        long_exit_zone = config.get('long_exit_zone', 60)
        short_exit_zone = config.get('short_exit_zone', 40)
        long_extreme_exit = config.get('long_extreme_exit', 70)
        short_extreme_exit = config.get('short_extreme_exit', 30)

        # 1. Екстремальні зони (швидкий вихід)
        if is_long and current_rsi >= long_extreme_exit:
            return True, f'RSI Extreme Exit ({current_rsi:.1f}>={long_extreme_exit})'
        elif not is_long and current_rsi <= short_extreme_exit:
            return True, f'RSI Extreme Exit ({current_rsi:.1f}<={short_extreme_exit})'

        # 2. Перетини RSI та SMA в зонах виходу
        if is_long and current_rsi >= long_exit_zone:
            if prev_rsi > prev_rsi_sma and current_rsi <= current_rsi_sma:
                return True, f'RSI Cross Down ({current_rsi:.1f}↓{current_rsi_sma:.1f})'
        elif not is_long and current_rsi <= short_exit_zone:
            if prev_rsi < prev_rsi_sma and current_rsi >= current_rsi_sma:
                return True, f'RSI Cross Up ({current_rsi:.1f}↑{current_rsi_sma:.1f})'

        # 3. MACD дивергенція (додатковий фільтр)
        if config.get('use_macd_exit', True):
            macd_diff = current_candle.get('macd_diff', 0)
            prev_macd_diff = prev_candle.get('macd_diff', 0)

            if not pd.isna(macd_diff) and not pd.isna(prev_macd_diff):
                if is_long and prev_macd_diff > 0 and macd_diff <= 0 and current_rsi > 50:
                    return True, f'MACD Bear Cross ({current_rsi:.1f})'
                elif not is_long and prev_macd_diff < 0 and macd_diff >= 0 and current_rsi < 50:
                    return True, f'MACD Bull Cross ({current_rsi:.1f})'

        # 4. Stochastic підтвердження виходу
        if config.get('use_stochastic_exit', True):
            stoch_k = current_candle.get('stoch_k', 50)
            prev_stoch_k = prev_candle.get('stoch_k', 50)

            if not pd.isna(stoch_k) and not pd.isna(prev_stoch_k):
                if is_long and prev_stoch_k > 80 and stoch_k <= 80:
                    return True, f'Stoch Exit ({stoch_k:.1f}↓80)'
                elif not is_long and prev_stoch_k < 20 and stoch_k >= 20:
                    return True, f'Stoch Exit ({stoch_k:.1f}↑20)'

        return False, ""

    def find_enhanced_entry_exit_points(self, df: pd.DataFrame, signal_time: datetime,
                                        direction: str, entry_rsi: float, config: Dict) -> Dict:
        """Покращена логіка пошуку точок входу та виходу"""

        # Знаходимо найближчу свічку до часу сигналу
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        entry_idx = df['time_diff'].idxmin()

        if pd.isna(entry_idx) or entry_idx >= len(df):
            return None

        # Застосовуємо фільтри перед входом
        filter_passed, filter_info = self.apply_signal_filters(df, entry_idx, direction, config)

        if not filter_passed and config.get('use_filters', True):
            return {
                'filtered_out': True,
                'filter_info': filter_info,
                'pnl_percent': 0,
                'status': 'Filtered'
            }

        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']

        is_long = direction.lower() == 'long'

        # Розрахунок динамічних рівнів
        dynamic_levels = self.calculate_dynamic_exits(df, entry_idx, direction, config)

        stop_loss_price = dynamic_levels['stop_loss_price']
        take_profit_price = dynamic_levels['take_profit_price']

        # Параметри виходу
        max_hold_hours = config.get('max_hold_hours', 24)
        max_exit_time = entry_time + timedelta(hours=max_hold_hours)

        # Пошук точки виходу з покращеною логікою
        exit_found = False
        trailing_stop_active = config.get('use_trailing_stop', False)
        best_price = entry_price
        trailing_stop_price = stop_loss_price

        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i]

            # Оновлюємо найкращу ціну для trailing stop
            if trailing_stop_active:
                if is_long and candle['high'] > best_price:
                    best_price = candle['high']
                    # Піднімаємо trailing stop на 75% від прибутку
                    trailing_distance = dynamic_levels['dynamic_stop_loss_pct'] / 100
                    trailing_stop_price = best_price * (1 - trailing_distance)
                elif not is_long and candle['low'] < best_price:
                    best_price = candle['low']
                    trailing_distance = dynamic_levels['dynamic_stop_loss_pct'] / 100
                    trailing_stop_price = best_price * (1 + trailing_distance)

            # Перевіряємо чи не вийшов час
            if candle['datetime'] > max_exit_time:
                exit_price = candle['close']
                exit_time = candle['datetime']
                exit_reason = f'Time Exit ({max_hold_hours}h)'
                exit_found = True
                break

            # Перевіряємо Stop Loss (оригінальний або trailing)
            current_stop = trailing_stop_price if trailing_stop_active else stop_loss_price

            if is_long:
                if candle['low'] <= current_stop:
                    exit_price = current_stop
                    exit_time = candle['datetime']
                    exit_reason = 'Trailing Stop' if trailing_stop_active else 'Stop Loss'
                    exit_found = True
                    break
                elif candle['high'] >= take_profit_price:
                    exit_price = take_profit_price
                    exit_time = candle['datetime']
                    exit_reason = 'Take Profit'
                    exit_found = True
                    break
            else:
                if candle['high'] >= current_stop:
                    exit_price = current_stop
                    exit_time = candle['datetime']
                    exit_reason = 'Trailing Stop' if trailing_stop_active else 'Stop Loss'
                    exit_found = True
                    break
                elif candle['low'] <= take_profit_price:
                    exit_price = take_profit_price
                    exit_time = candle['datetime']
                    exit_reason = 'Take Profit'
                    exit_found = True
                    break

            # Перевіряємо покращені RSI умови виходу
            enhanced_exit, enhanced_reason = self.check_enhanced_exit_conditions(df, i, direction, config)
            if enhanced_exit:
                exit_price = candle['close']
                exit_time = candle['datetime']
                exit_reason = enhanced_reason
                exit_found = True
                break

        if not exit_found:
            last_candle = df.iloc[-1]
            exit_price = last_candle['close']
            exit_time = last_candle['datetime']
            exit_reason = 'Data End'

        # Розрахунок PnL
        if is_long:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        # Розрахунок часу утримання
        time_diff = exit_time - entry_time
        hold_time_hours = time_diff.total_seconds() / 3600

        return {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'pnl_percent': pnl_pct,
            'hold_time_hours': hold_time_hours,
            'exit_reason': exit_reason,
            'status': 'Profit' if pnl_pct > 0 else 'Loss',
            'filter_info': filter_info,
            'dynamic_stop_loss_pct': dynamic_levels['dynamic_stop_loss_pct'],
            'dynamic_take_profit_pct': dynamic_levels['dynamic_take_profit_pct'],
            'volatility_multiplier': dynamic_levels['volatility_multiplier'],
            'filtered_out': False
        }

    async def analyze_signals(self, signals: List[Dict], config: Dict) -> List[Dict]:
        """Аналіз всіх сигналів з покращеною логікою"""
        results = []
        unique_pairs = list(set(signal['pair'] for signal in signals))

        logger.info(f"Початок аналізу {len(signals)} сигналів для {len(unique_pairs)} пар")

        # Визначаємо часовий діапазон
        signal_times = [self.parse_signal_time(signal['time']) for signal in signals]
        start_time = min(signal_times) - timedelta(days=3)  # +3 дні для індикаторів
        end_time = max(signal_times) + timedelta(days=2)  # +2 дні після останнього сигналу

        logger.info(f"Завантаження даних з {start_time} до {end_time}")

        # Завантажуємо дані для всіх пар
        market_data = {}
        for i, pair in enumerate(unique_pairs):
            logger.info(f"Завантаження даних для {pair} ({i + 1}/{len(unique_pairs)})")

            df = await self.fetch_market_data(
                pair,
                config.get('timeframe', '5m'),
                start_time,
                end_time
            )

            if not df.empty:
                market_data[pair] = df
                logger.info(f"✓ {pair}: {len(df)} свічок")
            else:
                logger.warning(f"✗ {pair}: немає даних")

            await asyncio.sleep(0.2)

        # Аналізуємо кожен сигнал
        for i, signal in enumerate(signals):
            logger.info(f"Аналіз сигналу {i + 1}/{len(signals)}: {signal['pair']} {signal['direction']}")

            pair = signal['pair']
            if pair not in market_data:
                logger.warning(f"Немає ринкових даних для {pair}")
                continue

            df = market_data[pair]
            signal_time = self.parse_signal_time(signal['time'])

            trade_result = self.find_enhanced_entry_exit_points(
                df, signal_time, signal['direction'],
                signal['rsi'], config
            )

            if trade_result:
                result = {
                    **signal,
                    'signal_time': signal_time.strftime('%d.%m.%Y %H:%M'),
                    **trade_result
                }

                if not trade_result.get('filtered_out', False):
                    result.update({
                        'entry_time': trade_result['entry_time'].strftime('%d.%m.%Y %H:%M'),
                        'exit_time': trade_result['exit_time'].strftime('%d.%m.%Y %H:%M'),
                        'entry_price': round(trade_result['entry_price'], 6),
                        'exit_price': round(trade_result['exit_price'], 6),
                        'pnl_percent': round(trade_result['pnl_percent'], 2),
                        'hold_time_hours': round(trade_result['hold_time_hours'], 2)
                    })
                    logger.info(f"✓ PnL: {result['pnl_percent']:.2f}%, Час: {result['hold_time_hours']:.1f}h")
                else:
                    logger.info(f"⚠️ Сигнал відфільтровано: {trade_result.get('filter_info', '')}")

                results.append(result)
            else:
                logger.warning(f"Не вдалося проаналізувати сигнал для {pair}")

        logger.info(f"Аналіз завершено. Оброблено {len(results)} сигналів")
        return results

    def optimize_parameters(self, signals: List[Dict], market_data: Dict) -> Dict:
        """Оптимізація параметрів на основі історичних даних"""
        logger.info("Початок оптимізації параметрів...")

        # Тестові діапазони параметрів
        stop_loss_range = [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
        take_profit_range = [0.04, 0.05, 0.06, 0.07, 0.08, 0.1]
        rsi_exit_zones = [(55, 35), (60, 40), (65, 35), (60, 35)]

        best_config = None
        best_score = -float('inf')

        total_combinations = len(stop_loss_range) * len(take_profit_range) * len(rsi_exit_zones)
        current_combination = 0

        for stop_loss in stop_loss_range:
            for take_profit in take_profit_range:
                for long_exit, short_exit in rsi_exit_zones:
                    current_combination += 1

                    test_config = {
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'long_exit_zone': long_exit,
                        'short_exit_zone': short_exit,
                        'use_filters': True,
                        'use_trailing_stop': True,
                        'min_filters_required': 2
                    }

                    # Тестуємо конфігурацію
                    test_results = []
                    for signal in signals[:50]:  # Тестуємо на підвибірці для швидкості
                        pair = signal['pair']
                        if pair not in market_data:
                            continue

                        df = market_data[pair]
                        signal_time = self.parse_signal_time(signal['time'])

                        result = self.find_enhanced_entry_exit_points(
                            df, signal_time, signal['direction'], signal['rsi'], test_config
                        )

                        if result and not result.get('filtered_out', False):
                            test_results.append(result)

                    if test_results:
                        # Розрахунок метрик
                        avg_pnl = np.mean([r['pnl_percent'] for r in test_results])
                        win_rate = len([r for r in test_results if r['pnl_percent'] > 0]) / len(test_results)

                        # Комбінована оцінка (середній PnL + win rate)
                        score = avg_pnl * 0.7 + win_rate * 30

                        if score > best_score:
                            best_score = score
                            best_config = test_config.copy()
                            best_config.update({
                                'avg_pnl': avg_pnl,
                                'win_rate': win_rate,
                                'total_trades': len(test_results),
                                'score': score
                            })

                    if current_combination % 10 == 0:
                        logger.info(f"Оптимізація: {current_combination}/{total_combinations}")

        if best_config:
            logger.info(f"Найкращі параметри знайдено:")
            logger.info(f"  Stop Loss: {best_config['stop_loss'] * 100:.1f}%")
            logger.info(f"  Take Profit: {best_config['take_profit'] * 100:.1f}%")
            logger.info(f"  Long Exit Zone: {best_config['long_exit_zone']}")
            logger.info(f"  Short Exit Zone: {best_config['short_exit_zone']}")
            logger.info(f"  Середній PnL: {best_config['avg_pnl']:.2f}%")
            logger.info(f"  Win Rate: {best_config['win_rate'] * 100:.1f}%")
            logger.info(f"  Оцінка: {best_config['score']:.2f}")

        return best_config or {}

    def save_enhanced_results(self, results: List[Dict], filename: str = 'enhanced_results.csv'):
        """Збереження покращених результатів"""
        try:
            if not results:
                logger.warning("Немає результатів для збереження")
                return

            fieldnames = [
                'signal_time', 'pair', 'direction', 'rsi', 'sma',
                'entry_time', 'entry_price', 'exit_time', 'exit_price',
                'pnl_percent', 'hold_time_hours', 'exit_reason', 'status',
                'filter_info', 'dynamic_stop_loss_pct', 'dynamic_take_profit_pct',
                'volatility_multiplier', 'filtered_out', 'comment'
            ]

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    filtered_result = {k: result.get(k, '') for k in fieldnames}
                    writer.writerow(filtered_result)

            logger.info(f"Покращені результати збережено в {filename}")
            self.print_enhanced_statistics(results)

        except Exception as e:
            logger.error(f"Помилка збереження результатів: {e}")

    def print_enhanced_statistics(self, results: List[Dict]):
        """Виведення розширеної статистики"""
        if not results:
            return

        # Розділяємо на відфільтровані та торговані
        traded_results = [r for r in results if not r.get('filtered_out', False)]
        filtered_results = [r for r in results if r.get('filtered_out', False)]

        print("\n" + "=" * 80)
        print("📊 РОЗШИРЕНА СТАТИСТИКА РЕЗУЛЬТАТІВ")
        print("=" * 80)

        print(f"Всього сигналів: {len(results)}")
        print(f"Відфільтровано: {len(filtered_results)} ({len(filtered_results) / len(results) * 100:.1f}%)")
        print(f"Торговано: {len(traded_results)} ({len(traded_results) / len(results) * 100:.1f}%)")

        if not traded_results:
            print("❌ Немає торгованих результатів для аналізу")
            return

        # Основна статистика
        total_trades = len(traded_results)
        profitable_trades = len([r for r in traded_results if float(r['pnl_percent']) > 0])
        losing_trades = total_trades - profitable_trades

        total_pnl = sum(float(r['pnl_percent']) for r in traded_results)
        avg_pnl = total_pnl / total_trades
        win_rate = (profitable_trades / total_trades) * 100

        # Максимальні прибутки та збитки
        max_profit = max(float(r['pnl_percent']) for r in traded_results)
        max_loss = min(float(r['pnl_percent']) for r in traded_results)

        # Середній час утримання
        avg_hold_time = sum(float(r['hold_time_hours']) for r in traded_results) / total_trades

        print(f"\n💰 ТОРГОВА СТАТИСТИКА:")
        print(f"  Всього угод: {total_trades}")
        print(f"  Прибуткових: {profitable_trades}")
        print(f"  Збиткових: {losing_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Середній PnL: {avg_pnl:.2f}%")
        print(f"  Загальний PnL: {total_pnl:.2f}%")
        print(f"  Максимальний прибуток: {max_profit:.2f}%")
        print(f"  Максимальний збиток: {max_loss:.2f}%")
        print(f"  Середній час утримання: {avg_hold_time:.1f} годин")

        # Статистика по напрямках
        long_results = [r for r in traded_results if r['direction'].lower() == 'long']
        short_results = [r for r in traded_results if r['direction'].lower() == 'short']

        if long_results:
            long_pnl = sum(float(r['pnl_percent']) for r in long_results)
            long_win_rate = len([r for r in long_results if float(r['pnl_percent']) > 0]) / len(long_results) * 100
            print(f"\n📈 LONG ПОЗИЦІЇ ({len(long_results)} угод):")
            print(f"  Win Rate: {long_win_rate:.1f}%")
            print(f"  Середній PnL: {long_pnl / len(long_results):.2f}%")
            print(f"  Загальний PnL: {long_pnl:.2f}%")

        if short_results:
            short_pnl = sum(float(r['pnl_percent']) for r in short_results)
            short_win_rate = len([r for r in short_results if float(r['pnl_percent']) > 0]) / len(short_results) * 100
            print(f"\n📉 SHORT ПОЗИЦІЇ ({len(short_results)} угод):")
            print(f"  Win Rate: {short_win_rate:.1f}%")
            print(f"  Середній PnL: {short_pnl / len(short_results):.2f}%")
            print(f"  Загальний PnL: {short_pnl:.2f}%")

        # Розподіл по причинах виходу
        exit_reasons = {}
        for result in traded_results:
            reason = result['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\n📊 ПРИЧИНИ ВИХОДУ:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            avg_pnl_reason = np.mean([float(r['pnl_percent']) for r in traded_results if r['exit_reason'] == reason])
            print(f"  {reason}: {count} ({count / total_trades * 100:.1f}%, avg PnL: {avg_pnl_reason:.2f}%)")

        # Топ пари по прибутковості
        pair_stats = {}
        for result in traded_results:
            pair = result['pair']
            if pair not in pair_stats:
                pair_stats[pair] = {'pnl': 0, 'count': 0, 'wins': 0}

            pair_stats[pair]['pnl'] += float(result['pnl_percent'])
            pair_stats[pair]['count'] += 1
            if float(result['pnl_percent']) > 0:
                pair_stats[pair]['wins'] += 1

        print(f"\n🏆 ТОП ПАРИ ПО ПРИБУТКОВОСТІ:")
        sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        for pair, stats in sorted_pairs[:10]:
            win_rate_pair = (stats['wins'] / stats['count']) * 100
            avg_pnl_pair = stats['pnl'] / stats['count']
            print(
                f"  {pair}: {stats['pnl']:.2f}% ({stats['count']} угод, WR:{win_rate_pair:.0f}%, avg:{avg_pnl_pair:.2f}%)")

        # Аналіз волатільності
        volatility_stats = [float(r.get('volatility_multiplier', 1)) for r in traded_results if
                            'volatility_multiplier' in r]
        if volatility_stats:
            print(f"\n🎯 ВОЛАТІЛЬНІСТЬ:")
            print(f"  Середній множник волатільності: {np.mean(volatility_stats):.2f}")
            print(f"  Діапазон: {min(volatility_stats):.2f} - {max(volatility_stats):.2f}")

    def generate_recommendations(self, results: List[Dict]) -> Dict:
        """Генерація рекомендацій для покращення стратегії"""
        traded_results = [r for r in results if not r.get('filtered_out', False)]

        if not traded_results:
            return {}

        recommendations = {
            'current_performance': {
                'total_trades': len(traded_results),
                'win_rate': len([r for r in traded_results if float(r['pnl_percent']) > 0]) / len(traded_results) * 100,
                'avg_pnl': np.mean([float(r['pnl_percent']) for r in traded_results]),
                'total_pnl': sum(float(r['pnl_percent']) for r in traded_results)
            }
        }

        # Аналіз причин збитків
        losing_trades = [r for r in traded_results if float(r['pnl_percent']) < 0]

        if losing_trades:
            exit_reasons_losses = {}
            for trade in losing_trades:
                reason = trade['exit_reason']
                exit_reasons_losses[reason] = exit_reasons_losses.get(reason, 0) + 1

            most_common_loss_reason = max(exit_reasons_losses.items(), key=lambda x: x[1])
            recommendations['main_loss_reason'] = most_common_loss_reason

        # Рекомендації по покращенню
        recommendations['suggestions'] = []

        # 1. Якщо багато Stop Loss
        stop_loss_count = len([r for r in traded_results if 'Stop Loss' in r['exit_reason']])
        if stop_loss_count / len(traded_results) > 0.4:
            recommendations['suggestions'].append({
                'type': 'stop_loss_adjustment',
                'message': f"Занадто багато Stop Loss виходів ({stop_loss_count}/{len(traded_results)}). Рекомендую збільшити stop loss або додати trailing stop.",
                'current_ratio': stop_loss_count / len(traded_results)
            })

        # 2. Якщо мало Take Profit
        take_profit_count = len([r for r in traded_results if 'Take Profit' in r['exit_reason']])
        if take_profit_count / len(traded_results) < 0.2:
            recommendations['suggestions'].append({
                'type': 'take_profit_adjustment',
                'message': f"Мало Take Profit виходів ({take_profit_count}/{len(traded_results)}). Можна зменшити target або покращити trailing stop.",
                'current_ratio': take_profit_count / len(traded_results)
            })

        # 3. Аналіз часу утримання
        avg_hold_time = np.mean([float(r['hold_time_hours']) for r in traded_results])
        if avg_hold_time > 20:
            recommendations['suggestions'].append({
                'type': 'hold_time_optimization',
                'message': f"Занадто довгий час утримання ({avg_hold_time:.1f}h). Рекомендую більш агресивні умови виходу.",
                'avg_hold_time': avg_hold_time
            })

        return recommendations


async def main():
    """Головна функція з покращеними опціями"""
    parser = argparse.ArgumentParser(description='Покращений аналізатор торгових сигналів')
    parser.add_argument('input_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='enhanced_results.csv', help='Файл для збереження результатів')

    # Основні параметри
    parser.add_argument('--stop-loss', type=float, default=3.0, help='Stop Loss у відсотках (за замовчуванням 3)')
    parser.add_argument('--take-profit', type=float, default=5.0, help='Take Profit у відсотках (за замовчуванням 5)')
    parser.add_argument('--max-hold', type=int, default=24, help='Максимальний час утримання в годинах')

    # RSI параметри
    parser.add_argument('--long-exit-zone', type=int, default=60, help='RSI зона виходу для лонгу')
    parser.add_argument('--short-exit-zone', type=int, default=40, help='RSI зона виходу для шорту')
    parser.add_argument('--long-extreme-exit', type=int, default=70, help='RSI екстремальний вихід для лонгу')
    parser.add_argument('--short-extreme-exit', type=int, default=30, help='RSI екстремальний вихід для шорту')

    # Фільтри
    parser.add_argument('--use-filters', action='store_true', default=True, help='Використовувати додаткові фільтри')
    parser.add_argument('--min-filters', type=int, default=2, help='Мінімум фільтрів що мають пройти')
    parser.add_argument('--use-trailing-stop', action='store_true', default=True, help='Використовувати trailing stop')

    # Фільтри волатільності та об'єму
    parser.add_argument('--min-volatility', type=float, default=0.5, help='Мінімальна волатільність (%)')
    parser.add_argument('--max-volatility', type=float, default=8.0, help='Максимальна волатільність (%)')
    parser.add_argument('--min-volume-ratio', type=float, default=0.8, help='Мінімальний коефіцієнт об\'єму')

    # Додаткові опції
    parser.add_argument('--optimize', action='store_true', help='Оптимізувати параметри перед аналізом')
    parser.add_argument('--timeframe', default='5m', help='Таймфрейм для аналізу')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ Файл {args.input_file} не знайдено")
        return

    # Конфігурація аналізу
    config = {
        'stop_loss': args.stop_loss / 100,
        'take_profit': args.take_profit / 100,
        'long_exit_zone': args.long_exit_zone,
        'short_exit_zone': args.short_exit_zone,
        'long_extreme_exit': args.long_extreme_exit,
        'short_extreme_exit': args.short_extreme_exit,
        'max_hold_hours': args.max_hold,
        'timeframe': args.timeframe,

        # Фільтри
        'use_filters': args.use_filters,
        'min_filters_required': args.min_filters,
        'use_trailing_stop': args.use_trailing_stop,
        'use_trend_filter': True,
        'use_volatility_filter': True,
        'use_volume_filter': True,
        'use_macd_filter': True,
        'use_stochastic_filter': True,
        'use_time_filter': True,
        'use_macd_exit': True,
        'use_stochastic_exit': True,

        # Параметри фільтрів
        'min_volatility': args.min_volatility,
        'max_volatility': args.max_volatility,
        'min_volume_ratio': args.min_volume_ratio,
        'avoid_hours': [23, 0, 1, 2, 3, 4, 5]  # UTC години з низькою ліквідністю
    }

    print("🚀 ПОЧАТОК ПОКРАЩЕНОГО АНАЛІЗУ ТОРГОВИХ СИГНАЛІВ")
    print(f"📁 Вхідний файл: {args.input_file}")
    print(f"📊 Налаштування:")
    print(f"   Stop Loss: {args.stop_loss}% (динамічний)")
    print(f"   Take Profit: {args.take_profit}% (динамічний)")
    print(f"   Trailing Stop: {'Увімкнено' if args.use_trailing_stop else 'Вимкнено'}")
    print(f"   Додаткові фільтри: {'Увімкнено' if args.use_filters else 'Вимкнено'}")
    print(f"   Мінімум фільтрів: {args.min_filters}")
    print(f"   Волатільність: {args.min_volatility}% - {args.max_volatility}%")
    print(f"   Оптимізація: {'Увімкнено' if args.optimize else 'Вимкнено'}")
    print("-" * 80)

    # Створюємо аналізатор
    analyzer = EnhancedSignalAnalyzer()

    # Читаємо сигнали
    signals = analyzer.read_signals_csv(args.input_file)
    if not signals:
        print("❌ Не вдалося прочитати сигнали з файлу")
        return

    print(f"✅ Завантажено {len(signals)} сигналів")

    try:
        # Завантажуємо ринкові дані
        unique_pairs = list(set(signal['pair'] for signal in signals))
        signal_times = [analyzer.parse_signal_time(signal['time']) for signal in signals]
        start_time = min(signal_times) - timedelta(days=3)
        end_time = max(signal_times) + timedelta(days=2)

        market_data = {}
        for i, pair in enumerate(unique_pairs):
            logger.info(f"Завантаження даних для {pair} ({i + 1}/{len(unique_pairs)})")
            df = await analyzer.fetch_market_data(pair, config['timeframe'], start_time, end_time)
            if not df.empty:
                market_data[pair] = df

        # Оптимізація параметрів (якщо потрібно)
        if args.optimize:
            print("🔧 Оптимізація параметрів...")
            optimized_config = analyzer.optimize_parameters(signals, market_data)
            if optimized_config:
                config.update(optimized_config)
                print("✅ Параметри оптимізовано")

        # Аналізуємо сигнали
        results = await analyzer.analyze_signals(signals, config)

        if results:
            # Зберігаємо результати
            analyzer.save_enhanced_results(results, args.output)

            # Генеруємо рекомендації
            recommendations = analyzer.generate_recommendations(results)

            print(f"\n🎯 РЕКОМЕНДАЦІЇ ДЛЯ ПОКРАЩЕННЯ:")
            for suggestion in recommendations.get('suggestions', []):
                print(f"  • {suggestion['message']}")

            print(f"\n✅ Покращений аналіз завершено!")
            print(f"📄 Результати збережено в {args.output}")
        else:
            print("❌ Не вдалося отримати результати аналізу")

    except Exception as e:
        logger.error(f"Критична помилка: {e}")
        print(f"❌ Критична помилка: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Аналіз перервано користувачем")
    except Exception as e:
        print(f"❌ Помилка: {e}")