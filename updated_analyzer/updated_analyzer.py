#!/usr/bin/env python3
"""
Аналізатор сигналів з режимами backtest та live.
Логіка: Вхід на перетині RSI(14)/SMA_RSI(14) на 5m.
Закриття на зворотному перетині в зоні (long_exit_zone=60 для Long тощо).
Буфер для даних: 120h до, 240h після для 5m; прогресія для інших TF.
Логування відкриття/закриття в trades_log.txt.
Вивід в CSV при закритті угоди у форматі: pair, direction, rsi, signal_time, entry_time_str, exit_time_str, entry_price, exit_price, pnl_percent, hold_time, dynamic_sl, status, exit_reason, filter_info, filtered_out.
Live: WebSocket для сигналів, асинхронний моніторинг угод (заглушка торгівлі).

Нові функції:
- Stop-loss логіка з динамічним трейлінг-стопом
- Фільтрація по secondary_timeframe для визначення тренду
"""

import ccxt
import pandas as pd
import asyncio
import csv
import json
import os
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
import websockets


# Налаштування логування
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = RotatingFileHandler('enhanced_analyzer.log', maxBytes=10 * 1024 * 1024, backupCount=3,
                                       encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()

# Окремий файл для логів угод
TRADES_LOG_FILE = 'trades_log.txt'


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(series, period=14):
    return series.rolling(window=period).mean()


class SignalAnalyzer:
    def __init__(self, config: Dict):
        exchange_name = config.get('exchange', 'bybit')
        self.exchange = ccxt.__getattribute__(exchange_name)({'enableRateLimit': True, 'rateLimit': 100})
        self.cache = {}
        self.open_positions = {}
        self.banned_pairs = config.get('banned_pairs', [])

    @staticmethod
    def create_default_config():
        default_config = {
            "trading_parameters": {
                "stop_loss": 0.018,
                "take_profit": 0.035,
                "max_hold_hours": 24,
                "use_trailing_stop": True,
                "trailing_stop_activation": 0.01,  # Активація трейлінг-стопу при прибутку 1%
                "trailing_stop_distance": 0.008,  # Дистанція трейлінг-стопу 0.8%
                "max_volume_per_trade": 1000
            },
            "rsi_parameters": {
                "rsi_period": 14,
                "rsi_sma_period": 14,
                "long_enter_zone": 30,
                "short_enter_zone": 70,
                "long_exit_zone": 60,
                "short_exit_zone": 40,
                "long_extreme_exit": 75,
                "short_extreme_exit": 25
            },
            "timeframe_settings": {
                "primary_timeframe": "5m",
                "secondary_timeframes": ["15m"]
            },
            "secondary_tf_filter": {
                "short_enter_zone_mid_tf": 60,  # Зона для перетину на вторинному TF
                "long_enter_zone_mid_tf": 40,  # Зона для перетину на вторинному TF
                "sma_change_periods": 5,  # N свічок для аналізу зміни SMA
                "sma_change_threshold": 0.5,  # M - поріг зміни SMA для низхідного тренду
                "rsi_extreme_threshold": 15,  # Поріг для екстремальних зростань RSI
                "cross_lookback_periods": 10  # X свічок для пошуку перетину RSI/SMA
            },
            "data_management": {
                "buffer_hours_before": 120,
                "buffer_hours_after": 240,
                "tf_multiplier": {"5m": 1, "15m": 3}
            },
            "exchange": "bybit",
            "banned_pairs": []
        }
        return default_config

    @staticmethod
    def load_config(config_file='updated_analyzer_config.json'):
        if not os.path.exists(config_file):
            logger.info(f"Конфіг {config_file} не знайдено, створюємо стандартний")
            config = SignalAnalyzer.create_default_config()
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return config
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def read_signals_csv(self, filename: str) -> List[Dict]:
        signals = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                pair = row['Pair'].replace('/', '')
                if row.get('Status') == 'open' and pair not in self.banned_pairs:
                    signal = {
                        'pair': pair,
                        'direction': row['Direction'],
                        'time': row['Signal_Time'],
                        'rsi': float(row['RSI_5m'].replace(',', '.')),
                        'rsi_sma': float(row['RSI_SMA_5m'].replace(',', '.')),
                    }
                    logger.debug(f"Знайдено сигнал: {signal}")
                    signals.append(signal)
                else:
                    logger.debug(f"Пропущено рядок: Status={row.get('Status')}, Pair={pair}")
        logger.info(f"Завантажено {len(signals)} валідних сигналів (open, не banned)")
        return signals

    def parse_time(self, time_str: str) -> datetime:
        formats = ['%d.%m.%Y %H:%M', '%Y-%m-%d %H:%M:%S']
        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt)
            except ValueError:
                pass
        logger.warning(f"Не вдалося парсити час: '{time_str}'. Використовуємо поточний.")
        return datetime.now()

    async def fetch_data_with_buffer(self, symbol: str, signals: List[Dict], config: Dict) -> Dict[str, pd.DataFrame]:
        pair_signals = [s for s in signals if s['pair'] == symbol]
        if not pair_signals:
            return {}

        times = [self.parse_time(s['time']) for s in pair_signals]
        first_time = min(times)
        last_time = max(times)

        data_config = config['data_management']
        tf_settings = config['timeframe_settings']
        primary_tf = tf_settings['primary_timeframe']

        data = {}
        for tf in [primary_tf] + tf_settings['secondary_timeframes']:
            multiplier = data_config['tf_multiplier'].get(tf, 1)
            start = first_time - timedelta(hours=data_config['buffer_hours_before'] * multiplier)
            end = last_time + timedelta(hours=data_config['buffer_hours_after'] * multiplier)
            cache_key = f"{symbol}_{tf}_{start.timestamp()}_{end.timestamp()}"
            if cache_key in self.cache:
                data[tf] = self.cache[cache_key]
                continue
            df = await self._fetch_ohlcv(symbol, tf, start, end)
            if not df.empty:
                df = self._calculate_indicators(df, config)
                data[tf] = df
                self.cache[cache_key] = df
            await asyncio.sleep(0.1)
        return data

    async def _fetch_ohlcv(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
        api_symbol = str(symbol).replace("USDT", "/USDT")
        since = int(start.timestamp() * 1000)
        until = int(end.timestamp() * 1000)
        all_data = []
        current = since
        while current < until:
            ohlcv = self.exchange.fetch_ohlcv(api_symbol, tf, since=current, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current = ohlcv[-1][0] + 1
            await asyncio.sleep(0.1)
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        return pd.DataFrame()

    def _calculate_indicators(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        rsi_params = config['rsi_parameters']
        df['rsi'] = calculate_rsi(df['close'], rsi_params['rsi_period'])
        df['rsi_sma'] = calculate_sma(df['rsi'], rsi_params['rsi_sma_period'])
        return df

    def check_secondary_tf_trend(self, secondary_df: pd.DataFrame, signal_time: datetime, direction: str,
                                 config: Dict) -> tuple[bool, str]:
        """
        Перевіряє тренд на вторинному таймфреймі
        Повертає (is_trend_favorable, filter_reason)
        """
        if secondary_df.empty:
            return True, ""

        # Знаходимо індекс найближчої свічки до часу сигналу
        secondary_df['time_diff'] = (secondary_df['datetime'] - signal_time).abs()
        signal_idx = secondary_df['time_diff'].idxmin()

        filter_config = config['secondary_tf_filter']
        is_long = direction.lower() == 'long'

        # Для long перевіряємо чи тренд не низхідний
        if is_long:
            # Метод 1: Аналіз середньої зміни SMA за останні N свічок
            if signal_idx >= filter_config['sma_change_periods']:
                start_idx = signal_idx - filter_config['sma_change_periods'] + 1
                end_idx = signal_idx + 1
                sma_segment = secondary_df.iloc[start_idx:end_idx]['rsi_sma']

                if len(sma_segment) >= 2:
                    # Середня зміна SMA
                    sma_changes = sma_segment.diff().dropna()
                    avg_sma_change = sma_changes.mean()

                    # Перевірка екстремальних зростань RSI
                    rsi_segment = secondary_df.iloc[start_idx:end_idx]['rsi']
                    rsi_extreme_growth = (rsi_segment.diff() > filter_config['rsi_extreme_threshold']).any()

                    if avg_sma_change <= -filter_config['sma_change_threshold'] and not rsi_extreme_growth:
                        return False, f"Downtrend: avg_sma_change={avg_sma_change:.2f}, no_rsi_extreme={not rsi_extreme_growth}"

            # Метод 2: Пошук перетину RSI/SMA вниз в зоні short_enter_zone_mid_tf
            lookback_start = max(0, signal_idx - filter_config['cross_lookback_periods'])
            lookback_df = secondary_df.iloc[lookback_start:signal_idx + 1]

            for i in range(1, len(lookback_df)):
                curr = lookback_df.iloc[i]
                prev = lookback_df.iloc[i - 1]

                # Перетин RSI/SMA вниз в зоні short_enter_zone_mid_tf
                if (prev['rsi'] >= prev['rsi_sma'] and
                        curr['rsi'] < curr['rsi_sma'] and
                        curr['rsi_sma'] >= filter_config['short_enter_zone_mid_tf']):
                    return False, f"Recent bear cross at RSI_SMA={curr['rsi_sma']:.2f}"

        # Для short аналогічно, але навпаки (перевіряємо чи тренд не висхідний)
        else:
            # Метод 1: Аналіз середньої зміни SMA за останні N свічок
            if signal_idx >= filter_config['sma_change_periods']:
                start_idx = signal_idx - filter_config['sma_change_periods'] + 1
                end_idx = signal_idx + 1
                sma_segment = secondary_df.iloc[start_idx:end_idx]['rsi_sma']

                if len(sma_segment) >= 2:
                    # Середня зміна SMA
                    sma_changes = sma_segment.diff().dropna()
                    avg_sma_change = sma_changes.mean()

                    # Перевірка екстремальних падінь RSI
                    rsi_segment = secondary_df.iloc[start_idx:end_idx]['rsi']
                    rsi_extreme_decline = (rsi_segment.diff() < -filter_config['rsi_extreme_threshold']).any()

                    if avg_sma_change >= filter_config['sma_change_threshold'] and not rsi_extreme_decline:
                        return False, f"Uptrend: avg_sma_change={avg_sma_change:.2f}, no_rsi_extreme={not rsi_extreme_decline}"

            # Метод 2: Пошук перетину RSI/SMA вгору в зоні long_enter_zone_mid_tf
            lookback_start = max(0, signal_idx - filter_config['cross_lookback_periods'])
            lookback_df = secondary_df.iloc[lookback_start:signal_idx + 1]

            for i in range(1, len(lookback_df)):
                curr = lookback_df.iloc[i]
                prev = lookback_df.iloc[i - 1]

                # Перетин RSI/SMA вгору в зоні long_enter_zone_mid_tf
                if (prev['rsi'] <= prev['rsi_sma'] and
                        curr['rsi'] > curr['rsi_sma'] and
                        curr['rsi_sma'] <= filter_config['short_enter_zone_mid_tf']):
                    return False, f"Recent bull cross at RSI_SMA={curr['rsi_sma']:.2f}"

        return True, ""

    def calculate_stop_loss(self, entry_price: float, direction: str, config: Dict) -> float:
        """Розраховує початковий рівень стоп-лосу"""
        stop_loss_pct = config['trading_parameters']['stop_loss']
        is_long = direction.lower() == 'long'

        if is_long:
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)

    def update_trailing_stop(self, current_price: float, entry_price: float, current_sl: float,
                             direction: str, config: Dict) -> float:
        """Оновлює трейлінг стоп-лос"""
        if not config['trading_parameters']['use_trailing_stop']:
            return current_sl

        is_long = direction.lower() == 'long'
        activation_pct = config['trading_parameters']['trailing_stop_activation']
        distance_pct = config['trading_parameters']['trailing_stop_distance']

        # Перевіряємо чи досягнуто активації трейлінг-стопу
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= activation_pct:
                new_sl = current_price * (1 - distance_pct)
                return max(current_sl, new_sl)  # Стоп-лос може тільки підвищуватися
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= activation_pct:
                new_sl = current_price * (1 + distance_pct)
                return min(current_sl, new_sl)  # Стоп-лос може тільки знижуватися

        return current_sl

    def check_stop_loss_hit(self, candle: pd.Series, stop_loss: float, direction: str) -> bool:
        """Перевіряє чи спрацював стоп-лос"""
        is_long = direction.lower() == 'long'

        if is_long:
            return candle['low'] <= stop_loss
        else:
            return candle['high'] >= stop_loss

    def log_trade_action(self, action: str, pair: str, time: datetime, price: float, rsi: float, rsi_sma: float):
        with open(TRADES_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{action};{pair};{time.strftime('%d.%m.%Y %H:%M')};{price:.2f};{rsi:.2f};{rsi_sma:.2f}\n")
        logger.info(f"{action} для {pair} в {time}: Ціна={price:.2f}, RSI={rsi:.2f}, SMA_RSI={rsi_sma:.2f}")

    def append_to_csv(self, filename: str, result: Dict):
        fields = ['pair', 'direction', 'rsi', 'signal_time', 'entry_time_str', 'exit_time_str', 'entry_price',
                  'exit_price', 'pnl_percent', 'hold_time', 'dynamic_sl', 'status', 'exit_reason', 'filter_info',
                  'filtered_out']
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, delimiter=';', fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    async def simulate_trade(self, data: Dict[str, pd.DataFrame], signal: Dict, config: Dict, output_csv: str):
        primary_tf = config['timeframe_settings']['primary_timeframe']
        secondary_tf = config['timeframe_settings']['secondary_timeframes'][0]
        df = data.get(primary_tf)
        secondary_df = data.get(secondary_tf)

        if df is None:
            return

        signal_time = self.parse_time(signal['time'])
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        entry_idx = df['time_diff'].idxmin()
        entry_candle = df.iloc[entry_idx]
        entry_time = entry_candle['datetime']
        entry_price = entry_candle['close']
        direction = signal['direction']
        is_long = direction.lower() == 'long'
        rsi_params = config['rsi_parameters']
        max_hold = config['trading_parameters']['max_hold_hours']
        max_time = entry_time + timedelta(hours=max_hold)

        # Перевірка входу (базова логіка)
        exit_reason = 'Timeout'
        exit_price = 0
        exit_time = entry_time
        filtered_out = 0
        filter_info = ""
        status = "Filtered"
        pnl = 0
        hold_time = 0
        dynamic_sl = 0

        # Базова фільтрація входу
        entry_valid = False
        if is_long and signal['rsi_sma'] < rsi_params['long_enter_zone'] and signal['rsi'] > signal['rsi_sma']:
            entry_valid = True
        elif not is_long and signal['rsi_sma'] > rsi_params['short_enter_zone'] and signal['rsi'] < signal['rsi_sma']:
            entry_valid = True

        if not entry_valid:
            filtered_out = 1
            filter_info = "Weak Entry Signal"
            logger.info(f"Сигнал для {signal['pair'], signal['direction']} відфільтровано: {filter_info}")
        else:
            # Перевірка тренду на вторинному таймфреймі
            trend_favorable, trend_reason = self.check_secondary_tf_trend(secondary_df, signal_time, direction, config)
            if not trend_favorable:
                filtered_out = 1
                filter_info = f"Secondary TF Filter: {trend_reason}"
                logger.info(f"Сигнал для {signal['pair']} відфільтровано по вторинному TF: {trend_reason}")

        if not filtered_out:
            # Розрахунок стоп-лосу
            stop_loss = self.calculate_stop_loss(entry_price, direction, config)
            dynamic_sl = stop_loss

            self.log_trade_action('OPEN', signal['pair'], entry_time, entry_price, entry_candle['rsi'],
                                  entry_candle['rsi_sma'])

            for i in range(entry_idx + 1, len(df)):
                candle = df.iloc[i]
                prev_candle = df.iloc[i - 1]
                curr_rsi, curr_sma = candle['rsi'], candle['rsi_sma']
                prev_rsi, prev_sma = prev_candle['rsi'], prev_candle['rsi_sma']

                # Оновлення трейлінг стопу
                stop_loss = self.update_trailing_stop(candle['close'], entry_price, stop_loss, direction, config)
                dynamic_sl = stop_loss

                # Перевірка стоп-лосу
                if self.check_stop_loss_hit(candle, stop_loss, direction):
                    exit_reason = 'Stop Loss'
                    exit_price = stop_loss
                    exit_time = candle['datetime']
                    status = 'Loss'
                    break

                # Перевірка зворотного сигналу
                if is_long:
                    if prev_rsi >= prev_sma and curr_rsi < curr_sma and curr_rsi > rsi_params['long_exit_zone']:
                        exit_reason = 'Reverse Signal (Short)'
                        exit_price = candle['close']
                        exit_time = candle['datetime']
                        status = 'Profit' if exit_price > entry_price else 'Loss'
                        break
                else:
                    if prev_rsi <= prev_sma and curr_rsi > curr_sma and curr_rsi < rsi_params['short_exit_zone']:
                        exit_reason = 'Reverse Signal (Long)'
                        exit_price = candle['close']
                        exit_time = candle['datetime']
                        status = 'Profit' if exit_price < entry_price else 'Loss'
                        break

                # Перевірка таймауту
                if candle['datetime'] > max_time:
                    exit_reason = 'Timeout'
                    exit_price = candle['close']
                    exit_time = candle['datetime']
                    pnl_check = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                                (entry_price - exit_price) / entry_price * 100)
                    status = 'Profit' if pnl_check > 0 else 'Loss'
                    break

            if exit_price == 0:  # Кінець даних
                last_candle = df.iloc[-1]
                exit_price = last_candle['close']
                exit_time = last_candle['datetime']
                exit_reason = 'End of Data'
                pnl_check = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                            (entry_price - exit_price) / entry_price * 100)
                status = 'Profit' if pnl_check > 0 else 'Loss'

            pnl = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                        (entry_price - exit_price) / entry_price * 100)
            hold_time = (exit_time - entry_time).total_seconds() / 3600

            if exit_time in df['datetime'].values:
                rsi_at_exit = df.loc[df['datetime'] == exit_time, 'rsi'].values[0]
                rsi_sma_at_exit = df.loc[df['datetime'] == exit_time, 'rsi_sma'].values[0]
                self.log_trade_action('CLOSE', signal['pair'], exit_time, exit_price, rsi_at_exit, rsi_sma_at_exit)
            else:
                logger.warning(f"Exit time {exit_time} not in df, skipping CLOSE log")

        result = {
            'pair': signal['pair'],
            'direction': direction,
            'rsi': signal['rsi'],
            'signal_time': signal_time.strftime('%d.%m.%Y %H:%M'),
            'entry_time_str': entry_time.strftime('%d.%m.%Y %H:%M'),
            'exit_time_str': exit_time.strftime('%d.%m.%Y %H:%M') if not filtered_out else '0',
            'entry_price': entry_price,
            'exit_price': exit_price if not filtered_out else 0,
            'pnl_percent': pnl if not filtered_out else 0,
            'hold_time': hold_time if not filtered_out else 0,
            'dynamic_sl': dynamic_sl if not filtered_out else 0,
            'status': status,
            'exit_reason': exit_reason if not filtered_out else 'Filtered',
            'filter_info': filter_info,
            'filtered_out': filtered_out
        }
        self.append_to_csv(output_csv, result)

    async def backtest(self, input_file: str, output_csv: str, config: Dict):
        signals = self.read_signals_csv(input_file)
        unique_pairs = set(s['pair'] for s in signals)
        data_dict = {}
        for pair in unique_pairs:
            data_dict[pair] = await self.fetch_data_with_buffer(pair, signals, config)

        for signal in signals:
            data = data_dict.get(signal['pair'])
            if data:
                await self.simulate_trade(data, signal, config, output_csv)

    # Live режим з стоп-лосом
    async def monitor_position(self, pair: str, direction: str, entry_time: datetime, entry_price: float, config: Dict):
        max_volume = config['trading_parameters']['max_volume_per_trade']
        logger.info(
            f"ЗАГЛУШКА: Відкрито {direction} для {pair} в {entry_time} за {entry_price}, об'єм до {max_volume} USDT")
        self.log_trade_action('OPEN (Live)', pair, entry_time, entry_price, 0, 0)

        primary_tf = config['timeframe_settings']['primary_timeframe']
        rsi_params = config['rsi_parameters']
        max_hold = config['trading_parameters']['max_hold_hours']
        max_time = entry_time + timedelta(hours=max_hold)
        is_long = direction.lower() == 'long'

        # Початковий стоп-лос
        stop_loss = self.calculate_stop_loss(entry_price, direction, config)

        while datetime.now() < max_time:
            start = entry_time - timedelta(hours=1)
            end = datetime.now() + timedelta(minutes=5)
            df = await self._fetch_ohlcv(pair, primary_tf, start, end)
            if not df.empty:
                df = self._calculate_indicators(df, config)
                last_candle = df.iloc[-1]
                prev_candle = df.iloc[-2] if len(df) > 1 else last_candle

                # Оновлення трейлінг стопу
                stop_loss = self.update_trailing_stop(last_candle['close'], entry_price, stop_loss, direction, config)

                # Перевірка стоп-лосу
                if self.check_stop_loss_hit(last_candle, stop_loss, direction):
                    exit_price = stop_loss
                    exit_time = last_candle['datetime']
                    logger.info(f"ЗАГЛУШКА: Закрито {direction} для {pair} в {exit_time} за {exit_price} (Stop Loss)")
                    self.log_trade_action('CLOSE (Live) - SL', pair, exit_time, exit_price, last_candle['rsi'],
                                          last_candle['rsi_sma'])

                    pnl = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                                (entry_price - exit_price) / entry_price * 100)
                    result = {
                        'pair': pair, 'direction': direction, 'rsi': 0,
                        'signal_time': entry_time.strftime('%d.%m.%Y %H:%M'),
                        'entry_time_str': entry_time.strftime('%d.%m.%Y %H:%M'),
                        'exit_time_str': exit_time.strftime('%d.%m.%Y %H:%M'),
                        'entry_price': entry_price, 'exit_price': exit_price, 'pnl_percent': pnl,
                        'hold_time': (exit_time - entry_time).total_seconds() / 3600,
                        'dynamic_sl': stop_loss, 'status': 'Loss', 'exit_reason': 'Stop Loss', 'filter_info': '',
                        'filtered_out': 0
                    }
                    self.append_to_csv('live_trades.csv', result)
                    break

                curr_rsi, curr_sma = last_candle['rsi'], last_candle['rsi_sma']
                prev_rsi, prev_sma = prev_candle['rsi'], prev_candle['rsi_sma']

                close = False
                if is_long and prev_rsi >= prev_sma and curr_rsi < curr_sma and curr_rsi > rsi_params['long_exit_zone']:
                    close = True
                elif not is_long and prev_rsi <= prev_sma and curr_rsi > curr_sma and curr_rsi < rsi_params[
                    'short_exit_zone']:
                    close = True

                if close:
                    exit_price = last_candle['close']
                    exit_time = last_candle['datetime']
                    logger.info(f"ЗАГЛУШКА: Закрито {direction} для {pair} в {exit_time} за {exit_price}")
                    self.log_trade_action('CLOSE (Live)', pair, exit_time, exit_price, curr_rsi, curr_sma)
                    pnl = ((exit_price - entry_price) / entry_price * 100) if is_long else (
                                (entry_price - exit_price) / entry_price * 100)
                    result = {
                        'pair': pair, 'direction': direction, 'rsi': 0,
                        'signal_time': entry_time.strftime('%d.%m.%Y %H:%M'),
                        'entry_time_str': entry_time.strftime('%d.%m.%Y %H:%M'),
                        'exit_time_str': exit_time.strftime('%d.%m.%Y %H:%M'),
                        'entry_price': entry_price, 'exit_price': exit_price, 'pnl_percent': pnl,
                        'hold_time': (exit_time - entry_time).total_seconds() / 3600,
                        'dynamic_sl': stop_loss, 'status': 'Profit', 'exit_reason': 'Reverse Signal', 'filter_info': '',
                        'filtered_out': 0
                    }
                    self.append_to_csv('live_trades.csv', result)
                    break

            await asyncio.sleep(300)

        if datetime.now() >= max_time:
            logger.info(f"ЗАГЛУШКА: Закрито {direction} для {pair} за таймаутом")

    async def live(self, ws_url: str, config: Dict):
        async with websockets.connect(ws_url) as ws:
            while True:
                message = await ws.recv()
                signal = json.loads(message)
                pair = signal.get('pair').replace('/', '')
                if pair in self.banned_pairs:
                    logger.info(f"Пропуск пари {pair} (banned)")
                    continue

                # Перевірка тренду на вторинному таймфреймі для live режиму
                direction = signal.get('direction')
                signal_time = self.parse_time(signal.get('time'))

                # Отримуємо дані для перевірки тренду
                secondary_tf = config['timeframe_settings']['secondary_timeframes'][0]
                start = signal_time - timedelta(hours=config['data_management']['buffer_hours_before'])
                end = signal_time + timedelta(hours=1)
                secondary_df = await self._fetch_ohlcv(pair, secondary_tf, start, end)

                if not secondary_df.empty:
                    secondary_df = self._calculate_indicators(secondary_df, config)
                    trend_favorable, trend_reason = self.check_secondary_tf_trend(secondary_df, signal_time, direction,
                                                                                  config)

                    if not trend_favorable:
                        logger.info(f"Live сигнал для {pair} відфільтровано по вторинному TF: {trend_reason}")
                        continue

                ticker = self.exchange.fetch_ticker(f"{pair}/USDT")
                entry_price = ticker['last']
                task = asyncio.create_task(self.monitor_position(pair, direction, signal_time, entry_price, config))
                self.open_positions[pair] = task


async def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Signal Analyzer with Stop-Loss and Secondary Timeframe Filter')
    parser.add_argument('mode', choices=['backtest', 'live'], help='Режим: backtest або live')
    parser.add_argument('input', help='Для backtest: CSV файл; Для live: WebSocket URL')
    parser.add_argument('-o', '--output', default='results.csv', help='CSV для результатів')
    parser.add_argument('-c', '--config', default='updated_analyzer_config.json', help='Конфіг')
    args = parser.parse_args()

    config = SignalAnalyzer.load_config(args.config)
    analyzer = SignalAnalyzer(config)

    if args.mode == 'backtest':
        logger.info(f"Запуск бектестінгу з файлом: {args.input}")
        logger.info(f"Результати будуть збережені в: {args.output}")
        await analyzer.backtest(args.input, args.output, config)
        logger.info("Бектестінг завершено")
    elif args.mode == 'live':
        logger.info(f"Запуск live режиму з WebSocket: {args.input}")
        await analyzer.live(args.input, config)


if __name__ == "__main__":
    asyncio.run(main())