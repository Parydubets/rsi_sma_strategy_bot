import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

@dataclass
class Signal:
    """Структура торгового сигналу"""
    pair: str
    direction: str
    time: datetime
    rsi: float
    rsi_sma: float


@dataclass
class TradeResult:
    """Результат торгівлі"""
    pair: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_percent: float
    hold_time: float
    status: str
    exit_reason: str
    filtered_out: bool = False
    filter_info: str = ""


class DataCache:
    """Високопродуктивний кеш даних з автоматичним управлінням пам'яттю"""

    def __init__(self, max_memory_mb: int = 4096):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._access_times: Dict[str, float] = {}
        self._max_memory = max_memory_mb * 1024 * 1024
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if key in self._cache:
                self._access_times[key] = datetime.now().timestamp()
                return self._cache[key]
        return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._cleanup_if_needed()
            self._cache[key] = df
            self._access_times[key] = datetime.now().timestamp()

    def _cleanup_if_needed(self):
        """Очищення кешу при необхідності"""
        current_memory = sum(df.memory_usage(deep=True).sum() for df in self._cache.values())

        if current_memory > self._max_memory:
            # Видаляємо 30% найменш використовуваних
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
            to_remove = len(sorted_items) // 3

            for key, _ in sorted_items[:to_remove]:
                if key in self._cache:
                    del self._cache[key]
                    del self._access_times[key]


class TechnicalIndicators:
    """Розрахунок технічних індикаторів"""

    @staticmethod
    def calculate_rsi_vectorized(close: pd.Series, period: int = 14) -> pd.Series:
        """ТОЧНА копія логіки з аналізатора"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_sma(series: pd.Series, period: int = 14) -> pd.Series:
        """ТОЧНА копія логіки з аналізатора"""
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Розрахунок ATR для volatility filter"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_price_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Розрахунок волатильності ціни"""
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std() * 100


class DataProvider:
    """Постачальник ринкових даних"""

    def __init__(self, exchange_config: Dict):
        import ccxt
        exchange_name = exchange_config.get('exchange', 'bybit')
        self.exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'rateLimit': 30,
            'options': {'defaultType': 'spot', 'recvWindow': 10000}
        })
        self.cache = DataCache()
        self.semaphore = asyncio.Semaphore(20)

    async def fetch_data_batch(self, symbols: List[str], signals: List[Signal],
                               config: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Пакетне завантаження даних для всіх символів ОДРАЗУ"""

        # Групуємо сигнали по парах
        pair_signals = {}
        for signal in signals:
            if signal.pair not in pair_signals:
                pair_signals[signal.pair] = []
            pair_signals[signal.pair].append(signal)

        # Створюємо задачі для паралельного завантаження
        tasks = []
        for symbol in symbols:
            if symbol in pair_signals:
                task = self._fetch_symbol_data(symbol, pair_signals[symbol], config)
                tasks.append((symbol, task))

        # Виконуємо ВСІ завантаження паралельно
        results = {}
        completed = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for (symbol, _), result in zip(tasks, completed):
            if not isinstance(result, Exception) and result:
                results[symbol] = result

        return results

    async def _fetch_symbol_data(self, symbol: str, signals: List[Signal],
                                 config: Dict) -> Dict[str, pd.DataFrame]:
        """Завантаження даних для одного символу"""
        async with self.semaphore:

            # Визначаємо діапазон часу
            times = [s.time for s in signals]
            start_time = min(times) - timedelta(hours=config['data_management']['buffer_hours_before'])
            end_time = max(times) + timedelta(hours=config['data_management']['buffer_hours_after'])

            data = {}
            timeframes = [config['timeframe_settings']['primary_timeframe']] + \
                         config['timeframe_settings']['secondary_timeframes']

            for tf in timeframes:
                cache_key = f"{symbol}_{tf}_{int(start_time.timestamp())}_{int(end_time.timestamp())}"

                # Перевіряємо кеш
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    data[tf] = cached_data
                    continue

                # Завантажуємо нові дані
                df = await self._fetch_ohlcv(symbol, tf, start_time, end_time)
                if not df.empty:
                    # Розраховуємо індикатори
                    df = self._add_indicators(df, config)
                    data[tf] = df
                    self.cache.set(cache_key, df)

                await asyncio.sleep(0.02)

            return data

    async def _fetch_ohlcv(self, symbol: str, tf: str, start: datetime, end: datetime) -> pd.DataFrame:
        """ТОЧНА копія логіки завантаження з аналізатора"""
        api_symbol = symbol.replace("USDT", "/USDT")
        since = int(start.timestamp() * 1000)
        until = int(end.timestamp() * 1000)

        all_data = []
        current = since
        batch_size = 1000
        max_requests = 50
        request_count = 0

        while current < until and request_count < max_requests:
            try:
                ohlcv = self.exchange.fetch_ohlcv(api_symbol, tf, since=current, limit=batch_size)
                if not ohlcv:
                    break

                all_data.extend(ohlcv)
                current = ohlcv[-1][0] + 1
                request_count += 1

                await asyncio.sleep(0.02)

            except Exception as e:
                logging.error(f"Помилка завантаження {symbol} {tf}: {e}")
                break

        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        return pd.DataFrame()

    def _add_indicators(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """ТОЧНА копія розрахунку індикаторів з аналізатора + volatility"""
        rsi_params = config['rsi_parameters']

        df['rsi'] = TechnicalIndicators.calculate_rsi_vectorized(df['close'], rsi_params['rsi_period'])
        df['rsi_sma'] = TechnicalIndicators.calculate_sma(df['rsi'], rsi_params['rsi_sma_period'])

        # Додаткові індикатори для volatility filter
        if 'volatility_filter' in config and config['volatility_filter'].get('enabled', False):
            vol_config = config['volatility_filter']
            method = vol_config.get('volatility_method', 'atr_percentage')
            period = vol_config.get('volatility_period', 20)

            if method == 'atr_percentage':
                df['atr'] = TechnicalIndicators.calculate_atr(df, period)
                df['volatility'] = (df['atr'] / df['close']) * 100
            elif method == 'price_range':
                df['volatility'] = ((df['high'] - df['low']) / df['close']).rolling(window=period).mean() * 100
            elif method == 'close_variance':
                df['volatility'] = TechnicalIndicators.calculate_price_volatility(df, period)

        return df


def load_signals_from_csv(filename: str) -> List[Signal]:
    """Завантаження сигналів з CSV"""
    signals = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if row.get('Status') == 'open':
                signals.append(Signal(
                    pair=row['Pair'].replace('/', ''),
                    direction=row['Direction'],
                    time=TradingEngine.parse_time(row['Signal_Time']),
                    rsi=float(row['RSI_5m'].replace(',', '.')),
                    rsi_sma=float(row['RSI_SMA_5m'].replace(',', '.'))
                ))
    return signals


def save_results_to_csv(results: List[TradeResult], filename: str):
    """Збереження результатів в CSV"""
    fieldnames = ['pair', 'direction', 'rsi', 'entry_time', 'exit_time', 'entry_price',
                  'exit_price', 'pnl_percent', 'hold_time', 'status', 'exit_reason',
                  'filtered_out', 'filter_info']

    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'pair': result.pair,
                'direction': result.direction,
                'rsi': 0,  # Placeholder
                'entry_time': result.entry_time.strftime('%d.%m.%Y %H:%M'),
                'exit_time': result.exit_time.strftime('%d.%m.%Y %H:%M'),
                'entry_price': result.entry_price,
                'exit_price': result.exit_price,
                'pnl_percent': result.pnl_percent,
                'hold_time': result.hold_time,
                'status': result.status,
                'exit_reason': result.exit_reason,
                'filtered_out': result.filtered_out,
                'filter_info': getattr(result, 'filter_info', '')
            })


def save_optimization_results(results: List[Dict], filename: str):
    """Збереження результатів оптимізації"""
    if not results:
        return

    fieldnames = ['rank', 'config_id', 'total_trades', 'win_rate', 'avg_pnl',
                  'total_pnl', 'profit_factor', 'max_drawdown', 'filtered_out',
                  'best_trade', 'worst_trade', 'avg_hold_time', 'sharpe_ratio']

    # Додаємо всі параметри конфігурації
    all_params = set()
    for result in results:
        all_params.update(result['parameters'].keys())
    fieldnames.extend(sorted(all_params))

    try:
        save_to_file(filename, results, fieldnames, all_params)
    except Exception as e:
        print(e)
        logging.info(f"The file with name {filename} already exists. Writing to temp.csv")
        filename = "temp.csv"
        save_to_file(filename, results, fieldnames, all_params)


def save_to_file(filename, results, fieldnames, all_params):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=fieldnames)
        writer.writeheader()

        for rank, result in enumerate(results, 1):
            metrics = result['metrics']
            row = {
                'rank': rank,
                'config_id': result['config_id'],
                'total_trades': metrics['total_trades'],
                'win_rate': round(metrics['win_rate'], 2),
                'avg_pnl': round(metrics['avg_pnl'], 4),
                'total_pnl': round(metrics['total_pnl'], 2),
                'profit_factor': round(metrics['profit_factor'], 2),
                'max_drawdown': round(metrics['max_drawdown'], 2),
                'filtered_out': metrics['filtered_out'],
                'best_trade': round(metrics['best_trade'], 2),
                'worst_trade': round(metrics['worst_trade'], 2),
                'avg_hold_time': round(metrics['avg_hold_time'], 2),
                'sharpe_ratio': round(metrics['sharpe_ratio'], 3)
            }

            # Додаємо параметри
            for param in all_params:
                row[param] = result['parameters'].get(param, '')

            writer.writerow(row)


def save_to_file_top10(results, filename, fieldnames):
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for rank, result in enumerate(results, 1):
            metrics = result['metrics']
            params = result['parameters']
            config_id_num = ''.join(filter(str.isdigit, result['config_id']))  # Витягуємо цифри для random_
            strategy_name = f"random_{config_id_num}" if config_id_num else f"random_{rank * 100 + random.randint(1, 99)}"

            row = {
                'rank': rank,
                'config_id': result['config_id'],
                'total_trades': metrics['total_trades'],
                'win_rate': round(metrics['win_rate'], 2),
                'avg_pnl': round(metrics['avg_pnl'], 4),
                'total_pnl': round(metrics['total_pnl'], 2),
                'profit_factor': round(metrics['profit_factor'], 2),
                'max_drawdown': round(metrics['max_drawdown'], 2),
                'filtered_out': metrics['filtered_out'],
                'best_trade': round(metrics['best_trade'], 2),
                'worst_trade': round(metrics['worst_trade'], 2),
                'execution_time': round(result.get('execution_time', 0), 3),
                'buffer_hours_after': params.get('buffer_hours_after', 240),
                'buffer_hours_before': params.get('buffer_hours_before', 120),
                'cross_lookback_periods': params.get('cross_lookback_periods', 10),
                'long_enter_zone_mid_tf': params.get('long_enter_zone_mid_tf', 40),
                'long_exit_zone': params.get('long_exit_zone', 65),
                'long_extreme_exit': params.get('long_extreme_exit', 80),
                'max_hold_hours': params.get('max_hold_hours', 24),
                'rsi_extreme_threshold': params.get('rsi_extreme_threshold', 15),
                'secondary_timeframes': str(params.get('secondary_timeframes', ['15m'])),
                'short_enter_zone_mid_tf': params.get('short_enter_zone_mid_tf', 60),
                'short_exit_zone': params.get('short_exit_zone', 35),
                'short_extreme_exit': params.get('short_extreme_exit', 20),
                'sma_change_periods': params.get('sma_change_periods', 5),
                'sma_change_threshold': params.get('sma_change_threshold', 0.5),
                'stop_loss': params.get('stop_loss', 0.018),
                'strategy_name': strategy_name,
                'take_profit': params.get('take_profit', 0.035),
                'tf_multiplier_15m': 3,  # Фіксоване з default config
                'tf_multiplier_1h': 12,
                'tf_multiplier_30m': 6,
                'trailing_stop_activation': params.get('trailing_stop_activation', 0.01),
                'trailing_stop_distance': params.get('trailing_stop_distance', 0.008),
                'use_trailing_stop': 'TRUE' if params.get('use_trailing_stop', True) else 'FALSE'
            }

            writer.writerow(row)


def save_top10(results: List[Dict], filename: str):
    """Збереження топ-10 конфігурацій у вказаному форматі (з табуляцією)"""
    if not results:
        return

    fieldnames = [
        'rank', 'config_id', 'total_trades', 'win_rate', 'avg_pnl', 'total_pnl', 'profit_factor', 'max_drawdown', 'filtered_out',
        'best_trade', 'worst_trade', 'execution_time', 'buffer_hours_after', 'buffer_hours_before', 'cross_lookback_periods',
        'long_enter_zone_mid_tf', 'long_exit_zone', 'long_extreme_exit', 'max_hold_hours', 'rsi_extreme_threshold',
        'secondary_timeframes', 'short_enter_zone_mid_tf', 'short_exit_zone', 'short_extreme_exit', 'sma_change_periods',
        'sma_change_threshold', 'stop_loss', 'strategy_name', 'take_profit', 'tf_multiplier_15m', 'tf_multiplier_1h',
        'tf_multiplier_30m', 'trailing_stop_activation', 'trailing_stop_distance', 'use_trailing_stop'
    ]

    try:
        save_to_file_top10(results, filename, fieldnames)
    except Exception as e:
        print(e)
        logging.info(f"The file with name {filename} already exists. Writing to temp_top10.csv")
        filename = "temp_top10.csv"
        save_to_file_top10(results, filename, fieldnames)


def print_top_results(results: List[Dict], top_n: int = 10):
    """Виведення топ результатів"""
    if not results:
        return

    print(f"\n{'=' * 80}")
    print(f"ТОП-{top_n} НАЙКРАЩИХ КОНФІГУРАЦІЙ")
    print(f"{'=' * 80}")

    for i, result in enumerate(results[:top_n], 1):
        metrics = result['metrics']
        params = result['parameters']

        print(f"\n#{i} - {result['config_id']}")
        print(f"Угод: {metrics['total_trades']} | "
              f"Відфільтровано: {metrics['filtered_out']} | "
              f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Avg PnL: {metrics['avg_pnl']:.3f}% | "
              f"Total PnL: {metrics['total_pnl']:.2f}% | "
              f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max DD: {metrics['max_drawdown']:.2f}% | "
              f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"Avg Hold: {metrics['avg_hold_time']:.1f}h")

        # Ключові параметри
        key_params = []
        if 'use_trailing_stop' in params:
            ts_status = "ON" if params['use_trailing_stop'] else "OFF"
            key_params.append(f"TrailingStop={ts_status}")

        if 'volatility_enabled' in params:
            vol_status = "ON" if params['volatility_enabled'] else "OFF"
            key_params.append(f"VolFilter={vol_status}")

        key_params.extend([
            f"SL={params.get('stop_loss', 0):.3f}",
            f"TP={params.get('take_profit', 0):.3f}",
            f"MaxHold={params.get('max_hold_hours', 0)}h"
        ])

        if key_params:
            print(f"Параметри: {' | '.join(key_params[:6])}")


def save_best_config(results: List[Dict], output_file: str = None):
    """Збереження найкращої конфігурації"""
    if not results:
        return None

    best_result = results[0]

    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'best_config_{timestamp}.json'

    # Створюємо повну конфігурацію
    optimizer = ConfigOptimizer({})
    base_config = optimizer.create_default_config()

    # Застосовуємо найкращі параметри
    best_config = optimizer._apply_config_params(best_result['parameters'])

    # Додаємо метадані
    best_config['optimization_metadata'] = {
        'config_id': best_result['config_id'],
        'rank': 1,
        'performance_metrics': best_result['metrics'],
        'optimization_date': datetime.now().isoformat(),
        'optimizer_version': "enhanced_modular_with_genetic"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"Найкращу конфігурацію збережено в: {output_file}")
    return output_file
