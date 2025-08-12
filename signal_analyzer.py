#!/usr/bin/env python3
"""
Аналізатор торгових сигналів з реальними ринковими даними
Читає CSV файл з сигналами, завантажує реальні дані з біржі та розраховує результати
"""

import ccxt
import pandas as pd
import asyncio
import csv
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import logging
from typing import Dict, List, Optional, Tuple
import argparse
import os

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


class SignalAnalyzer:
    """Аналізатор торгових сигналів з реальними ринковими даними"""

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
            # Спробуємо різні кодування
            encodings = ['utf-8', 'cp1251', 'windows-1251']
            df = None

            for encoding in encodings:
                try:
                    # Спробуємо різні роздільники
                    for sep in [';', ',', '\t']:
                        try:
                            df = pd.read_csv(filename, encoding=encoding, sep=sep)
                            if len(df.columns) > 3:  # Якщо знайшли правильний формат
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

            # Знаходимо відповідні колонки
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

                    # Фільтруємо валідні сигнали
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
                # Видаляємо префікси типу "Сер" та замінюємо кому на крапку
                value = value.replace('Сер', '').replace(',', '.')
            return float(value)
        except:
            return 0.0

    def parse_signal_time(self, time_str: str) -> datetime:
        """Парсинг часу сигналу"""
        print(time_str)
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

            # Перетворення символу для API біржі
            api_symbol = f"{symbol}/USDT" if not symbol.endswith('USDT') else f"{symbol[:-4]}/USDT"

            # Розрахунок кількості свічок
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)

            all_data = []
            current_since = since

            # Завантажуємо дані порціями
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
                    current_since = ohlcv[-1][0] + 1  # Наступна мілісекунда

                    # Затримка для Rate Limit
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Помилка завантаження даних для {api_symbol}: {e}")
                    break

            if not all_data:
                return pd.DataFrame()

            # Створення DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            # Розрахунок RSI та RSI SMA
            df = self.calculate_indicators(df)

            logger.info(f"Завантажено {len(df)} свічок для {api_symbol}")
            return df

        except Exception as e:
            logger.error(f"Помилка завантаження даних для {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок технічних індикаторів"""
        if len(df) < 30:  # Мінімум для розрахунку індикаторів
            return df

        try:
            # RSI
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            df['rsi'] = rsi_indicator.rsi()

            # RSI SMA
            df['rsi_sma'] = df['rsi'].rolling(window=14).mean()

        except Exception as e:
            logger.warning(f"Помилка розрахунку індикаторів: {e}")

        return df

    def check_rsi_crossover_exit(self, df: pd.DataFrame, current_idx: int,
                                 direction: str, config: Dict) -> Tuple[bool, str]:
        """
        Перевіряє умови виходу на основі RSI та його SMA

        Для лонга:
        - Перетин RSI вниз через RSI SMA в зоні 60+ (RSI був вище SMA, став нижче)
        - Або RSI досяг зони 68+

        Для шорта:
        - Перетин RSI вгору через RSI SMA в зоні 40- (RSI був нижче SMA, став вище)
        - Або RSI досяг зони 30-
        """
        if current_idx < 1:  # Потрібна попередня свічка для перетину
            return False, ""

        current_candle = df.iloc[current_idx]
        prev_candle = df.iloc[current_idx - 1]

        current_rsi = current_candle.get('rsi', 0)
        current_rsi_sma = current_candle.get('rsi_sma', 0)
        prev_rsi = prev_candle.get('rsi', 0)
        prev_rsi_sma = prev_candle.get('rsi_sma', 0)

        # Перевіряємо чи є всі необхідні дані
        if pd.isna(current_rsi) or pd.isna(current_rsi_sma) or pd.isna(prev_rsi) or pd.isna(prev_rsi_sma):
            return False, ""

        is_long = direction.lower() == 'long'

        long_exit_zone = config.get('long_exit_zone', 60)
        short_exit_zone = config.get('short_exit_zone', 40)
        long_extreme_exit = config.get('long_extreme_exit', 68)
        short_extreme_exit = config.get('short_extreme_exit', 30)

        if is_long:
            # Для лонга: вихід при досягненні екстремальної зони
            if current_rsi >= long_extreme_exit:
                return True, f'RSI Extreme Exit ({current_rsi:.1f}>=68)'

            # Для лонга: перетин вниз в зоні 60+
            if prev_rsi >= long_exit_zone:
                # Перевіряємо перетин: RSI був вище SMA, став нижче або дорівнює
                if prev_rsi > prev_rsi_sma and current_rsi <= current_rsi_sma:
                    return True, f'RSI Crossover Down ({current_rsi:.1f} cross {current_rsi_sma:.1f})'

        else:  # Short
            # Для шорта: вихід при досягненні екстремальної зони
            if current_rsi <= short_extreme_exit:
                return True, f'RSI Extreme Exit ({current_rsi:.1f}<=30)'

            # Для шорта: перетин вгору в зоні 40-
            if prev_rsi <= short_exit_zone:
                # Перевіряємо перетин: RSI був нижче SMA, став вище або дорівнює
                if prev_rsi < prev_rsi_sma and current_rsi >= current_rsi_sma:
                    return True, f'RSI Crossover Up ({current_rsi:.1f} cross {current_rsi_sma:.1f})'

        return False, ""

    def find_entry_exit_points(self, df: pd.DataFrame, signal_time: datetime,
                               direction: str, entry_rsi: float, config: Dict) -> Dict:
        """Знаходження точок входу та виходу"""

        # Знаходимо найближчу свічку до часу сигналу
        df['time_diff'] = (df['datetime'] - signal_time).abs()
        entry_idx = df['time_diff'].idxmin()

        if pd.isna(entry_idx) or entry_idx >= len(df):
            return None

        entry_candle = df.iloc[entry_idx]
        entry_price = entry_candle['close']
        entry_time = entry_candle['datetime']

        is_long = direction.lower() == 'long'

        # Параметри виходу
        stop_loss_pct = config.get('stop_loss', 0.03)  # 3%
        take_profit_pct = config.get('take_profit', 0.05)  # 5%
        max_hold_hours = config.get('max_hold_hours', 24)  # 24 години

        # Розрахунок рівнів
        stop_loss_price = entry_price * (1 - stop_loss_pct) if is_long else entry_price * (1 + stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct) if is_long else entry_price * (1 - take_profit_pct)

        # Пошук точки виходу
        max_exit_time = entry_time + timedelta(hours=max_hold_hours)
        exit_found = False

        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i]

            # Перевіряємо чи не вийшов час
            if candle['datetime'] > max_exit_time:
                exit_price = candle['close']
                exit_time = candle['datetime']
                exit_reason = f'Time Exit ({max_hold_hours}h)'
                exit_found = True
                break

            # Перевіряємо Stop Loss та Take Profit
            if is_long:
                if candle['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_time = candle['datetime']
                    exit_reason = 'Stop Loss'
                    exit_found = True
                    break
                elif candle['high'] >= take_profit_price:
                    exit_price = take_profit_price
                    exit_time = candle['datetime']
                    exit_reason = 'Take Profit'
                    exit_found = True
                    break
            else:
                if candle['high'] >= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_time = candle['datetime']
                    exit_reason = 'Stop Loss'
                    exit_found = True
                    break
                elif candle['low'] <= take_profit_price:
                    exit_price = take_profit_price
                    exit_time = candle['datetime']
                    exit_reason = 'Take Profit'
                    exit_found = True
                    break

            # Перевіряємо RSI вихід з покращеною логікою
            rsi_exit, rsi_exit_reason = self.check_rsi_crossover_exit(df, i, direction, config)
            if rsi_exit:
                exit_price = candle['close']
                exit_time = candle['datetime']
                exit_reason = rsi_exit_reason
                exit_found = True
                break

        if not exit_found:
            # Закриваємо на останній доступній ціні
            last_candle = df.iloc[-1]
            exit_price = last_candle['close']
            exit_time = last_candle['datetime']
            exit_reason = 'Data End'

        # Розрахунок PnL
        if is_long:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        # Розрахунок часу між входом і виходом
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
            'status': 'Profit' if pnl_pct > 0 else 'Loss'
        }

    async def analyze_signals(self, signals: List[Dict], config: Dict) -> List[Dict]:
        """Аналіз всіх сигналів"""
        results = []
        unique_pairs = list(set(signal['pair'] for signal in signals))

        logger.info(f"Початок аналізу {len(signals)} сигналів для {len(unique_pairs)} пар")

        # Визначаємо часовий діапазон для завантаження даних
        signal_times = [self.parse_signal_time(signal['time']) for signal in signals]
        start_time = min(signal_times) - timedelta(days=2)  # +2 дні для індикаторів
        end_time = max(signal_times) + timedelta(days=1)  # +1 день після останнього сигналу

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

            # Затримка між запитами
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

            trade_result = self.find_entry_exit_points(
                df, signal_time, signal['direction'],
                signal['rsi'], config
            )

            if trade_result:
                result = {
                    **signal,
                    'signal_time': signal_time.strftime('%d.%m.%Y %H:%M'),
                    'entry_time': trade_result['entry_time'].strftime('%d.%m.%Y %H:%M'),
                    'exit_time': trade_result['exit_time'].strftime('%d.%m.%Y %H:%M'),
                    'entry_price': round(trade_result['entry_price'], 4),
                    'exit_price': round(trade_result['exit_price'], 4),
                    'pnl_percent': round(trade_result['pnl_percent'], 2),
                    'hold_time_hours': round(trade_result['hold_time_hours'], 2),
                    'exit_reason': trade_result['exit_reason'],
                    'status': trade_result['status']
                }
                results.append(result)

                logger.info(f"✓ PnL: {result['pnl_percent']:.2f}%, Час: {result['hold_time_hours']:.1f}h")
            else:
                logger.warning(f"Не вдалося проаналізувати сигнал для {pair}")

        logger.info(f"Аналіз завершено. Оброблено {len(results)} сигналів")
        return results

    def save_results(self, results: List[Dict], filename: str = 'results.csv'):
        """Збереження результатів у CSV файл"""
        try:
            if not results:
                logger.warning("Немає результатів для збереження")
                return

            # Визначаємо колонки для збереження
            fieldnames = [
                'signal_time', 'pair', 'direction', 'rsi', 'sma',
                'entry_time', 'entry_price', 'exit_time', 'exit_price',
                'pnl_percent', 'hold_time_hours', 'exit_reason', 'status', 'comment'
            ]

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile,delimiter=';', fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    # Фільтруємо тільки потрібні поля
                    filtered_result = {k: result.get(k, '') for k in fieldnames}
                    writer.writerow(filtered_result)

            logger.info(f"Результати збережено в {filename}")

            # Виводимо статистику
            self.print_statistics(results)

        except Exception as e:
            logger.error(f"Помилка збереження результатів: {e}")

    def print_statistics(self, results: List[Dict]):
        """Виведення статистики результатів"""
        if not results:
            return

        total_trades = len(results)
        profitable_trades = len([r for r in results if float(r['pnl_percent']) > 0])
        losing_trades = total_trades - profitable_trades

        total_pnl = sum(float(r['pnl_percent']) for r in results)
        win_rate = (profitable_trades / total_trades) * 100

        avg_hold_time = sum(float(r['hold_time_hours']) for r in results) / total_trades

        # Розподіл по причинах виходу
        exit_reasons = {}
        for result in results:
            reason = result['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА РЕЗУЛЬТАТІВ")
        print("=" * 60)
        print(f"Всього угод: {total_trades}")
        print(f"Прибуткових: {profitable_trades}")
        print(f"Збиткових: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Загальний PnL: {total_pnl:.2f}%")
        print(f"Середній час утримання: {avg_hold_time:.1f} годин")

        print(f"\n📈 РОЗПОДІЛ ПО ПРИЧИНАХ ВИХОДУ:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({count / total_trades * 100:.1f}%)")

        # Топ пари по прибутковості
        pair_pnl = {}
        for result in results:
            pair = result['pair']
            pair_pnl[pair] = pair_pnl.get(pair, 0) + float(result['pnl_percent'])

        print(f"\n🏆 ТОП ПАРИ ПО ПРИБУТКОВОСТІ:")
        for pair, pnl in sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)[:10]:
            count = len([r for r in results if r['pair'] == pair])
            print(f"  {pair}: {pnl:.2f}% ({count} угод)")


async def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description='Аналізатор торгових сигналів')
    parser.add_argument('input_file', help='CSV файл з сигналами')
    parser.add_argument('-o', '--output', default='results.csv', help='Файл для збереження результатів')
    parser.add_argument('--stop-loss', type=float, default=5.0, help='Stop Loss у відсотках (за замовчуванням 3)')
    parser.add_argument('--take-profit', type=float, default=5.0, help='Take Profit у відсотках (за замовчуванням 5)')
    parser.add_argument('--long-exit-zone', type=int, default=60,
                        help='RSI зона для виходу з лонгу (за замовчуванням 60)')
    parser.add_argument('--short-exit-zone', type=int, default=40,
                        help='RSI зона для виходу з шорту (за замовчуванням 40)')
    parser.add_argument('--long-extreme-exit', type=int, default=68,
                        help='RSI екстремальний вихід для лонгу (за замовчуванням 68)')
    parser.add_argument('--short-extreme-exit', type=int, default=30,
                        help='RSI екстремальний вихід для шорту (за замовчуванням 30)')
    parser.add_argument('--max-hold', type=int, default=24,
                        help='Максимальний час утримання в годинах (за замовчуванням 24)')
    parser.add_argument('--timeframe', default='5m', help='Таймфрейм для аналізу (за замовчуванням 5m)')

    args = parser.parse_args()

    # Перевіряємо чи існує файл
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
        'timeframe': args.timeframe
    }

    print("🚀 ПОЧАТОК АНАЛІЗУ ТОРГОВИХ СИГНАЛІВ")
    print(f"📁 Вхідний файл: {args.input_file}")
    print(f"📊 Налаштування:")
    print(f"   Stop Loss: {args.stop_loss}%")
    print(f"   Take Profit: {args.take_profit}%")
    print(f"   Long Exit Zone: {args.long_exit_zone}+ (Extreme: {args.long_extreme_exit}+)")
    print(f"   Short Exit Zone: {args.short_exit_zone}- (Extreme: {args.short_extreme_exit}-)")
    print(f"   Max Hold Time: {args.max_hold} годин")
    print(f"   Timeframe: {args.timeframe}")
    print("-" * 60)

    # Створюємо аналізатор
    analyzer = SignalAnalyzer()

    # Читаємо сигнали
    signals = analyzer.read_signals_csv(args.input_file)
    if not signals:
        print("❌ Не вдалося прочитати сигнали з файлу")
        return

    print(f"✅ Завантажено {len(signals)} сигналів")

    # Аналізуємо сигнали
    try:
        results = await analyzer.analyze_signals(signals, config)

        if results:
            # Зберігаємо результати
            analyzer.save_results(results, args.output)
            print(f"\n✅ Аналіз завершено успішно!")
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