import ccxt
import pandas as pd
import asyncio
import aiofiles
import json
import websockets
import socket
import time
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum


# === КОНФІГУРАЦІЯ ===
@dataclass
class MonitorConfig:
    # Біржа
    EXCHANGE_NAME: str = 'bybit'

    # Торгові пари для моніторингу
    PAIRS: List[str] = field(default_factory=lambda: [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT',
        'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
        'LTC/USDT', 'BCH/USDT', 'XRP/USDT', 'DOGE/USDT', 'SHIB/USDT'
    ])

    # Таймфрейм
    TIMEFRAME: str = '5m'

    # RSI параметри
    RSI_PERIOD: int = 14
    RSI_SMA_PERIOD: int = 14

    # Зони для сигналів
    LONG_ZONE_MAX: float = 40.0  # RSI < 40 для лонг сигналів
    SHORT_ZONE_MIN: float = 60.0  # RSI > 60 для шорт сигналів

    # Інтервал оновлення (секунди)
    UPDATE_INTERVAL: int = 30

    # Файли
    SIGNALS_OUTPUT: str = 'live_signals.csv'
    LOG_FILE: str = 'market_monitor.log'

    # WebSocket/Socket
    SEND_WEBSOCKET: bool = True
    WS_HOST: str = '127.0.0.1'
    WS_PORT: int = 65433

    # TCP Socket для сигналів
    SEND_SOCKET: bool = True
    SOCKET_HOST: str = '127.0.0.1'
    SOCKET_PORT: int = 65432


class SignalType(Enum):
    LONG = "Long"
    SHORT = "Short"


@dataclass
class MarketSignal:
    pair: str
    direction: SignalType
    rsi: float
    rsi_sma: float
    price: float
    timestamp: datetime
    comment: str = ""


class MarketData:
    """Клас для зберігання ринкових даних пари"""

    def __init__(self, pair: str):
        self.pair = pair
        self.df = pd.DataFrame()
        self.last_signal_time = None
        self.last_rsi_above_sma = None  # Для відстеження перетинів

    def update_data(self, new_df: pd.DataFrame):
        """Оновлення даних з новими свічками"""
        if self.df.empty:
            self.df = new_df
        else:
            # Об'єднання з новими даними та видалення дублікатів
            combined = pd.concat([self.df, new_df])
            self.df = combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

            # Залишаємо останні 100 свічок для ефективності
            if len(self.df) > 100:
                self.df = self.df.tail(100).reset_index(drop=True)


class MarketMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.exchange = self._init_exchange()
        self.logger = self._init_logger()
        self.market_data: Dict[str, MarketData] = {}
        self.signals_sent = []

        # Ініціалізація даних для кожної пари
        for pair in self.config.PAIRS:
            self.market_data[pair] = MarketData(pair)

    def _init_exchange(self):
        """Ініціалізація біржі"""
        return ccxt.bybit({
            'enableRateLimit': True,
            'sandbox': False  # Змініть на True для тестування
        })

    def _init_logger(self):
        """Ініціалізація логера"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    async def fetch_ohlcv_data(self, symbol: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """Отримання останніх OHLCV даних"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.config.TIMEFRAME, limit=limit)

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df

        except Exception as e:
            self.logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Розрахунок RSI та RSI SMA"""
        if len(df) < max(self.config.RSI_PERIOD, self.config.RSI_SMA_PERIOD):
            return df

        df = df.copy()

        # RSI
        rsi_indicator = RSIIndicator(close=df["close"], window=self.config.RSI_PERIOD)
        df["rsi"] = rsi_indicator.rsi()

        # SMA від RSI
        df["rsi_sma"] = df["rsi"].rolling(window=self.config.RSI_SMA_PERIOD).mean()

        return df

    def detect_signal(self, market_data: MarketData) -> Optional[MarketSignal]:
        """Виявлення сигналу на основі перетину RSI та SMA"""
        df = market_data.df

        if len(df) < 2:
            return None

        current = df.iloc[-1]
        previous = df.iloc[-2]

        # Перевірка чи є валідні значення індикаторів
        if pd.isna(current["rsi"]) or pd.isna(current["rsi_sma"]) or \
                pd.isna(previous["rsi"]) or pd.isna(previous["rsi_sma"]):
            return None

        # Захист від дублювання сигналів (не частіше ніж раз на 15 хвилин)
        if (market_data.last_signal_time and
                current["datetime"] - market_data.last_signal_time < timedelta(minutes=15)):
            return None

        signal = None

        # LONG сигнал: RSI пересікає SMA знизу вгору в зоні < 40
        if (previous["rsi"] < previous["rsi_sma"] and
                current["rsi"] > current["rsi_sma"] and
                current["rsi"] < self.config.LONG_ZONE_MAX):

            signal = MarketSignal(
                pair=market_data.pair,
                direction=SignalType.LONG,
                rsi=current["rsi"],
                rsi_sma=current["rsi_sma"],
                price=current["close"],
                timestamp=current["datetime"],
                comment=f"RSI cross above SMA in oversold zone"
            )

        # SHORT сигнал: RSI пересікає SMA зверху вниз в зоні > 60
        elif (previous["rsi"] > previous["rsi_sma"] and
              current["rsi"] < current["rsi_sma"] and
              current["rsi"] > self.config.SHORT_ZONE_MIN):

            signal = MarketSignal(
                pair=market_data.pair,
                direction=SignalType.SHORT,
                rsi=current["rsi"],
                rsi_sma=current["rsi_sma"],
                price=current["close"],
                timestamp=current["datetime"],
                comment=f"RSI cross below SMA in overbought zone"
            )

        if signal:
            market_data.last_signal_time = current["datetime"]

        return signal

    async def send_signal_socket(self, signal: MarketSignal):
        """Відправка сигналу через TCP Socket"""
        if not self.config.SEND_SOCKET:
            return

        try:
            # Підготовка даних для відправки (формат як у вашому валідаційному боті)
            signal_data = {
                'Pair': signal.pair.replace('/', ''),
                'Exchange': 'BYBIT',
                'Direction': signal.direction.value,
                'RSI': round(signal.rsi, 2),
                'SMA': round(signal.rsi_sma, 2),
                'Comment': 'ПЕРЕПРОДАНО' if signal.direction == SignalType.LONG else 'ПЕРЕТИНЕННЯ',
                'Time': signal.timestamp.strftime("%d.%m.%Y %H:%M")
            }

            # Створення TCP з'єднання
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # 5 секунд таймаут

            try:
                sock.connect((self.config.SOCKET_HOST, self.config.SOCKET_PORT))

                # Відправка JSON даних
                message = json.dumps(signal_data, ensure_ascii=False).encode('utf-8')
                sock.sendall(message)

                self.logger.info(f"Сигнал відправлено через Socket: {signal.pair} {signal.direction.value}")

            finally:
                sock.close()

        except Exception as e:
            self.logger.warning(f"Не вдалося відправити сигнал через Socket: {e}")

    async def send_signal_websocket(self, signal: MarketSignal):
        """Відправка сигналу через WebSocket"""
        if not self.config.SEND_WEBSOCKET:
            return

        try:
            # Підготовка даних для відправки
            signal_data = {
                'Pair': signal.pair.replace('/', ''),  # BTC/USDT -> BTCUSDT
                'Exchange': 'BYBIT',
                'Direction': signal.direction.value,
                'RSI': round(signal.rsi, 2),
                'SMA': round(signal.rsi_sma, 2),
                'Price': signal.price,
                'Time': signal.timestamp.strftime("%d.%m.%Y %H:%M"),
                'Comment': 'ПЕРЕПРОДАНО' if signal.direction == SignalType.LONG else 'ПЕРЕТИНЕННЯ'
            }

            # Відправка через WebSocket
            uri = f"ws://{self.config.WS_HOST}:{self.config.WS_PORT}"
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps(signal_data, ensure_ascii=False))
                self.logger.info(f"Сигнал відправлено через WebSocket: {signal.pair} {signal.direction.value}")

        except Exception as e:
            self.logger.warning(f"Не вдалося відправити сигнал через WebSocket: {e}")

    async def send_signal(self, signal: MarketSignal):
        """Універсальна функція відправки сигналу через різні канали"""
        # Відправка через Socket (TCP)
        if self.config.SEND_SOCKET:
            await self.send_signal_socket(signal)

        # Відправка через WebSocket
        if self.config.SEND_WEBSOCKET:
            await self.send_signal_websocket(signal)

    async def save_signal_to_csv(self, signal: MarketSignal):
        """Збереження сигналу в CSV файл"""
        try:
            # Визначення коментаря залежно від напрямку
            if signal.direction == SignalType.LONG:
                comment = "ПЕРЕПРОДАНО"
            else:
                comment = "ПЕРЕТИНЕННЯ"

            # Підготовка даних
            signal_data = {
                'Pair': signal.pair.replace('/', ''),
                'Exchange': 'BYBIT',
                'Timeframe': '5m',
                'Direction': signal.direction.value,
                'Comment': comment,
                'RSI': round(signal.rsi, 2),
                'SMA': round(signal.rsi_sma, 2),
                'Time': signal.timestamp.strftime("%d.%m.%Y %H:%M")
            }

            # Перевірка чи існує файл
            try:
                df_existing = pd.read_csv(self.config.SIGNALS_OUTPUT, encoding='cp1251', sep=';')
            except FileNotFoundError:
                df_existing = pd.DataFrame()

            # Додавання нового сигналу
            df_new = pd.DataFrame([signal_data])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

            # Правильний порядок колонок
            column_order = ['Pair', 'Exchange', 'Timeframe', 'Direction', 'Comment', 'RSI', 'SMA', 'Time']
            df_combined = df_combined[column_order]

            # Збереження
            df_combined.to_csv(self.config.SIGNALS_OUTPUT, index=False, encoding='cp1251', sep=';')

        except Exception as e:
            self.logger.error(f"Помилка збереження сигналу в CSV: {e}")

    async def update_pair_data(self, pair: str):
        """Оновлення даних для однієї пари"""
        try:
            # Отримання нових даних
            df_new = await self.fetch_ohlcv_data(pair)
            if df_new is None:
                return

            # Розрахунок індикаторів
            df_new = self.calculate_indicators(df_new)

            # Оновлення в market_data
            self.market_data[pair].update_data(df_new)

            # Перевірка на наявність сигналу
            signal = self.detect_signal(self.market_data[pair])

            if signal:
                self.logger.info(f"🚨 НОВИЙ СИГНАЛ: {signal.pair} {signal.direction.value} "
                                 f"RSI:{signal.rsi:.2f} SMA:{signal.rsi_sma:.2f} Ціна:{signal.price}")

                # Збереження сигналу
                await self.save_signal_to_csv(signal)
                self.signals_sent.append(signal)

                # Відправка через Socket/WebSocket
                await self.send_signal(signal)

        except Exception as e:
            self.logger.error(f"Помилка оновлення {pair}: {e}")

    async def monitor_market(self):
        """Основний цикл моніторингу ринку"""
        self.logger.info("=== ПОЧАТОК МОНІТОРИНГУ РИНКУ ===")
        self.logger.info(f"Моніторинг {len(self.config.PAIRS)} пар на таймфреймі {self.config.TIMEFRAME}")
        self.logger.info(f"Умови сигналів:")
        self.logger.info(f"  LONG: RSI cross above SMA в зоні < {self.config.LONG_ZONE_MAX}")
        self.logger.info(f"  SHORT: RSI cross below SMA в зоні > {self.config.SHORT_ZONE_MIN}")

        # Початкове завантаження даних
        self.logger.info("Початкове завантаження даних...")
        for pair in self.config.PAIRS:
            await self.update_pair_data(pair)
            await asyncio.sleep(0.1)  # Невелика затримка між запитами

        # Основний цикл моніторингу
        while True:
            try:
                start_time = time.time()

                # Створення завдань для всіх пар
                tasks = [self.update_pair_data(pair) for pair in self.config.PAIRS]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Статистика
                total_signals = len(self.signals_sent)
                long_signals = len([s for s in self.signals_sent if s.direction == SignalType.LONG])
                short_signals = total_signals - long_signals

                elapsed_time = time.time() - start_time
                self.logger.info(f"Оновлення завершено за {elapsed_time:.2f}с. "
                                 f"Всього сигналів: {total_signals} (L:{long_signals} S:{short_signals})")

                # Очікування до наступного оновлення
                await asyncio.sleep(self.config.UPDATE_INTERVAL)

            except KeyboardInterrupt:
                self.logger.info("Зупинка моніторингу...")
                break
            except Exception as e:
                self.logger.error(f"Помилка в основному циклі: {e}")
                await asyncio.sleep(60)  # Пауза при помилці

    async def get_market_overview(self) -> Dict:
        """Отримання огляду ринку"""
        overview = {
            'timestamp': datetime.now(),
            'pairs_monitored': len(self.config.PAIRS),
            'pairs_data': {}
        }

        for pair, data in self.market_data.items():
            if not data.df.empty and len(data.df) > 0:
                latest = data.df.iloc[-1]
                overview['pairs_data'][pair] = {
                    'price': latest['close'],
                    'rsi': latest.get('rsi', None),
                    'rsi_sma': latest.get('rsi_sma', None),
                    'last_update': latest['datetime'],
                    'last_signal': data.last_signal_time
                }

        return overview

    async def generate_historical_signals(self, days_back: int = 7, output_file: str = None) -> List[MarketSignal]:
        """Генерація сигналів на основі історичних даних за останні N днів"""
        if output_file is None:
            output_file = f'historical_signals_{days_back}d.csv'

        self.logger.info(f"=== ГЕНЕРАЦІЯ ІСТОРИЧНИХ СИГНАЛІВ ЗА {days_back} ДНІВ ===")

        # Розрахунок періоду
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back + 2)  # +2 дні для індикаторів

        all_signals = []

        for pair in self.config.PAIRS:
            try:
                self.logger.info(f"Обробка історичних даних для {pair}...")

                # Отримання історичних даних
                limit = days_back * 24 * 12 + 100  # 12 свічок на годину для 5m + запас
                since_ms = int(start_time.timestamp() * 1000)

                ohlcv = self.exchange.fetch_ohlcv(pair, self.config.TIMEFRAME, since_ms, limit)

                if not ohlcv or len(ohlcv) < self.config.RSI_PERIOD + self.config.RSI_SMA_PERIOD:
                    self.logger.warning(f"Недостатньо даних для {pair}")
                    continue

                # Створення DataFrame
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Розрахунок індикаторів
                df = self.calculate_indicators(df)

                # Фільтрація за часовим періодом
                target_start = end_time - timedelta(days=days_back)
                df_period = df[df["datetime"] >= target_start].copy()

                if len(df_period) < 2:
                    continue

                # Пошук сигналів у історичних даних
                signals = self._scan_historical_signals(df_period, pair)
                all_signals.extend(signals)

                self.logger.info(f"Знайдено {len(signals)} сигналів для {pair}")
                await asyncio.sleep(0.1)  # Пауза між запитами

            except Exception as e:
                self.logger.error(f"Помилка обробки історичних даних для {pair}: {e}")
                continue

        # Сортування сигналів за часом
        all_signals.sort(key=lambda x: x.timestamp)

        # Збереження в CSV файл
        if all_signals:
            await self._save_historical_signals_to_csv(all_signals, output_file)

            # Відправка історичних сигналів через Socket/WebSocket (опціонально)
            if self.config.SEND_SOCKET or self.config.SEND_WEBSOCKET:
                await self._send_historical_signals(all_signals)

        self.logger.info(f"=== ЗАВЕРШЕНО: знайдено {len(all_signals)} історичних сигналів ===")

        return all_signals

    async def _send_historical_signals(self, signals: List[MarketSignal]):
        """Відправка історичних сигналів через Socket/WebSocket з затримкою"""
        self.logger.info("Відправка історичних сигналів...")

        for i, signal in enumerate(signals):
            try:
                await self.send_signal(signal)

                # Прогрес
                if (i + 1) % 10 == 0 or i == len(signals) - 1:
                    self.logger.info(f"Відправлено {i + 1}/{len(signals)} сигналів")

                # Невелика затримка між сигналами (щоб не перевантажити приймач)
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Помилка відправки історичного сигналу {i + 1}: {e}")
                continue

        self.logger.info("Відправка історичних сигналів завершена")

    def _scan_historical_signals(self, df: pd.DataFrame, pair: str) -> List[MarketSignal]:
        """Сканування історичних даних на предмет сигналів"""
        signals = []
        last_signal_time = None

        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]

            # Перевірка валідності індикаторів
            if (pd.isna(current["rsi"]) or pd.isna(current["rsi_sma"]) or
                    pd.isna(previous["rsi"]) or pd.isna(previous["rsi_sma"])):
                continue

            # Захист від дублювання (мінімум 15 хвилин між сигналами)
            if (last_signal_time and
                    current["datetime"] - last_signal_time < timedelta(minutes=15)):
                continue

            signal = None

            # LONG сигнал: RSI пересікає SMA знизу вгору в зоні < 40
            if (previous["rsi"] < previous["rsi_sma"] and
                    current["rsi"] > current["rsi_sma"] and
                    current["rsi"] < self.config.LONG_ZONE_MAX):

                signal = MarketSignal(
                    pair=pair,
                    direction=SignalType.LONG,
                    rsi=current["rsi"],
                    rsi_sma=current["rsi_sma"],
                    price=current["close"],
                    timestamp=current["datetime"],
                    comment="Historical RSI cross above SMA in oversold zone"
                )

            # SHORT сигнал: RSI пересікає SMA зверху вниз в зоні > 60
            elif (previous["rsi"] > previous["rsi_sma"] and
                  current["rsi"] < current["rsi_sma"] and
                  current["rsi"] > self.config.SHORT_ZONE_MIN):

                signal = MarketSignal(
                    pair=pair,
                    direction=SignalType.SHORT,
                    rsi=current["rsi"],
                    rsi_sma=current["rsi_sma"],
                    price=current["close"],
                    timestamp=current["datetime"],
                    comment="Historical RSI cross below SMA in overbought zone"
                )

            if signal:
                signals.append(signal)
                last_signal_time = current["datetime"]

        return signals

    async def _save_historical_signals_to_csv(self, signals: List[MarketSignal], filename: str):
        """Збереження історичних сигналів у CSV файл у форматі сумісному з валідаційним ботом"""
        try:
            data = []
            for signal in signals:
                # Визначення коментаря залежно від напрямку
                if signal.direction == SignalType.LONG:
                    comment = "ПЕРЕПРОДАНО"
                else:
                    comment = "ПЕРЕТИНЕННЯ"

                data.append({
                    'Pair': signal.pair.replace('/', ''),  # BTC/USDT -> BTCUSDT
                    'Exchange': 'BYBIT',
                    'Timeframe': '5m',
                    'Direction': signal.direction.value,
                    'Comment': comment,
                    'RSI': round(signal.rsi, 2),
                    'SMA': round(signal.rsi_sma, 2),
                    'Time': signal.timestamp.strftime("%d.%m.%Y %H:%M")
                })

            df = pd.DataFrame(data)

            # Збереження з правильним порядком колонок
            column_order = ['Pair', 'Exchange', 'Timeframe', 'Direction', 'Comment', 'RSI', 'SMA', 'Time']
            df = df[column_order]

            df.to_csv(filename, index=False, encoding='cp1251', sep=';')

            self.logger.info(f"Історичні сигнали збережено в {filename}")

            # Статистика по сигналах
            long_count = len([s for s in signals if s.direction == SignalType.LONG])
            short_count = len(signals) - long_count

            print(f"\n📊 СТАТИСТИКА ІСТОРИЧНИХ СИГНАЛІВ:")
            print(f"   Всього сигналів: {len(signals)}")
            print(f"   LONG сигналів: {long_count}")
            print(f"   SHORT сигналів: {short_count}")
            print(f"   Файл збережено: {filename}")

            # Розподіл по парах
            pair_counts = {}
            for signal in signals:
                pair = signal.pair.replace('/', '')
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            print(f"\n📈 РОЗПОДІЛ ПО ПАРАХ:")
            for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {pair}: {count} сигналів")

        except Exception as e:
            self.logger.error(f"Помилка збереження історичних сигналів: {e}")

    async def analyze_historical_period(self, days_back: int = 7):
        """Аналіз історичного періоду з детальною статистикою"""
        self.logger.info(f"=== АНАЛІЗ ІСТОРИЧНОГО ПЕРІОДУ {days_back} ДНІВ ===")

        signals = await self.generate_historical_signals(days_back)

        if not signals:
            self.logger.warning("Не знайдено історичних сигналів для аналізу")
            return

        # Детальна статистика
        print(f"\n📋 ДЕТАЛЬНИЙ АНАЛІЗ ЗА {days_back} ДНІВ:")
        print(
            f"   Період: {(datetime.now() - timedelta(days=days_back)).strftime('%d.%m.%Y')} - {datetime.now().strftime('%d.%m.%Y')}")

        # Розподіл по дням
        daily_counts = {}
        for signal in signals:
            day = signal.timestamp.strftime('%d.%m.%Y')
            daily_counts[day] = daily_counts.get(day, 0) + 1

        print(f"\n📅 РОЗПОДІЛ ПО ДНЯХ:")
        for day, count in sorted(daily_counts.items()):
            print(f"   {day}: {count} сигналів")

        # Розподіл по годинах
        hourly_counts = {}
        for signal in signals:
            hour = signal.timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        print(f"\n🕐 НАЙАКТИВНІШІ ГОДИНИ (UTC):")
        for hour, count in sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {hour:02d}:00: {count} сигналів")

        # Середні значення RSI
        long_signals = [s for s in signals if s.direction == SignalType.LONG]
        short_signals = [s for s in signals if s.direction == SignalType.SHORT]

        if long_signals:
            long_rsi_avg = sum([s.rsi for s in long_signals]) / len(long_signals)
            print(f"\n📊 СЕРЕДНІ ЗНАЧЕННЯ RSI:")
            print(f"   LONG сигнали: {long_rsi_avg:.2f}")

        if short_signals:
            short_rsi_avg = sum([s.rsi for s in short_signals]) / len(short_signals)
            if not long_signals:
                print(f"\n📊 СЕРЕДНІ ЗНАЧЕННЯ RSI:")
            print(f"   SHORT сигнали: {short_rsi_avg:.2f}")

        return signals

    def print_market_status(self):
        """Виведення поточного статусу ринку"""
        print("\n" + "=" * 80)
        print(f"СТАТУС РИНКУ - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)

        for pair, data in self.market_data.items():
            if not data.df.empty and len(data.df) > 0:
                latest = data.df.iloc[-1]
                rsi = latest.get('rsi', 0)
                rsi_sma = latest.get('rsi_sma', 0)
                price = latest.get('close', 0)

                # Визначення зони RSI
                if rsi < 30:
                    zone = "📉 OVERSOLD"
                elif rsi < self.config.LONG_ZONE_MAX:
                    zone = "📊 LONG_ZONE"
                elif rsi > 70:
                    zone = "📈 OVERBOUGHT"
                elif rsi > self.config.SHORT_ZONE_MIN:
                    zone = "📊 SHORT_ZONE"
                else:
                    zone = "⚖️  NEUTRAL"

                # Напрямок RSI відносно SMA
                direction = "↗️" if rsi > rsi_sma else "↘️"

                print(f"{pair:12} | ${price:8.2f} | RSI:{rsi:5.1f} {direction} SMA:{rsi_sma:5.1f} | {zone}")


class SignalWebSocketServer:
    """WebSocket сервер для відправки сигналів зовнішнім системам"""

    def __init__(self, host: str, port: int, logger):
        self.host = host
        self.port = port
        self.logger = logger
        self.clients = set()

    async def register_client(self, websocket):
        """Реєстрація нового клієнта"""
        self.clients.add(websocket)
        self.logger.info(f"Новий WebSocket клієнт підключився. Всього: {len(self.clients)}")

    async def unregister_client(self, websocket):
        """Видалення клієнта"""
        self.clients.discard(websocket)
        self.logger.info(f"WebSocket клієнт відключився. Залишилось: {len(self.clients)}")

    async def broadcast_signal(self, signal: MarketSignal):
        """Розсилка сигналу всім підключеним клієнтам"""
        if not self.clients:
            return

        signal_data = {
            'type': 'trading_signal',
            'pair': signal.pair.replace('/', ''),
            'direction': signal.direction.value,
            'rsi': round(signal.rsi, 2),
            'sma': round(signal.rsi_sma, 2),
            'price': signal.price,
            'timestamp': signal.timestamp.isoformat(),
            'comment': signal.comment
        }

        message = json.dumps(signal_data)
        disconnected = []

        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)

        # Видалення відключених клієнтів
        for client in disconnected:
            await self.unregister_client(client)

    async def handle_client(self, websocket, path):
        """Обробка підключення клієнта"""
        await self.register_client(websocket)
        try:
            await websocket.wait_closed()
        except:
            pass
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        """Запуск WebSocket сервера"""
        self.logger.info(f"Запуск WebSocket сервера на {self.host}:{self.port}")
        return await websockets.serve(self.handle_client, self.host, self.port)


class MarketMonitorWithWebSocket(MarketMonitor):
    """Розширена версія моніторингу з WebSocket сервером"""

    def __init__(self, config: MonitorConfig):
        super().__init__(config)
        self.ws_server = SignalWebSocketServer(
            config.WS_HOST,
            config.WS_PORT + 1,  # Використовуємо інший порт для вхідних WebSocket
            self.logger
        )

    async def send_signal_websocket(self, signal: MarketSignal):
        """Відправка сигналу через WebSocket сервер (broadcasting)"""
        await self.ws_server.broadcast_signal(signal)

        # Також відправляємо через звичайний WebSocket клієнт
        await super().send_signal_websocket(signal)

    async def start_monitoring(self):
        """Запуск моніторингу з WebSocket сервером"""
        # Запуск WebSocket сервера
        server = await self.ws_server.start_server()

        # Запуск моніторингу ринку
        try:
            await self.monitor_market()
        finally:
            server.close()
            await server.wait_closed()

    async def generate_historical_signals(self, days_back: int = 7, output_file: str = None) -> List[MarketSignal]:
        """Генерація сигналів на основі історичних даних за останні N днів"""
        if output_file is None:
            output_file = f'historical_signals_{days_back}d.csv'

        self.logger.info(f"=== ГЕНЕРАЦІЯ ІСТОРИЧНИХ СИГНАЛІВ ЗА {days_back} ДНІВ ===")

        # Розрахунок періоду
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back + 2)  # +2 дні для індикаторів

        all_signals = []

        for pair in self.config.PAIRS:
            try:
                self.logger.info(f"Обробка історичних даних для {pair}...")

                # Отримання історичних даних
                limit = days_back * 24 * 12 + 100  # 12 свічок на годину для 5m + запас
                since_ms = int(start_time.timestamp() * 1000)

                ohlcv = self.exchange.fetch_ohlcv(pair, self.config.TIMEFRAME, since_ms, limit)

                if not ohlcv or len(ohlcv) < self.config.RSI_PERIOD + self.config.RSI_SMA_PERIOD:
                    self.logger.warning(f"Недостатньо даних для {pair}")
                    continue

                # Створення DataFrame
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Розрахунок індикаторів
                df = self.calculate_indicators(df)

                # Фільтрація за часовим періодом
                target_start = end_time - timedelta(days=days_back)
                df_period = df[df["datetime"] >= target_start].copy()

                if len(df_period) < 2:
                    continue

                # Пошук сигналів у історичних даних
                signals = self._scan_historical_signals(df_period, pair)
                all_signals.extend(signals)

                self.logger.info(f"Знайдено {len(signals)} сигналів для {pair}")
                await asyncio.sleep(0.1)  # Пауза між запитами

            except Exception as e:
                self.logger.error(f"Помилка обробки історичних даних для {pair}: {e}")
                continue

        # Сортування сигналів за часом
        all_signals.sort(key=lambda x: x.timestamp)

        # Збереження в CSV файл
        if all_signals:
            await self._save_historical_signals_to_csv(all_signals, output_file)

        self.logger.info(f"=== ЗАВЕРШЕНО: знайдено {len(all_signals)} історичних сигналів ===")

        return all_signals

    def _scan_historical_signals(self, df: pd.DataFrame, pair: str) -> List[MarketSignal]:
        """Сканування історичних даних на предмет сигналів"""
        signals = []
        last_signal_time = None

        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]

            # Перевірка валідності індикаторів
            if (pd.isna(current["rsi"]) or pd.isna(current["rsi_sma"]) or
                    pd.isna(previous["rsi"]) or pd.isna(previous["rsi_sma"])):
                continue

            # Захист від дублювання (мінімум 15 хвилин між сигналами)
            if (last_signal_time and
                    current["datetime"] - last_signal_time < timedelta(minutes=15)):
                continue

            signal = None

            # LONG сигнал: RSI пересікає SMA знизу вгору в зоні < 40
            if (previous["rsi"] < previous["rsi_sma"] and
                    current["rsi"] > current["rsi_sma"] and
                    current["rsi"] < self.config.LONG_ZONE_MAX):

                signal = MarketSignal(
                    pair=pair,
                    direction=SignalType.LONG,
                    rsi=current["rsi"],
                    rsi_sma=current["rsi_sma"],
                    price=current["close"],
                    timestamp=current["datetime"],
                    comment="Historical RSI cross above SMA in oversold zone"
                )

            # SHORT сигнал: RSI пересікає SMA зверху вниз в зоні > 60
            elif (previous["rsi"] > previous["rsi_sma"] and
                  current["rsi"] < current["rsi_sma"] and
                  current["rsi"] > self.config.SHORT_ZONE_MIN):

                signal = MarketSignal(
                    pair=pair,
                    direction=SignalType.SHORT,
                    rsi=current["rsi"],
                    rsi_sma=current["rsi_sma"],
                    price=current["close"],
                    timestamp=current["datetime"],
                    comment="Historical RSI cross below SMA in overbought zone"
                )

            if signal:
                signals.append(signal)
                last_signal_time = current["datetime"]

        return signals

    async def _save_historical_signals_to_csv(self, signals: List[MarketSignal], filename: str):
        """Збереження історичних сигналів у CSV файл у форматі сумісному з валідаційним ботом"""
        try:
            data = []
            for signal in signals:
                # Форматування SMA як у оригінальному файлі (з префіксом "Сер")
                sma_formatted = f"Сер{signal.rsi_sma:.2f}".replace('.', '')

                data.append({
                    'Time': signal.timestamp.strftime("%d.%m.%Y %H:%M"),
                    'Pair': signal.pair.replace('/', ''),  # BTC/USDT -> BTCUSDT
                    'Exchange': 'BYBIT',
                    'Direction': signal.direction.value,
                    'RSI': round(signal.rsi, 2),
                    'SMA': sma_formatted,
                    'Comment': signal.comment
                })

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='cp1251', sep=';')

            self.logger.info(f"Історичні сигнали збережено в {filename}")

            # Статистика по сигналах
            long_count = len([s for s in signals if s.direction == SignalType.LONG])
            short_count = len(signals) - long_count

            print(f"\n📊 СТАТИСТИКА ІСТОРИЧНИХ СИГНАЛІВ:")
            print(f"   Всього сигналів: {len(signals)}")
            print(f"   LONG сигналів: {long_count}")
            print(f"   SHORT сигналів: {short_count}")
            print(f"   Файл збережено: {filename}")

            # Розподіл по парах
            pair_counts = {}
            for signal in signals:
                pair = signal.pair.replace('/', '')
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            print(f"\n📈 РОЗПОДІЛ ПО ПАРАХ:")
            for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {pair}: {count} сигналів")

        except Exception as e:
            self.logger.error(f"Помилка збереження історичних сигналів: {e}")

    async def analyze_historical_period(self, days_back: int = 7):
        """Аналіз історичного періоду з детальною статистикою"""
        self.logger.info(f"=== АНАЛІЗ ІСТОРИЧНОГО ПЕРІОДУ {days_back} ДНІВ ===")

        signals = await self.generate_historical_signals(days_back)

        if not signals:
            self.logger.warning("Не знайдено історичних сигналів для аналізу")
            return

        # Детальна статистика
        print(f"\n📋 ДЕТАЛЬНИЙ АНАЛІЗ ЗА {days_back} ДНІВ:")
        print(
            f"   Період: {(datetime.now() - timedelta(days=days_back)).strftime('%d.%m.%Y')} - {datetime.now().strftime('%d.%m.%Y')}")

        # Розподіл по дням
        daily_counts = {}
        for signal in signals:
            day = signal.timestamp.strftime('%d.%m.%Y')
            daily_counts[day] = daily_counts.get(day, 0) + 1

        print(f"\n📅 РОЗПОДІЛ ПО ДНЯХ:")
        for day, count in sorted(daily_counts.items()):
            print(f"   {day}: {count} сигналів")

        # Розподіл по годинах
        hourly_counts = {}
        for signal in signals:
            hour = signal.timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        print(f"\n🕐 НАЙАКТИВНІШІ ГОДИНИ (UTC):")
        for hour, count in sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {hour:02d}:00: {count} сигналів")

        # Середні значення RSI
        long_rsi_avg = sum([s.rsi for s in signals if s.direction == SignalType.LONG]) / max(1,
                                                                                             len([s for s in signals if
                                                                                                  s.direction == SignalType.LONG]))
        short_rsi_avg = sum([s.rsi for s in signals if s.direction == SignalType.SHORT]) / max(1,
                                                                                               len([s for s in signals
                                                                                                    if
                                                                                                    s.direction == SignalType.SHORT]))

        print(f"\n📊 СЕРЕДНІ ЗНАЧЕННЯ RSI:")
        print(f"   LONG сигнали: {long_rsi_avg:.2f}")
        print(f"   SHORT сигнали: {short_rsi_avg:.2f}")

        return signals


async def main():
    """Головна функція"""
    config = MonitorConfig()

    print("=== МОНІТОРИНГ РИНКУ ТА ГЕНЕРАЦІЯ СИГНАЛІВ ===")
    print("1 - Запустити live моніторинг")
    print("2 - Згенерувати історичні сигнали")
    print("3 - Тестувати з обмеженим списком пар")
    print("4 - Переглянути налаштування")
    print("5 - Переглянути поточний стан ринку")
    print("6 - Повний аналіз історичного періоду")

    choice = input("Оберіть опцію (1-6): ").strip()

    if choice == '1':
        monitor = MarketMonitorWithWebSocket(config)
        await monitor.start_monitoring()

    elif choice == '2':
        # Генерація історичних сигналів
        days = input("Скільки днів назад проаналізувати? (за замовчуванням 7): ").strip()
        days_back = int(days) if days.isdigit() else 7

        custom_file = input("Назва файлу (Enter для автоматичної): ").strip()
        output_file = custom_file if custom_file else None

        # Опціонально відправити через Socket/WebSocket
        send_signals = input("Відправити сигнали через Socket/WebSocket? (y/n, за замовчуванням n): ").strip().lower()
        if send_signals != 'y':
            config.SEND_SOCKET = False
            config.SEND_WEBSOCKET = False

        monitor = MarketMonitor(config)
        signals = await monitor.generate_historical_signals(days_back, output_file)

        if signals:
            print(f"\n✅ Успішно згенеровано {len(signals)} історичних сигналів!")
            print(f"   Файл можна використати для бектестів у вашому основному боті.")

            if send_signals == 'y':
                print(f"   Сигнали також відправлено через Socket/WebSocket.")

    elif choice == '3':
        # Тестування з обмеженим списком
        config.PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        config.UPDATE_INTERVAL = 15  # Частіше оновлення для тесту

        monitor = MarketMonitorWithWebSocket(config)
        await monitor.start_monitoring()

    elif choice == '4':
        print("\nПоточні налаштування:")
        print(f"  Таймфрейм: {config.TIMEFRAME}")
        print(f"  RSI період: {config.RSI_PERIOD}")
        print(f"  RSI SMA період: {config.RSI_SMA_PERIOD}")
        print(f"  LONG зона: RSI < {config.LONG_ZONE_MAX}")
        print(f"  SHORT зона: RSI > {config.SHORT_ZONE_MIN}")
        print(f"  Інтервал оновлення: {config.UPDATE_INTERVAL}с")
        print(f"  Кількість пар: {len(config.PAIRS)}")
        print(f"  WebSocket сервер: {config.WS_HOST}:{config.WS_PORT + 1}")
        print(f"  WebSocket клієнт: {config.WS_HOST}:{config.WS_PORT}")
        print(f"  TCP Socket: {config.SOCKET_HOST}:{config.SOCKET_PORT}")
        print(f"  Відправка через Socket: {'Так' if config.SEND_SOCKET else 'Ні'}")
        print(f"  Відправка через WebSocket: {'Так' if config.SEND_WEBSOCKET else 'Ні'}")
        print(f"  Торгові пари: {', '.join(config.PAIRS[:5])}...")

    elif choice == '5':
        monitor = MarketMonitor(config)
        # Швидке оновлення для показу статусу
        for pair in config.PAIRS[:8]:  # Перші 8 пар для демонстрації
            await monitor.update_pair_data(pair)
        monitor.print_market_status()

    elif choice == '6':
        # Повний аналіз з детальною статистикою
        days = input("Скільки днів проаналізувати? (за замовчуванням 14): ").strip()
        days_back = int(days) if days.isdigit() else 14

        monitor = MarketMonitor(config)
        await monitor.analyze_historical_period(days_back)

    else:
        print("Невірний вибір.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nМоніторинг зупинено користувачем.")
    except Exception as e:
        print(f"Критична помилка: {e}")