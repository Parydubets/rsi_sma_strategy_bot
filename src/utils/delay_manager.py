"""
Менеджер системи делею для торгових позицій
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional


class DelayManager:
    """Менеджер системи делею між закриттям і відкриттям нових позицій"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Налаштування делею
        self.enabled = config.get('enabled', True)
        self.default_delay_candles = config.get('default_delay_candles', 5)
        self.custom_delays = config.get('custom_delays', {})

        # Словник для збереження часу останнього закриття позицій
        # Формат ключа: "exchange_pair_direction"
        self.last_positions: Dict[str, datetime] = {}

        self.logger.info(f"DelayManager ініціалізований: enabled={self.enabled}, "
                         f"default_delay={self.default_delay_candles} candles")

    def _get_delay_key(self, exchange: str, pair: str, direction: str) -> str:
        """Генерація ключа для збереження делею"""
        return f"{exchange.lower()}_{pair.upper()}_{direction.upper()}"

    def get_delay_candles(self, pair: str) -> int:
        """Отримання кількості свічок делею для конкретної пари"""
        return self.custom_delays.get(pair.upper(), self.default_delay_candles)

    def update_delay(self, exchange: str, pair: str, direction: str, timestamp: datetime):
        """Оновлення часу останньої позиції"""
        if not self.enabled:
            return

        delay_key = self._get_delay_key(exchange, pair, direction)
        self.last_positions[delay_key] = timestamp

        self.logger.debug(f"Оновлено делей для {delay_key}: {timestamp}")

    def check_delay(self, exchange: str, pair: str, direction: str,
                    current_time: Optional[datetime] = None) -> bool:
        """
        Перевірка чи можна відкривати нову позицію

        Args:
            exchange: Назва біржі
            pair: Торгова пара
            direction: Напрямок позиції (LONG/SHORT)
            current_time: Поточний час (якщо не вказано, використовується datetime.now())

        Returns:
            True якщо делей пройшов і можна відкривати позицію, False - інакше
        """
        if not self.enabled:
            return True

        if current_time is None:
            current_time = datetime.now()

        delay_key = self._get_delay_key(exchange, pair, direction)

        # Якщо немає записів про попередні позиції - можна відкривати
        if delay_key not in self.last_positions:
            return True

        last_position_time = self.last_positions[delay_key]
        delay_candles = self.get_delay_candles(pair)

        # Розрахунок часу делею (припускаємо 1-хвилинні свічки)
        delay_duration = timedelta(minutes=delay_candles)
        required_time = last_position_time + delay_duration

        can_open = current_time >= required_time

        if not can_open:
            remaining = required_time - current_time
            self.logger.debug(f"Делей не пройшов для {delay_key}. "
                              f"Залишилось: {remaining}")

        return can_open

    def get_remaining_delay(self, exchange: str, pair: str, direction: str,
                            current_time: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Отримання часу, що залишився до закінчення делею

        Returns:
            timedelta з часом що залишився, або None якщо делей не активний
        """
        if not self.enabled:
            return None

        if current_time is None:
            current_time = datetime.now()

        delay_key = self._get_delay_key(exchange, pair, direction)

        if delay_key not in self.last_positions:
            return None

        last_position_time = self.last_positions[delay_key]
        delay_candles = self.get_delay_candles(pair)
        delay_duration = timedelta(minutes=delay_candles)
        required_time = last_position_time + delay_duration

        if current_time >= required_time:
            return None

        return required_time - current_time

    def get_all_active_delays(self, current_time: Optional[datetime] = None) -> Dict[str, Dict]:
        """Отримання всіх активних делеїв"""
        if current_time is None:
            current_time = datetime.now()

        active_delays = {}

        for delay_key, last_time in self.last_positions.items():
            # Парсинг ключа
            parts = delay_key.split('_')
            if len(parts) >= 3:
                exchange = parts[0]
                pair = '_'.join(parts[1:-1])  # Пара може містити підкреслення
                direction = parts[-1]

                remaining = self.get_remaining_delay(exchange, pair, direction, current_time)

                if remaining and remaining > timedelta(0):
                    active_delays[delay_key] = {
                        'exchange': exchange,
                        'pair': pair,
                        'direction': direction,
                        'last_position_time': last_time,
                        'remaining_time': remaining
                    }

        return active_delays

    def clear_delay(self, exchange: str, pair: str, direction: str):
        """Очищення делею для конкретної комбінації"""
        delay_key = self._get_delay_key(exchange, pair, direction)

        if delay_key in self.last_positions:
            del self.last_positions[delay_key]
            self.logger.info(f"Делей очищено для {delay_key}")

    def clear_all_delays(self):
        """Очищення всіх делеїв"""
        count = len(self.last_positions)
        self.last_positions.clear()
        self.logger.info(f"Очищено всі делеї ({count} записів)")

    def clear_expired_delays(self, current_time: Optional[datetime] = None):
        """Очищення застарілих делеїв"""
        if current_time is None:
            current_time = datetime.now()

        expired_keys = []

        for delay_key, last_time in self.last_positions.items():
            # Максимальний час зберігання делею - 24 години
            max_age = timedelta(hours=24)
            if current_time - last_time > max_age:
                expired_keys.append(delay_key)

        for key in expired_keys:
            del self.last_positions[key]

        if expired_keys:
            self.logger.info(f"Очищено {len(expired_keys)} застарілих делеїв")

    def get_status(self) -> Dict:
        """Отримання статусу менеджера делею"""
        current_time = datetime.now()
        active_delays = self.get_all_active_delays(current_time)

        return {
            'enabled': self.enabled,
            'default_delay_candles': self.default_delay_candles,
            'custom_delays': self.custom_delays,
            'total_delays': len(self.last_positions),
            'active_delays': len(active_delays),
            'active_delays_details': active_delays
        }

    def set_enabled(self, enabled: bool):
        """Увімкнення/вимкнення системи делею"""
        self.enabled = enabled
        self.logger.info(f"Система делею {'увімкнена' if enabled else 'вимкнена'}")

    def set_default_delay(self, candles: int):
        """Встановлення делею за замовчуванням"""
        if candles < 0:
            raise ValueError("Делей не може бути негативним")

        self.default_delay_candles = candles
        self.logger.info(f"Делей за замовчуванням встановлено: {candles} свічок")

    def set_custom_delay(self, pair: str, candles: int):
        """Встановлення індивідуального делею для пари"""
        if candles < 0:
            raise ValueError("Делей не може бути негативним")

        self.custom_delays[pair.upper()] = candles
        self.logger.info(f"Індивідуальний делей для {pair}: {candles} свічок")

    def remove_custom_delay(self, pair: str):
        """Видалення індивідуального делею для пари"""
        pair_upper = pair.upper()
        if pair_upper in self.custom_delays:
            del self.custom_delays[pair_upper]
            self.logger.info(f"Індивідуальний делей для {pair} видалено")

    def __repr__(self) -> str:
        return (f"DelayManager(enabled={self.enabled}, "
                f"default_delay={self.default_delay_candles}, "
                f"active_delays={len(self.last_positions)})")