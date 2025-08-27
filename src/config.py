"""
Клас для роботи з конфігурацією бота
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv


class ConfigError(Exception):
    """Помилка конфігурації"""
    pass


class Config:
    """Клас для роботи з конфігурацією"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config_data: Dict = {}
        self.logger = logging.getLogger(__name__)

        # Завантажуємо змінні середовища з .env файлу
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)

    async def load(self):
        """Завантаження конфігурації з файлу"""
        try:
            if not self.config_path.exists():
                raise ConfigError(f"Конфігураційний файл {self.config_path} не знайдено")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            print(self.config_data)
            # Підстановка змінних середовища
            self._substitute_env_vars(self.config_data)

            self.logger.info(f"Конфігурацію завантажено з {self.config_path}")

        except yaml.YAMLError as e:
            raise ConfigError(f"Помилка парсингу YAML файлу: {e}")
        except Exception as e:
            raise ConfigError(f"Помилка завантаження конфігурації: {e}")

    def _substitute_env_vars(self, data: Any):
        """Рекурсивна підстановка змінних середовища"""
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._substitute_env_vars(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                data[i] = self._substitute_env_vars(item)
        elif isinstance(data, str):
            # Підстановка змінних виду ${VAR_NAME}
            if data.startswith('${') and data.endswith('}'):
                env_var = data[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    self.logger.warning(f"Змінна середовища {env_var} не встановлена")
                    return data
                return env_value

        return data

    def get(self, key: str, default: Any = None) -> Any:
        """Отримання значення з конфігурації по ключу (підтримує точкову нотацію)"""
        try:
            keys = key.split('.')
            value = self.config_data

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

        except Exception:
            return default

    def set(self, key: str, value: Any):
        """Встановлення значення в конфігурації"""
        keys = key.split('.')
        data = self.config_data

        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        data[keys[-1]] = value

    def has(self, key: str) -> bool:
        """Перевірка існування ключа в конфігурації"""
        return self.get(key) is not None

    async def validate(self) -> bool:
        """Валідація конфігурації"""
        try:
            validation_errors = []

            # Перевірка обов'язкових секцій
            required_sections = ['strategy', 'exchanges', 'pairs', 'logging']
            for section in required_sections:
                if not self.has(section):
                    validation_errors.append(f"Відсутня обов'язкова секція: {section}")

            # Валідація параметрів стратегії
            await self._validate_strategy_params(validation_errors)

            # Валідація налаштувань бірж
            await self._validate_exchanges(validation_errors)

            # Валідація торгових пар
            await self._validate_pairs(validation_errors)

            # Валідація налаштувань логування
            await self._validate_logging(validation_errors)

            if validation_errors:
                for error in validation_errors:
                    self.logger.error(f"Помилка валідації: {error}")
                return False

            self.logger.info("Конфігурація валідна")
            return True

        except Exception as e:
            self.logger.error(f"Помилка валідації конфігурації: {e}")
            return False

    async def _validate_strategy_params(self, errors: List[str]):
        """Валідація параметрів стратегії"""
        strategy = self.get('strategy', {})

        # Перевірка параметрів індикаторів
        indicators = strategy.get('indicators', {})
        if not indicators.get('rsi_period'):
            errors.append("strategy.indicators.rsi_period не встановлений")
        elif not isinstance(indicators['rsi_period'], int) or indicators['rsi_period'] < 1:
            errors.append("strategy.indicators.rsi_period повинен бути цілим числом > 0")

        if not indicators.get('rsi_sma_period'):
            errors.append("strategy.indicators.rsi_sma_period не встановлений")
        elif not isinstance(indicators['rsi_sma_period'], int) or indicators['rsi_sma_period'] < 1:
            errors.append("strategy.indicators.rsi_sma_period повинен бути цілим числом > 0")

        # Перевірка основних параметрів
        params = strategy.get('parameters', {})
        numeric_params = [
            'min_diff', 'long_entry_max_rsi', 'short_entry_min_rsi',
            'overbought_level', 'oversold_level'
        ]

        for param in numeric_params:
            value = params.get(param)
            if value is None:
                errors.append(f"strategy.parameters.{param} не встановлений")
            elif not isinstance(value, (int, float)):
                errors.append(f"strategy.parameters.{param} повинен бути числом")

        # Перевірка логічності значень RSI
        if params.get('overbought_level', 0) <= params.get('oversold_level', 100):
            errors.append("overbought_level повинен бути більше oversold_level")

        if params.get('long_entry_max_rsi', 0) >= params.get('short_entry_min_rsi', 100):
            errors.append("long_entry_max_rsi повинен бути менше short_entry_min_rsi")

    async def _validate_exchanges(self, errors: List[str]):
        """Валідація налаштувань бірж"""
        exchanges = self.get('exchanges', {})

        if not exchanges:
            errors.append("Не налаштовано жодної біржі")
            return

        enabled_exchanges = []
        for exchange_name, exchange_config in exchanges.items():
            if exchange_config.get('enabled', False):
                enabled_exchanges.append(exchange_name)

                # Перевірка API ключів
                api_key = exchange_config.get('api_key')
                api_secret = exchange_config.get('api_secret')

                if not api_key or api_key.startswith('${'):
                    self.logger.warning(f"API ключ для {exchange_name} не встановлений")

                if not api_secret or api_secret.startswith('${'):
                    self.logger.warning(f"API секрет для {exchange_name} не встановлений")

        if not enabled_exchanges:
            errors.append("Не активовано жодної біржі")

    async def _validate_pairs(self, errors: List[str]):
        """Валідація торгових пар"""
        pairs = self.get('pairs', [])

        if not pairs:
            errors.append("Не налаштовано торгових пар")
            return

        enabled_pairs = []
        for pair_config in pairs:
            if not isinstance(pair_config, dict):
                errors.append("Неправильний формат конфігурації пари")
                continue

            symbol = pair_config.get('symbol')
            if not symbol:
                errors.append("Відсутній символ торгової пари")
                continue

            if pair_config.get('enabled', True):
                enabled_pairs.append(symbol)

        if not enabled_pairs:
            errors.append("Не активовано жодної торгової пари")

    async def _validate_logging(self, errors: List[str]):
        """Валідація налаштувань логування"""
        logging_config = self.get('logging', {})

        # Перевірка рівня логування
        level = logging_config.get('level', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level not in valid_levels:
            errors.append(f"Неправильний рівень логування: {level}")

        # Перевірка шляхів до файлів
        files = logging_config.get('files', {})
        csv_files = logging_config.get('csv_files', {})

        all_files = {**files, **csv_files}
        for file_type, file_path in all_files.items():
            if file_path:
                file_dir = Path(file_path).parent
                try:
                    file_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Не можна створити директорію для {file_type}: {e}")

    def get_enabled_exchanges(self) -> List[str]:
        """Отримання списку активних бірж"""
        exchanges = self.get('exchanges', {})
        enabled = []

        for name, config in exchanges.items():
            if config.get('enabled', False):
                enabled.append(name)

        # Сортування по пріоритету
        enabled.sort(key=lambda x: exchanges[x].get('priority', 999))

        return enabled

    def get_enabled_pairs(self) -> List[Dict]:
        """Отримання списку активних торгових пар"""
        pairs = self.get('pairs', [])
        enabled = []

        for pair_config in pairs:
            if pair_config.get('enabled', True):
                enabled.append(pair_config)

        return enabled

    def get_exchange_config(self, exchange_name: str) -> Optional[Dict]:
        """Отримання конфігурації конкретної біржі"""
        exchanges = self.get('exchanges', {})
        return exchanges.get(exchange_name.lower())

    def get_pair_config(self, symbol: str) -> Optional[Dict]:
        """Отримання конфігурації конкретної пари"""
        pairs = self.get('pairs', [])

        for pair_config in pairs:
            if pair_config.get('symbol') == symbol.upper():
                return pair_config

        return None

    def save(self, output_path: Optional[str] = None):
        """Збереження конфігурації в файл"""
        try:
            save_path = Path(output_path) if output_path else self.config_path

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False,
                          allow_unicode=True, indent=2)

            self.logger.info(f"Конфігурацію збережено в {save_path}")

        except Exception as e:
            self.logger.error(f"Помилка збереження конфігурації: {e}")
            raise ConfigError(f"Не вдалося зберегти конфігурацію: {e}")

    def to_dict(self) -> Dict:
        """Повернення конфігурації як словника"""
        return self.config_data.copy()

    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, sections={list(self.config_data.keys())})"