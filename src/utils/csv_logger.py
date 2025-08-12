"""
CSV логгер для запису торгових сигналів та детальної інформації
"""

import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import aiofiles


class CSVLogger:
    """Асинхронний CSV логгер для торгових сигналів"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Шляхи до файлів
        self.signals_file = config.get('signals', 'data/signals.csv')
        self.detailed_file = config.get('detailed_log', 'data/detailed_log.csv')
        self.historical_file = config.get('historical_signals', 'data/historical_signals.csv')

        # Створення директорій
        self._create_directories()

        # Ініціалізація заголовків файлів
        self._initialized = False

        # Буфери для асинхронного запису
        self._write_queue = asyncio.Queue()
        self._writer_task = None

        self.logger.info(f"CSV Logger ініціалізований")

    def _create_directories(self):
        """Створення необхідних директорій"""
        files = [self.signals_file, self.detailed_file, self.historical_file]

        for file_path in files:
            if file_path:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Ініціалізація CSV файлів з заголовками"""