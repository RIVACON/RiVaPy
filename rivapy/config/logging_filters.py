# -*- coding: utf-8 -*-


from logging import LogRecord
import logging


class InfoFilter(logging.Filter):
    def __init__(self):
        """
        Constructor
        """
        super().__init__(name="filter_info_logs")

    def filter(
        self,
        record: LogRecord,
    ) -> LogRecord:
        """
        Return Log Record Object based on condition - Return only info logs

        Args:
            record: Log Record Object

        Returns:
            Log Record Object
        """
        assert isinstance(record, LogRecord)
        if record.levelno == logging.INFO:
            return record
