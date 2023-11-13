import logging
import sys
import os


class Logger:
    """
    Logger class. Logs message to file and console.

    Use the same names for loggers if you need to store information in one file (with the name of the logger)

    Use different names for loggers if you need to store information separately from each other (in different files)

    """
    def __init__(self, name: str):
        """
        Constructor.
        """
        self._allowed_levels = ['info', 'error', 'debug', 'exception']
        self._log_core = logging.getLogger(name)
        self._log_core.setLevel(level=logging.DEBUG)
        self._logs_path = os.path.join(os.path.dirname(__file__)[:-4], 'Logs', f'{name}.log')
        if self._log_core.hasHandlers():
            self._log_core.handlers.clear()
        self._file_handler = logging.FileHandler(self._logs_path)
        self._console_handler = logging.StreamHandler(stream=sys.stdout)
        self._set_format()
        self._add_all_handlers()

    def _add_all_handlers(self):
        """
        Apply all handlers to logger.
        :return:
        """
        self._log_core.addHandler(self._file_handler)
        self._log_core.addHandler(self._console_handler)

    def _set_format(self):
        """
        Sets logging format.
        """
        for_file = '[%(asctime)s: %(levelname)s %(message)s]'
        for_console = '[%(asctime)s: %(levelname)s %(message)s]'
        file_format = logging.Formatter(fmt=for_file)
        console_format = logging.Formatter(fmt=for_console)
        self._file_handler.setFormatter(file_format)
        self._console_handler.setFormatter(console_format)

    def send_message(self, message: str, level: str) -> None:
        """
        Sends message to file by logger.
        :param message: String information message
        :param level: 'info', 'error', 'debug' , 'exception'
        :return: None
        """
        if level == 'info':
            self._log_core.info(message)
        elif level == 'error':
            self._log_core.error(message)
        elif level == 'debug':
            self._log_core.debug(message)
        elif level == 'exception':
            self._log_core.exception(message)
        else:
            print("[!]ERROR[!] Invalid output level! Please use 'info', 'error', 'debug' or"
                  " 'exception' [!]ERROR[!].")

    def get_logs_path(self):
        """
        Returns path to this logger path file
        :return: String path
        """
        return self._logs_path

    def get_allowed_levels(self) -> list:
        """
        Returns list of allowed levels
        :return: List of allowed levels
        """
        return self._allowed_levels
