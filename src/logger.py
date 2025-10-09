from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime

from rich.console import Console, Text

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class Log:
    message: str
    level: LogLevel

LEVEL_STYLES = {
    LogLevel.DEBUG: "grey50",
    LogLevel.INFO: "white",
    LogLevel.SUCCESS: "green",
    LogLevel.WARNING: "orange3",
    LogLevel.ERROR: "red",
}

class Logger:
    def __init__(self, debug: bool = False):
        self.console = Console()
        self.saved_logs: List[Log] = []
        self.failed = 0
        self.debug = debug

    def log(self, level: LogLevel, message: str, timestamped: bool = True, save: bool = False):
        timestamp = datetime.now().strftime("[%H:%M:%S] - ") if timestamped else ''
        style = LEVEL_STYLES.get(level, "grey50")
        self.console.print(Text(timestamp, style="grey50") + Text(message, style=style))
        if save:
            self.saved_logs.append(Log(message, level))
            if level == LogLevel.ERROR:
                self.failed += 1

    def print_final_log(self):
        if self.failed:
            self.log(LogLevel.ERROR, f"failed: {self.failed}\n", timestamped=False)
        for entry in self.saved_logs:
            self.log(entry.level, entry.message, timestamped=False)

    def debug(self, msg: str, timestamped: bool = True, save: bool = False):
        if self.debug:
            self.log(LogLevel.DEBUG, msg, timestamped, save)

    def info(self, msg: str, timestamped: bool = True, save: bool = False):
        self.log(LogLevel.INFO, msg, timestamped, save)

    def success(self, msg: str, timestamped: bool = True, save: bool = False):
        self.log(LogLevel.SUCCESS, msg, timestamped, save)

    def warning(self, msg: str, timestamped: bool = True, save: bool = False):
        self.log(LogLevel.WARNING, msg, timestamped, save)

    def error(self, msg: str, timestamped: bool = True, save: bool = False):
        self.log(LogLevel.ERROR, msg, timestamped, save)
