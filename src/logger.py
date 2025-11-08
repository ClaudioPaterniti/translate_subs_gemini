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

console = Console()
saved_logs: List[Log] = []
failed = 0
debug_enabled = False

def log(level: LogLevel, message: str, timestamped: bool = True, save: bool = False):
    global failed
    timestamp = datetime.now().strftime("[%H:%M:%S] - ") if timestamped else ''
    style = LEVEL_STYLES.get(level, "grey50")
    console.print(Text(timestamp, style="grey50") + Text(message, style=style))
    if save:
        saved_logs.append(Log(message, level))
        if level == LogLevel.ERROR:
           failed += 1

def print_final_log():
    if failed:
        log(LogLevel.ERROR, f"failed: {failed}\n", timestamped=False)
    for entry in saved_logs:
        log(entry.level, entry.message, timestamped=False)

def debug(msg: str, timestamped: bool = True, save: bool = False):
    if debug_enabled:
        log(LogLevel.DEBUG, msg, timestamped, save)

def info(msg: str, timestamped: bool = True, save: bool = False):
    log(LogLevel.INFO, msg, timestamped, save)

def success(msg: str, timestamped: bool = True, save: bool = False):
    log(LogLevel.SUCCESS, msg, timestamped, save)

def warning(msg: str, timestamped: bool = True, save: bool = False):
    log(LogLevel.WARNING, msg, timestamped, save)

def error(msg: str, timestamped: bool = True, save: bool = False):
    log(LogLevel.ERROR, msg, timestamped, save)