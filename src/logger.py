from dataclasses import dataclass
from typing import Callable

from rich.console import Console, Text

@dataclass
class Log: s: str; level_call: Callable[[str], None]

console = Console()
saved_logs: list[Log] = []
failed = 0

def info(s: str, save: bool = False):
    console.print(Text(s, style='white'))
    if save: saved_logs.append(Log(s, info))

def success(s: str, save: bool = False):
    console.print(Text(s, style='green'))
    if save: saved_logs.append(Log(s, success))

def warning(s: str, save: bool = False):
    console.print(Text(s, style='orange3'))
    if save: saved_logs.append(Log(s, warning))

def error(s: str, save: bool = False):
    console.print(Text(s, style='red'))
    if save:
        saved_logs.append(Log(s, error))
        failed += 1

def print_final_log():
    if failed:
        error(f"failed: {failed}\n")
    for log in saved_logs:
        log.level_call(log.s)
