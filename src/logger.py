from rich.console import Console, Text

console = Console()

def info(s: str) : console.print(Text(s, style='white'))
def success(s: str) : console.print(Text(s, style='green'))
def warning(s: str) : console.print(Text(s, style='orange3'))
def error(s: str) : console.print(Text(s, style='red'))