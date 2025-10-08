import re
from typing import Optional, Any, ClassVar, Protocol

from pydantic import BaseModel, Field

from src.logger import Logger

class Config(BaseModel):
    original_language: str
    translate_to: str
    outfile_suffix: str
    model: str = "gemini-2.0-flash"
    dialogue_chunks_size: int = 10
    json_max_chars: int = 300000
    json_reduced_chars: int = 8000
    requests_per_minutes: int = 15
    token_per_minutes: int = 1000000
    max_concurrent_requests: Optional[int] = None
    content_config: dict[str, Any] = {}
    max_retries: int = 50

class MisalignmentException(Exception):
    pass

class RetriableException(Exception):
    pass

class InvalidJsonException(Exception):
    pass

class DialogueChunk(BaseModel):
    from_line: int
    to_line: int
    dialogue: list[str]
    _translated: list[str] = None

class DialogueChunks(BaseModel):
    chunks: list[DialogueChunk]

class SubsTranslation:

    def __init__(
            self,
            dialogue: list[str],
            chunk_size: int = 10,
            logger: Logger = None):
        self.chunk_size = chunk_size
        self.logger = logger or Logger()
        self._translated = False

        self.chunks: DialogueChunks = DialogueChunks(chunks=[
            DialogueChunk(
                from_line= i,
                to_line= i + chunk_size,
                dialogue= dialogue[i: i + chunk_size],
            )
            for i in range(0, len(dialogue), chunk_size)
        ])
        self.misaligned_chunks: list[int] = []

    def add_translation(self, chunks: DialogueChunks):
        if len(chunks.chunks) != len(self.chunks.chunks):
            raise MisalignmentException("The number of translated chunks returned does not match original chunks")
        for i, (original, translated) in enumerate(zip(self.chunks.chunks, chunks.chunks)):
            if len(original.dialogue) != len(translated.dialogue):
                self.misaligned_chunks.append(i)
                diff = len(original.dialogue) - len(translated.dialogue)
                if diff > 0:
                    translated.dialogue.extend(original.dialogue[-diff:])
                else:
                    translated.dialogue = translated.dialogue[:-diff]
            original._translated = translated.dialogue
        self._translated = True

    def get_misaligned_chunks(self) -> DialogueChunks:
        return DialogueChunks(chunks=[self.chunks.chunks[i] for i in self.misaligned_chunks])

    def apply_corrections(self, chunks: DialogueChunks):
        if len(chunks.chunks) != len(self.misaligned_chunks):
            return
        temp_misaligned = []
        for i, index in enumerate(self.misaligned_chunks):
            original, translated = self.chunks.chunks[index], chunks.chunks[i]
            if len(original.dialogue) != len(translated.dialogue):
                temp_misaligned.append(index)
                diff = len(original.dialogue) - len(translated.dialogue)
                if diff > 0:
                    translated.dialogue.extend(original.dialogue[-diff:])
                else:
                    translated.dialogue = translated.dialogue[:-diff]
            original._translated = translated.dialogue
        self.misaligned_chunks = temp_misaligned

    def get_translated_dialogue(self) -> list[str]:
        if not self._translated: raise Exception("No translation available")
        return [l for chunk in self.chunks.chunks for l in chunk._translated]

class TranslationFile(Protocol):
    def get_dialogue(self) -> list[str]:
        ...

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        """Returns the corresponding line number in the final file"""
        ...

    def get_translation(self, translation: list[str]) -> str:
        ...

class AssTranslationFile(TranslationFile):
    command_regex: ClassVar[re.Pattern] = re.compile(r'\{[^{}]+\}')

    def __init__(self, text: str):
        splitted = text.split('Dialogue:', 1)
        self._header = splitted[0]
        self._fields: list[str]
        self._commands: dict[str, str] = {}
        dialogue = 'Dialogue: '+ splitted[1].strip()
        lines =  dialogue.split('\n')
        self._fields: list[str] = [','.join(l.split(',', 9)[:9]) for l in lines if l]
        self._dialogue: list[str] = [
            self.command_regex.sub(self._sub_commands, l.split(',', 9)[9])
            for l in lines if l]

    def _sub_commands(self, m: re.Match) -> str:
        token = f"{{format {len(self._commands)}}}"
        self._commands[token] = m.group(0)
        return token

    def _restore_commands(self, m: re.Match) -> str:
        return self._commands.get(m.group(0), '{}')

    def get_dialogue(self):
        return self._dialogue

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        offset = self._header.count('\n') + 1
        return [offset + i for i in lines]

    def get_translation(self, translation: list[str]):
        if len(self._fields) != len(translation): raise Exception("Lines count mismatch")
        return self._header +\
            '\n'.join([
                f"{f},{self.command_regex.sub(self._restore_commands, l)}"
                for f, l in zip(self._fields, translation)])

class SrtTranslationFile(TranslationFile):

    def __init__(self, text: str):
        blocks = text.split('\n\n')
        splitted = [b.split('\n', 2) for b in blocks]
        self._dialogue: list[str] = [l[-1] for l in splitted]
        self._timestamps: list[str] = ['\n'.join(l[:2]) for l in splitted]

    def get_dialogue(self):
        return self._dialogue

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        return list(lines)

    def get_translation(self, translation: list[str]):
        if len(self._timestamps) != len(translation): raise Exception("Lines count mismatch")
        return '\n\n'.join([f"{f}\n{l}" for f, l in zip(self._timestamps, translation)])