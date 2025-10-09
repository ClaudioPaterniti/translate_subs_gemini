import re
from typing import Optional, Any, ClassVar, Protocol

from pydantic import BaseModel, Field

from src.logger import Logger

class AssIgnore(BaseModel):
    field: str
    values: set[str]
    _field_i: int


class AssSettings(BaseModel):
    ignore: Optional[list[AssIgnore]] = None


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
    ass_settings: AssSettings

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

    def __init__(self, text: str, settings: AssSettings):
        splitted = text.strip().split('[Events]', 1)
        subs = [s for s in splitted[1].split('\n') if s.strip()]
        self._header = splitted[0] + '[Events]\n' + subs[0]
        self._format = {
            s.strip().lower(): i
            for i, s in enumerate(subs[0].replace('Format:', '').split(','))
        }

        for rule in settings.ignore:
            rule._field_i = self._format.get(rule.field.lower())
        self._ignore = [r for r in settings.ignore if r._field_i is not None]

        self._fields: list[str]
        self._commands: dict[str, str] = {}
        self._ignored: list[str] = []
        self._ignored_i: list[int] = []

        sections = self._apply_ignores(subs[1:])

        self._fields: list[str] = [f"{l[0]}:" + ','.join(l[1:len(self._format)]) for l in sections]
        self._dialogue: list[str] = [
            self.command_regex.sub(self._sub_commands, l[len(self._format)])
            for l in sections]


    def _apply_ignores(self, lines: list[str]) -> list[list[str]]:
        subs = []
        for i, l in enumerate(lines):
            ignored = False
            event, value = l.split(':', 1)
            fields = value.split(',', len(self._format)-1)
            if event.strip().lower() == 'comment': ignored = True
            else:
                for rule in self._ignore:
                    if fields[rule._field_i].strip() in rule.values: ignored = True

            if ignored:
                self._ignored_i.append(i)
                self._ignored.append(l)
            else:
                subs.append([event] + fields)

        return subs

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
        final = []
        i = 0
        for l in lines:
            t = l + i
            while  i < len(self._ignored_i) and self._ignored_i[i] <= t:
                i += 1
                t += 1
            final.append(offset + t)
        return final

    def get_translation(self, translation: list[str]):
        if len(self._fields) != len(translation): raise Exception("Lines count mismatch")
        lines = [
            f"{f},{self.command_regex.sub(self._restore_commands, l)}"
            for f, l in zip(self._fields, translation)]
        final = []
        j = h = 0
        for i in range(len(self._ignored) + len(lines)) :
            if j < len(self._ignored) and i == self._ignored_i[j]:
                final.append(self._ignored[j])
                j += 1
            else:
                final.append(lines[h])
                h += 1
        return  '\n'.join([self._header] + final)

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