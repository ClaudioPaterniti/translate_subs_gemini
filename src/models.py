import os
import re
from typing import Optional, Any, ClassVar
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src import logger

class Config(BaseModel):
    original_language: str
    translate_to: str
    outfile_suffix: str
    model: str = "gemini-2.0-flash"
    dialogue_chunks_size: int = 10
    max_context_window: int = 300000
    reduced_context_window: int = 8000
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
    _metadata: Any
    _translated: list[str]

class DialogueChunks(BaseModel):
    chunks: list[DialogueChunk]

class SubsTranslation(BaseModel):
    path: str
    folder: str
    filename: str
    out_path: str
    chunk_size: int = 10
    chunks: DialogueChunks = DialogueChunks(chunks=[])
    misaligned_chunks: list[int] = []
    translation_tokens_estimate: int = None

    def _load(self):
        raise NotImplementedError()

    def _dump(self) -> str:
        raise NotImplementedError()

    def get_chunks(self) -> DialogueChunks:
        if not self.chunks.chunks:
            self._load()
        return self.chunks

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

    def to_file(self, path = None) -> str:
        out_path = path or self.out_path
        text = self._dump()
        with open(
                out_path,
                'w+', encoding='utf-8-sig') as fp:
            fp.write(text)
        logger.success(f"{self.filename}: Generated {self.out_path}", True)
        del self.chunks

    @staticmethod
    def from_file(file_path: str, out_path: str, chunk_size: int = 10) -> 'SubsTranslation':
        folder, filename = os.path.split(file_path)
        if filename.endswith('.ass'):
            return AssTranslation(
                path = file_path, folder=folder,
                filename=filename, out_path=out_path, chunk_size=chunk_size
            )
        else:
            return SrtTranslation(
                path = file_path, folder=folder,
                filename=filename, out_path=out_path, chunk_size=chunk_size
            )

class AssTranslation(SubsTranslation):
    header: str = None
    command_regex: ClassVar[re.Pattern] = re.compile(r'\{[^{}]+\}')

    @dataclass
    class ChunkMetadata:
        fields: list[str]
        commands: dict[str, str]

        def sub_commands(self, m: re.Match) -> str:
            token = f"{{format {len(self.commands)}}}"
            self.commands[token] = m.group(0)
            return token

        def restore_commands(self, m: re.Match) -> str:
            return self.commands.get(m.group(0), '{}')

    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            text = fp.read()

        splitted = text.split('Dialogue:', 1)
        self.header = splitted[0]
        text = 'Dialogue: '+ splitted[1].strip()
        splitted =  text.split('\n')
        for i in range(0, len(splitted), self.chunk_size):
            self.chunks.chunks.append(self._lines_to_chunk(i, splitted[i:i+self.chunk_size]))

    def _lines_to_chunk(self, from_line: int, lines: list[str]) -> 'DialogueChunk':
        metadata = AssTranslation.ChunkMetadata(
            fields=[','.join(l.split(',', 9)[:9]) for l in lines if l],
            commands= {})
        chunk = DialogueChunk(
            from_line=from_line,
            to_line=from_line + len(lines) - 1,
            dialogue= [
                self.command_regex.sub(metadata.sub_commands, l.split(',', 9)[9])
                for l in lines if l]
        )
        chunk._metadata = metadata
        return chunk

    def _dump(self) -> str:
        if self.misaligned_chunks:
            offset = self.header.count('\n') + 1
            misalignments = [f"{offset + self.chunk_size*i}-{offset + self.chunk_size*(i+1)}" for i in self.misaligned_chunks]
            logger.warning(f"{self.filename} - misilignments at lines [{', '.join(misalignments)}]", True)

        return self.header +\
            '\n'.join([self._chunk_to_lines(c) for c in self.chunks.chunks])

    def _chunk_to_lines(self, chunk: DialogueChunk) -> str:
         lines = [
             self.command_regex.sub(chunk._metadata.restore_commands, l) for l in chunk._translated
         ]
         return '\n'.join([f"{f},{l}" for f, l in zip(chunk._metadata.fields, lines)])


class SrtTranslation(SubsTranslation):

    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            text = fp.read()

        blocks = text.split('\n\n')
        splitted = [b.split('\n', 2) for b in blocks]

        for i in range(0, len(splitted), self.chunk_size):
            self.chunks.chunks.append(self._lines_to_chunk(i, splitted[i:i+self.chunk_size]))

    def _lines_to_chunk(self, from_line: int, splitted: list[list[str]]) -> 'DialogueChunk':
        chunk = DialogueChunk(
            from_line=from_line,
            to_line=from_line + len(splitted) - 1,
            dialogue = [l[-1] for l in splitted]
        )
        chunk._metadata = ['\n'.join(l[:2]) for l in splitted]
        return chunk

    def _dump(self) -> str:
        if self.misaligned_chunks:
            misalignments = [f"{self.chunk_size*i}-{self.chunk_size*(i+1)}" for i in self.misaligned_chunks]
            logger.warning(f"{self.filename} - misilignments at blocks [{', '.join(misalignments)}]", True)

        return '\n\n'.join([self._chunk_to_blocks(c) for c in self.chunks.chunks])

    def _chunk_to_blocks(self, chunk: DialogueChunk) -> str:
         return '\n\n'.join([f"{f}\n{l}" for f, l in zip(chunk._metadata, chunk._translated)])