import os
from typing import Optional, Any

from pydantic import BaseModel, Field

from src import logger

class Config(BaseModel):
    original_language: str
    translate_to: str
    outfile_suffix: str
    model: str = "gemini-2.0-flash"
    dialogue_chunks_size: int = 10
    max_context_window: int = 300000
    requests_per_minutes: int = 15
    token_per_minutes: int = 1000000
    max_concurrent_requests: Optional[int] = None
    content_config: dict[str, Any] = {}
    max_retries: int = 50

class MisalignmentException(Exception):
    pass

class RetriableException(Exception):
    pass

class DialogueChunk(BaseModel):
    from_line: int
    to_line: int
    dialogue: list[str]
    _fields: list[str]
    _translated: list[str]

    @staticmethod
    def from_ass_lines(from_line: int, lines: list[str]) -> 'DialogueChunk':
        chunk = DialogueChunk(
            from_line=from_line,
            to_line=from_line + len(lines) - 1,
            dialogue = [''.join(l.split(',', 10)[9:]) for l in lines if l]
        )
        chunk._fields = [','.join(l.split(',', 10)[:9]) for l in lines if l]
        return chunk

    def get_translated_ass_line(self) -> str:
         return '\n'.join([f"{f},{l}" for f, l in zip(self._fields, self._translated)])

class DialogChunks(BaseModel):
    chunks: list[DialogueChunk]

class AssTranslation(BaseModel):
    path: str
    folder: str
    filename: str
    out_path: str
    chunk_size: int = 10
    header: str = None
    chunks: DialogChunks = DialogChunks(chunks=[])
    misaligned_chunks: list[int] = []
    translation_tokens_estimate: int = None

    def _load(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            text = fp.read()

        splitted = text.split('Dialogue:', 1)
        self.header = splitted[0]
        text = 'Dialogue: '+ splitted[1].strip()
        splitted =  text.split('\n')
        for i in range(0, len(splitted), self.chunk_size):
            self.chunks.chunks.append(DialogueChunk.from_ass_lines(i, splitted[i:i+self.chunk_size]))

    def get_chunks(self) -> DialogChunks:
        if not self.header:
            self._load()
        return self.chunks

    def add_translation(self, chunks: DialogChunks):
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

    def get_misaligned_chunks(self) -> DialogChunks:
        return DialogChunks(chunks=[self.chunks.chunks[i] for i in self.misaligned_chunks])

    def apply_corrections(self, chunks: DialogChunks):
        if len(chunks.chunks) != len(self.misaligned_chunks):
            return
        temp_misaligned = []
        for i in self.misaligned_chunks:
            original, translated = self.chunks.chunks[i], chunks.chunks[i]
            if len(original.dialogue) != len(translated.dialogue):
                temp_misaligned.append(i)
                diff = len(original.dialogue) - len(translated.dialogue)
                if diff > 0:
                    translated.dialogue.extend(original.dialogue[-diff:])
                else:
                    translated.dialogue = translated.dialogue[:-diff]
            original._translated = translated.dialogue
        self.misaligned_chunks = temp_misaligned

    def to_file(self, path = None) -> str:
        out_path = path or self.out_path
        message = f"{self.filename}: Generated {self.out_path}"
        if self.misaligned_chunks:
            misalignments = [f"{self.chunk_size*i}-{self.chunk_size*(i+1)}" for i in self.misaligned_chunks]
            logger.warning(message + f" - misilignments at lines [{', '.join(misalignments)}]", True)
        else:
            logger.success(message, True)
        text =  self.header +\
            '\n'.join([c.get_translated_ass_line() for c in self.chunks.chunks])
        with open(
                out_path,
                'w+', encoding='utf-8-sig') as fp:
            fp.write(text)

    @staticmethod
    def from_file(file_path: str, out_path: str, chunk_size: int = 10) -> 'AssTranslation':
        folder, filename = os.path.split(file_path)
        return AssTranslation(
            path = file_path, folder=folder,
            filename=filename, out_path=out_path, chunk_size=chunk_size)