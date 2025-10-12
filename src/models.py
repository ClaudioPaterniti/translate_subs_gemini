from typing import Optional, Any, Protocol

from pydantic import BaseModel

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
    lines_per_chunk: int = 30
    chunks_per_request: int = 10
    reduced_chunks_per_request: int = 5
    requests_per_minutes: int = 15
    token_per_minutes: int = 1000000
    max_concurrent_requests: Optional[int] = None
    content_config: dict[str, Any] = {}
    max_retries: int = 50
    ass_settings: AssSettings
    debug: bool = False

class MisalignmentException(Exception):
    pass

class RetriableException(Exception):
    pass

class InvalidJsonException(Exception):
    pass

class TranslationFile(Protocol):
    def get_dialogue(self) -> list[str]:
        """Return a simple dialogue as a list of lines"""
        ...

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        """Map simple dialogue line numbers to the corresponding lines in the final file"""
        ...

    def get_translation(self, translation: list[str]) -> str:
        """Recompose the final file structure with the translated dialogue"""
        ...

class DialogueChunk(BaseModel):
    from_line: int
    to_line: int
    dialogue: list[str]
    _translated: list[str] = None

class DialogueChunks(BaseModel):
    chunks: list[DialogueChunk]