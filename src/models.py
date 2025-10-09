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

class TranslationFile(Protocol):
    def get_dialogue(self) -> list[str]:
        ...

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        """Returns the corresponding line number in the final file"""
        ...

    def get_translation(self, translation: list[str]) -> str:
        ...

class DialogueChunk(BaseModel):
    from_line: int
    to_line: int
    dialogue: list[str]
    _translated: list[str] = None

class DialogueChunks(BaseModel):
    chunks: list[DialogueChunk]