import os
from typing import Optional, Any

from pydantic import BaseModel

class Config(BaseModel):
    model: str
    requests_per_minutes: int
    token_per_minutes: int
    max_concurrent_requests: Optional[int] = None
    content_config: dict[str, Any]
    outfile_suffix: str
    max_retries: int

class MisalignmentException(Exception):
    pass

class RetriableException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class Dialogue(BaseModel):
    lines: list[str]

class Ass(BaseModel):
    path: str
    filename: str
    header: str
    fields: list[str]
    dialogue: Dialogue
    translation_tokens_estimate: Optional[int] = None

    def _strip_dialogue(self):
        s = e = 0
        while not self.dialogue.lines[s]:
            s += 1
        while not self.dialogue.lines[e]:
            e -= 1
        self.dialogue.lines = self.dialogue.lines[s: e or None]

    def to_file(self, path = None) -> str:
        self._strip_dialogue()
        if (len(self.fields) != len(self.dialogue.lines)):
            raise MisalignmentException("Dialog lines do not match fields")
        text =  self.header +\
            '\n'.join([f"{f},{l}" for f, l in zip(self.fields, self.dialogue.lines)])
        with open(
                path or os.path.join(self.path, self.filename + self.ext),
                'w+', encoding='utf-8-sig') as fp:
            fp.write(text)

    @staticmethod
    def from_file(file_path: str) -> 'Ass':
        path, file = os.path.split(file_path)
        filename, ext = os.path.splitext(file)

        with open(file_path, 'r', encoding='utf-8') as fp:
            text = fp.read()

        splitted = text.split('Dialogue:', 1)
        header = splitted[0]
        text = 'Dialogue:'+ splitted[1].strip()
        fields = [','.join(l.split(',', 10)[:9]) for l in text.split('\n')]
        dialogue = Dialogue(lines=[''.join(l.split(',', 10)[9:]) for l in text.split('\n')])

        return Ass(
            path = path,
            filename=filename,
            header=header,
            fields=fields,
            dialogue=dialogue)

    def __hash__(self):
        return hash(self.filename)