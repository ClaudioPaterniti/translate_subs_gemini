import os
from typing import Optional

from pydantic import BaseModel

class MisalignmentException(Exception):
    pass

class Dialogue(BaseModel):
    lines: list[str]

class Ass(BaseModel):
    path: str
    filename: str
    ext: str
    header: str
    fields: list[str]
    dialogue: Dialogue
    translation_tokens_estimate: Optional[int] = None

    def to_string(self) -> str:
        if (len(self.fields) != len(self.dialogue.lines)):
            raise MisalignmentException("Dialog lines do not match fields")
        return self.header +\
            '\n'.join([f"{f},{l}" for f, l in zip(self.fields, self.dialogue.lines)])

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
            ext=ext,
            header=header,
            fields=fields,
            dialogue=dialogue)

    def __hash__(self):
        return hash(self.filename)

class RetriableException(Exception):
    def __init__(self, ass: Ass, message: str):
        self.ass = ass
        self.message = message
        super().__init__(message)