import asyncio
import traceback
import re

from string import Template
from math import ceil
from itertools import chain

from src.models import *
from src.rate_limiter import RateLimitedLLM
import src.logger as logger

from importlib import resources

prompt = Template(resources.files(__package__).joinpath("prompt.md").read_text())
split_regex = re.compile(r'\s*Line\s+\d+\s*-\s*')

class TextTranslator:

    def __init__(
            self,
            llm: RateLimitedLLM,
            chunk_lines: int):
        self.llm = llm
        self.chunk_lines = chunk_lines

    async def __call__(self, filename: str, dialogue: list[str]) -> TranslationOutput:
        try:

            translated = await self._split_and_translate(
                filename, dialogue, self.chunk_lines)

            return TranslationOutput(filename, translated)

        except Exception as ex:
            logger.error(f"{filename}: {ex}", save=True)
            logger.debug(traceback.format_exc())

    async def _split_and_translate(
            self, chunk_id: str, dialogue: list[str], chunk_lines: int) -> list[str]:

        blocks = ceil(len(dialogue)/chunk_lines)
        q, r = divmod(len(dialogue), blocks)
        chunks = [
            dialogue[i*q + min(i, r):(i+1)*q + min(i+1, r)]
            for i in range(blocks)]

        chunk_id = f"{chunk_id}.{{}}" if len(chunks) > 1 else chunk_id
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._translate_block(chunk_id.format(i+1), chunk))
                    for i, chunk in enumerate(chunks)]

        except* Exception as exs:
            raise exs.exceptions[0]

        return list(chain.from_iterable(t.result() for t in tasks))

    async def _translate_block(
            self, chunk_id: str, dialogue: list[str]) -> list[str]:
        text = '\n'.join([f"Line {i} - {line}" for i, line in enumerate(dialogue)])
        question = prompt.substitute(lines_per_chunk= self.chunk_lines, text= text)
        resp = await self.llm.ask(chunk_id, question)
        lines = [line for line in split_regex.split(resp)][1:]
        if len(lines) != len(dialogue):
            if len(dialogue) > self.chunk_lines/2:
                logger.warning(f"{chunk_id}: response lines number does not match original dialogue, retrying with reduced context")
                return await self._split_and_translate(chunk_id, dialogue, self.chunk_lines/2)
            else:
                raise MisalignmentException(f"{chunk_id}: response lines number does not match original dialogue")
        return lines