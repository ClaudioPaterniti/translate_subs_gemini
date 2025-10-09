import os
import asyncio
import traceback

from math import ceil
from itertools import chain

from src.models import *
from src.rate_limiter import RateLimitedLLM
from src.srt_parser import SrtTranslationFile
from src.ass_parser import AssTranslationFile
from src.chunker import ChunkedTranslation, split_chunks, flatten_chunks
from src.logger import Logger

class FileTranslationTask:

    def __init__(
            self,
            path: str,
            out_path: str,
            llm: RateLimitedLLM,
            chunk_size: int,
            max_chars: int,
            reduced_chars: int,
            ass_settings: AssSettings,
            logger: Logger):
        self.path = path
        self.out_path = out_path
        self.llm = llm
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.reduced_chars = reduced_chars
        self.ass_settings = ass_settings
        self.logger = logger


        _, self._filename = os.path.split(self.path)

    async def __call__(self):
        try:
            sub_file = self._load_file()
            dialogue = sub_file.get_dialogue()
            translation = ChunkedTranslation(dialogue, self.chunk_size, self.logger)
            chunks = split_chunks(translation.chunks, self.max_chars)

            chunk_id = f"{self._filename} - part {{}}" if len(chunks) > 1 else self._filename
            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = [
                        tg.create_task(self._translate_chunk(chunk_id.format(i+1), chunk))
                        for i, chunk in enumerate(chunks)
                    ]
            except* Exception as exs:
                raise exs.exceptions[0]

            translation.add_translation(flatten_chunks([t.result() for t in tasks]))
            await self._handle_misalignments(translation, sub_file)

            translated = sub_file.get_translation(translation.get_translated_dialogue())
            with open(self.out_path, 'w+', encoding='utf-8') as fp:
                fp.write(translated)

            self.logger.success(f"{self._filename}: Generated {self.out_path}", save=True)

        except Exception as ex:
            self.logger.error(f"{self._filename}: {ex}", save=True)
            self.logger.debug(traceback.format_exc())

    def _load_file(self) -> TranslationFile:
        with open(self.path, 'r', encoding='utf-8') as fp:
            if self.path.endswith('.ass'):
                return AssTranslationFile(fp.read(), self.ass_settings)
            else:
                return SrtTranslationFile(fp.read())

    async def _handle_misalignments(self, subs: ChunkedTranslation, sub_file: TranslationFile):
        misaligned = subs.get_misaligned_chunks()
        if misaligned.chunks:
            text = misaligned.model_dump_json(indent=2)
            if len(text) > self.max_chars:
                raise MisalignmentException(f"{self._filename}: result translation did no match original structure")

            self.logger.warning(
                f"{self._filename}: result translation has some line misalignment, trying correction")

            result = await self.llm.structured_output(
                f"{self._filename} corrections", text, DialogueChunks)
            subs.apply_corrections(result)

        misalignments = [self.chunk_size*i for i in subs.misaligned_chunks]
        misalignments_warnings = [
            f"{l}-{l + self.chunk_size}"
            for l in sub_file.map_dialogue_lines(misalignments)]

        if misalignments:
            self.logger.warning(
                f"{self._filename} - misilignments at lines [{', '.join(misalignments_warnings)}]",
                save=True)

    async def _translate_chunk(
            self, chunk_id: str, chunks: DialogueChunks) -> DialogueChunks:
        text = chunks.model_dump_json(indent=2)
        try:
            return await self.llm.structured_output(chunk_id, text, DialogueChunks)
        except InvalidJsonException:
            if len(text) > self.reduced_chars:
                self.logger.warning(f"{chunk_id}: Gemini returned an invalid json, retrying with reduced context window")
                splitted = split_chunks(chunks, self.reduced_chars)
                async with asyncio.TaskGroup() as tg:
                    tasks = [
                        tg.create_task(self.llm.structured_output(
                            f'{chunk_id}.{i}',
                            chunk.model_dump_json(indent=2),
                            DialogueChunks))
                        for i, chunk in enumerate(splitted)]
                return flatten_chunks([t.result() for t in tasks])
            else:
                raise
