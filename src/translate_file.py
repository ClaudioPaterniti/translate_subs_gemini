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
import src.logger as logger

class FileTranslationTask:

    def __init__(
            self,
            path: str,
            out_path: str,
            llm: RateLimitedLLM,
            chunk_lines: int,
            request_chunks: int,
            reduced_request_chunks: int,
            ass_settings: AssSettings):
        self.path = path
        self.out_path = out_path
        self.llm = llm
        self.chunk_lines = chunk_lines
        self.request_chunks = request_chunks
        self.reduced_request_chunks = reduced_request_chunks
        self.ass_settings = ass_settings


        _, self._filename = os.path.split(self.path)

    async def __call__(self):
        try:
            sub_file = self._load_file()
            dialogue = sub_file.get_dialogue()
            translation = ChunkedTranslation(dialogue, self.chunk_lines)

            result = await self._split_and_translate(
                self._filename, translation.chunks, self.request_chunks)

            translation.add_translation(result)
            await self._handle_misalignments(translation, sub_file)

            translated = sub_file.get_translation(translation.get_translated_dialogue())

            with open(self.out_path, 'w+', encoding='utf-8') as fp:
                fp.write(translated)

            logger.success(f"{self._filename}: Generated {self.out_path}", save=True)

        except Exception as ex:
            logger.error(f"{self._filename}: {ex}", save=True)
            logger.debug(traceback.format_exc())

    def _load_file(self) -> TranslationFile:
        with open(self.path, 'r', encoding='utf-8') as fp:
            if self.path.endswith('.ass'):
                return AssTranslationFile(fp.read(), self.ass_settings)
            else:
                return SrtTranslationFile(fp.read())

    async def _split_and_translate(
            self, chunk_id: str, chunks: DialogueChunks, request_chunks: int) -> DialogueChunks:

        splitted = split_chunks(chunks, request_chunks)
        chunk_id = f"{chunk_id}.{{}}" if len(splitted) > 1 else chunk_id
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._translate_block(chunk_id.format(i+1), chunk))
                    for i, chunk in enumerate(splitted)]

        except* Exception as exs:
            raise exs.exceptions[0]

        return flatten_chunks([t.result() for t in tasks])

    async def _translate_block(
            self, chunk_id: str, chunks: DialogueChunks) -> DialogueChunks:
        try:
            text = chunks.model_dump_json(indent=2)
            resp: DialogueChunks = await self.llm.structured_output(chunk_id, text, DialogueChunks)
            if len(resp.chunks) != len(chunks.chunks): raise InvalidJsonException("Number of translated chunks does not match")
            return resp
        except InvalidJsonException:
            if len(chunks.chunks) > self.reduced_request_chunks:
                logger.warning(f"{chunk_id}: Gemini returned an invalid json, retrying with reduced context window")
                return await self._split_and_translate(chunk_id, chunks, self.reduced_request_chunks)
            else:
                raise

    async def _handle_misalignments(self, subs: ChunkedTranslation, sub_file: TranslationFile):
        misaligned = subs.get_misaligned_chunks()
        if misaligned.chunks:
            if len(misaligned.chunks) > self.request_chunks:
                raise MisalignmentException(f"{self._filename}: result translation did no match original structure")

            logger.warning(
                f"{self._filename}: result translation has some line misalignment, trying correction")

            text = misaligned.model_dump_json(indent=2)
            result = await self.llm.structured_output(
                f"{self._filename} corrections", text, DialogueChunks)
            subs.apply_corrections(result)

        misalignments = [self.chunk_lines*i for i in subs.misaligned_chunks]
        misalignments_warnings = [
            f"{l}-{l + self.chunk_lines}"
            for l in sub_file.map_dialogue_lines(misalignments)]

        if misalignments:
            logger.warning(
                f"{self._filename} - misilignments at lines [{', '.join(misalignments_warnings)}]",
                save=True)
