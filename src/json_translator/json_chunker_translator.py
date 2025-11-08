import asyncio
import traceback
from string import Template

from src.models import *
from src.rate_limiter import RateLimitedLLM
from src.json_translator.json_chunker import ChunkedTranslation, split_chunks, flatten_chunks
import src.logger as logger

from importlib import resources

prompt = Template(resources.files(__package__).joinpath("prompt.md").read_text())

class JsonChunkerTranslator:

    def __init__(
            self,
            llm: RateLimitedLLM,
            chunk_lines: int,
            request_chunks: int,
            reduced_request_chunks: int):
        self.llm = llm
        self.chunk_lines = chunk_lines
        self.request_chunks = request_chunks
        self.reduced_request_chunks = reduced_request_chunks

    async def __call__(self, filename: str, dialogue: list[str]) -> TranslationOutput:
        try:
            translation = ChunkedTranslation(dialogue, self.chunk_lines)

            result = await self._split_and_translate(
                filename, translation.chunks, self.request_chunks)

            translation.add_translation(result)
            misalignments = await self._handle_misalignments(filename, translation)

            translated =  translation.get_translated_dialogue()

            return TranslationOutput(filename, translated, misalignments)

        except Exception as ex:
            logger.error(f"{filename}: {ex}", save=True)
            logger.debug(traceback.format_exc())

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
            json_str = chunks.model_dump_json(indent=2)
            text = prompt.substitute(lines_per_chunk= self.chunk_lines, json= json_str)
            resp: DialogueChunks = await self.llm.structured_output(chunk_id, text, DialogueChunks)
            if len(resp.chunks) != len(chunks.chunks): raise InvalidJsonException("Number of translated chunks does not match")
            return resp
        except InvalidJsonException:
            if len(chunks.chunks) > self.reduced_request_chunks:
                logger.warning(f"{chunk_id}: Gemini returned an invalid json, retrying with reduced context window")
                return await self._split_and_translate(chunk_id, chunks, self.reduced_request_chunks)
            else:
                raise

    async def _handle_misalignments(
            self, filename: str, subs: ChunkedTranslation) -> list[tuple[int, int]]:
        misaligned = subs.get_misaligned_chunks()
        if misaligned.chunks:
            if len(misaligned.chunks) > self.request_chunks:
                raise MisalignmentException(f"{filename}: result translation did no match original structure")

            logger.warning(
                f"{filename}: result translation has some line misalignment, trying correction")

            json_str = misaligned.model_dump_json(indent=2)
            text = prompt.substitute(lines_per_chunk= self.chunk_lines, json= json_str)
            result = await self.llm.structured_output(
                f"{filename} corrections", text, DialogueChunks)
            subs.apply_corrections(result)

        misalignments = [self.chunk_lines*i for i in subs.misaligned_chunks]
        return [(l, l + self.chunk_lines) for l in misalignments]