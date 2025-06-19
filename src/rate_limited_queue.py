import asyncio

from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass
from math import inf, ceil
from itertools import chain

from src.gemini import GeminiClient
from src.models import *
from src import logger

@dataclass
class LogEntry:
    utc: datetime
    tokens: int

class RateLimitedQueue:

    def __init__(self,
            client: GeminiClient, max_context: int, reduced_context: int,
            requests_per_minute: int, tokens_per_minute: int, max_retries: int,
            max_concurrent_requests: int = None, wait_window: timedelta = timedelta(seconds=50)):

        self.client = client
        self.rpm = requests_per_minute
        self.max_context = max_context
        self.reduced_context = reduced_context
        self.tpm = tokens_per_minute
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests or inf
        self.wait_window = wait_window

        self._retries = 0
        self._minute_tokens = 0
        self._minute_requests = 0
        self._running = 0
        self._completed_log: deque[LogEntry] = deque()
        self._waiting_warning = False

    def _clean_window(self):
        while (
                self._completed_log
                and datetime.now(tz=timezone.utc) - self._completed_log[0].utc > self.wait_window):
            self._minute_tokens -= self._completed_log.popleft().tokens
            self._minute_requests -= 1

    async def translate_all(self, to_translate: list[SubsTranslation]):
        async with asyncio.TaskGroup() as tg:
            for file in to_translate:
                while (
                        self._minute_requests >= self.rpm
                        or self._running >= self.max_concurrent_requests): # to not load all files in memory at the same time
                    await asyncio.sleep(2)
                    self._clean_window()
                tg.create_task(self.translate_file(file))
                await asyncio.sleep(0)

    def _flatten_chunks(self, chunks: list[DialogueChunks]) -> DialogueChunks:
        chunks = list(chain.from_iterable([c.chunks for c in chunks]))
        return DialogueChunks(chunks= chunks)

    async def translate_file(self, subs: SubsTranslation):
        try:
            chunks = subs.get_chunks()
            chunks = self._split(chunks, self.max_context)

            chunk_id = f"{subs.filename} - part {{}}" if len(chunks) > 1 else subs.filename
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._call_llm(chunk_id.format(i+1), chunk))
                         for i,chunk in enumerate(chunks)]
            subs.add_translation(self._flatten_chunks([t.result() for t in tasks]))
            await self._handle_misalignments(subs)
            subs.to_file()

        except* Exception as exs:
            logger.error(f"{subs.filename}: {exs.exceptions[0]}",  True)


    async def _handle_misalignments(self, subs: SubsTranslation):
        misaligned = subs.get_misaligned_chunks()
        if misaligned.chunks:
            text = misaligned.model_dump_json(indent=2)
            tokens = self.client.estimate_question_tokens(text)
            if tokens > self.max_context:
                raise MisalignmentException(f"{subs.filename}: result translation did no match original structure")

            logger.warning(
                f"{subs.filename}: result translation has some line misalignment, trying correction")

            result = await self._call_llm(f"{subs.filename} corrections", misaligned)
            subs.apply_corrections(result)

    def _split(self, chunks: DialogueChunks, max_context: int) -> list[DialogueChunks]:
        text = chunks.model_dump_json(indent=2)
        tokens = self.client.estimate_question_tokens(text)
        if tokens > max_context:
            split = int(ceil(tokens/max_context))
            chunks_n = int(ceil(len(chunks.chunks)/split))
            return [
                DialogueChunks(chunks=chunks.chunks[i: i + chunks_n])
                for i in range(0, len(chunks.chunks), chunks_n)]
        return [chunks]

    def _try_start(self, tokens_n: int) -> bool:

        self._clean_window()

        if (
            self._minute_requests < self.rpm
            and self._running < self.max_concurrent_requests
            and self._minute_tokens + tokens_n <= self.tpm
        ):
            self._minute_requests += 1
            self._minute_tokens += tokens_n
            self._running += 1
            self._waiting_warning = False
            return True

        elif self._running == 0 and not self._waiting_warning:
            wait = self.wait_window + self._completed_log[0].utc - datetime.now(tz=timezone.utc)
            logger.warning(f"Waiting {max(wait.seconds, 1)} seconds for rate limits")
            self._waiting_warning = True

        return False

    def _complete(self, tokens_n: int):
        self._running -= 1
        self._completed_log.append(
            LogEntry(datetime.now(tz=timezone.utc), tokens_n))

    async def _call_llm(self, chunk_id: str, chunks: DialogueChunks) -> DialogueChunks:
        text = chunks.model_dump_json(indent=2)
        tokens = int(self.client.estimate_question_tokens(text)*2.1)

        queued = False
        while not self._try_start(tokens):
            if not queued:
                queued = True
                logger.info(f"{chunk_id}: in queue")
            await asyncio.sleep(2)

        try:
            logger.info(f"{chunk_id}: calling Gemini")
            result: DialogueChunks = (await self.client.ask_question(text, DialogueChunks)).parsed
            if result is None:
                if tokens > self.reduced_context and self._retries < self.max_retries:
                    logger.warning(f"{chunk_id}: Gemini returned an invalid json, retrying with reduced context window")
                    chunks = self._split(chunks, self.reduced_context)
                    self._retries += len(chunks)
                    async with asyncio.TaskGroup() as tg:
                        tasks = [tg.create_task(self._call_llm(f'{chunk_id}.{i}', chunk))
                            for i, chunk in enumerate(chunks)]
                    result = self._flatten_chunks([t.result() for t in tasks])
                else:
                    raise InvalidJsonException("Gemini response could not be parsed")
            self._complete(tokens)
            return result
        except RetriableException as ex:
            if self._retries < self.max_retries:
                logger.warning(f"{chunk_id}: rescheduling after - {ex}")
                self._retries += 1
                self._complete(tokens)
                return await self._call_llm(chunk_id, text)
            else:
                self._complete(tokens)
                raise ex

