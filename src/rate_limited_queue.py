import asyncio

from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass
from math import inf

from src.gemini import GeminiClient
from src.ass import *
from src import logger

@dataclass
class LogEntry:
    utc: datetime
    tokens: int

class RateLimitedQueue:

    def __init__(self,
                 client: GeminiClient, requests_per_minute: int,
                 tokens_per_minute: int, max_retries: int, max_concurrent_requests: int = None,
                 wait_window: timedelta = timedelta(seconds=50)):
        self.client = client
        self.rpm = requests_per_minute
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


    def _try_start(self, ass: Ass) -> bool:
        while (
                self._completed_log
                and datetime.now(tz=timezone.utc) - self._completed_log[0].utc > self.wait_window):
            self._minute_tokens -= self._completed_log.popleft().tokens
            self._minute_requests -= 1
        if (
            self._minute_requests < self.rpm
            and self._minute_tokens + ass.translation_tokens_estimate <= self.tpm
            and self._running < self.max_concurrent_requests
        ):
            self._minute_requests += 1
            self._minute_tokens += ass.translation_tokens_estimate
            self._running += 1
            self._waiting_warning = False
            return True
        elif self._running == 0 and not self._waiting_warning:
            wait = self.wait_window + self._completed_log[0].utc - datetime.now(tz=timezone.utc)
            logger.warning(f"Waiting {wait.seconds} seconds for rate limits")
            self._waiting_warning = True
        return False

    def _complete(self, ass: Ass):
        self._running -= 1
        self._completed_log.append(
            LogEntry(datetime.now(tz=timezone.utc), ass.translation_tokens_estimate))

    async def queue_translation(self, ass: Ass) -> Ass:
        if not ass.translation_tokens_estimate:
            ass.translation_tokens_estimate = int(self.client.estimate_question_tokens(ass)*2.1)

        if ass.translation_tokens_estimate > self.tpm:
            raise Exception(f"{ass.filename}: Cannot translate, text too long")

        queued = False
        while not self._try_start(ass):
            if not queued:
                queued = True
                logger.info(f"{ass.filename}: in queue")
            await asyncio.sleep(2)

        try:
            result = await self.client.translate_ass(ass)
            self._complete(ass)
            return result
        except RetriableException as ex:
            if self._retries < self.max_retries:
                logger.warning(f"{ass.filename}: rescheduling after - {ex.message}")
                self._retries += 1
                self._complete(ass)
                return await self.queue_translation(ass)
            else:
                self._complete(ass)
                raise ex

