import asyncio

from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass

from src.gemini import GeminiClient
from src.ass import *

@dataclass
class LogEntry:
    utc: datetime
    tokens: int

class RateLimitedQueue:

    def __init__(self,
                 client: GeminiClient, rpm: int, tpm: int,
                 max_retries: int, wait_window: timedelta = timedelta(seconds=50)):
        self.client = client
        self.rpm = rpm
        self.tpm = tpm
        self.max_retries = max_retries
        self.wait_window = wait_window

        self._retries = 0
        self._minute_tokens = 0
        self._minute_requests = 0
        self._running = 0
        self._completed_log: deque[LogEntry] = deque()


    def _try_start(self, ass: Ass) -> bool:
        while (
                self._completed_log
                and datetime.now(tz=timezone.utc) - self._completed_log[0].utc > self.wait_window):
            self._minute_tokens -= self._completed_log.popleft().tokens
            self._minute_requests -= 1
        if (
            self._minute_requests + 1 <= self.rpm
            and self._minute_tokens + ass.translation_tokens_estimate < self.tpm
        ):
            self._minute_requests += 1
            self._minute_tokens += ass.translation_tokens_estimate
            self._running += 1
            return True
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

        while not self._try_start(ass):
            await asyncio.sleep(2)

        try:
            result = await self.client.translate_ass(ass)
            self._complete(ass)
            return result
        except RetriableException as ex:
            if self._retries < self.max_retries:
                print(f"{ass.filename}: {ex.message} - rescheduling")
                self._retries += 1
                self._complete(ass)
                return await self.queue_translation(ass)
            else:
                self._complete(ass)
                raise ex

