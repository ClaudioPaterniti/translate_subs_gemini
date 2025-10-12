import asyncio

from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass
from math import inf

from src.gemini import GeminiClient, Structure
from src.models import *
from src.logger import Logger

@dataclass
class LogEntry:
    utc: datetime
    tokens: int

class RateLimitedLLM:

    def __init__(self,
            client: GeminiClient,
            requests_per_minute: int,
            tokens_per_minute: int,
            max_retries: int,
            max_concurrent_requests: int = None,
            wait_window: timedelta = timedelta(seconds=60),
            logger: Logger = None):

        self.client = client
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests or inf
        self.wait_window = wait_window
        self.logger = logger or Logger()

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
            self.logger.warning(f"Waiting {max(wait.seconds, 1)} seconds for rate limits")
            self._waiting_warning = True

        return False

    def _complete(self, tokens_n: int) -> bool:
        self._running -= 1
        self._completed_log.append(
            LogEntry(datetime.now(tz=timezone.utc), tokens_n))
        self.logger.debug(f"Completed {tokens_n} tokens")
        return True

    async def structured_output(
            self, request_id: str, text: str, structure: Structure, _retry: int = 0) -> Structure:
        tokens = int(self.client.estimate_question_tokens(text) * 2.1)

        queued = False
        complete = False
        while not self._try_start(tokens):
            if not queued:
                queued = True
                self.logger.info(f"{request_id}: in queue")
            await asyncio.sleep(2)

        try:
            self.logger.info(f"{request_id}: calling Gemini")
            return await self.client.structured_output(text, structure)

        except RetriableException as ex:
            if _retry < self.max_retries:
                self.logger.warning(f"{request_id}: rescheduling after - {ex}")
                if not complete: complete = self._complete(tokens)
                return await self.structured_output(request_id, text, structure, _retry + 1)
            else:
                raise

        finally:
            if not complete: self._complete(tokens)