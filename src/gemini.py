from pydantic import BaseModel
from typing import TypeVar

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentResponse

from src.models import RetriableException, InvalidJsonException
from src.logger import Logger


Structure = TypeVar('Structure', bound=BaseModel)

class GeminiClient:

    def __init__(self,
            key: str, model: str, prompt: str, config: dict = None, logger: Logger = None):
        self.model = model
        self.prompt = prompt
        self.config = config or {}
        self.logger = logger or Logger()

        self.client = genai.Client(api_key=key)


    async def structured_output(self, question: str, structure: Structure) -> Structure:

        config= self.config | {
            "response_mime_type": "application/json",
            "response_schema": structure
        }

        try:
            full_question = self.prompt + '\n' + question
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=full_question,
                config=config
            )
        except (ClientError, ServerError) as ex:
            if ex.status in {'RESOURCE_EXHAUSTED', 'UNAVAILABLE'}:
                raise RetriableException(ex.message)
            else:
                raise ex

        if response.parsed is None:
            raise InvalidJsonException("Gemini response could not be parsed")

        return response.parsed

    async def compute_question_tokens(self, question: str) -> int:
        question = self.prompt + '\n' + question
        response = await self.client.aio.models.count_tokens(
            model=self.model,
            contents=question,
        )
        return response.total_tokens

    def estimate_question_tokens(self, question: str) -> int:
        question = self.prompt + '\n' + question
        return int(len(question)*0.5)