from pydantic import BaseModel

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentResponse

from src.models import RetriableException

class GeminiClient:

    def __init__(self, key: str, model: str, prompt: str, config: dict = None):
        self.model = model
        self.prompt = prompt
        self.config = config or {}

        self.client = genai.Client(api_key=key)


    async def ask_question(self,
            question: str, structure: BaseModel = None) -> GenerateContentResponse:

        config= self.config | {
            "response_mime_type": "application/json",
            "response_schema": structure
        } if structure is not None else self.config

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

        return response

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