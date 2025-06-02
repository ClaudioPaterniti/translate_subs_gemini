from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentResponse

from src.ass import *

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
        response = await self.client.aio.models.generate_content(
            model=self.model, contents=question,
            config=config
        )
        return response

    async def translate_ass(self, ass: Ass) -> Ass:
        question = self.prompt + '\n' + ass.dialogue.model_dump_json(indent=2)

        print(f"{ass.filename}: calling gemini")
        try:
            response = await self.ask_question(question, Dialogue)
        except (ClientError, ServerError) as ex:
            if ex.status in {'RESOURCE_EXHAUSTED', 'UNAVAILABLE'}:
                raise RetriableException(ass, ex.message)
            else:
                raise ex

        translated: Dialogue = response.parsed
        out = ass.model_copy()
        out.dialogue = translated
        print(f"{ass.filename}: Gemini call terminated")
        return out

    async def compute_question_tokens(self, ass: Ass) -> int:
        question = self.prompt + '\n' + ass.dialogue.model_dump_json(indent=2)
        response = await self.client.aio.models.count_tokens(
            model=self.model,
            contents=question,
        )
        return response.total_tokens

    def estimate_question_tokens(self, ass: Ass) -> int:
        question = self.prompt + '\n' + ass.dialogue.model_dump_json(indent=2)
        return int(len(question)*0.5)