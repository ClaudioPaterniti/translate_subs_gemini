import asyncio

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentResponse

from src.ass import *


async def ask_gemini_with_retry(
        client: genai.Client, question: str, retries: int = 1) -> GenerateContentResponse:
    while retries + 1 > 0:
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash", contents=question,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Dialogue,
                }
            )
            break
        except (ClientError, ServerError) as ex:
            retries -= 1
            if ex.status in {'RESOURCE_EXHAUSTED', '503 UNAVAILABLE'} and retries > 0:
                wait_time = 70 * (2 - retries)
                print(f"Gemini returned {ex.status}, retrying in {wait_time} seconds")
                await asyncio.sleep(wait_time)
            else:
                raise ex
    return response

async def translate_ass(client: genai.Client, prompt: str, ass: Ass, retries: int = 0) -> Ass:
    question = prompt + '\n' + ass.dialogue.model_dump_json(indent=2)

    print(f"Calling Gemini for {ass.filename}")
    response = await ask_gemini_with_retry(client, question, retries)
    translated: Dialogue = response.parsed
    out = ass.model_copy()
    out.dialogue = translated
    print(f"Gemini call terminated for {ass.filename}")
    return out

async def compute_question_tokens(client: genai.Client, prompt: str, ass: Ass) -> int:
    question = prompt + '\n' + ass.dialogue.model_dump_json(indent=2)
    response = await client.aio.models.count_tokens(
        model='gemini-2.0-flash',
        contents=question,
    )
    return response.total_tokens

def estimate_question_tokens(prompt: str, ass: Ass) -> int:
    question = prompt + '\n' + ass.dialogue.model_dump_json(indent=2)
    return int(len(question)*0.5)