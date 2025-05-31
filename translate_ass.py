import os
import sys
import time

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel

class MisalignmentException(Exception):
    pass

class Dialogue(BaseModel):
    lines: list[str]

class Ass(BaseModel):
    header: str
    fields: list[str]
    dialogue: Dialogue

    def to_string(self) -> str:
        if (len(self.fields) != len(self.dialogue.lines)):
            raise MisalignmentException("Dialog lines do not match fields")
        return self.header +\
            '\n'.join([f"{f},{l}" for f, l in zip(self.fields, self.dialogue.lines)])

    @staticmethod
    def from_string(text: str) -> 'Ass':
        splitted = text.split('Dialogue:', 1)
        header = splitted[0]
        text = 'Dialogue:'+ splitted[1].strip()
        fields = [','.join(l.split(',', 10)[:9]) for l in text.split('\n')]
        dialogue = Dialogue(lines=[''.join(l.split(',', 10)[9:]) for l in text.split('\n')])
        return Ass(header=header, fields=fields, dialogue=dialogue)

def ask_gemini_with_retry(client: genai.Client, question: str, retries: int = 2) -> GenerateContentResponse:
    while retries > 0:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents=question,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Dialogue,
                }
            )
            break
        except (ClientError, ServerError) as ex:
            if ex.status in {'RESOURCE_EXHAUSTED', '503 UNAVAILABLE'} and retries > 0:
                retries -= 1
                wait_time = 70 * (2 - retries)
                print(f"Gemini returned {ex.status}, retrying in {wait_time} seconds")
                time.sleep(wait_time)
            else:
                print("Call to Gemini failed")
                raise ex
    return response

def translate_ass(client: genai.Client, text: str, prompt: str) -> Ass:
    ass = Ass.from_string(text)

    question = prompt + '\n' + ass.dialogue.model_dump_json(indent=2)

    print(f"Calling Gemini")
    response = ask_gemini_with_retry(client, question)
    translated: Dialogue = response.parsed
    ass.dialogue = translated

    return ass

if __name__ == '__main__':

    suffix = '_ita'
    file_paths = [f for f in sys.argv[1:] if f.endswith('.ass') and not f.endswith(f'{suffix}.ass')]

    with (
            open('prompt.txt', 'r') as prompt_fp,
            open('gemini.key', 'r') as key_fp,
        ):
        prompt = prompt_fp.read()
        key = key_fp.read()

    client = genai.Client(api_key=key)

    for file_path in file_paths:
        print(f'Translating {file_path}')
        path, file = os.path.split(file_path)
        filename, ext = os.path.splitext(file)

        response_cache_file = os.path.join('data', f'{filename}_cache.json')

        if os.path.isfile(response_cache_file):
            print(f"Reading response from cache")
            with open(response_cache_file, 'r', encoding='utf-8') as fp:
                translated = Ass.model_validate_json(fp.read())
        else:
            with open(file_path, 'r', encoding='utf-8-sig') as subs_fp:
                text = subs_fp.read()

            translated = translate_ass(client, text, prompt)
            os.makedirs(os.path.dirname(response_cache_file), exist_ok=True)
            print(f"Saving response cache {response_cache_file}")
            with open(response_cache_file, 'w+', encoding='utf-8') as fp:
                fp.write(translated.model_dump_json(indent=2))

        try:
            final = translated.to_string()
        except MisalignmentException as ex:
                print(ex.message)
                continue

        translated_file = os.path.join(path, f'{filename}{suffix}{ext}')
        print(f"Generating final {translated_file}")
        with open(translated_file, 'w+', encoding='utf-8-sig') as fp:
            fp.write(final)