import os
import sys
import time

from google import genai
from google.genai.errors import ClientError

file_paths = [f for f in sys.argv[1:] if f.endswith('.ass')]

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

    with open(file_path, 'r', encoding='utf-8-sig') as subs_fp:
        text = subs_fp.read()

    splitted = text.split('Dialogue:', 1)
    header = splitted[0]
    dialogue = 'Dialogue:'+ splitted[1]

    lines = [''.join(l.split(',', 10)[9:]) for l in dialogue.split('\n')]

    question = '\n'.join([prompt] + lines)

    response_cache_file = os.path.join('data', f'{filename}_reponse.txt')

    if os.path.isfile(response_cache_file):
        print(f"Reading response from cache")
        with open(response_cache_file, 'r', encoding='utf-8') as fp:
            translated = fp .read()
    else:
        print(f"Calling Gemini")
        retries = 2
        while retries > 0:
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash", contents=question
                )
                break
            except ClientError as ex:
                if ex.status == 'RESOURCE_EXHAUSTED' and retries > 0:
                    retries -= 1
                    wait_time = 70 * (2 - retries)
                    print(f"Gemini returned RESOURCE_EXHAUSTED, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    print("Call to Gemini failed")
                    raise ex

        translated = response.text

        os.makedirs(os.path.dirname(response_cache_file), exist_ok=True)
        print(f"Saving response cache {response_cache_file}")
        with open(response_cache_file, 'w+', encoding='utf-8') as fp:
            fp.write(translated)

    translated_file = os.path.join(path, f'{filename}_ita{ext}')
    print(f"Generating final {translated_file}")

    translated_lines = translated.split('\n')

    fields = [','.join(l.split(',', 10)[:9]) for l in dialogue.split('\n')]

    final = header + '\n'.join([f"{f},{l}" for f, l in zip(fields, translated_lines)])

    with open(translated_file, 'w+', encoding='utf-8-sig') as fp:
        fp.write(final)
