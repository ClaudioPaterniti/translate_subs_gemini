import os
import sys
import asyncio
import math

from collections.abc import Coroutine

import src.gemini as gemini

from src.ass import *

RPM = 10
TPM = 250_000
wait_seconds = 60
suffix = '_ita'
script_path, _ = os.path.split(__file__)

async def translate_and_reschedule(tasks: dict[Ass, Coroutine], reschedule: bool = True) -> list[Ass]:
    print(f"\nExecuting {len(tasks)} translations\n")
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    failed = 0; rescheduled = []
    for ass, result in zip(tasks.keys(), results):
        if isinstance(result, BaseException):
            if reschedule and result.status in {'RESOURCE_EXHAUSTED', 'UNAVAILABLE'}:
                print(f"{ass.filename} translation failed: {result.status}, rescheduling for next batch")
                rescheduled.append(ass)
            else:
                print(f"\n\n{ass.filename} translation failed: {result.status} {result.message}\n\n")
                failed += 1
        else:
            response_cache_file = os.path.join(script_path, 'data', 'cache', f'{result.filename}_cache.json')
            os.makedirs(os.path.dirname(response_cache_file), exist_ok=True)
            print(f"Saving response cache {response_cache_file}")
            with open(response_cache_file, 'w+', encoding='utf-8') as fp:
                fp.write(result.model_dump_json(indent=2))

            try:
                final = result.to_string()
            except MisalignmentException as ex:
                    print(ex.message)
                    failed += 1
                    continue

            translated_file = os.path.join(result.path, f'{result.filename}{suffix}{result.ext}')
            print(f"Generating final {translated_file}")
            with open(translated_file, 'w+', encoding='utf-8-sig') as fp:
                fp.write(final)
    print(f"\n\n{failed} translations failed, {len(rescheduled)} rescheduled\n\n")
    return rescheduled

async def main(
        client: gemini.genai.Client, prompt: str, files: list[str]):
    max_calls = len(files)*2
    tasks = {}; tasks_tokens = 0; gemini_tasks = 0; calls = 0
    for file_path in file_paths:
        print(f'Parsing {file_path}')
        ass = Ass.from_file(file_path)

        response_cache_file = os.path.join(script_path, 'data', 'cache', f'{ass.filename}_cache.json')

        if os.path.isfile(response_cache_file):
            print(f"Reading response from cache")
            with open(response_cache_file, 'r', encoding='utf-8') as fp:
                tasks[ass] = asyncio.sleep(0, Ass.model_validate_json(fp.read()))
        else:
            ass.translation_tokens_estimate = math.ceil(gemini.estimate_question_tokens(prompt, ass)*2.1)
            while tasks_tokens + ass.translation_tokens_estimate > TPM or gemini_tasks >= RPM:
                to_reschedule = await translate_and_reschedule(tasks, reschedule=calls < max_calls)
                calls += len(tasks)
                print("Waiting 60 seconds for next batch")
                await asyncio.sleep(wait_seconds)
                tasks = {a: gemini.translate_ass(client, prompt, a) for a in to_reschedule}
                gemini_tasks = len(tasks)
                tasks_tokens = sum(a.translation_tokens_estimate for a in to_reschedule)

            gemini_tasks += 1
            tasks_tokens += ass.translation_tokens_estimate
            tasks[ass] = gemini.translate_ass(client, prompt, ass)

    while tasks:
        to_reschedule = await translate_and_reschedule(tasks, reschedule=calls < max_calls)
        if to_reschedule:
            print("Waiting 60 seconds for next batch")
            await asyncio.sleep(wait_seconds)
        calls += len(tasks)
        tasks = {a: gemini.translate_ass(client, prompt, a) for a in to_reschedule}

if __name__ == '__main__':
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    else:
        file_paths = [f for f in sys.argv[1:] if f.endswith('.ass') and not f.endswith(f'{suffix}.ass')]

    with (
            open('prompt.txt', 'r') as prompt_fp,
            open('gemini.key', 'r') as key_fp,
        ):
        prompt = prompt_fp.read()
        key = key_fp.read()

    client = gemini.genai.Client(api_key=key)

    asyncio.run(main(client, prompt, file_paths))