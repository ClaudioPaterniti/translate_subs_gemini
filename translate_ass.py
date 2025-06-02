import os
import sys
import asyncio

from collections.abc import Awaitable

from src.ass import *
from src.gemini import GeminiClient
from src.rate_limited_queue import RateLimitedQueue

RPM  = 10
TPM = 250_000
suffix = '_ita'

async def complete_translation(ass: Ass, translation: Awaitable[Ass]) -> str:
    try:
        result = await translation
    except Exception as ex:
        return f"{ass.filename} failed: {ex}"

    response_cache_file = os.path.join('data', 'cache', f'{result.filename}_cache.json')
    os.makedirs(os.path.dirname(response_cache_file), exist_ok=True)
    print(f"{ass.filename}: Saving response cache {response_cache_file}")
    with open(response_cache_file, 'w+', encoding='utf-8') as fp:
        fp.write(result.model_dump_json(indent=2))

    try:
        final = result.to_string()
    except MisalignmentException as ex:
        return f"{ass.filename} failed: {ex}"

    translated_file = os.path.join(result.path, f'{result.filename}{suffix}{result.ext}')
    with open(translated_file, 'w+', encoding='utf-8-sig') as fp:
        fp.write(final)

    return f"{ass.filename}: Generated {translated_file}"


async def main(queue: RateLimitedQueue, file_paths: list[str]):
    tasks = []; uncached = 0
    for file_path in file_paths:
        print(f'Parsing {file_path}')
        ass = Ass.from_file(file_path)

        response_cache_file = os.path.join('data', 'cache', f'{ass.filename}_cache.json')

        if os.path.isfile(response_cache_file):
            print(f"{ass.filename}: Reading response from cache")
            with open(response_cache_file, 'r', encoding='utf-8') as fp:
                tasks.append(complete_translation(
                    ass, asyncio.sleep(0, Ass.model_validate_json(fp.read()))))
        else:
            tasks.append(complete_translation(ass, queue.queue_translation(ass)))
            uncached += 1

    queue.max_retries = min(queue.max_retries, uncached)
    results = await asyncio.gather(*tasks)

    print('\nTerminated\n', '\n'.join(results))

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.split(__file__)[0]))
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    else:
        file_paths = sys.argv[1:]

    file_paths = [f for f in file_paths if f.endswith('.ass') and not f.endswith(f'{suffix}.ass')]

    with (
            open('prompt.txt', 'r') as prompt_fp,
            open('gemini.key', 'r') as key_fp,
        ):
        prompt = prompt_fp.read()
        key = key_fp.read()

    client = GeminiClient(key, "gemini-2.0-flash", prompt)
    queue = RateLimitedQueue(client, RPM, TPM, 10)

    asyncio.run(main(queue, file_paths))