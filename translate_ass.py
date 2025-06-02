import os
import sys
import asyncio

from collections.abc import Awaitable
from typing import Any
from pydantic import BaseModel

from src.ass import *
from src.gemini import GeminiClient
from src.rate_limited_queue import RateLimitedQueue

logs = []
failed = 0
script_path = os.path.abspath(os.path.split(__file__)[0])

class Config(BaseModel):
    model: str
    requests_per_minutes: int
    token_per_minutes: int
    content_config: dict[str, Any]
    outfile_suffix: str
    max_requests: int

def log(s: str, success: bool = True):
    global failed
    if not success: failed += 1
    print(s)
    logs.append(s)
    return success

async def complete_translation(ass: Ass, translation: Awaitable[Ass], out_path: str) -> str:
    try:
        result = await translation

        response_cache_file = os.path.join(
            script_path, 'data', 'cache', f'{result.filename}_cache.json')
        os.makedirs(os.path.dirname(response_cache_file), exist_ok=True)
        print(f"{ass.filename}: Saving response cache {response_cache_file}")
        with open(response_cache_file, 'w+', encoding='utf-8') as fp:
            fp.write(result.model_dump_json(indent=2))

        result.to_file(out_path)

        return log(f"{ass.filename}: Generated {out_path}")

    except Exception as ex:
        return log(f"{ass.filename} failed: {ex}", False),


async def main(queue: RateLimitedQueue, file_paths: list[str], config: Config):
    tasks = []
    for file_path in file_paths:
        try:
            ass = Ass.from_file(file_path)

            response_cache_file = os.path.join(
                script_path, 'data', 'cache', f'{ass.filename}_cache.json')
            translated_file = os.path.join(ass.path, f'{ass.filename}{config.outfile_suffix}{ass.ext}')

            if os.path.isfile(response_cache_file):
                with open(response_cache_file, 'r', encoding='utf-8') as fp:
                    translated = Ass.model_validate_json(fp.read())
                    translated.to_file(translated_file)
                    log(f"{ass.filename}: Generated from cache {translated_file}")
            else:
                tasks.append(complete_translation(ass, queue.queue_translation(ass), translated_file))

        except Exception as ex:
            log(f"{ass.filename} failed: {ex}", False)

    queue.max_retries = min(queue.max_retries, len(tasks))
    results = await asyncio.gather(*tasks)

    print(f'\nTerminated, {failed} failed\n', '\n'.join(logs))

if __name__ == '__main__':

    with (
            open('prompt.txt', 'r') as prompt_fp,
            open('gemini.key', 'r') as key_fp,
            open('gemini_config.json', 'r') as config_fp,
        ):
        prompt = prompt_fp.read()
        key = key_fp.read()
        config = Config.model_validate_json(config_fp.read())

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    else:
        file_paths = sys.argv[1:]

    file_paths = [f for f in file_paths
                  if f.endswith('.ass') and not f.endswith(f'{config.outfile_suffix}.ass')]

    client = GeminiClient(key, config.model, prompt, config.content_config)
    queue = RateLimitedQueue(
        client, config.requests_per_minutes, config.token_per_minutes, config.max_requests)

    asyncio.run(main(queue, file_paths, config))