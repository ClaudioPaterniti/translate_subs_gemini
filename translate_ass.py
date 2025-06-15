import os
import sys
import asyncio

from collections.abc import Awaitable
from typing import Any
from pydantic import BaseModel
from glob import glob

from src.models import *
from src.gemini import GeminiClient
from src.rate_limited_queue import RateLimitedQueue
from src import logger

def translated_path(file_path: str, suffix: str) -> tuple[str, str]:
    path, file = os.path.split(file_path)
    filename, ext = os.path.splitext(file)
    out_file_name = f'{filename}{suffix}{ext}'
    full_path = os.path.join(path, out_file_name)
    return full_path, out_file_name

async def complete_translation(ass: Ass, translation: Awaitable[Ass], out_file_name: str) -> str:
    try:
        result = await translation

        result.to_file(os.path.join(ass.path, out_file_name))

        logger.success(f"{ass.filename}: Generated {out_file_name}", True)

    except Exception as ex:
        logger.error(f"{ass.filename} failed: {ex}", True),


async def main(queue: RateLimitedQueue, file_paths: list[str], config: Config):
    tasks = []
    for file_path in file_paths:
        try:
            ass = Ass.from_file(file_path)
            out_path, out_file_name = translated_path(file_path, config.outfile_suffix)
            tasks.append(complete_translation(ass, queue.queue_translation(ass), out_file_name))

        except Exception as ex:
            logger.error(f"{ass.filename} failed: {ex}", True)

    queue.max_retries = min(queue.max_retries, len(tasks))
    results = await asyncio.gather(*tasks)

    logger.info(f'\nTerminated')
    logger.print_final_log()


if __name__ == '__main__':

    script_path = os.path.abspath(os.path.split(__file__)[0])
    key = os.environ.get('GEMINI_KEY')
    if key is None and os.path.exists(os.path.join(script_path, 'gemini.key')):
            with open(os.path.join(script_path, 'gemini.key'), 'r') as key_fp:
                key = key_fp.read()

    if not key:
        logger.error("Could not retrieve gemini key, populate env variable GEMINI_KEY or file gemini.key")
        sys.exit()

    with (
            open(os.path.join(script_path, 'prompt.txt'), 'r') as prompt_fp,
            open(os.path.join(script_path, 'gemini_config.json'), 'r') as config_fp,
        ):
        prompt = prompt_fp.read()
        config = Config.model_validate_json(config_fp.read())

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        file_paths = glob(f'{folder}/*.ass')
    else:
        file_paths = [f for f in sys.argv[1:] if f.endswith('.ass')]
        folder, _ = os.path.split(file_paths[0])


    translated = glob(f'{folder}/*{config.outfile_suffix}.ass')

    to_translate = [f for f in file_paths
                    if not f.endswith(f'{config.outfile_suffix}.ass')
                    and translated_path(f, config.outfile_suffix)[0] not in translated]

    if not to_translate:
        logger.warning("Found no file to translate, already translated files are ignored.")
        sys.exit()

    client = GeminiClient(key, config.model, prompt, config.content_config)
    queue = RateLimitedQueue(
        client, config.requests_per_minutes, config.token_per_minutes,
        config.max_retries, config.max_concurrent_requests)

    asyncio.run(main(queue, file_paths, config))