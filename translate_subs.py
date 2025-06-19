import os
import sys
import asyncio

from collections.abc import Awaitable
from string import Template
from glob import glob

from src.models import *
from src.gemini import GeminiClient
from src.rate_limited_queue import RateLimitedQueue
from src import logger

def translated_path(file_path: str, suffix: str) -> str:
    path, file = os.path.split(file_path)
    filename, ext = os.path.splitext(file)
    out_file_name = f'{filename}{suffix}{ext}'
    full_path = os.path.join(path, out_file_name)
    return full_path

async def main(queue: RateLimitedQueue, file_paths: list[str], config: Config):
    files = []
    for file_path in file_paths:
        try:
            out_path = translated_path(file_path, config.outfile_suffix)
            files.append(SubsTranslation.from_file(file_path, out_path, config.dialogue_chunks_size))
        except Exception as ex:
            logger.error(f"{file_path} failed: {ex}", True)

    await queue.translate_all(files)

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
            open(os.path.join(script_path, 'gemini_config.json'), 'r') as config_fp,
            open(os.path.join(script_path, 'user_prompt.txt'), 'r') as user_prompt_fp,
            open(os.path.join(script_path, 'system_prompt.txt'), 'r') as system_prompt_fp,
        ):
        config = Config.model_validate_json(config_fp.read())
        user_prompt = user_prompt_fp.read()
        system_prompt = Template(system_prompt_fp.read()).substitute(dict(config))
    prompt = user_prompt + '\n' + system_prompt

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = sys.argv[1]
        file_paths = glob(f'{folder}/*.ass') + glob(f'{folder}/*.srt')
    else:
        file_paths = [f for f in sys.argv[1:] if f.endswith('.ass') or f.endswith('.srt')]
        folder, _ = os.path.split(file_paths[0])


    translated = glob(f'{folder}/*{config.outfile_suffix}.ass') + glob(f'{folder}/*{config.outfile_suffix}.srt')

    to_translate = [f for f in file_paths
                    if not f[:-4].endswith(f'{config.outfile_suffix}')
                    and translated_path(f, config.outfile_suffix) not in translated]

    if not to_translate:
        logger.warning("Found no file to translate, already translated files are ignored.")
        sys.exit()

    client = GeminiClient(key, config.model, prompt, config.content_config)

    max_retries = config.max_retries
    queue = RateLimitedQueue(
        client, config.max_context_window, config.reduced_context_window,
        config.requests_per_minutes, config.token_per_minutes, max_retries,
        config.max_concurrent_requests)

    asyncio.run(main(queue, to_translate, config))