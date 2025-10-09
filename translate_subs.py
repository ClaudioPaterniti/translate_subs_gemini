import os
import sys
import asyncio
import glob
import traceback

from collections.abc import Awaitable
from string import Template

from src.models import *
from src.gemini import GeminiClient
from src.rate_limiter import RateLimitedLLM
from src.translate_file import FileTranslationTask
from src.logger import Logger

def translated_path(file_path: str, suffix: str) -> str:
    path, file = os.path.split(file_path)
    filename, ext = os.path.splitext(file)
    out_file_name = f'{filename}{suffix}{ext}'
    full_path = os.path.join(path, out_file_name)
    return full_path

async def worker(semaphore: asyncio.Semaphore, task: FileTranslationTask):
    async with semaphore:
        await task()

async def main(llm: RateLimitedLLM, file_paths: list[str], config: Config, logger: Logger):
    semaphore = asyncio.Semaphore(config.max_concurrent_requests or config.requests_per_minutes)

    async with asyncio.TaskGroup() as tg:
        for file_path in file_paths:
            try:
                out_path = translated_path(file_path, config.outfile_suffix)
                translation_task = FileTranslationTask(
                    file_path, out_path, llm,
                    config.dialogue_chunks_size, config.json_max_chars,
                    config.json_reduced_chars, config.ass_settings, logger
                )
                tg.create_task(worker(semaphore, translation_task))
            except Exception as ex:
                logger.error(f"{file_path} failed: {ex}", save=True)
                logger.debug(traceback.format_exc())

    print('\n')
    logger.info(f'Terminated - final log:')
    logger.print_final_log()


if __name__ == '__main__':
    rich_logger = Logger()
    script_path = os.path.abspath(os.path.split(__file__)[0])
    key = os.environ.get('GEMINI_KEY')
    if key is None and os.path.exists(os.path.join(script_path, 'gemini.key')):
            with open(os.path.join(script_path, 'gemini.key'), 'r') as key_fp:
                key = key_fp.read()

    if not key:
        rich_logger.error("Could not retrieve gemini key, populate env variable GEMINI_KEY or file gemini.key")
        sys.exit()

    with (
            open(os.path.join(script_path, 'config.json'), 'r') as config_fp,
            open(os.path.join(script_path, 'user_prompt.txt'), 'r') as user_prompt_fp,
            open(os.path.join(script_path, 'system_prompt.txt'), 'r') as system_prompt_fp,
        ):
        config = Config.model_validate_json(config_fp.read())
        user_prompt = user_prompt_fp.read()
        system_prompt = Template(system_prompt_fp.read()).substitute(dict(config))

    prompt = user_prompt + '\n' + system_prompt

    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        folder = glob.escape(sys.argv[1])
        file_paths = glob.glob(f'{folder}/*.ass') + glob.glob(f'{folder}/*.srt')
    else:
        file_paths = [f for f in sys.argv[1:] if f.endswith('.ass') or f.endswith('.srt')]
        folder, _ = os.path.split(file_paths[0])
        folder = glob.escape(folder)

    translated = glob.glob(f'{folder}/*{config.outfile_suffix}.ass') + glob.glob(f'{folder}/*{config.outfile_suffix}.srt')

    to_translate = [f for f in file_paths
                    if not f[:-4].endswith(f'{config.outfile_suffix}')
                    and translated_path(f, config.outfile_suffix) not in translated]

    if not to_translate:
        rich_logger.warning("Found no file to translate, already translated files are ignored.")
        sys.exit()

    client = GeminiClient(
        key=key,
        model=config.model,
        prompt=prompt,
        config=config.content_config,
        logger=rich_logger
    )

    max_retries = config.max_retries
    queue = RateLimitedLLM(
        client=client,
        requests_per_minute=config.requests_per_minutes,
        tokens_per_minute=config.token_per_minutes,
        max_retries=max_retries,
        max_concurrent_requests=config.max_concurrent_requests,
        logger=rich_logger
    )

    asyncio.run(main(queue, to_translate, config, rich_logger))