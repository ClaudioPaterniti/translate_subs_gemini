import os
import sys
import asyncio
import glob
import traceback

from src.srt_parser import SrtTranslationFile
from src.ass_parser import AssTranslationFile
from string import Template

from src.models import *
from src.gemini import GeminiClient
from src.rate_limiter import RateLimitedLLM
from src.srt_parser import SrtTranslationFile
from src.ass_parser import AssTranslationFile

import src.logger as logger

def translated_path(file_path: str, suffix: str) -> str:
    path, file = os.path.split(file_path)
    filename, ext = os.path.splitext(file)
    out_file_name = f'{filename}{suffix}{ext}'
    full_path = os.path.join(path, out_file_name)
    return full_path

def load_file(path: str, ass_settings: AssSettings) -> TranslationFile:
        with open(path, 'r', encoding='utf-8') as fp:
            if path.endswith('.ass'):
                return AssTranslationFile(fp.read(), ass_settings)
            else:
                return SrtTranslationFile(fp.read())

async def worker(
        semaphore: asyncio.Semaphore,
        translator: Translator,
        file_path: str,
        out_path: str,
        ass_settings: AssSettings):

    async with semaphore: # avoid loading all files at once
        _, filename = os.path.split(file_path)
        sub_file = load_file(file_path, ass_settings)
        dialogue = sub_file.get_dialogue()
        translation = await translator(filename, dialogue)

        translated = sub_file.get_translation(translation.dialogue)

        if translation.misalignments:
            misalignments = sub_file.map_dialogue_lines(
                [x for a, b in translation.misalignments for x in (a, b)])

            misalignments_warnings = [
                f"{misalignments[i]}-{misalignments[i+1]}"
                for i in range(0, len(misalignments), 2)]

            if misalignments:
                logger.warning(
                    f"{filename} - misilignments at lines [{', '.join(misalignments_warnings)}]",
                    save=True)

        with open(out_path, 'w+', encoding='utf-8') as fp:
            fp.write(translated)

        logger.success(f"{filename}: Generated {out_path}", save=True)

async def main(llm: RateLimitedLLM, file_paths: list[str], config: Config):
    semaphore = asyncio.Semaphore(config.max_concurrent_requests or config.requests_per_minutes)

    if config.translator_type == 'json':
        from src.json_translator.translator import JsonChunkerTranslator
        translator = JsonChunkerTranslator(llm, config.lines_per_chunk, config.chunks_per_request)
    else:
        from src.text_translator.translator import TextTranslator
        translator = TextTranslator (llm, config.lines_per_chunk)

    async with asyncio.TaskGroup() as tg:
        for file_path in file_paths:
            try:
                out_path = translated_path(file_path, config.outfile_suffix)
                tg.create_task(worker(semaphore, translator, file_path, out_path, config.ass_settings))
            except Exception as ex:
                logger.error(f"{file_path} failed: {ex}", save=True)
                logger.debug(traceback.format_exc())

    print('\n')
    logger.info(f'Terminated - final log:')
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
            open(os.path.join(script_path, 'config.json'), 'r') as config_fp,
            open(os.path.join(script_path, 'user_prompt.md'), 'r') as user_prompt_fp,
            open(os.path.join(script_path, 'system_prompt.md'), 'r') as system_prompt_fp,
        ):
        config = Config.model_validate_json(config_fp.read())
        user_prompt = user_prompt_fp.read()
        system_prompt = Template(system_prompt_fp.read()).substitute(dict(config))

    logger._debug = config.debug
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
        logger.warning("Found no file to translate, already translated files are ignored.")
        sys.exit()

    client = GeminiClient(
        key=key,
        model=config.model,
        prompt=prompt,
        config=config.content_config
    )

    queue = RateLimitedLLM(
        client=client,
        requests_per_minute=config.requests_per_minutes,
        tokens_per_minute=config.token_per_minutes,
        max_retries=config.max_retries,
        max_concurrent_requests=config.max_concurrent_requests
    )

    asyncio.run(main(queue, to_translate, config))