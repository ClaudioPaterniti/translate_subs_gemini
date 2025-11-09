import os
import asyncio

from src.models import *
from src.srt_parser import SrtTranslationFile
from src.ass_parser import AssTranslationFile

import src.logger as logger

class TranslateFileTask:

    def __init__(self,
            translator: Translator,
            file_path: str,
            out_path: str,
            ass_settings: AssSettings):
        self.translator = translator
        self.file_path = file_path
        self.out_path = out_path
        self.ass_settings = ass_settings
        _, self.filename = os.path.split(self.file_path)

    def _load_file(self) -> TranslationFile:
        with open(self.file_path, 'r', encoding='utf-8') as fp:
            if self.file_path.endswith('.ass'):
                return AssTranslationFile(fp.read(), self.ass_settings)
            else:
                return SrtTranslationFile(fp.read())

    async def __call__(self):
        sub_file = self._load_file()
        dialogue = sub_file.get_dialogue()
        translation = await self.translator(self.filename, dialogue)

        translated = sub_file.get_translation(translation.dialogue)

        if translation.misalignments:
            misalignments = sub_file.map_dialogue_lines(
                [x for a, b in translation.misalignments for x in (a, b)])

            misalignments_warnings = [
                f"{misalignments[i]}-{misalignments[i+1]}"
                for i in range(0, len(misalignments), 2)]

            if misalignments:
                logger.warning(
                    f"{self.filename} - misilignments at lines [{', '.join(misalignments_warnings)}]",
                    save=True)

        with open(self.out_path, 'w+', encoding='utf-8') as fp:
            fp.write(translated)

        logger.success(f"{self.filename}: Generated {self.out_path}", save=True)