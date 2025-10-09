from src.models import TranslationFile

class SrtTranslationFile(TranslationFile):

    def __init__(self, text: str):
        blocks = text.split('\n\n')
        splitted = [b.split('\n', 2) for b in blocks]
        self._dialogue: list[str] = [l[-1] for l in splitted]
        self._timestamps: list[str] = ['\n'.join(l[:2]) for l in splitted]

    def get_dialogue(self):
        return self._dialogue

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        return list(lines)

    def get_translation(self, translation: list[str]):
        if len(self._timestamps) != len(translation): raise Exception("Lines count mismatch")
        return '\n\n'.join([f"{f}\n{l}" for f, l in zip(self._timestamps, translation)])