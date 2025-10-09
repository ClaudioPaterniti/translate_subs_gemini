import re

from typing import ClassVar

from src.models import TranslationFile, AssSettings

class AssTranslationFile(TranslationFile):
    command_regex: ClassVar[re.Pattern] = re.compile(r'\{[^{}]+\}')

    def __init__(self, text: str, settings: AssSettings):
        splitted = text.strip().split('[Events]', 1)
        subs = [s for s in splitted[1].split('\n') if s.strip()]
        self._header = splitted[0] + '[Events]\n' + subs[0]
        self._format = {
            s.strip().lower(): i
            for i, s in enumerate(subs[0].replace('Format:', '').split(','))
        }

        for rule in settings.ignore:
            rule._field_i = self._format.get(rule.field.lower())
        self._ignore = [r for r in settings.ignore if r._field_i is not None]

        self._fields: list[str]
        self._commands: dict[str, str] = {}
        self._ignored: list[str] = []
        self._ignored_i: list[int] = []

        sections = self._apply_ignores(subs[1:])

        self._fields: list[str] = [f"{l[0]}:" + ','.join(l[1:len(self._format)]) for l in sections]
        self._dialogue: list[str] = [
            self.command_regex.sub(self._sub_commands, l[len(self._format)])
            for l in sections]


    def _apply_ignores(self, lines: list[str]) -> list[list[str]]:
        subs = []
        for i, l in enumerate(lines):
            ignored = False
            event, value = l.split(':', 1)
            fields = value.split(',', len(self._format)-1)
            if event.strip().lower() == 'comment': ignored = True
            else:
                for rule in self._ignore:
                    if fields[rule._field_i].strip() in rule.values: ignored = True

            if ignored:
                self._ignored_i.append(i)
                self._ignored.append(l)
            else:
                subs.append([event] + fields)

        return subs

    def _sub_commands(self, m: re.Match) -> str:
        token = f"{{format {len(self._commands)}}}"
        self._commands[token] = m.group(0)
        return token

    def _restore_commands(self, m: re.Match) -> str:
        return self._commands.get(m.group(0), '{}')

    def get_dialogue(self):
        return self._dialogue

    def map_dialogue_lines(self, lines: list[int]) -> list[int]:
        offset = self._header.count('\n') + 1
        final = []
        i = 0
        for l in lines:
            t = l + i
            while  i < len(self._ignored_i) and self._ignored_i[i] <= t:
                i += 1
                t += 1
            final.append(offset + t)
        return final

    def get_translation(self, translation: list[str]):
        if len(self._fields) != len(translation): raise Exception("Lines count mismatch")
        lines = [
            f"{f},{self.command_regex.sub(self._restore_commands, l)}"
            for f, l in zip(self._fields, translation)]
        final = []
        j = h = 0
        for i in range(len(self._ignored) + len(lines)) :
            if j < len(self._ignored) and i == self._ignored_i[j]:
                final.append(self._ignored[j])
                j += 1
            else:
                final.append(lines[h])
                h += 1
        return  '\n'.join([self._header] + final)