from math import ceil
from itertools import chain

from src.models import DialogueChunk, DialogueChunks, MisalignmentException


def split_chunks(chunks: DialogueChunks, chunks_per_block: int) -> list[DialogueChunks]:
    blocks = ceil(len(chunks.chunks)/chunks_per_block)
    q, r = divmod(len(chunks.chunks), blocks)
    return [
        DialogueChunks(chunks=chunks.chunks[i*q + min(i, r):(i+1)*q + min(i+1, r)])
        for i in range(blocks)]

def flatten_chunks(chunks: list[DialogueChunks]) -> DialogueChunks:
    chunks = list(chain.from_iterable([c.chunks for c in chunks]))
    return DialogueChunks(chunks= chunks)

class ChunkedTranslation:

    def __init__(
            self,
            dialogue: list[str],
            chunk_size: int = 10):
        self.chunk_size = chunk_size
        self._translated = False

        self.chunks: DialogueChunks = DialogueChunks(chunks=[
            DialogueChunk(
                from_line= i,
                to_line= i + chunk_size,
                dialogue= dialogue[i: i + chunk_size],
            )
            for i in range(0, len(dialogue), chunk_size)
        ])
        self.misaligned_chunks: list[int] = []

    def add_translation(self, chunks: DialogueChunks):
        if len(chunks.chunks) != len(self.chunks.chunks):
            raise MisalignmentException("The number of translated chunks returned does not match original chunks")
        for i, (original, translated) in enumerate(zip(self.chunks.chunks, chunks.chunks)):
            if len(original.dialogue) != len(translated.dialogue):
                self.misaligned_chunks.append(i)
                diff = len(original.dialogue) - len(translated.dialogue)
                if diff > 0:
                    translated.dialogue.extend(original.dialogue[-diff:])
                else:
                    translated.dialogue = translated.dialogue[:-diff]
            original._translated = translated.dialogue
        self._translated = True

    def get_misaligned_chunks(self) -> DialogueChunks:
        return DialogueChunks(chunks=[self.chunks.chunks[i] for i in self.misaligned_chunks])

    def apply_corrections(self, chunks: DialogueChunks):
        if len(chunks.chunks) != len(self.misaligned_chunks):
            return
        temp_misaligned = []
        for i, index in enumerate(self.misaligned_chunks):
            original, translated = self.chunks.chunks[index], chunks.chunks[i]
            if len(original.dialogue) != len(translated.dialogue):
                temp_misaligned.append(index)
                diff = len(original.dialogue) - len(translated.dialogue)
                if diff > 0:
                    translated.dialogue.extend(original.dialogue[-diff:])
                else:
                    translated.dialogue = translated.dialogue[:-diff]
            original._translated = translated.dialogue
        self.misaligned_chunks = temp_misaligned

    def get_translated_dialogue(self) -> list[str]:
        if not self._translated: raise Exception("No translation available")
        return [l for chunk in self.chunks.chunks for l in chunk._translated]