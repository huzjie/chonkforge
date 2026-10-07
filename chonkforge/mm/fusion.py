"""Native multimodal fusion: one sequence, one backbone.

Builds a single interleaved token sequence from text + images, carrying:
  - token ids (text bytes + visual patch ids)
  - embeddings (text embedding + projected patch vectors)
  - segment ids (0 = text, 1 = vision) for modality-aware processing

This is the "native" part of native multimodal: there is no cross-attention
between two encoders -- everything is one stream through the same transformer.
"""
from dataclasses import dataclass, field

from .tokenizer import TextTokenizer
from .vision import PatchEmbedder


@dataclass
class MultimodalSequence:
    token_ids: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)  # seq x dim
    segment_ids: list = field(default_factory=list)  # 0 text / 1 vision

    def __len__(self):
        return len(self.token_ids)


class NativeFusion:
    def __init__(self, text_embedder, patch_embedder, tokenizer=None):
        self.text_embedder = text_embedder  # callable(id) -> vector
        self.patch_embedder = patch_embedder
        self.tokenizer = tokenizer or TextTokenizer()

    def build(self, segments):
        """segments: list of ("text", str) | ("image", ImageToken)."""
        seq = MultimodalSequence()
        seq.token_ids.append(SPECIAL_TOKENS["<bos>"])
        seq.embeddings.append(self.text_embedder(SPECIAL_TOKENS["<bos>"]))
        seq.segment_ids.append(0)
        for kind, val in segments:
            if kind == "text":
                for tid in self.tokenizer.encode(val):
                    seq.token_ids.append(tid)
                    seq.embeddings.append(self.text_embedder(tid))
                    seq.segment_ids.append(0)
            else:
                seq.token_ids.append(SPECIAL_TOKENS["<img>"])
                seq.embeddings.append(self.text_embedder(SPECIAL_TOKENS["<img>"]))
                seq.segment_ids.append(0)
                for pid, vec in zip(val.patch_ids, val.vectors):
                    seq.token_ids.append(pid)
                    seq.embeddings.append(vec)
                    seq.segment_ids.append(1)
                seq.token_ids.append(SPECIAL_TOKENS["</img>"])
                seq.embeddings.append(self.text_embedder(SPECIAL_TOKENS["</img>"]))
                seq.segment_ids.append(0)
        seq.token_ids.append(SPECIAL_TOKENS["<eos>"])
        seq.embeddings.append(self.text_embedder(SPECIAL_TOKENS["<eos>"]))
        seq.segment_ids.append(0)
        return seq


# import at bottom to avoid circular import
from .tokenizer import SPECIAL_TOKENS  # noqa: E402
