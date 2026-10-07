"""Deterministic byte-level text tokenizer.

The reusable idea: a byte-level tokenizer with a few special control tokens means
*any* text (including code, symbols, and non-Latin scripts) maps to a small fixed
vocabulary without a learned merge table. Combined with vision patch tokens, this
is what lets text and images share a single token space.
"""
SPECIAL_TOKENS = {
    "<bos>": 256,
    "<eos>": 257,
    "<pad>": 258,
    "<img>": 259,
    "</img>": 260,
    "<sep>": 261,
    "<unk>": 262,
}
SPECIAL_IDS = {v: k for k, v in SPECIAL_TOKENS.items()}
VOCAB_SIZE = 512  # 256 bytes + specials + headroom for visual tokens


class TextTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size

    def encode(self, text):
        ids = []
        for ch in text.encode("utf-8"):
            ids.append(ch)  # 0..255
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            if i < 256:
                out.append(i)
        return bytes(out).decode("utf-8", errors="replace")

    def wrap(self, text):
        """Wrap text with BOS/EOS and image placeholders."""
        return [SPECIAL_TOKENS["<bos>"]] + self.encode(text) + [SPECIAL_TOKENS["<eos>"]]

    def interleave(self, segments):
        """segments: list of ("text", str) or ("image", ImageToken).

        Returns a flat id sequence where images become <img> + patch ids + </img>.
        """
        out = [SPECIAL_TOKENS["<bos>"]]
        for kind, val in segments:
            if kind == "text":
                out.extend(self.encode(val))
            else:
                out.append(SPECIAL_TOKENS["<img>"])
                out.extend(val.patch_ids)  # visual token ids (>= 256)
                out.append(SPECIAL_TOKENS["</img>"])
        out.append(SPECIAL_TOKENS["<eos>"])
        return out

    @staticmethod
    def is_visual(token_id):
        return token_id >= 300
