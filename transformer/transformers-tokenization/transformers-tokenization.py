class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        special = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        words = sorted(set(w.lower() for t in texts for w in t.split()))
        words = [w for w in words if w not in special]

        self.word_to_id = {w: i for i, w in enumerate(special + words)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.word_to_id.get(w, 1) for w in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_word.get(i, self.unk_token) for i in ids)