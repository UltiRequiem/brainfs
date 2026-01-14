class Tokenizer:
    vocab: dict[str, int]
    TRANSLATION_TABLE: dict[int, int | None] = str.maketrans('', '', "!.?,;:\"'()[]{}<>")
    SUFFIXES: list[tuple[str,str]] = [
          ('ing', ''), ('ly', ''), ('ed', ''), ('ies', 'y'), ('ied', 'y'),
          ('ies', 'y'), ('s', ''), ('es', ''), ('er', ''), ('est', ''),
          ('tion', ''), ('ness', ''), ('ment', ''), ('able', ''), ('ible', '')
      ]

    def __init__(self):
        self.vocab = {}

    def fit(self, documents:list[str]):
        unique_words: set[str] = set()

        for doc in documents:
            words = self._clean(doc)
            unique_words.update(words)

        sorted_words = sorted(unique_words)

        self.vocab = {word: idx for idx, word in enumerate(sorted_words)}

    def embed(self, text: str) -> list[float]:
        words = self._clean(text)
        vector = self._create_empty_vector()
        
        for word in words:
            if word in self.vocab:
                index = self.vocab[word]
                vector[index] += 1

        return vector

    def _create_empty_vector(self) -> list[float]:
        return [0] * len(self.vocab)

    @staticmethod
    def _stem_word(word:str, extra: int = 2)->str:
        for suffix, replacement in Tokenizer.SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + extra:
                return word[:-len(suffix)] + replacement

        return word

    @staticmethod
    def _clean(word:str)->list[str]:
        clean = word.lower().translate(Tokenizer.TRANSLATION_TABLE)
        words = clean.split()
        stemed = [Tokenizer._stem_word(w) for w in words]

        return stemed
