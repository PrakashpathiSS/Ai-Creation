from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokensizer.tokenizer import (  # noqa: E402
    SimpleTokenizer,
    UNK_TOKEN,
    build_vocabulary,
    detokenize,
    load_vocabulary,
    save_vocabulary,
    wordpiece_tokenizer,
)


class TokenizerTests(unittest.TestCase):
    def test_word_tokenizer_round_trip(self) -> None:
        tokenizer = SimpleTokenizer()
        tokenizer.fit("Hello, world!")

        encoded = tokenizer.encode("Hello, world!", add_special_tokens=True)

        self.assertEqual(tokenizer.decode(encoded), "Hello, world!")

    def test_wordpiece_tokenizer_uses_vocabulary(self) -> None:
        tokenizer = SimpleTokenizer(tokenize=wordpiece_tokenizer)
        tokenizer.fit("hello world")

        encoded = tokenizer.encode("hello world")

        self.assertNotIn(tokenizer.vocabulary[UNK_TOKEN], encoded)
        self.assertEqual(tokenizer.decode(encoded), "hello world")

    def test_batch_helpers(self) -> None:
        tokenizer = SimpleTokenizer()
        tokenizer.fit(["hello world", "another example"])

        encoded_batch = tokenizer.encode_batch(
            ["hello world", "another example"],
            add_special_tokens=True,
        )

        self.assertEqual(
            tokenizer.decode_batch(encoded_batch),
            ["hello world", "another example"],
        )

    def test_detokenize_handles_common_token_styles(self) -> None:
        self.assertEqual(detokenize(["Hello", ",", "world", "!"]), "Hello, world!")
        self.assertEqual(detokenize(["token", "##izer"]), "tokenizer")
        self.assertEqual(detokenize(["▁Hello", "▁world"]), "Hello world")
        self.assertEqual(detokenize(["<0x41>", "<0x49>"]), "AI")

    def test_save_and_load_vocabulary(self) -> None:
        tokenizer = SimpleTokenizer()
        tokenizer.fit("hello world")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vocabulary.json"
            save_vocabulary(tokenizer.vocabulary, path)
            loaded_vocabulary = load_vocabulary(path)

        self.assertEqual(loaded_vocabulary, tokenizer.vocabulary)

    def test_build_vocabulary_is_stable(self) -> None:
        vocabulary = build_vocabulary(["hello", "world", "hello"])

        self.assertEqual(vocabulary["<PAD>"], 0)
        self.assertEqual(vocabulary["<UNK>"], 1)
        self.assertEqual(vocabulary["<BOS>"], 2)
        self.assertEqual(vocabulary["<EOS>"], 3)
        self.assertEqual(len(vocabulary), len(set(vocabulary.values())))


if __name__ == "__main__":
    unittest.main()
