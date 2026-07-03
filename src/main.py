from pathlib import Path
from importlib import import_module

from tokensizer import (
    SimpleTokenizer,
    bpe_tokenizer,
    byte_level_tokenizer,
    character_tokenizer,
    regex_tokenizer,
    sentence_tokenizer,
    sentencepiece_tokenizer,
    subword_level_tokenizer,
    unigram_tokenizer,
    whitespace_tokenizer,
    word_tokenizer,
    wordpiece_tokenizer,
)


def show_gpt_tokenizer_demo(text: str) -> None:
    """Show GPT-style token IDs using OpenAI's tiktoken library."""
    try:
        tiktoken = import_module("tiktoken")
    except ImportError:
        print("\nGPT tokenizer demo")
        print("------------------")
        print("Install dependency first: python3 -m pip install -r requirements.txt")
        return

    try:
        encoding = tiktoken.get_encoding("o200k_base")
    except Exception as error:
        print("\nGPT tokenizer demo")
        print("------------------")
        print("Could not load GPT encoding:", error)
        return

    gpt_token_ids = encoding.encode(text)
    gpt_tokens = [encoding.decode([token_id]) for token_id in gpt_token_ids]

    print("\nGPT tokenizer demo")
    print("------------------")
    print("Encoding:", encoding.name)
    print("GPT tokens:", gpt_tokens)
    print("GPT token IDs:", gpt_token_ids)


def main() -> None:
    user_text = input("Enter text to tokenize: ").strip()

    training_text = user_text

    vocabulary_path = Path(__file__).parent / "tokensizer" / "vocabulary.json"

    tokenizer = SimpleTokenizer(tokenize=word_tokenizer)
    if vocabulary_path.exists():
        tokenizer.load(vocabulary_path)

    tokenizer.pretrain()
    vocabulary = tokenizer.train(training_text)

    tokens = word_tokenizer(user_text)
    token_ids = tokenizer.encode(user_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(token_ids)
    tokenizer.save(vocabulary_path)
    loaded_tokenizer = SimpleTokenizer(tokenize=word_tokenizer)
    loaded_vocabulary = loaded_tokenizer.load(vocabulary_path)

    print("\nOutput")
    print("------")
    print("Training text:", training_text)
    print("Your text:", user_text)
    print("Character tokens:", character_tokenizer(user_text))
    print("Whitespace tokens:", whitespace_tokenizer(user_text))
    print("Word tokens:", tokens)
    print("Sentence tokens:", sentence_tokenizer(training_text))
    print("Regex tokens:", regex_tokenizer(user_text, r"\w+|[^\w\s]"))
    print("Byte-level tokens:", byte_level_tokenizer("AI"))
    print("Subword tokens:", subword_level_tokenizer("tokenizer"))
    print("BPE tokens:", bpe_tokenizer("low lower lowest"))
    print("WordPiece tokens:", wordpiece_tokenizer("tokenizer"))
    print("SentencePiece tokens:", sentencepiece_tokenizer(user_text))
    print("Unigram tokens:", unigram_tokenizer("tokenizer"))
    # print("Vocabulary:", vocabulary)
    print("Token IDs:", token_ids)
    print("Decoded text:", decoded_text)
    # print("Saved vocabulary:", vocabulary_path)
    print("Loaded vocabulary:", loaded_vocabulary)
    show_gpt_tokenizer_demo(user_text)


if __name__ == "__main__":
    main()
