from pathlib import Path

from tokensizer import TokenizerWrapper, ids_to_tokens


def main() -> None:
    user_text = input("Enter text to tokenize: ").strip()
    model_path = Path(__file__).parent / "tokensizer" / "tokenizer_model.json"

    tokenizer = TokenizerWrapper(target_vocab_size=2000, min_frequency=2)

    if model_path.exists():
        tokenizer.load(model_path)
    else:
        # This creates a starter model for local smoke tests.
        # For production, train with your full corpus before saving.
        tokenizer.fit(user_text, reset=True)
        tokenizer.save(model_path)

    subword_tokens = ids_to_tokens(tokenizer.encode(user_text), tokenizer.vocabulary)
    token_ids = tokenizer.encode(user_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(token_ids)

    print("\nTokenizer Output")
    print("----------------")
    print("Input text:", user_text)
    print("Subword tokens:", subword_tokens)
    print("Token IDs:", token_ids)
    print("Decoded text:", decoded_text)
    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Model path:", model_path)


if __name__ == "__main__":
    main()
