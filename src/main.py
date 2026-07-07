from pathlib import Path

from corpus import build_inventory_corpus
from dataset import build_tokenized_dataset, extract_input_ids_to_text
from scraper import scrape_website_to_file
from tokensizer import TokenizerWrapper, word_tokenizer, character_tokenizer, whitespace_tokenizer, sentence_tokenizer, regex_tokenizer, byte_level_tokenizer, subword_level_tokenizer, bpe_tokenizer, wordpiece_tokenizer, sentencepiece_tokenizer, unigram_tokenizer
from training import create_dataloader

URL = "https://ascsoftware.com/features/inventory-management/"
OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
CORPUS_DIR = Path(__file__).parent / "data" / "inventory_corpus"
DATASET_DIR = Path(__file__).parent / "data" / "dataset"
DECODED_DIR = Path(__file__).parent / "data" / "decoded"
TOKENIZER_MODEL_PATH = Path(__file__).parent / "tokensizer" / "tokenizer_model.json"


def main() -> None:
    # scraped_path = scrape_website_to_file(URL, OUTPUT_DIR)
    # print(f"Saved scraped text to: {scraped_path}")

    # corpus_path = build_inventory_corpus(OUTPUT_DIR, CORPUS_DIR)
    # print(f"Saved corpus to: {corpus_path}")

    # dataset_path = build_tokenized_dataset(
    #     corpus_path,
    #     DATASET_DIR,
    #     tokenizer_model_path=TOKENIZER_MODEL_PATH,
    # )
    # print(f"Saved tokenized dataset to: {dataset_path}")

    # decoded_path = extract_input_ids_to_text(
    #     DATASET_DIR / "inventory_tokenized_dataset.jsonl",
    #     DECODED_DIR,
    #     tokenizer_model_path=TOKENIZER_MODEL_PATH,
    # )
    # print(f"Saved decoded text to: {decoded_path}")

    train_loader = create_dataloader(
        DATASET_DIR / "inventory_tokenized_dataset.jsonl",
        batch_size=8,
        shuffle=True,
        seed=42,
    )
    first_batch = next(iter(train_loader))
    print(first_batch["input_ids"][1])
    print(first_batch["target_ids"][1])

    # user_text = input("Enter text to tokenize: ").strip()

    # training_text = user_text

    # vocabulary_path = Path(__file__).parent / "tokensizer" / "tokenizer_model.json"

    # tokenizer = TokenizerWrapper(tokenize=word_tokenizer)
    # if vocabulary_path.exists():
    #     tokenizer.load(vocabulary_path)

    # tokenizer.pretrain()
    # vocabulary = tokenizer.train(training_text)

    # tokens = word_tokenizer(user_text)
    # token_ids = tokenizer.encode(user_text, add_special_tokens=True)
    # decoded_text = tokenizer.decode(token_ids)
    # tokenizer.save(vocabulary_path)
    # loaded_tokenizer = TokenizerWrapper(tokenize=word_tokenizer)
    # loaded_vocabulary = loaded_tokenizer.load(vocabulary_path)

    # print("\nOutput")
    # print("------")
    # print("Training text:", training_text)
    # print("Your text:", user_text)
    # print("Character tokens:", character_tokenizer(user_text))
    # print("Whitespace tokens:", whitespace_tokenizer(user_text))
    # print("Word tokens:", tokens)
    # print("Sentence tokens:", sentence_tokenizer(training_text))
    # print("Regex tokens:", regex_tokenizer(user_text, r"\w+|[^\w\s]"))
    # print("Byte-level tokens:", byte_level_tokenizer("AI"))
    # print("Subword tokens:", subword_level_tokenizer("tokenizer"))
    # print("BPE tokens:", bpe_tokenizer("low lower lowest"))
    # print("WordPiece tokens:", wordpiece_tokenizer("tokenizer"))
    # print("SentencePiece tokens:", sentencepiece_tokenizer(user_text))
    # print("Unigram tokens:", unigram_tokenizer("tokenizer"))
    # print("Vocabulary:", vocabulary)
    # print("Token IDs:", token_ids)
    # print("Decoded text:", decoded_text)
    # print("Saved vocabulary:", vocabulary_path)
    # print("Loaded vocabulary:", loaded_vocabulary)


if __name__ == "__main__":
    main()
