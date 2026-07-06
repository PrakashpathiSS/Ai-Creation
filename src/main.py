from pathlib import Path

from corpus import build_inventory_corpus
from dataset import build_tokenized_dataset
from scraper import scrape_website_to_file


URL = "https://ascsoftware.com/features/inventory-management/"
OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
CORPUS_DIR = Path(__file__).parent / "data" / "inventory_corpus"
DATASET_DIR = Path(__file__).parent / "data" / "dataset"
TOKENIZER_MODEL_PATH = Path(__file__).parent / "tokensizer" / "tokenizer_model.json"


def main() -> None:
    scraped_path = scrape_website_to_file(URL, OUTPUT_DIR)
    print(f"Saved scraped text to: {scraped_path}")

    corpus_path = build_inventory_corpus(OUTPUT_DIR, CORPUS_DIR)
    print(f"Saved corpus to: {corpus_path}")

    dataset_path = build_tokenized_dataset(
        corpus_path,
        DATASET_DIR,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
    )
    print(f"Saved tokenized dataset to: {dataset_path}")


if __name__ == "__main__":
    main()
