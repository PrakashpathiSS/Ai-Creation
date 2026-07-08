from pathlib import Path

from corpus import build_inventory_corpus
from dataset import build_tokenized_dataset, extract_input_ids_to_text
from inference import GenerationConfig, generate_text
from model import GPTConfig, GPTLanguageModel, load_vocab_size
from scraper import DEFAULT_RAW_CSV_DIR, DEFAULT_RAW_DIR, DEFAULT_RAW_PDF_DIR, scrape_csv_directory_to_files, scrape_google_sheet_to_file, scrape_pdf_directory_to_files, scrape_website_to_file
from tokensizer import TokenizerWrapper, word_tokenizer, character_tokenizer, whitespace_tokenizer, sentence_tokenizer, regex_tokenizer, byte_level_tokenizer, subword_level_tokenizer, bpe_tokenizer, wordpiece_tokenizer, sentencepiece_tokenizer, unigram_tokenizer
from training import TrainerConfig, create_train_validation_dataloaders, train_model

URL = "https://ascsoftware.com/features/inventory-management/"
OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
CORPUS_DIR = Path(__file__).parent / "data" / "inventory_corpus"
DATASET_DIR = Path(__file__).parent / "data" / "dataset"
DECODED_DIR = Path(__file__).parent / "data" / "decoded"
TOKENIZER_MODEL_PATH = Path(__file__).parent / "tokensizer" / "tokenizer_model.json"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "gpt_inventory.pt"
PDF_SOURCE_DIR = DEFAULT_RAW_PDF_DIR if DEFAULT_RAW_PDF_DIR.exists() else DEFAULT_RAW_DIR
CSV_SOURCE_DIR = DEFAULT_RAW_CSV_DIR if DEFAULT_RAW_CSV_DIR.exists() else DEFAULT_RAW_DIR


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




    # pdf_text_paths = scrape_pdf_directory_to_files(PDF_SOURCE_DIR, OUTPUT_DIR)
    # print(f"Saved PDF text files: {len(pdf_text_paths)}")
    # for pdf_text_path in pdf_text_paths:
    #     print(f" - {pdf_text_path}")





    # csv_text_paths = scrape_csv_directory_to_files(CSV_SOURCE_DIR, OUTPUT_DIR)
    # print(f"Saved CSV text files: {len(csv_text_paths)}")
    # for csv_text_path in csv_text_paths:
    #     print(f" - {csv_text_path}")




    # google_sheet_path = scrape_google_sheet_to_file(
    #     "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit#gid=0",
    #     OUTPUT_DIR,
    # )
    # print(f"Saved Google Sheet text to: {google_sheet_path}")








    # train_loader, validation_loader = create_train_validation_dataloaders(
    #     DATASET_DIR / "inventory_tokenized_dataset.jsonl",
    #     batch_size=32,
    #     validation_fraction=0.1,
    #     shuffle_train=True,
    #     seed=42,
    # )
    # first_batch = next(iter(train_loader))
    # # print("input_ids", first_batch["input_ids"][1])
    # # print("target_ids", first_batch["target_ids"][1])
    # # print("labels", first_batch["labels"])

    # vocab_size = load_vocab_size(TOKENIZER_MODEL_PATH)
    # model = GPTLanguageModel(
    #     GPTConfig(
    #         vocab_size=vocab_size,
    #         context_length=first_batch["input_ids"].shape[1],
    #     )
    # )
    # resume_checkpoint = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None
    # if resume_checkpoint is None:
    #     output = model(
    #         first_batch["input_ids"],
    #         attention_mask=first_batch["attention_mask"],
    #         labels=first_batch["labels"],
    #     )
    #     logits = output["logits"]
    #     loss = output["loss"]
    #     if logits is None or loss is None:
    #         raise ValueError("Model forward pass did not return logits and loss.")
    #     print("model logits shape", logits.shape)
    #     print("model loss", loss)
    # else:
    #     print(f"Resuming training from: {resume_checkpoint}")

    # history = train_model(
    #     model,
    #     train_loader,
    #     TrainerConfig(
    #         epochs=45,
    #         checkpoint_path=CHECKPOINT_PATH,
    #         resume_from_checkpoint=resume_checkpoint,
    #     ),
    #     validation_dataloader=validation_loader,
    # )
    # print("training history", history)

    generated_text = generate_text(
        "Inventory management is",
        checkpoint_path=CHECKPOINT_PATH,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        config=GenerationConfig(
            max_new_tokens=50,
            temperature=0.45,
            top_k=10,
            top_p=0.8,
            repetition_penalty=1.25,
            no_repeat_ngram_size=4,
        ),
    )
    print("generated text:------>", generated_text)














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
