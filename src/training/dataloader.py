"""PyTorch DataLoader utilities for tokenized next-token datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    from torch.utils.data import Dataset as _TorchDataset
except ModuleNotFoundError:
    _TorchDataset = object

DEFAULT_DATASET_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "dataset"
    / "inventory_tokenized_dataset.jsonl"
)
DEFAULT_BATCH_SIZE = 8
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_LABEL_PAD_TOKEN_ID = -100


@dataclass(frozen=True)
class TokenizedSample:
    """A single training row from the JSONL dataset."""

    sample_id: int | str
    input_ids: list[int]
    target_ids: list[int]


class InventoryTokenizedDataset(_TorchDataset):
    """PyTorch-compatible map dataset for ``inventory_tokenized_dataset.jsonl``."""

    def __init__(
        self,
        dataset_file: str | Path = DEFAULT_DATASET_FILE,
        *,
        input_field: str = "input_ids",
        target_field: str = "target_ids",
    ) -> None:
        _require_torch()

        self.dataset_file = Path(dataset_file)
        self.input_field = input_field
        self.target_field = target_field

        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Tokenized dataset does not exist: {self.dataset_file}")

        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(f"No training samples were found in: {self.dataset_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        sample = self.samples[index]
        input_ids = torch.tensor(sample.input_ids, dtype=torch.long)
        target_ids = torch.tensor(sample.target_ids, dtype=torch.long)

        return {
            "sample_id": sample.sample_id,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "labels": target_ids,
        }

    def _load_samples(self) -> list[TokenizedSample]:
        samples: list[TokenizedSample] = []
        for line_number, row in _read_jsonl(self.dataset_file):
            input_ids = _extract_token_ids(row, self.input_field, line_number)
            target_ids = _extract_token_ids(row, self.target_field, line_number)

            if len(input_ids) != len(target_ids):
                raise ValueError(
                    f"Line {line_number} has mismatched input/target lengths: "
                    f"{len(input_ids)} != {len(target_ids)}."
                )
            if not input_ids:
                raise ValueError(f"Line {line_number} contains an empty token sequence.")

            sample_id = row.get("_id", row.get("id", len(samples)))
            samples.append(
                TokenizedSample(
                    sample_id=sample_id if isinstance(sample_id, (int, str)) else len(samples),
                    input_ids=input_ids,
                    target_ids=target_ids,
                )
            )

        return samples


def create_dataloader(
    dataset_file: str | Path = DEFAULT_DATASET_FILE,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    label_pad_token_id: int = DEFAULT_LABEL_PAD_TOKEN_ID,
    seed: int | None = None,
) -> Any:
    """Create a PyTorch ``DataLoader`` for the tokenized JSONL dataset."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    torch = _require_torch()
    dataloader_class = _require_torch_dataloader()
    dataset = InventoryTokenizedDataset(dataset_file)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return dataloader_class(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=build_collate_fn(
            pad_token_id=pad_token_id,
            label_pad_token_id=label_pad_token_id,
        ),
    )


def create_train_validation_dataloaders(
    dataset_file: str | Path = DEFAULT_DATASET_FILE,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_fraction: float = 0.1,
    shuffle_train: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    label_pad_token_id: int = DEFAULT_LABEL_PAD_TOKEN_ID,
    seed: int | None = None,
) -> tuple[Any, Any]:
    """Create train and validation DataLoaders from one tokenized dataset."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    torch = _require_torch()
    dataloader_class = _require_torch_dataloader()
    dataset = InventoryTokenizedDataset(dataset_file)
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for a train/validation split.")

    validation_size = max(1, int(len(dataset) * validation_fraction))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("The validation split leaves no samples for training.")

    split_generator = None
    train_generator = None
    if seed is not None:
        split_generator = torch.Generator()
        split_generator.manual_seed(seed)
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [training_size, validation_size],
        generator=split_generator,
    )
    collate_fn = build_collate_fn(
        pad_token_id=pad_token_id,
        label_pad_token_id=label_pad_token_id,
    )

    train_loader = dataloader_class(
        training_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=train_generator,
        collate_fn=collate_fn,
    )
    validation_loader = dataloader_class(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, validation_loader


def build_dataloader(*args: Any, **kwargs: Any) -> Any:
    """Alias for ``create_dataloader``."""
    return create_dataloader(*args, **kwargs)


def build_collate_fn(
    *,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    label_pad_token_id: int = DEFAULT_LABEL_PAD_TOKEN_ID,
) -> Callable[[Iterable[dict[str, Any]]], dict[str, Any]]:
    """Build a PyTorch collate function with configured padding values."""
    return partial(
        collate_tokenized_batch,
        pad_token_id=pad_token_id,
        label_pad_token_id=label_pad_token_id,
    )


def collate_tokenized_batch(
    samples: Iterable[dict[str, Any]],
    *,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    label_pad_token_id: int = DEFAULT_LABEL_PAD_TOKEN_ID,
) -> dict[str, Any]:
    """Pad token sequences and return tensors ready for model training."""
    torch = _require_torch()
    from torch.nn.utils.rnn import pad_sequence

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Cannot collate an empty batch.")

    input_tensors = [sample["input_ids"] for sample in sample_list]
    label_tensors = [sample["labels"] for sample in sample_list]
    for sample_index, (input_tensor, label_tensor) in enumerate(
        zip(input_tensors, label_tensors),
        start=1,
    ):
        if input_tensor.size(0) != label_tensor.size(0):
            raise ValueError(
                f"Batch sample {sample_index} has mismatched input/label lengths: "
                f"{input_tensor.size(0)} != {label_tensor.size(0)}."
            )

    lengths = torch.tensor([tensor.size(0) for tensor in input_tensors], dtype=torch.long)
    input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(label_tensors, batch_first=True, padding_value=label_pad_token_id)

    max_length = input_ids.size(1)
    positions = torch.arange(max_length).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).long()

    return {
        "input_ids": input_ids,
        "target_ids": labels,
        "labels": labels,
        "attention_mask": attention_mask,
        "sample_ids": _collate_sample_ids([sample["sample_id"] for sample in sample_list]),
    }


def _read_jsonl(file_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number} is not valid JSON: {exc}") from exc

        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object.")

        yield line_number, row


def _extract_token_ids(row: dict[str, Any], field_name: str, line_number: int) -> list[int]:
    token_ids = row.get(field_name)
    if not isinstance(token_ids, list):
        raise ValueError(f"Line {line_number} is missing list field {field_name!r}.")

    try:
        return [int(token_id) for token_id in token_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Line {line_number} field {field_name!r} must contain integer token IDs."
        ) from exc


def _collate_sample_ids(sample_ids: list[int | str]) -> Any:
    if all(isinstance(sample_id, int) for sample_id in sample_ids):
        torch = _require_torch()
        return torch.tensor(sample_ids, dtype=torch.long)
    return sample_ids


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for the training DataLoader. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch


def _require_torch_dataloader() -> Any:
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for the training DataLoader. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return DataLoader


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DATASET_FILE",
    "DEFAULT_LABEL_PAD_TOKEN_ID",
    "DEFAULT_PAD_TOKEN_ID",
    "InventoryTokenizedDataset",
    "TokenizedSample",
    "build_collate_fn",
    "build_dataloader",
    "collate_tokenized_batch",
    "create_dataloader",
    "create_train_validation_dataloaders",
]
