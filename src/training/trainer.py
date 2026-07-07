"""Training loop utilities for the GPT language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    """Small training configuration for next-token prediction."""

    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "auto"
    log_every: int = 10
    checkpoint_path: str | Path | None = None


def train_model(
    model: Any,
    dataloader: Any,
    config: TrainerConfig | None = None,
) -> list[dict[str, float]]:
    """Train the model for one or more epochs and return epoch metrics."""
    torch = _require_torch()
    config = config or TrainerConfig()
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")

    device = _resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        metrics = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device=device,
            max_grad_norm=config.max_grad_norm,
            log_every=config.log_every,
        )
        metrics["epoch"] = float(epoch)
        history.append(metrics)

        if config.checkpoint_path is not None:
            save_checkpoint(
                model,
                optimizer,
                config.checkpoint_path,
                epoch=epoch,
                loss=metrics["loss"],
            )

    return history


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    *,
    device: Any,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
) -> dict[str, float]:
    """Run one training epoch over the DataLoader."""
    torch = _require_torch()
    model.train()

    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(dataloader, start=1):
        input_ids = _batch_get(batch, "input_ids").to(device)
        labels = _batch_get(batch, "labels").to(device)
        attention_mask = _batch_get(batch, "attention_mask").to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.get("loss") if isinstance(output, dict) else output.loss
        if loss is None:
            raise ValueError("Model output must contain a loss when labels are provided.")

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_steps += 1

        if log_every > 0 and step % log_every == 0:
            average_loss = total_loss / total_steps
            print(f"step={step} loss={average_loss:.4f}")

    if total_steps == 0:
        raise ValueError("The DataLoader did not yield any batches.")

    return {"loss": total_loss / total_steps, "steps": float(total_steps)}


def save_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    loss: float,
) -> Path:
    """Save model and optimizer state to disk."""
    torch = _require_torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": getattr(model, "config", None),
        },
        path,
    )
    return path


def _batch_get(batch: Any, key: str) -> Any:
    if isinstance(batch, dict):
        return batch[key]
    return getattr(batch, key)


def _resolve_device(device: str) -> Any:
    torch = _require_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for training. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch


__all__ = [
    "TrainerConfig",
    "save_checkpoint",
    "train_model",
    "train_one_epoch",
]
