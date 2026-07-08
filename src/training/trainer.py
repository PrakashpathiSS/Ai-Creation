"""Training loop utilities for the GPT language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
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
    resume_from_checkpoint: str | Path | None = None


def train_model(
    model: Any,
    dataloader: Any,
    config: TrainerConfig | None = None,
    *,
    validation_dataloader: Any | None = None,
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

    start_epoch = 1
    if config.resume_from_checkpoint is not None:
        checkpoint = load_checkpoint(config.resume_from_checkpoint, device=device)
        _validate_resume_config(model, checkpoint, config.resume_from_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, start_epoch + config.epochs):
        metrics = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device=device,
            max_grad_norm=config.max_grad_norm,
            log_every=config.log_every,
        )
        metrics["epoch"] = float(epoch)
        if validation_dataloader is not None:
            validation_metrics = evaluate_model(
                model,
                validation_dataloader,
                device=device,
            )
            metrics["validation_loss"] = validation_metrics["loss"]
            metrics["validation_steps"] = validation_metrics["steps"]
        history.append(metrics)

        if config.checkpoint_path is not None:
            save_checkpoint(
                model,
                optimizer,
                config.checkpoint_path,
                epoch=epoch,
                loss=metrics["loss"],
                validation_loss=metrics.get("validation_loss"),
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


def evaluate_model(
    model: Any,
    dataloader: Any,
    *,
    device: Any,
) -> dict[str, float]:
    """Evaluate average loss over a DataLoader without updating weights."""
    torch = _require_torch()
    model.eval()

    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = _batch_get(batch, "input_ids").to(device)
            labels = _batch_get(batch, "labels").to(device)
            attention_mask = _batch_get(batch, "attention_mask").to(device)

            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.get("loss") if isinstance(output, dict) else output.loss
            if loss is None:
                raise ValueError("Model output must contain a loss when labels are provided.")

            total_loss += float(loss.detach().cpu().item())
            total_steps += 1

    if total_steps == 0:
        raise ValueError("The validation DataLoader did not yield any batches.")

    return {"loss": total_loss / total_steps, "steps": float(total_steps)}


def save_checkpoint(
    model: Any,
    optimizer: Any,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    loss: float,
    validation_loss: float | None = None,
) -> Path:
    """Save model and optimizer state to disk."""
    torch = _require_torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": _serialize_model_config(getattr(model, "config", None)),
        },
        path,
    )
    return path


def load_checkpoint(checkpoint_path: str | Path, *, device: Any | None = None) -> dict[str, Any]:
    """Load a training checkpoint dictionary."""
    torch = _require_torch()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    map_location = device if device is not None else "cpu"
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    except Exception:
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint is missing model_state_dict: {path}")
    return checkpoint


def _validate_resume_config(model: Any, checkpoint: dict[str, Any], checkpoint_path: str | Path) -> None:
    saved_config = checkpoint.get("model_config")
    if is_dataclass(saved_config):
        saved_config = asdict(saved_config)

    current_config = _serialize_model_config(getattr(model, "config", None))
    if saved_config is None or current_config is None:
        return
    if saved_config != current_config:
        raise ValueError(
            "Checkpoint model_config does not match the current model. "
            f"Checkpoint: {checkpoint_path}. "
            f"Saved config: {saved_config}. "
            f"Current config: {current_config}."
        )


def _serialize_model_config(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    return config


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
    "evaluate_model",
    "load_checkpoint",
    "save_checkpoint",
    "train_model",
    "train_one_epoch",
]
