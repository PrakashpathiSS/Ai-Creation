"""Text generation utilities for trained GPT checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from model import GPTConfig, GPTLanguageModel
from tokensizer import TokenizerWrapper


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 60
    temperature: float = 0.9
    top_k: int | None = 20
    device: str = "auto"


def generate_text(
    prompt: str,
    *,
    checkpoint_path: str | Path,
    tokenizer_model_path: str | Path,
    config: GenerationConfig | None = None,
) -> str:
    """Generate text from a saved GPT checkpoint and tokenizer model."""
    torch = _require_torch()
    config = config or GenerationConfig()
    if not prompt.strip():
        raise ValueError("prompt cannot be empty.")

    device = _resolve_device(config.device)
    tokenizer = TokenizerWrapper()
    tokenizer.load(tokenizer_model_path)

    model = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt did not produce any token IDs.")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        eos_token_id=tokenizer.vocabulary.get("<EOS>"),
    )

    return tokenizer.decode(generated_ids[0].detach().cpu().tolist())


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: Any | None = None,
) -> GPTLanguageModel:
    """Load a GPT model from a saved checkpoint."""
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
        # Older project checkpoints stored GPTConfig directly, which requires pickle loading.
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)
    config = _load_model_config(checkpoint)
    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)
    return model


def _load_model_config(checkpoint: dict[str, Any]) -> GPTConfig:
    config_data = checkpoint.get("model_config")
    if isinstance(config_data, GPTConfig):
        return config_data
    if is_dataclass(config_data):
        return GPTConfig(**asdict(config_data))
    if isinstance(config_data, dict):
        return GPTConfig(**config_data)

    raise ValueError("Checkpoint is missing a readable model_config.")


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
            "PyTorch is required for inference. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch
