"""Generate Benign or DDoS predictions with a locally saved Qwen LoRA adapter."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel

from src.prepare_data import build_text_rows, prepare_dataset
from src.train_llm import load_base_model
from src.utils import ROOT, load_config, pick_device, resolve_path

CLASS_PATTERN = re.compile(r"\b(Benign|DDoS)\b", re.IGNORECASE)


def parse_class_name(answer: str) -> str | None:
    """Return the first valid class name in a generated answer, if present."""
    match = CLASS_PATTERN.search(answer)
    if match is None:
        return None
    return "Benign" if match.group(1).lower() == "benign" else "DDoS"


def load_finetuned_model(
    adapter_dir: str | Path | None = None,
) -> tuple[torch.nn.Module, object, str]:
    """Load the local base model and saved LoRA adapter for generation."""
    config = load_config(ROOT / "config.yaml")
    device = pick_device()
    adapter_path = (
        ROOT / "results" / "llm_adapter"
        if adapter_dir is None
        else Path(adapter_dir)
    )
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. Run `python -m src.train_llm` first."
        )
    model, tokenizer = load_base_model(str(config["model_name"]), device)
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        local_files_only=True,
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    return model, tokenizer, device


def generate_answers(
    prompts: list[str],
    model: torch.nn.Module,
    tokenizer: object,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    max_input_length: int,
) -> list[str | None]:
    """Generate greedy answers in batches and parse the first valid class name."""
    predictions: list[str | None] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            input_length = encoded["input_ids"].shape[1]
            answers = tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )
            predictions.extend(parse_class_name(answer) for answer in answers)
    return predictions


def predict_unique_prompts(
    prompts: list[str],
    model: torch.nn.Module,
    tokenizer: object,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    max_input_length: int,
) -> list[str | None]:
    """Generate each distinct prompt once and restore the original row order."""
    unique_prompts = list(dict.fromkeys(prompts))
    unique_predictions = generate_answers(
        unique_prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
    )
    lookup = dict(zip(unique_prompts, unique_predictions))
    return [lookup[prompt] for prompt in prompts]


def predict_labels() -> tuple[pd.Series, pd.Series]:
    """Generate predictions and return integer classes plus their ground-truth labels."""
    config = load_config(ROOT / "config.yaml")
    csv_path = resolve_path(config["csv_path"])
    _, _, _, split_map, _ = prepare_dataset(csv_path)
    test_rows = split_map["test"]
    text_rows = build_text_rows(test_rows)
    model, tokenizer, device = load_finetuned_model()
    names = predict_unique_prompts(
        text_rows["text"].tolist(),
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(config["prediction_batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        max_input_length=int(config["max_seq_length"]),
    )
    class_ids = {"Benign": 0, "DDoS": 1}
    predictions = pd.Series(
        [class_ids.get(name, -1) for name in names],
        index=test_rows.index,
        dtype=int,
    )
    labels = test_rows["label"].astype(int)
    return predictions, labels


def main() -> None:
    """Print test predictions, invalid-answer count, and misclassification count."""
    predictions, labels = predict_labels()
    invalid_count = int((predictions == -1).sum())
    misclassification_count = int((predictions != labels).sum())
    print(f"Invalid answers: {invalid_count}")
    print(f"Misclassifications: {misclassification_count}")


if __name__ == "__main__":
    main()
