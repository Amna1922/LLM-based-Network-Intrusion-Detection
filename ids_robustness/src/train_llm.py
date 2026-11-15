"""Fine-tune Qwen with LoRA on flow prompts and answer-only causal-language loss."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.prepare_data import build_text_rows, prepare_dataset
from src.utils import ROOT, load_config, pick_device, resolve_path, set_seed


def load_base_model(
    model_name: str,
    device: str,
    training: bool = False,
) -> tuple[torch.nn.Module, object]:
    """Load the configured base model and download it from Hugging Face if needed."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=False,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    except OSError as error:
        raise FileNotFoundError(
            f"Could not load {model_name!r}. Check the model name, internet connection, "
            "and available disk space."
        ) from error

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = not training
    model.to(device)
    if device == "cuda" and training:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, tokenizer


def sample_training_rows(frame: pd.DataFrame, row_limit: int, seed: int) -> pd.DataFrame:
    """Select a reproducible, approximately class-balanced training subset."""
    groups = [group for _, group in frame.groupby("label", sort=True)]
    if not groups:
        raise ValueError("The training split contains no labeled rows.")
    per_class = max(1, row_limit // len(groups))
    selected = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for group in groups
    ]
    return pd.concat(selected).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_training_examples(
    tokenizer: object,
    text_rows: pd.DataFrame,
    max_length: int,
) -> list[dict[str, list[int]]]:
    """Tokenize prompts and mask all labels except the class answer and its EOS."""
    examples = []
    for row in text_rows.itertuples(index=False):
        answer = " DDoS" if int(row.label) == 1 else " Benign"
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)
        prompt_ids = tokenizer.encode(
            row.text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length - len(answer_ids),
        )
        input_ids = prompt_ids + answer_ids
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(prompt_ids) + answer_ids,
            }
        )
    return examples


def make_collator(tokenizer: object):
    """Create a dynamic-padding collator that ignores prompt tokens in the loss."""
    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Pad a batch of tokenized examples and their masked labels."""
        max_length = max(len(example["input_ids"]) for example in examples)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_masks, labels = [], [], []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [pad_id] * padding)
            attention_masks.append(example["attention_mask"] + [0] * padding)
            labels.append(example["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_training_arguments(
    config: dict,
    quick: bool,
    device: str,
    seed: int,
    output_dir: str,
) -> TrainingArguments:
    """Create training settings for the quick, CPU, or CUDA run."""
    is_cuda = device == "cuda"
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(config["warmup_ratio"]),
        num_train_epochs=int(config["quick_epochs"] if quick else config["epochs"]),
        max_steps=int(config["quick_max_steps"]) if quick else -1,
        per_device_train_batch_size=int(
            config["cuda_micro_batch_size"] if is_cuda else config["cpu_micro_batch_size"]
        ),
        gradient_accumulation_steps=int(
            config["cuda_gradient_accumulation_steps"]
            if is_cuda
            else config["cpu_gradient_accumulation_steps"]
        ),
        fp16=is_cuda,
        gradient_checkpointing=is_cuda,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


def build_trainer(
    model: torch.nn.Module,
    tokenizer: object,
    examples: list[dict[str, list[int]]],
    arguments: TrainingArguments,
) -> Trainer:
    """Create a Trainer with answer-only labels and dynamic batch padding."""
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=examples,
        data_collator=make_collator(tokenizer),
        tokenizer=tokenizer,
        callbacks=[TrainingProgressCallback()],
    )


class TrainingProgressCallback(TrainerCallback):
    """Print step loss and an elapsed-time estimate during fine-tuning."""

    def __init__(self) -> None:
        """Initialize the timer used to estimate remaining training time."""
        self.started_at = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        """Start the training timer."""
        self.started_at = time.monotonic()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Print the current step, loss, and estimated time remaining."""
        if not logs or "loss" not in logs:
            return control
        elapsed = time.monotonic() - self.started_at
        steps = max(int(state.global_step), 1)
        remaining_steps = max(int(state.max_steps) - steps, 0)
        eta = elapsed / steps * remaining_steps
        print(
            f"Training progress: step {state.global_step}/{state.max_steps}, "
            f"loss={logs['loss']:.6f}, ETA={eta:.0f}s"
        )
        return control


def save_training_artifacts(
    model: torch.nn.Module,
    tokenizer: object,
    trainer: Trainer,
    training_loss: float,
    device: str,
    quick: bool,
    row_count: int,
    epoch_count: int,
    adapter_dir: str | None = None,
    training_time_s: float | None = None,
    log_path: str | None = None,
) -> dict[str, object]:
    """Save the LoRA adapter and write the training summary."""
    adapter_dir = (
        ROOT / "results" / "llm_adapter"
        if adapter_dir is None
        else ROOT / adapter_dir
    )
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary_path = (
        ROOT / "results" / "training_log.txt"
        if log_path is None
        else Path(log_path)
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"device={device}\n")
        handle.write(f"quick={quick}\n")
        handle.write(f"train_rows={row_count}\n")
        handle.write(f"epochs={epoch_count}\n")
        handle.write(f"optimizer_steps={trainer.state.global_step}\n")
        handle.write(f"train_loss={training_loss:.6f}\n")
        if training_time_s is not None:
            handle.write(f"training_time_seconds={training_time_s:.3f}\n")
    return {
        "device": device,
        "train_rows": row_count,
        "optimizer_steps": trainer.state.global_step,
        "train_loss": training_loss,
        "adapter_path": str(adapter_dir),
        "log_path": str(summary_path),
    }


def train_model(
    quick: bool = False,
    prepared_data: tuple | None = None,
    seed: int | None = None,
    adapter_dir: str | None = None,
) -> dict[str, object]:
    """Fine-tune and save a LoRA adapter using the configured CPU or CUDA settings."""
    config = load_config(ROOT / "config.yaml")
    run_seed = int(config["seed"] if seed is None else seed)
    if adapter_dir is None:
        adapter_path = ROOT / "results" / "llm_adapter"
    else:
        adapter_path = Path(adapter_dir)
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
    set_seed(run_seed)
    device = pick_device()
    print(f"Device: {device}")
    data = prepared_data or prepare_dataset(resolve_path(config["csv_path"]))
    _, _, _, split_map, _ = data
    row_limit = int(config["quick_train_rows"]) if quick else int(config["max_train_rows"])
    if not quick and device == "cpu":
        row_limit = min(row_limit, int(config["cpu_max_train_rows"]))
    training_rows = sample_training_rows(
        split_map["train"],
        row_limit=row_limit,
        seed=run_seed,
    )
    model, tokenizer = load_base_model(
        str(config["model_name"]), device, training=True
    )
    lora_config = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=config["lora_target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    examples = build_training_examples(
        tokenizer,
        build_text_rows(training_rows),
        max_length=int(config["max_seq_length"]),
    )
    training_output = (
        ROOT / "results" / "training"
        if adapter_dir is None
        else adapter_path / "trainer"
    )
    arguments = build_training_arguments(
        config,
        quick,
        device,
        seed=run_seed,
        output_dir=str(training_output),
    )
    trainer = build_trainer(model, tokenizer, examples, arguments)
    training_started = time.monotonic()
    training_loss = trainer.train().training_loss
    training_time_s = time.monotonic() - training_started
    return save_training_artifacts(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        training_loss=training_loss,
        device=device,
        quick=quick,
        row_count=len(training_rows),
        epoch_count=int(arguments.num_train_epochs),
        adapter_dir=str(adapter_path) if adapter_dir is not None else None,
        training_time_s=training_time_s,
        log_path=(
            str(adapter_path / "training_log.txt")
            if adapter_dir is not None
            else None
        ),
    )


def main() -> None:
    """Parse command-line options and start the requested training run."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA")
    parser.add_argument("--quick", action="store_true", help="Run the configured smoke test")
    args = parser.parse_args()
    result = train_model(quick=args.quick)
    print(f"Training rows: {result['train_rows']}")
    print(f"Optimizer steps: {result['optimizer_steps']}")
    print(f"Training loss: {result['train_loss']:.6f}")
    print(f"Saved adapter: {result['adapter_path']}")
    print(f"Training log: {result['log_path']}")


if __name__ == "__main__":
    main()
