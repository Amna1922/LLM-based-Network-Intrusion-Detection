"""Tests for answer parsing and answer-only training labels."""

from __future__ import annotations

import pandas as pd

from src.predict_llm import parse_class_name
from src.train_llm import build_training_examples


def test_parser_returns_the_first_valid_class_name() -> None:
    """Return the earliest class mention, regardless of its casing."""
    assert parse_class_name("maybe benign, but DDoS") == "Benign"


def test_parser_returns_none_when_no_class_is_present() -> None:
    """Mark generated output without either class as invalid."""
    assert parse_class_name("uncertain") is None


def test_training_labels_mask_the_prompt_tokens() -> None:
    """Keep loss only on the class answer and its end token."""
    class Tokenizer:
        """Minimal tokenizer stand-in for checking label masking."""

        eos_token_id = 999

        def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            truncation: bool = False,
            max_length: int | None = None,
        ) -> list[int]:
            """Encode characters as deterministic token IDs for the test."""
            token_ids = [ord(character) for character in text]
            return token_ids[:max_length] if max_length is not None else token_ids

    text_rows = pd.DataFrame([{"text": "prompt", "label": 1}])
    example = build_training_examples(Tokenizer(), text_rows, max_length=32)[0]
    prompt_size = len("prompt")
    assert example["labels"][:prompt_size] == [-100] * prompt_size
    assert example["labels"][prompt_size:] == example["input_ids"][prompt_size:]
