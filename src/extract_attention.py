"""Extract token-level attention weights from Hugging Face transformer models."""

from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


def load_model_and_tokenizer(model_name: str = "bert-base-uncased"):
    """Load a pre-trained model and its tokenizer.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return model, tokenizer


def get_attention(
    text: str,
    model_name: str = "bert-base-uncased",
    model=None,
    tokenizer=None,
) -> Tuple[List[str], torch.Tensor]:
    """Extract attention tensors and tokens for a given input text.

    Args:
        text: Input sentence.
        model_name: Hugging Face model identifier (used if model/tokenizer not provided).
        model: Pre-loaded model (optional).
        tokenizer: Pre-loaded tokenizer (optional).

    Returns:
        tokens: List of token strings.
        attentions: Stacked attention tensor of shape
                    (num_layers, num_heads, seq_len, seq_len).
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_name)

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple: one tensor per layer,
    # each of shape (batch, num_heads, seq_len, seq_len).
    attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
    # Final shape: (num_layers, num_heads, seq_len, seq_len)
    return tokens, attentions


def get_mean_attention_per_layer(attentions: torch.Tensor) -> torch.Tensor:
    """Average attention weights across all heads for each layer.

    Args:
        attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len).

    Returns:
        Tensor of shape (num_layers, seq_len, seq_len).
    """
    return attentions.mean(dim=1)


def get_mean_attention_all_layers(attentions: torch.Tensor) -> torch.Tensor:
    """Average attention weights across all layers and heads.

    Args:
        attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len).

    Returns:
        Tensor of shape (seq_len, seq_len).
    """
    return attentions.mean(dim=(0, 1))
