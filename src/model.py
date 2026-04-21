"""Encoder with classification head for CWE detection.

Supports T5-family (CodeT5-small, CodeT5-base) and RoBERTa-family (CodeBERT-base)
encoders.

Architecture:
    Encoder → mean pooling → dropout → Linear(hidden_size, num_classes)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel

# Model families that use T5EncoderModel (encoder-only).
# All others default to AutoModel.
_T5_PREFIXES = ("Salesforce/codet5",)


def _is_t5_family(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _T5_PREFIXES)


class CWEClassifier(nn.Module):
    """Multi-class CWE classifier supporting CodeT5 and CodeBERT encoders."""

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        if _is_t5_family(model_name):
            self.encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        # T5 exposes d_model; RoBERTa exposes hidden_size
        cfg = self.encoder.config
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_size)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        pooled = sum_hidden / count  # (batch, hidden_size)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits
