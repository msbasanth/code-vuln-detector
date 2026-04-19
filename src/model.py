"""CodeT5-small encoder with classification head for CWE detection.

Architecture:
    CodeT5-small encoder → mean pooling → dropout → Linear(512, num_classes)
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel


class CWEClassifier(nn.Module):
    """CodeT5-small based multi-class CWE classifier."""

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model  # 512 for codet5-small

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
