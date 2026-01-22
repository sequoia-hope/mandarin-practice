"""Tone classifier with attention pooling and lightweight MLP head."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over sequence dimension.

    Learns to weight different time steps based on their importance for classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize attention pooling.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention network
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = min(64, input_dim // 8)  # Smaller for large inputs

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            x: Input features (batch, time, dim)
            mask: Optional mask (batch, time), True for valid positions

        Returns:
            Pooled features (batch, dim)
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # (batch, time)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for richer representations.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        self.output_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-head attention pooling.

        Args:
            x: (batch, time, dim)
            mask: (batch, time)

        Returns:
            (batch, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project keys and values
        keys = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Expand query for batch: (batch, heads, 1, head_dim)
        query = self.query.expand(batch_size, -1, -1).unsqueeze(2)

        # Compute attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.squeeze(2)  # (batch, heads, seq)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: (batch, heads, head_dim)
        context = torch.bmm(
            attn_weights.view(batch_size * self.num_heads, 1, seq_len),
            values.view(batch_size * self.num_heads, seq_len, self.head_dim),
        ).squeeze(1).view(batch_size, -1)

        return self.output_proj(context)


class ToneClassifier(nn.Module):
    """
    Lightweight tone classifier head.

    Architecture:
    - Optional attention pooling (if input is sequence)
    - FiLM fusion with F0 features
    - 2-layer MLP with dropout
    - 5-class softmax output

    Target: ~300K parameters
    """

    def __init__(
        self,
        input_dim: int = 2048,  # MMS output dimension
        hidden_dim: int = 256,
        num_classes: int = 5,
        f0_dim: int = 3,
        use_f0: bool = True,
        use_attention_pooling: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize tone classifier.

        Args:
            input_dim: Input feature dimension (from MMS encoder)
            hidden_dim: Hidden layer dimension
            num_classes: Number of tone classes (5 for Mandarin)
            f0_dim: F0 feature dimension
            use_f0: Whether to use F0 fusion
            use_attention_pooling: Use attention pooling for sequences
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_f0 = use_f0
        self.use_attention_pooling = use_attention_pooling

        # Attention pooling for sequence inputs
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(input_dim)

        # Bottleneck to reduce MMS dimension for parameter efficiency
        bottleneck_dim = 256
        self.bottleneck = nn.Linear(input_dim, bottleneck_dim)

        # F0 fusion - use lightweight concatenation instead of FiLM for ~300K param target
        if use_f0:
            # Project F0 to small hidden dim and concatenate
            self.f0_projection = nn.Sequential(
                nn.Linear(f0_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
            )
            classifier_input_dim = bottleneck_dim + 64
        else:
            classifier_input_dim = bottleneck_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        mms_features: torch.Tensor,
        f0_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mms_features: MMS features (batch, dim) or (batch, time, dim)
            f0_features: Optional F0 features, same shape as mms_features
            mask: Optional attention mask for sequences

        Returns:
            Logits (batch, num_classes)
        """
        x = mms_features

        # Handle sequence input
        if x.dim() == 3:
            if self.use_attention_pooling:
                x = self.attention_pool(x, mask)
            else:
                # Simple mean pooling
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1).float()
                    x = (x * mask_expanded).sum(1) / mask_expanded.sum(1)
                else:
                    x = x.mean(dim=1)

            # Pool F0 features as well if they exist
            if f0_features is not None and f0_features.dim() == 3:
                f0_features = f0_features.mean(dim=1)

        # Apply bottleneck
        x = self.bottleneck(x)

        # Apply F0 fusion (concatenation-based)
        if self.use_f0 and f0_features is not None:
            f0_projected = self.f0_projection(f0_features)
            x = torch.cat([x, f0_projected], dim=-1)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict(
        self,
        mms_features: torch.Tensor,
        f0_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict tone class and confidence.

        Returns:
            (predicted_class, confidence)
        """
        logits = self.forward(mms_features, f0_features)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        return predicted, confidence

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ToneClassifierWithMMS(nn.Module):
    """
    End-to-end model combining frozen MMS encoder with tone classifier.

    For training and evaluation convenience.
    """

    def __init__(
        self,
        mms_model_name: str = "facebook/mms-300m",
        hidden_dim: int = 256,
        num_classes: int = 5,
        use_f0: bool = True,
        dropout: float = 0.3,
        device: Optional[str] = None,
    ):
        super().__init__()

        from ..features.mms_encoder import MMSEncoder
        from ..features.f0_extract import F0Extractor

        # Frozen MMS encoder
        self.mms_encoder = MMSEncoder(model_name=mms_model_name, device=device)

        # F0 extractor (not a nn.Module, just a utility)
        self.f0_extractor = F0Extractor(method="pyin") if use_f0 else None

        # Trainable classifier
        self.classifier = ToneClassifier(
            input_dim=self.mms_encoder.output_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_f0=use_f0,
            dropout=dropout,
        )

    def forward(
        self,
        audio: torch.Tensor,
        f0_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass from raw audio to tone logits.

        Args:
            audio: Audio tensor (batch, samples)
            f0_features: Optional pre-extracted F0 features

        Returns:
            Logits (batch, num_classes)
        """
        # Extract MMS features
        mms_output = self.mms_encoder(audio)
        mms_features = mms_output["pooled"]

        return self.classifier(mms_features, f0_features)


# Tone label names
TONE_NAMES = {
    0: "Tone 1 (high level)",
    1: "Tone 2 (rising)",
    2: "Tone 3 (dipping)",
    3: "Tone 4 (falling)",
    4: "Tone 5 (neutral)",
}


if __name__ == "__main__":
    # Test classifier
    batch_size = 8
    seq_len = 50
    input_dim = 2048
    f0_dim = 3

    print("Testing ToneClassifier...")

    classifier = ToneClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=5,
        use_f0=True,
    )

    print(f"Trainable parameters: {classifier.count_parameters():,}")

    # Test with pooled features
    mms_features = torch.randn(batch_size, input_dim)
    f0_features = torch.randn(batch_size, f0_dim)

    logits = classifier(mms_features, f0_features)
    print(f"Pooled input: {mms_features.shape} -> Output: {logits.shape}")

    # Test with sequence features
    mms_seq = torch.randn(batch_size, seq_len, input_dim)
    f0_seq = torch.randn(batch_size, seq_len, f0_dim)

    logits = classifier(mms_seq, f0_seq)
    print(f"Sequence input: {mms_seq.shape} -> Output: {logits.shape}")

    # Test prediction
    predicted, confidence = classifier.predict(mms_features, f0_features)
    print(f"Predictions: {predicted}")
    print(f"Confidences: {confidence}")
