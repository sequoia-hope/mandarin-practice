"""FiLM (Feature-wise Linear Modulation) fusion for F0 and MMS features."""

from typing import Optional

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Modulates input features x with conditioning signal c:
        output = gamma(c) * x + beta(c)

    Where gamma and beta are learned transformations of the conditioning signal.
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize FiLM layer.

        Args:
            feature_dim: Dimension of features to modulate
            condition_dim: Dimension of conditioning signal (F0 features)
            hidden_dim: Hidden dimension for gamma/beta networks
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = condition_dim * 4

        # Network to produce gamma (scale)
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Network to produce beta (shift)
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Initialize gamma close to 1, beta close to 0
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            features: Input features (batch, time, feature_dim) or (batch, feature_dim)
            condition: Conditioning signal (batch, time, condition_dim) or (batch, condition_dim)

        Returns:
            Modulated features with same shape as input
        """
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)

        return gamma * features + beta


class FiLMFusion(nn.Module):
    """
    FiLM-based fusion of MMS features with F0 features.

    F0 features (3-dim: f0, delta_f0, confidence) modulate MMS embeddings.
    """

    def __init__(
        self,
        mms_dim: int = 2048,
        f0_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        """
        Initialize FiLM fusion module.

        Args:
            mms_dim: Dimension of MMS features (prosody + phonetic = 2048)
            f0_dim: Dimension of F0 features (f0, delta_f0, confidence = 3)
            hidden_dim: Hidden dimension in FiLM networks
            num_layers: Number of FiLM layers to apply
        """
        super().__init__()

        self.mms_dim = mms_dim
        self.f0_dim = f0_dim

        # Project F0 features to higher dimension
        self.f0_projection = nn.Sequential(
            nn.Linear(f0_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stack of FiLM layers
        self.film_layers = nn.ModuleList([
            FiLMLayer(mms_dim, hidden_dim, hidden_dim * 2)
            for _ in range(num_layers)
        ])

        # Layer norm between FiLM layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(mms_dim)
            for _ in range(num_layers)
        ])

        # Output dimension stays the same as MMS
        self.output_dim = mms_dim

    def forward(
        self,
        mms_features: torch.Tensor,
        f0_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse MMS and F0 features using FiLM.

        Args:
            mms_features: MMS features (batch, time, mms_dim) or (batch, mms_dim)
            f0_features: F0 features (batch, time, f0_dim) or (batch, f0_dim)

        Returns:
            Fused features with same shape as mms_features
        """
        # Project F0 features
        f0_projected = self.f0_projection(f0_features)

        # Apply FiLM layers with residual connections
        x = mms_features
        for film_layer, layer_norm in zip(self.film_layers, self.layer_norms):
            # FiLM modulation
            x_modulated = film_layer(x, f0_projected)
            # Residual connection and layer norm
            x = layer_norm(x + x_modulated)

        return x


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to weight F0 vs MMS features.

    Uses attention mechanism to determine how much to rely on F0 information.
    """

    def __init__(
        self,
        mms_dim: int = 2048,
        f0_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.mms_dim = mms_dim
        self.f0_dim = f0_dim

        # Project F0 to match MMS dimension
        self.f0_to_mms = nn.Sequential(
            nn.Linear(f0_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mms_dim),
        )

        # Attention weights for fusion
        self.attention = nn.Sequential(
            nn.Linear(mms_dim + mms_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output_projection = nn.Linear(mms_dim, mms_dim)

        self.output_dim = mms_dim

    def forward(
        self,
        mms_features: torch.Tensor,
        f0_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptively fuse MMS and F0 features.

        Args:
            mms_features: (batch, mms_dim) or (batch, time, mms_dim)
            f0_features: (batch, f0_dim) or (batch, time, f0_dim)

        Returns:
            Fused features
        """
        # Project F0 to MMS dimension
        f0_projected = self.f0_to_mms(f0_features)

        # Compute attention weights
        combined = torch.cat([mms_features, f0_projected], dim=-1)
        weights = self.attention(combined)

        # Weighted sum
        fused = weights[..., 0:1] * mms_features + weights[..., 1:2] * f0_projected

        return self.output_projection(fused)


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion.

    Concatenates MMS and projected F0 features.
    """

    def __init__(
        self,
        mms_dim: int = 2048,
        f0_dim: int = 3,
        f0_hidden_dim: int = 64,
    ):
        super().__init__()

        self.f0_projection = nn.Sequential(
            nn.Linear(f0_dim, f0_hidden_dim),
            nn.ReLU(),
            nn.Linear(f0_hidden_dim, f0_hidden_dim),
        )

        self.output_dim = mms_dim + f0_hidden_dim

    def forward(
        self,
        mms_features: torch.Tensor,
        f0_features: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate MMS and projected F0 features."""
        f0_projected = self.f0_projection(f0_features)
        return torch.cat([mms_features, f0_projected], dim=-1)


if __name__ == "__main__":
    # Test fusion modules
    batch_size = 4
    seq_len = 50
    mms_dim = 2048
    f0_dim = 3

    mms_features = torch.randn(batch_size, seq_len, mms_dim)
    f0_features = torch.randn(batch_size, seq_len, f0_dim)

    print("Testing FiLMFusion...")
    film_fusion = FiLMFusion(mms_dim, f0_dim)
    output = film_fusion(mms_features, f0_features)
    print(f"  Input MMS: {mms_features.shape}")
    print(f"  Input F0: {f0_features.shape}")
    print(f"  Output: {output.shape}")

    print("\nTesting AdaptiveFusion...")
    adaptive_fusion = AdaptiveFusion(mms_dim, f0_dim)
    output = adaptive_fusion(mms_features, f0_features)
    print(f"  Output: {output.shape}")

    print("\nTesting ConcatFusion...")
    concat_fusion = ConcatFusion(mms_dim, f0_dim)
    output = concat_fusion(mms_features, f0_features)
    print(f"  Output: {output.shape}, dim: {concat_fusion.output_dim}")

    # Test with pooled features (batch, dim)
    print("\nTesting with pooled features...")
    mms_pooled = torch.randn(batch_size, mms_dim)
    f0_pooled = torch.randn(batch_size, f0_dim)
    output = film_fusion(mms_pooled, f0_pooled)
    print(f"  Pooled output: {output.shape}")
