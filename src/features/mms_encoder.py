"""Frozen MMS encoder for extracting prosodic and phonetic features."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class MMSEncoder(nn.Module):
    """
    Frozen MMS encoder extracting features from specific layers.

    Uses layers 8-12 (prosody) and 15-17 (phonetic) concatenated → 2048-dim.
    MMS-300M has 24 transformer layers with 1024-dim hidden states.
    """

    # Default layer groups for feature extraction
    PROSODY_LAYERS = [8, 9, 10, 11, 12]  # Prosodic information
    PHONETIC_LAYERS = [15, 16, 17]  # Phonetic information

    def __init__(
        self,
        model_name: str = "facebook/mms-300m",
        prosody_layers: Optional[list[int]] = None,
        phonetic_layers: Optional[list[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize frozen MMS encoder.

        Args:
            model_name: HuggingFace model identifier
            prosody_layers: Layer indices for prosody features (default: 8-12)
            phonetic_layers: Layer indices for phonetic features (default: 15-17)
            device: Device to use (auto-detected if None)
        """
        super().__init__()

        self.prosody_layers = prosody_layers or self.PROSODY_LAYERS
        self.phonetic_layers = phonetic_layers or self.PHONETIC_LAYERS
        self.all_layers = sorted(set(self.prosody_layers + self.phonetic_layers))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

        # Get hidden size from config
        self.hidden_size = self.model.config.hidden_size  # 1024 for MMS-300M

        # Output dimension: prosody layers averaged + phonetic layers averaged
        # Both are 1024-dim, so total is 2048-dim
        self.output_dim = self.hidden_size * 2

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract features from audio.

        Args:
            audio: Audio waveform tensor of shape (batch, samples) at 16kHz
            attention_mask: Optional attention mask

        Returns:
            Dict with:
                - 'prosody': Prosody features (batch, time, hidden_size)
                - 'phonetic': Phonetic features (batch, time, hidden_size)
                - 'combined': Concatenated features (batch, time, 2*hidden_size)
                - 'pooled': Mean-pooled combined features (batch, 2*hidden_size)
        """
        # Ensure audio is on correct device
        audio = audio.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass through model
        outputs = self.model(
            audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states  # Tuple of (batch, time, hidden)

        # Extract and average prosody layers
        prosody_features = torch.stack(
            [hidden_states[i] for i in self.prosody_layers],
            dim=0,
        ).mean(dim=0)

        # Extract and average phonetic layers
        phonetic_features = torch.stack(
            [hidden_states[i] for i in self.phonetic_layers],
            dim=0,
        ).mean(dim=0)

        # Concatenate prosody and phonetic features
        combined = torch.cat([prosody_features, phonetic_features], dim=-1)

        # Mean pooling over time dimension
        if attention_mask is not None:
            # Compute output length after conv layers
            output_lengths = self.model._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            mask = torch.arange(combined.size(1), device=self.device)
            mask = mask.unsqueeze(0) < output_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (combined * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = combined.mean(dim=1)

        return {
            "prosody": prosody_features,
            "phonetic": phonetic_features,
            "combined": combined,
            "pooled": pooled,
        }

    def encode_audio(
        self,
        audio_array,
        sample_rate: int = 16000,
    ) -> dict[str, torch.Tensor]:
        """
        Encode raw audio array to features.

        Args:
            audio_array: numpy array or list of audio samples
            sample_rate: Sample rate of audio (resampled to 16kHz if different)

        Returns:
            Feature dict from forward()
        """
        import numpy as np

        # Handle single audio or batch
        if isinstance(audio_array, np.ndarray):
            if audio_array.ndim == 1:
                audio_array = [audio_array]

        # Process with HuggingFace feature extractor
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        audio_tensor = inputs.input_values
        attention_mask = inputs.get("attention_mask")

        return self.forward(audio_tensor, attention_mask)

    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output sequence length for given input length.

        MMS uses a convolutional feature extractor that downsamples.
        """
        # MMS-300M uses conv layers with specific strides
        # Total downsampling factor is approximately 320 (20ms per frame)
        return self.model._get_feat_extract_output_lengths(
            torch.tensor([input_length])
        ).item()


class MMSEncoderLight(nn.Module):
    """
    Lightweight MMS encoder using fewer layers for faster inference.

    For real-time use, only extracts from layer 12 (best prosody) and 16 (best phonetic).
    """

    def __init__(
        self,
        model_name: str = "facebook/mms-300m",
        device: Optional[str] = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

        self.hidden_size = self.model.config.hidden_size
        self.output_dim = self.hidden_size * 2

        # Use only best layers for speed
        self.prosody_layer = 12
        self.phonetic_layer = 16

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract combined features from single prosody and phonetic layer."""
        audio = audio.to(self.device)

        outputs = self.model(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        prosody = hidden_states[self.prosody_layer]
        phonetic = hidden_states[self.phonetic_layer]

        combined = torch.cat([prosody, phonetic], dim=-1)
        pooled = combined.mean(dim=1)

        return pooled


if __name__ == "__main__":
    import numpy as np

    # Test encoder
    print("Loading MMS encoder...")
    encoder = MMSEncoder()

    # Generate test audio (1 second at 16kHz)
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1

    print("Encoding test audio...")
    features = encoder.encode_audio(test_audio)

    print(f"Prosody features shape: {features['prosody'].shape}")
    print(f"Phonetic features shape: {features['phonetic'].shape}")
    print(f"Combined features shape: {features['combined'].shape}")
    print(f"Pooled features shape: {features['pooled'].shape}")
    print(f"Output dimension: {encoder.output_dim}")
