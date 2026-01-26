"""Frozen Omnilingual ASR encoder for extracting audio features.

NOTE: Layer analysis (2024-01-24) showed this encoder achieves ~71% accuracy
on tone classification, compared to 88% for MMS encoder. The MMS encoder is
recommended for production use. This module is kept for reference and
potential future experimentation.

Analysis results:
  - Best single layer: 6 (70.85%)
  - Best layer combination: Top 8 concat (71.35%)
  - MMS comparison: 88.23%
"""

from typing import Optional

import torch
import torch.nn as nn


class OmnilingualEncoder(nn.Module):
    """
    Frozen Omnilingual ASR encoder for feature extraction.

    Uses Meta's Omnilingual ASR wav2vec2 model (omniASR_W2V_300M) to extract
    audio features for tone classification. This can be used as an alternative
    to the MMS encoder.

    Like MMS, extracts features from specific layers:
    - Prosody layers (8-12): Capture tonal/suprasegmental information
    - Phonetic layers (15-17): Capture phonetic content

    Output dimension: 2048 (prosody 1024 + phonetic 1024)
    """

    # Default layer groups for feature extraction (same as MMS)
    PROSODY_LAYERS = [8, 9, 10, 11, 12]  # Prosodic information
    PHONETIC_LAYERS = [15, 16, 17]  # Phonetic information

    def __init__(
        self,
        model_name: str = "omniASR_W2V_300M",
        prosody_layers: Optional[list[int]] = None,
        phonetic_layers: Optional[list[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize frozen Omnilingual encoder.

        Args:
            model_name: Omnilingual model card name. Options:
                - omniASR_W2V_300M (~1.2GB, 1024-dim)
                - omniASR_W2V_1B (~4GB, 1280-dim)
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
        self.model_name = model_name

        # Import here to avoid loading fairseq2 if not needed
        try:
            import omnilingual_asr  # Register model cards
            from fairseq2.models import load_model
            from fairseq2.nn.batch_layout import BatchLayout
        except ImportError:
            raise ImportError(
                "omnilingual-asr is required for OmnilingualEncoder. "
                "Install with: pip install omnilingual-asr"
            )

        self._BatchLayout = BatchLayout

        # Load model
        print(f"Loading Omnilingual encoder: {model_name}...")
        self.model = load_model(model_name)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

        # Determine hidden size by running a test forward pass
        with torch.no_grad():
            test_audio = torch.randn(1, 16000).to(self.device)
            test_layout = self._BatchLayout.of(batch=test_audio, seq_lens=[16000])
            frontend = self.model.encoder_frontend
            seqs, seqs_layout, _ = frontend.extract_features(test_audio, test_layout)
            seqs, _ = frontend.process_features(seqs, seqs_layout, masker=None)
            encoder_out = self.model.encoder(seqs, seqs_layout)
            self.hidden_size = encoder_out.shape[-1]

        # Output dimension: prosody layers averaged + phonetic layers averaged
        # Both are hidden_size-dim, so total is 2*hidden_size (2048 for 300M model)
        self.output_dim = self.hidden_size * 2

        print(f"Omnilingual encoder loaded. Hidden size: {self.hidden_size}, Output dim: {self.output_dim}")
        print(f"  Prosody layers: {self.prosody_layers}, Phonetic layers: {self.phonetic_layers}")

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract features from audio using layer-specific extraction.

        Uses forward hooks to capture intermediate layer outputs, then averages
        prosody layers and phonetic layers separately (like MMS encoder).

        Args:
            audio: Audio waveform tensor of shape (batch, samples) at 16kHz
            attention_mask: Optional attention mask (1 for valid, 0 for padding)

        Returns:
            Dict with:
                - 'prosody': Prosody features (batch, time, hidden_size)
                - 'phonetic': Phonetic features (batch, time, hidden_size)
                - 'combined': Concatenated features (batch, time, 2*hidden_size)
                - 'pooled': Mean-pooled combined features (batch, 2*hidden_size)
        """
        audio = audio.to(self.device)

        # Compute sequence lengths
        batch_size = audio.shape[0]
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            seq_lens = attention_mask.sum(dim=1).tolist()
        else:
            seq_lens = [audio.shape[1]] * batch_size

        # Create batch layout for fairseq2
        batch_layout = self._BatchLayout.of(batch=audio, seq_lens=seq_lens)

        # Set up forward hooks to capture intermediate layer outputs
        hidden_states = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                hidden_states[layer_idx] = output.detach()
            return hook

        # Register hooks on layers we need
        for layer_idx in self.all_layers:
            handle = self.model.encoder.layers[layer_idx].register_forward_hook(
                get_hook(layer_idx)
            )
            handles.append(handle)

        try:
            # Extract features manually to avoid masking issues
            # Step 1: Run frontend (feature extraction + positional encoding)
            frontend = self.model.encoder_frontend
            seqs, seqs_layout, _ = frontend.extract_features(audio, batch_layout)

            # Step 2: Process features WITHOUT masker (masker=None for inference)
            seqs, _ = frontend.process_features(seqs, seqs_layout, masker=None)

            # Step 3: Run encoder (this triggers the hooks)
            encoder_out = self.model.encoder(seqs, seqs_layout)
            out_seq_lens = seqs_layout.seq_lens
        finally:
            # Clean up hooks to prevent memory leaks
            for handle in handles:
                handle.remove()

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
            # Create mask for output sequence lengths
            max_len = combined.shape[1]
            mask = torch.arange(max_len, device=self.device).unsqueeze(0)
            mask = mask < torch.tensor(out_seq_lens, device=self.device).unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            pooled = (combined * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
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
            sample_rate: Sample rate of audio (must be 16kHz)

        Returns:
            Feature dict from forward()
        """
        import numpy as np

        if sample_rate != 16000:
            raise ValueError(f"OmnilingualEncoder requires 16kHz audio, got {sample_rate}Hz")

        # Handle single audio or batch
        if isinstance(audio_array, np.ndarray):
            if audio_array.ndim == 1:
                audio_array = audio_array[np.newaxis, :]
            audio_tensor = torch.from_numpy(audio_array).float()
        elif isinstance(audio_array, list):
            # Pad to same length
            max_len = max(len(a) for a in audio_array)
            padded = np.zeros((len(audio_array), max_len), dtype=np.float32)
            for i, a in enumerate(audio_array):
                padded[i, :len(a)] = a
            audio_tensor = torch.from_numpy(padded)
        else:
            audio_tensor = audio_array

        return self.forward(audio_tensor)

    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output sequence length for given input length.

        Omnilingual uses a convolutional frontend with ~320x downsampling.
        """
        # Approximate downsampling factor
        return input_length // 320


class OmnilingualEncoderLight(nn.Module):
    """
    Lightweight Omnilingual encoder using fewer layers for faster inference.

    For real-time use, only extracts from layer 12 (best prosody) and layer 16 (best phonetic),
    matching the approach used by MMSEncoderLight.

    Output dimension: 2048 (prosody 1024 + phonetic 1024)
    """

    def __init__(
        self,
        model_name: str = "omniASR_W2V_300M",
        device: Optional[str] = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        # Use only best layers for speed (same as MMSEncoderLight)
        self.prosody_layer = 12
        self.phonetic_layer = 16

        try:
            import omnilingual_asr
            from fairseq2.models import load_model
            from fairseq2.nn.batch_layout import BatchLayout
        except ImportError:
            raise ImportError(
                "omnilingual-asr is required for OmnilingualEncoderLight. "
                "Install with: pip install omnilingual-asr"
            )

        self._BatchLayout = BatchLayout

        print(f"Loading Omnilingual encoder (light): {model_name}...")
        self.model = load_model(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

        # Determine hidden size
        with torch.no_grad():
            test_audio = torch.randn(1, 16000).to(self.device)
            test_layout = self._BatchLayout.of(batch=test_audio, seq_lens=[16000])
            frontend = self.model.encoder_frontend
            seqs, seqs_layout, _ = frontend.extract_features(test_audio, test_layout)
            seqs, _ = frontend.process_features(seqs, seqs_layout, masker=None)
            encoder_out = self.model.encoder(seqs, seqs_layout)
            self.hidden_size = encoder_out.shape[-1]

        self.output_dim = self.hidden_size * 2

        print(f"Omnilingual encoder (light) loaded. Output dim: {self.output_dim}")
        print(f"  Using prosody layer {self.prosody_layer}, phonetic layer {self.phonetic_layer}")

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract combined features from single prosody and phonetic layer."""
        audio = audio.to(self.device)

        batch_size = audio.shape[0]
        seq_lens = [audio.shape[1]] * batch_size

        batch_layout = self._BatchLayout.of(batch=audio, seq_lens=seq_lens)

        # Set up forward hooks to capture layer outputs
        hidden_states = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                hidden_states[layer_idx] = output.detach()
            return hook

        # Register hooks on the two layers we need
        for layer_idx in [self.prosody_layer, self.phonetic_layer]:
            handle = self.model.encoder.layers[layer_idx].register_forward_hook(
                get_hook(layer_idx)
            )
            handles.append(handle)

        try:
            # Extract features manually to avoid masking issues
            frontend = self.model.encoder_frontend
            seqs, seqs_layout, _ = frontend.extract_features(audio, batch_layout)
            seqs, _ = frontend.process_features(seqs, seqs_layout, masker=None)
            encoder_out = self.model.encoder(seqs, seqs_layout)
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

        # Get prosody and phonetic features from specific layers
        prosody = hidden_states[self.prosody_layer]
        phonetic = hidden_states[self.phonetic_layer]

        # Concatenate and pool
        combined = torch.cat([prosody, phonetic], dim=-1)
        pooled = combined.mean(dim=1)

        return pooled


if __name__ == "__main__":
    import numpy as np

    # Test full encoder
    print("=" * 60)
    print("Testing OmnilingualEncoder (full)")
    print("=" * 60)
    encoder = OmnilingualEncoder()

    # Generate test audio (1 second at 16kHz)
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1

    print("\nEncoding test audio...")
    features = encoder.encode_audio(test_audio)

    print(f"Prosody features shape: {features['prosody'].shape}")
    print(f"Phonetic features shape: {features['phonetic'].shape}")
    print(f"Combined features shape: {features['combined'].shape}")
    print(f"Pooled features shape: {features['pooled'].shape}")
    print(f"Output dimension: {encoder.output_dim}")

    # Test light encoder
    print("\n" + "=" * 60)
    print("Testing OmnilingualEncoderLight")
    print("=" * 60)
    encoder_light = OmnilingualEncoderLight()

    audio_tensor = torch.from_numpy(test_audio).unsqueeze(0)
    pooled_light = encoder_light(audio_tensor)
    print(f"Pooled (light) shape: {pooled_light.shape}")
    print(f"Output dimension: {encoder_light.output_dim}")
