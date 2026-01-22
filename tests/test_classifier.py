"""Tests for tone classifier."""

import pytest
import torch

from src.model.classifier import ToneClassifier, AttentionPooling


class TestAttentionPooling:
    """Tests for AttentionPooling module."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        batch_size = 4
        seq_len = 50
        input_dim = 256

        pooling = AttentionPooling(input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)

        output = pooling(x)

        assert output.shape == (batch_size, input_dim)

    def test_with_mask(self):
        """Test pooling with attention mask."""
        batch_size = 4
        seq_len = 50
        input_dim = 256

        pooling = AttentionPooling(input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)

        # Create mask (first half valid, second half masked)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :seq_len // 2] = True

        output = pooling(x, mask)

        assert output.shape == (batch_size, input_dim)


class TestToneClassifier:
    """Tests for ToneClassifier."""

    def test_forward_pooled(self):
        """Test forward pass with pooled features."""
        batch_size = 8
        input_dim = 2048
        f0_dim = 3

        classifier = ToneClassifier(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=5,
            use_f0=True,
        )

        mms_features = torch.randn(batch_size, input_dim)
        f0_features = torch.randn(batch_size, f0_dim)

        logits = classifier(mms_features, f0_features)

        assert logits.shape == (batch_size, 5)

    def test_forward_sequence(self):
        """Test forward pass with sequence features."""
        batch_size = 8
        seq_len = 50
        input_dim = 2048
        f0_dim = 3

        classifier = ToneClassifier(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=5,
            use_f0=True,
            use_attention_pooling=True,
        )

        mms_features = torch.randn(batch_size, seq_len, input_dim)
        f0_features = torch.randn(batch_size, seq_len, f0_dim)

        logits = classifier(mms_features, f0_features)

        assert logits.shape == (batch_size, 5)

    def test_without_f0(self):
        """Test forward pass without F0 features."""
        batch_size = 8
        input_dim = 2048

        classifier = ToneClassifier(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=5,
            use_f0=False,
        )

        mms_features = torch.randn(batch_size, input_dim)

        logits = classifier(mms_features, f0_features=None)

        assert logits.shape == (batch_size, 5)

    def test_predict(self):
        """Test predict method."""
        batch_size = 8
        input_dim = 2048
        f0_dim = 3

        classifier = ToneClassifier(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=5,
            use_f0=True,
        )

        mms_features = torch.randn(batch_size, input_dim)
        f0_features = torch.randn(batch_size, f0_dim)

        predicted, confidence = classifier.predict(mms_features, f0_features)

        assert predicted.shape == (batch_size,)
        assert confidence.shape == (batch_size,)
        assert (predicted >= 0).all() and (predicted < 5).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_parameter_count(self):
        """Test that parameter count is reasonable (~300K target)."""
        classifier = ToneClassifier(
            input_dim=2048,
            hidden_dim=256,
            num_classes=5,
            use_f0=True,
        )

        param_count = classifier.count_parameters()

        # Should be roughly 300K-500K parameters
        assert 100_000 < param_count < 1_000_000


class TestFusion:
    """Tests for fusion modules."""

    def test_film_fusion(self):
        """Test FiLM fusion."""
        from src.features.fusion import FiLMFusion

        batch_size = 4
        mms_dim = 2048
        f0_dim = 3

        fusion = FiLMFusion(mms_dim, f0_dim)

        mms_features = torch.randn(batch_size, mms_dim)
        f0_features = torch.randn(batch_size, f0_dim)

        output = fusion(mms_features, f0_features)

        assert output.shape == mms_features.shape

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        from src.features.fusion import ConcatFusion

        batch_size = 4
        mms_dim = 2048
        f0_dim = 3

        fusion = ConcatFusion(mms_dim, f0_dim, f0_hidden_dim=64)

        mms_features = torch.randn(batch_size, mms_dim)
        f0_features = torch.randn(batch_size, f0_dim)

        output = fusion(mms_features, f0_features)

        assert output.shape == (batch_size, fusion.output_dim)
        assert fusion.output_dim == mms_dim + 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
