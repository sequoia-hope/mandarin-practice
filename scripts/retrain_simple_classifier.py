#!/usr/bin/env python3
"""
Retrain a simple tone classifier from scratch using TTS-generated audio.

This tests whether T3 can be learned correctly with clean, verified labels.
If this succeeds, the issue is in the original training data.
If this fails, the issue is in the architecture or F0 features.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TONE_COACH_DEVICE"] = "cpu"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Expanded training set with verified tone labels
TRAINING_CHARS = [
    # Tone 1 (high level) - 20 chars
    ("妈", 1), ("诗", 1), ("书", 1), ("汤", 1), ("温", 1),
    ("一", 1), ("都", 1), ("八", 1), ("他", 1), ("飞", 1),
    ("高", 1), ("开", 1), ("天", 1), ("山", 1), ("花", 1),
    ("家", 1), ("猫", 1), ("吃", 1), ("喝", 1), ("说", 1),

    # Tone 2 (rising) - 20 chars
    ("麻", 2), ("十", 2), ("熟", 2), ("糖", 2), ("文", 2),
    ("姨", 2), ("读", 2), ("拔", 2), ("拿", 2), ("肥", 2),
    ("田", 2), ("华", 2), ("夹", 2), ("来", 2), ("年", 2),
    ("人", 2), ("学", 2), ("茶", 2), ("词", 2), ("红", 2),

    # Tone 3 (dipping) - 20 chars
    ("马", 3), ("史", 3), ("鼠", 3), ("躺", 3), ("吻", 3),
    ("椅", 3), ("堵", 3), ("把", 3), ("哪", 3), ("匪", 3),
    ("搞", 3), ("凯", 3), ("舔", 3), ("闪", 3), ("假", 3),
    ("买", 3), ("你", 3), ("我", 3), ("好", 3), ("小", 3),

    # Tone 4 (falling) - 20 chars
    ("骂", 4), ("是", 4), ("树", 4), ("烫", 4), ("问", 4),
    ("意", 4), ("度", 4), ("爸", 4), ("那", 4), ("费", 4),
    ("告", 4), ("看", 4), ("善", 4), ("画", 4), ("价", 4),
    ("卖", 4), ("大", 4), ("去", 4), ("对", 4), ("做", 4),
]


def generate_training_data():
    """Generate TTS audio for all training chars."""
    from src.coach.tts import LightTTS

    print("Generating TTS audio...")
    tts = LightTTS()

    samples = []
    for char, tone in tqdm(TRAINING_CHARS, desc="Generating"):
        try:
            audio = tts.speak(char, language="zh", speed="normal", play=False)
            if audio is not None and len(audio) > 1600:
                samples.append((char, tone, audio))
        except:
            pass

    # Count distribution
    dist = defaultdict(int)
    for _, t, _ in samples:
        dist[t] += 1
    print(f"Generated {len(samples)} samples")
    print(f"Distribution: {dict(sorted(dist.items()))}")

    return samples


def extract_features(samples, device="cpu"):
    """Extract MMS and F0 features."""
    from src.features.mms_encoder import MMSEncoder
    from src.features.f0_extract import F0Extractor

    print("\nLoading feature extractors...")
    encoder = MMSEncoder(device=device)
    f0_extractor = F0Extractor(method="pyin")

    mms_features = []
    f0_features = []
    labels = []

    print("Extracting features...")
    for char, tone, audio in tqdm(samples, desc="Extracting"):
        # MMS features
        features = encoder.encode_audio(audio, sample_rate=16000)
        mms_feat = features["pooled"].squeeze().cpu().numpy()
        mms_features.append(mms_feat)

        # F0 features
        f0_result = f0_extractor.extract(audio)
        f0_raw = f0_result.get("features", np.zeros((1, 3)))
        f0_feat = f0_raw.mean(axis=0) if len(f0_raw) > 0 else np.zeros(3)
        f0_features.append(f0_feat)

        labels.append(tone - 1)  # 0-indexed

    return (
        np.array(mms_features),
        np.array(f0_features),
        np.array(labels)
    )


class SimpleToneClassifier(nn.Module):
    """Simplified classifier for testing."""

    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=4, use_f0=True, f0_dim=3):
        super().__init__()
        self.use_f0 = use_f0

        # Bottleneck
        self.bottleneck = nn.Linear(input_dim, hidden_dim)

        # F0 projection (optional)
        if use_f0:
            self.f0_proj = nn.Sequential(
                nn.Linear(f0_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
            )
            classifier_input = hidden_dim + 64
        else:
            classifier_input = hidden_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, mms, f0=None):
        x = self.bottleneck(mms)
        if self.use_f0 and f0 is not None:
            f0_feat = self.f0_proj(f0)
            x = torch.cat([x, f0_feat], dim=-1)
        return self.classifier(x)


def train_classifier(X_train, y_train, X_val, y_val, use_f0=True, epochs=200):
    """Train classifier and return best model."""

    # Separate MMS and F0
    if use_f0:
        X_train_mms = X_train[:, :2048]
        X_train_f0 = X_train[:, 2048:]
        X_val_mms = X_val[:, :2048]
        X_val_f0 = X_val[:, 2048:]
    else:
        X_train_mms = X_train[:, :2048]
        X_train_f0 = None
        X_val_mms = X_val[:, :2048]
        X_val_f0 = None

    # Convert to tensors
    train_mms = torch.tensor(X_train_mms, dtype=torch.float32)
    train_f0 = torch.tensor(X_train_f0, dtype=torch.float32) if use_f0 else None
    train_y = torch.tensor(y_train, dtype=torch.long)

    val_mms = torch.tensor(X_val_mms, dtype=torch.float32)
    val_f0 = torch.tensor(X_val_f0, dtype=torch.float32) if use_f0 else None
    val_y = torch.tensor(y_val, dtype=torch.long)

    # Model
    model = SimpleToneClassifier(use_f0=use_f0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Class weights for imbalance
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()

        if use_f0:
            logits = model(train_mms, train_f0)
        else:
            logits = model(train_mms, None)

        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            if use_f0:
                val_logits = model(val_mms, val_f0)
            else:
                val_logits = model(val_mms, None)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_y).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, val_acc={val_acc:.1%}")

    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_acc


def evaluate_model(model, X_test, y_test, use_f0=True, name="Model"):
    """Evaluate model and print per-tone accuracy."""

    if use_f0:
        test_mms = torch.tensor(X_test[:, :2048], dtype=torch.float32)
        test_f0 = torch.tensor(X_test[:, 2048:], dtype=torch.float32)
    else:
        test_mms = torch.tensor(X_test[:, :2048], dtype=torch.float32)
        test_f0 = None
    test_y = y_test

    model.eval()
    with torch.no_grad():
        if use_f0:
            logits = model(test_mms, test_f0)
        else:
            logits = model(test_mms, None)
        preds = logits.argmax(dim=-1).numpy()

    # Overall accuracy
    overall = (preds == test_y).mean()
    print(f"\n{name}: {overall:.1%} overall")

    # Per-tone
    for tone in range(4):
        mask = test_y == tone
        if mask.sum() > 0:
            acc = (preds[mask] == test_y[mask]).mean()
            n_correct = (preds[mask] == test_y[mask]).sum()
            n_total = mask.sum()
            print(f"  T{tone+1}: {acc:.1%} ({n_correct}/{n_total})")

    # Confusion matrix
    print("\n  Confusion matrix:")
    print("      T1  T2  T3  T4")
    for true in range(4):
        row = []
        for pred in range(4):
            count = ((test_y == true) & (preds == pred)).sum()
            row.append(f"{count:3d}")
        print(f"  T{true+1} {' '.join(row)}")

    return overall


def main():
    print("=" * 60)
    print("RETRAIN CLASSIFIER WITH TTS DATA")
    print("=" * 60)
    print("\nGoal: Test if T3 can be learned with clean, verified labels\n")

    # Generate data
    samples = generate_training_data()

    if len(samples) < 50:
        print("ERROR: Not enough samples")
        return

    # Extract features
    mms_features, f0_features, labels = extract_features(samples)

    # Combine features
    combined = np.concatenate([mms_features, f0_features], axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        combined, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"\nTrain: {len(y_train)}, Test: {len(y_test)}")
    print(f"Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Train with F0
    print("\n" + "=" * 60)
    print("TRAINING WITH F0 FEATURES")
    print("=" * 60)

    model_with_f0, best_acc = train_classifier(X_train, y_train, X_test, y_test, use_f0=True)
    evaluate_model(model_with_f0, X_test, y_test, use_f0=True, name="Classifier with F0")

    # Train without F0
    print("\n" + "=" * 60)
    print("TRAINING WITHOUT F0 FEATURES")
    print("=" * 60)

    model_no_f0, best_acc = train_classifier(X_train, y_train, X_test, y_test, use_f0=False)
    evaluate_model(model_no_f0, X_test, y_test, use_f0=False, name="Classifier without F0")

    # Summary
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)


if __name__ == "__main__":
    main()
