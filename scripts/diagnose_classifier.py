#!/usr/bin/env python3
"""
Diagnose why the trained classifier fails on Tone 3 while linear probes succeed.

Tests:
1. Our trained MLP classifier (with/without F0)
2. A fresh linear classifier trained on same features
3. A fresh MLP trained from scratch

This helps isolate whether the issue is:
- The MLP architecture
- The F0 fusion
- The training data/procedure
- Class imbalance
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Same training data as layer analysis
TRAINING_CHARS = [
    ("妈", 1), ("麻", 2), ("马", 3), ("骂", 4),
    ("诗", 1), ("十", 2), ("史", 3), ("是", 4),
    ("书", 1), ("熟", 2), ("鼠", 3), ("树", 4),
    ("汤", 1), ("糖", 2), ("躺", 3), ("烫", 4),
    ("温", 1), ("文", 2), ("吻", 3), ("问", 4),
    ("一", 1), ("姨", 2), ("椅", 3), ("意", 4),
    ("都", 1), ("读", 2), ("堵", 3), ("度", 4),
    ("八", 1), ("拔", 2), ("把", 3), ("爸", 4),
    ("他", 1), ("拿", 2), ("哪", 3), ("那", 4),
    ("飞", 1), ("肥", 2), ("匪", 3), ("费", 4),
    ("高", 1), ("搞", 3), ("告", 4),
    ("开", 1), ("凯", 3), ("看", 4),
    ("天", 1), ("甜", 2), ("舔", 3), ("田", 2),
    ("山", 1), ("闪", 3), ("善", 4),
    ("花", 1), ("华", 2), ("画", 4),
    ("家", 1), ("夹", 2), ("假", 3), ("价", 4),
]


def generate_audio():
    """Generate TTS audio."""
    from src.coach.tts import LightTTS

    print("Generating TTS audio...")
    tts = LightTTS()

    samples = []
    for char, tone in tqdm(TRAINING_CHARS, desc="Generating"):
        try:
            audio = tts.speak(char, language="zh", speed="normal", play=False)
            if audio is not None and len(audio) > 1600:
                samples.append((char, tone, audio))
        except Exception as e:
            pass

    print(f"Generated {len(samples)} samples")
    return samples


def extract_features_with_evaluator(samples):
    """Extract features using our evaluator's encoder."""
    from src.coach.evaluator import Evaluator

    print("\nLoading evaluator...")
    evaluator = Evaluator(use_asr=False, lazy_load=False)

    mms_features = []
    f0_features = []
    labels = []

    print("Extracting features...")
    for char, tone, audio in tqdm(samples, desc="Extracting"):
        # MMS features
        features = evaluator.encoder.encode_audio(audio, sample_rate=16000)
        mms_feat = features["pooled"].squeeze().cpu().numpy()
        mms_features.append(mms_feat)

        # F0 features
        if evaluator.f0_extractor:
            f0_result = evaluator.f0_extractor.extract(audio)
            f0_raw = f0_result.get("features", np.zeros((1, 3)))
            f0_feat = f0_raw.mean(axis=0) if len(f0_raw) > 0 else np.zeros(3)
        else:
            f0_feat = np.zeros(3)
        f0_features.append(f0_feat)

        labels.append(tone - 1)  # 0-indexed

    return (
        np.array(mms_features),
        np.array(f0_features),
        np.array(labels),
        evaluator
    )


def test_trained_classifier(mms_features, f0_features, labels, evaluator):
    """Test our pre-trained classifier."""
    print("\n" + "=" * 60)
    print("TEST 1: Pre-trained Classifier")
    print("=" * 60)

    classifier = evaluator.classifier
    device = evaluator.device

    # Test with F0
    print("\nWith F0 features:")
    predictions_with_f0 = []
    for i in range(len(labels)):
        mms = torch.tensor(mms_features[i]).unsqueeze(0).to(device)
        f0 = torch.tensor(f0_features[i]).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if classifier.use_f0:
                logits = classifier(mms, f0)
            else:
                logits = classifier(mms, None)
            pred = logits.argmax(dim=-1).item()
            predictions_with_f0.append(pred)

    predictions_with_f0 = np.array(predictions_with_f0)
    report_accuracy(predictions_with_f0, labels, "Trained classifier (with F0)")

    # Test without F0
    print("\nWithout F0 features:")
    predictions_no_f0 = []
    for i in range(len(labels)):
        mms = torch.tensor(mms_features[i]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(mms, None)
            pred = logits.argmax(dim=-1).item()
            predictions_no_f0.append(pred)

    predictions_no_f0 = np.array(predictions_no_f0)
    report_accuracy(predictions_no_f0, labels, "Trained classifier (no F0)")


def test_fresh_linear(mms_features, f0_features, labels):
    """Train a fresh linear classifier on same features."""
    print("\n" + "=" * 60)
    print("TEST 2: Fresh Linear Classifier (sklearn)")
    print("=" * 60)

    # With F0
    combined = np.concatenate([mms_features, f0_features], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        combined, labels, test_size=0.3, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\nWith F0 features (MMS+F0):")
    report_accuracy(preds, y_test, "Linear (with F0)")

    # Without F0
    X_train, X_test, y_train, y_test = train_test_split(
        mms_features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\nWithout F0 features (MMS only):")
    report_accuracy(preds, y_test, "Linear (no F0)")


def test_fresh_mlp(mms_features, f0_features, labels):
    """Train a fresh MLP similar to our architecture."""
    print("\n" + "=" * 60)
    print("TEST 3: Fresh MLP (same architecture as ToneClassifier)")
    print("=" * 60)

    # Convert to tensors
    X_mms = torch.tensor(mms_features, dtype=torch.float32)
    X_f0 = torch.tensor(f0_features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Train/test split
    n = len(y)
    indices = torch.randperm(n)
    train_size = int(0.7 * n)

    train_idx, test_idx = indices[:train_size], indices[train_size:]

    # Build model matching ToneClassifier architecture
    # Input: 2048 (MMS) -> 256 (bottleneck) + 64 (F0 projected) = 320 -> 256 -> 128 -> 4

    class SimpleMLP(nn.Module):
        def __init__(self, use_f0=True):
            super().__init__()
            self.use_f0 = use_f0

            self.bottleneck = nn.Linear(2048, 256)

            if use_f0:
                self.f0_proj = nn.Sequential(
                    nn.Linear(3, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                )
                input_dim = 256 + 64
            else:
                input_dim = 256

            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 4),
            )

        def forward(self, mms, f0=None):
            x = self.bottleneck(mms)
            if self.use_f0 and f0 is not None:
                f0_feat = self.f0_proj(f0)
                x = torch.cat([x, f0_feat], dim=-1)
            return self.classifier(x)

    def train_and_eval(model, X_mms, X_f0, y, train_idx, test_idx, name, use_f0):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Train
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            if use_f0:
                logits = model(X_mms[train_idx], X_f0[train_idx])
            else:
                logits = model(X_mms[train_idx], None)
            loss = criterion(logits, y[train_idx])
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            if use_f0:
                logits = model(X_mms[test_idx], X_f0[test_idx])
            else:
                logits = model(X_mms[test_idx], None)
            preds = logits.argmax(dim=-1).numpy()

        report_accuracy(preds, y[test_idx].numpy(), name)

    # With F0
    print("\nWith F0 features:")
    model_with_f0 = SimpleMLP(use_f0=True)
    train_and_eval(model_with_f0, X_mms, X_f0, y, train_idx, test_idx, "Fresh MLP (with F0)", True)

    # Without F0
    print("\nWithout F0 features:")
    model_no_f0 = SimpleMLP(use_f0=False)
    train_and_eval(model_no_f0, X_mms, X_f0, y, train_idx, test_idx, "Fresh MLP (no F0)", False)


def report_accuracy(predictions, labels, name):
    """Print detailed accuracy report."""
    overall = (predictions == labels).mean()
    print(f"\n{name}: {overall:.1%} overall")

    # Per-tone
    for tone in range(4):
        mask = labels == tone
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).mean()
            count = mask.sum()
            print(f"  Tone {tone+1}: {acc:.1%} ({(predictions[mask] == labels[mask]).sum()}/{count})")

    # Confusion matrix
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print("      T1  T2  T3  T4")
    for true_tone in range(4):
        row = []
        for pred_tone in range(4):
            count = ((labels == true_tone) & (predictions == pred_tone)).sum()
            row.append(f"{count:3d}")
        print(f"  T{true_tone+1} {' '.join(row)}")


def analyze_f0_features(f0_features, labels):
    """Analyze F0 features by tone to see if they're problematic."""
    print("\n" + "=" * 60)
    print("F0 FEATURE ANALYSIS")
    print("=" * 60)

    print("\nF0 feature statistics by tone:")
    print("(Features are typically: mean_f0, f0_range, f0_slope)")

    for tone in range(4):
        mask = labels == tone
        tone_f0 = f0_features[mask]

        print(f"\nTone {tone+1} (n={mask.sum()}):")
        for feat_idx, feat_name in enumerate(["mean_f0", "range", "slope"]):
            values = tone_f0[:, feat_idx]
            print(f"  {feat_name}: mean={values.mean():.3f}, std={values.std():.3f}, "
                  f"min={values.min():.3f}, max={values.max():.3f}")


def main():
    print("=" * 60)
    print("CLASSIFIER DIAGNOSIS")
    print("=" * 60)
    print("\nGoal: Find why trained classifier fails on T3 while linear probes succeed\n")

    # Generate data
    samples = generate_audio()

    # Extract features
    mms_features, f0_features, labels, evaluator = extract_features_with_evaluator(samples)

    print(f"\nData shape: MMS={mms_features.shape}, F0={f0_features.shape}, labels={labels.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Run tests
    test_trained_classifier(mms_features, f0_features, labels, evaluator)
    test_fresh_linear(mms_features, f0_features, labels)
    test_fresh_mlp(mms_features, f0_features, labels)
    analyze_f0_features(f0_features, labels)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("""
If trained classifier fails but fresh models succeed:
  -> Training data issue (class imbalance, mislabeled T3 samples)

If all MLPs fail but linear succeeds:
  -> Architecture issue (bottleneck losing T3 info, dropout too high)

If F0 features hurt performance:
  -> F0 extraction issue or T3 has unusual F0 that confuses model

If nothing works on T3:
  -> TTS audio quality issue for T3
""")


if __name__ == "__main__":
    main()
