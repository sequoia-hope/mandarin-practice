#!/usr/bin/env python3
"""
Analyze which MMS layers best capture tone information.

MMS-300M has 24 transformer layers. This script:
1. Generates TTS audio for minimal pairs (all 4 tones)
2. Extracts features from each layer
3. Trains a simple linear probe on each layer
4. Reports per-layer accuracy to find optimal layers for tone classification

The goal is to find which layers encode prosodic/tonal information vs phonetic information.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TONE_COACH_DEVICE"] = "cpu"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


# Expanded minimal pairs for better training signal
TRAINING_CHARS = [
    # (character, tone)
    # ma series
    ("妈", 1), ("麻", 2), ("马", 3), ("骂", 4),
    # shi series
    ("诗", 1), ("十", 2), ("史", 3), ("是", 4),
    # shu series
    ("书", 1), ("熟", 2), ("鼠", 3), ("树", 4),
    # tang series
    ("汤", 1), ("糖", 2), ("躺", 3), ("烫", 4),
    # wen series
    ("温", 1), ("文", 2), ("吻", 3), ("问", 4),
    # yi series
    ("一", 1), ("姨", 2), ("椅", 3), ("意", 4),
    # du series
    ("都", 1), ("读", 2), ("堵", 3), ("度", 4),
    # ba series
    ("八", 1), ("拔", 2), ("把", 3), ("爸", 4),
    # Additional common words
    ("他", 1), ("拿", 2), ("哪", 3), ("那", 4),
    ("飞", 1), ("肥", 2), ("匪", 3), ("费", 4),
    ("高", 1), ("搞", 3), ("告", 4),
    ("开", 1), ("凯", 3), ("看", 4),
    ("天", 1), ("甜", 2), ("舔", 3), ("田", 2),
    ("山", 1), ("闪", 3), ("善", 4),
    ("花", 1), ("华", 2), ("画", 4),
    ("家", 1), ("夹", 2), ("假", 3), ("价", 4),
]


def generate_training_audio():
    """Generate TTS audio for all training characters."""
    from src.coach.tts import LightTTS

    print("Generating TTS audio for training samples...")
    tts = LightTTS()

    samples = []
    for char, tone in tqdm(TRAINING_CHARS, desc="Generating audio"):
        try:
            audio = tts.speak(char, language="zh", speed="normal", play=False)
            if audio is not None and len(audio) > 1600:  # At least 0.1s
                samples.append((char, tone, audio))
        except Exception as e:
            print(f"  Failed for {char}: {e}")

    print(f"Generated {len(samples)} audio samples")

    # Count per tone
    tone_counts = defaultdict(int)
    for _, tone, _ in samples:
        tone_counts[tone] += 1
    print(f"Tone distribution: {dict(sorted(tone_counts.items()))}")

    return samples


def extract_all_layer_features(samples, device="cpu"):
    """Extract features from all MMS layers for each sample."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    print("\nLoading MMS model...")
    model_name = "facebook/mms-300m"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(device)

    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_size = model.config.hidden_size

    print(f"MMS has {num_layers} layers (including embeddings), hidden_size={hidden_size}")

    # Extract features for each sample from each layer
    all_features = {i: [] for i in range(num_layers)}
    all_labels = []

    print("\nExtracting features from all layers...")
    with torch.no_grad():
        for char, tone, audio in tqdm(samples, desc="Extracting features"):
            # Process audio
            inputs = feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(device)

            # Forward pass
            outputs = model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, time, hidden)

            # Extract mean-pooled features from each layer
            for layer_idx, layer_output in enumerate(hidden_states):
                # Mean pool over time dimension
                pooled = layer_output.mean(dim=1).squeeze().cpu().numpy()
                all_features[layer_idx].append(pooled)

            all_labels.append(tone - 1)  # Convert to 0-indexed

    # Convert to numpy arrays
    for layer_idx in all_features:
        all_features[layer_idx] = np.array(all_features[layer_idx])
    all_labels = np.array(all_labels)

    return all_features, all_labels, num_layers


def train_linear_probe(features, labels, layer_idx):
    """Train a simple linear classifier and return cross-validated accuracy."""
    # Use logistic regression with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # 5-fold cross-validation
    scores = cross_val_score(clf, features, labels, cv=5, scoring='accuracy')

    return scores.mean(), scores.std()


def train_mlp_probe(features, labels, layer_idx, hidden_dim=64):
    """Train a small MLP and return accuracy."""
    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Simple train/test split (80/20)
    n_samples = len(y)
    indices = torch.randperm(n_samples)
    train_size = int(0.8 * n_samples)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Build MLP
    input_dim = features.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 4)  # 4 tones
    )

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy


def analyze_layer_combinations(all_features, all_labels, top_layers):
    """Test combinations of top-performing layers."""
    print("\n" + "=" * 70)
    print("LAYER COMBINATION ANALYSIS")
    print("=" * 70)

    # Test pairs
    print("\nTesting layer pairs...")
    pair_results = []
    for i, layer_i in enumerate(top_layers):
        for layer_j in top_layers[i+1:]:
            combined = np.concatenate([all_features[layer_i], all_features[layer_j]], axis=1)
            acc, std = train_linear_probe(combined, all_labels, f"{layer_i}+{layer_j}")
            pair_results.append((f"L{layer_i}+L{layer_j}", acc, std))

    pair_results.sort(key=lambda x: -x[1])
    print("\nTop layer pairs:")
    for name, acc, std in pair_results[:10]:
        print(f"  {name}: {acc:.1%} (+/- {std:.1%})")

    # Test triplets of top 5
    print("\nTesting layer triplets (top 5 layers)...")
    triplet_results = []
    top5 = top_layers[:5]
    for i, layer_i in enumerate(top5):
        for j, layer_j in enumerate(top5[i+1:], i+1):
            for layer_k in top5[j+1:]:
                combined = np.concatenate([
                    all_features[layer_i],
                    all_features[layer_j],
                    all_features[layer_k]
                ], axis=1)
                acc, std = train_linear_probe(combined, all_labels, f"{layer_i}+{layer_j}+{layer_k}")
                triplet_results.append((f"L{layer_i}+L{layer_j}+L{layer_k}", acc, std))

    triplet_results.sort(key=lambda x: -x[1])
    print("\nTop layer triplets:")
    for name, acc, std in triplet_results[:10]:
        print(f"  {name}: {acc:.1%} (+/- {std:.1%})")

    return pair_results, triplet_results


def analyze_per_tone_accuracy(all_features, all_labels, layer_idx):
    """Analyze accuracy breakdown by tone for a specific layer."""
    from sklearn.model_selection import train_test_split

    features = all_features[layer_idx]

    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Per-tone accuracy
    tone_acc = {}
    for tone in range(4):
        mask = y_test == tone
        if mask.sum() > 0:
            tone_acc[tone + 1] = (y_pred[mask] == y_test[mask]).mean()

    return tone_acc


def main():
    print("=" * 70)
    print("MMS LAYER ANALYSIS FOR TONE CLASSIFICATION")
    print("=" * 70)

    # Generate training data
    samples = generate_training_audio()

    if len(samples) < 20:
        print("ERROR: Not enough samples generated")
        return

    # Extract features from all layers
    all_features, all_labels, num_layers = extract_all_layer_features(samples)

    # Train linear probes on each layer
    print("\n" + "=" * 70)
    print("LINEAR PROBE ACCURACY BY LAYER")
    print("=" * 70)

    layer_results = []
    for layer_idx in tqdm(range(num_layers), desc="Training probes"):
        features = all_features[layer_idx]
        acc, std = train_linear_probe(features, all_labels, layer_idx)
        layer_results.append((layer_idx, acc, std))

    # Sort by accuracy
    layer_results.sort(key=lambda x: -x[1])

    print("\nLayer rankings (best to worst):")
    print("-" * 40)
    for layer_idx, acc, std in layer_results:
        bar = "#" * int(acc * 40)
        print(f"  Layer {layer_idx:2d}: {acc:.1%} (+/- {std:.1%}) {bar}")

    # Identify layer groups
    print("\n" + "=" * 70)
    print("LAYER GROUP ANALYSIS")
    print("=" * 70)

    # Group by accuracy
    top_layers = [l for l, a, s in layer_results if a > 0.5]
    mid_layers = [l for l, a, s in layer_results if 0.3 <= a <= 0.5]
    low_layers = [l for l, a, s in layer_results if a < 0.3]

    print(f"\nHigh accuracy (>50%): Layers {sorted(top_layers)}")
    print(f"Medium accuracy (30-50%): Layers {sorted(mid_layers)}")
    print(f"Low accuracy (<30%): Layers {sorted(low_layers)}")

    # Current configuration comparison
    print("\n" + "=" * 70)
    print("CURRENT VS OPTIMAL CONFIGURATION")
    print("=" * 70)

    # Current: prosody layers 8-12, phonetic 15-17
    current_prosody = [8, 9, 10, 11, 12]
    current_phonetic = [15, 16, 17]
    current_all = current_prosody + current_phonetic

    print(f"\nCurrent configuration:")
    print(f"  Prosody layers: {current_prosody}")
    print(f"  Phonetic layers: {current_phonetic}")

    # Check accuracy of current layers
    current_accs = [(l, acc, std) for l, acc, std in layer_results if l in current_all]
    current_accs.sort(key=lambda x: x[0])
    print(f"\nCurrent layer accuracies:")
    for l, acc, std in current_accs:
        label = "prosody" if l in current_prosody else "phonetic"
        print(f"  Layer {l} ({label}): {acc:.1%}")

    # Recommended layers based on analysis
    top5_layers = [l for l, a, s in layer_results[:5]]
    print(f"\nTop 5 layers by accuracy: {top5_layers}")

    # Test combination of top layers
    if len(top_layers) >= 2:
        pair_results, triplet_results = analyze_layer_combinations(
            all_features, all_labels, [l for l, a, s in layer_results[:8]]
        )

    # Per-tone analysis for best layer
    best_layer = layer_results[0][0]
    print("\n" + "=" * 70)
    print(f"PER-TONE ACCURACY FOR BEST LAYER (Layer {best_layer})")
    print("=" * 70)

    tone_acc = analyze_per_tone_accuracy(all_features, all_labels, best_layer)
    for tone, acc in sorted(tone_acc.items()):
        print(f"  Tone {tone}: {acc:.1%}")

    # Compare with worst-performing top layer
    if len(top_layers) > 1:
        comparison_layer = layer_results[4][0] if len(layer_results) > 4 else layer_results[-1][0]
        print(f"\nPER-TONE ACCURACY FOR COMPARISON LAYER (Layer {comparison_layer})")
        tone_acc2 = analyze_per_tone_accuracy(all_features, all_labels, comparison_layer)
        for tone, acc in sorted(tone_acc2.items()):
            print(f"  Tone {tone}: {acc:.1%}")

    # Summary recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    best_single = layer_results[0]
    print(f"\n1. Best single layer: Layer {best_single[0]} ({best_single[1]:.1%} accuracy)")

    if 'pair_results' in dir() and pair_results:
        best_pair = pair_results[0]
        print(f"2. Best layer pair: {best_pair[0]} ({best_pair[1]:.1%} accuracy)")

    if 'triplet_results' in dir() and triplet_results:
        best_triplet = triplet_results[0]
        print(f"3. Best layer triplet: {best_triplet[0]} ({best_triplet[1]:.1%} accuracy)")

    # Check if current config is suboptimal
    current_combined = np.concatenate(
        [all_features[l] for l in current_all], axis=1
    )
    current_acc, current_std = train_linear_probe(current_combined, all_labels, "current")
    print(f"\n4. Current configuration (L{current_all}): {current_acc:.1%} accuracy")

    if pair_results and pair_results[0][1] > current_acc:
        print(f"\n   -> Best pair outperforms current config by {pair_results[0][1] - current_acc:.1%}")


if __name__ == "__main__":
    main()
