#!/usr/bin/env python3
"""
Test tone classification accuracy using sound-alike phrases (minimal pairs).

This test generates audio using both QwenTTS and LightTTS (edge-tts) to ensure
variance in samples, then runs them through the evaluation system to measure
how accurately the tone classifier detects tones.

We test:
1. Classifier-only accuracy (without ASR)
2. Full evaluator accuracy (with ASR override logic)
3. Breakdown by tone and TTS backend
"""

import os
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

# Set CPU mode for testing
os.environ["TONE_COACH_DEVICE"] = "cpu"


@dataclass
class MinimalPair:
    """A set of characters that share the same pinyin base but differ in tone."""
    pinyin_base: str  # e.g., "ma"
    variants: list  # [(character, tone, meaning), ...]


# Minimal pairs for testing - words that sound alike but differ only in tone
MINIMAL_PAIRS = [
    MinimalPair(
        pinyin_base="ma",
        variants=[
            ("妈", 1, "mother"),
            ("麻", 2, "hemp"),
            ("马", 3, "horse"),
            ("骂", 4, "scold"),
        ]
    ),
    MinimalPair(
        pinyin_base="shi",
        variants=[
            ("诗", 1, "poem"),
            ("十", 2, "ten"),
            ("史", 3, "history"),
            ("是", 4, "is"),
        ]
    ),
    MinimalPair(
        pinyin_base="shu",
        variants=[
            ("书", 1, "book"),
            ("熟", 2, "cooked"),
            ("鼠", 3, "rat"),
            ("树", 4, "tree"),
        ]
    ),
    MinimalPair(
        pinyin_base="mai",
        variants=[
            ("买", 3, "buy"),
            ("卖", 4, "sell"),
        ]
    ),
    MinimalPair(
        pinyin_base="tang",
        variants=[
            ("汤", 1, "soup"),
            ("糖", 2, "sugar"),
            ("躺", 3, "lie down"),
            ("烫", 4, "hot"),
        ]
    ),
    MinimalPair(
        pinyin_base="wen",
        variants=[
            ("温", 1, "warm"),
            ("文", 2, "culture"),
            ("吻", 3, "kiss"),
            ("问", 4, "ask"),
        ]
    ),
    MinimalPair(
        pinyin_base="yi",
        variants=[
            ("一", 1, "one"),
            ("姨", 2, "aunt"),
            ("椅", 3, "chair"),
            ("意", 4, "meaning"),
        ]
    ),
    MinimalPair(
        pinyin_base="du",
        variants=[
            ("都", 1, "capital"),
            ("读", 2, "read"),
            ("堵", 3, "block"),
            ("度", 4, "degree"),
        ]
    ),
]


@dataclass
class TestResult:
    """Result of a single tone classification test."""
    character: str
    expected_tone: int
    classifier_tone: int
    classifier_confidence: float
    asr_tone: Optional[int]
    asr_text: str
    final_correct: bool  # After all override logic
    tts_backend: str
    pinyin_base: str


def generate_audio_samples(pairs: list[MinimalPair], output_dir: Path) -> dict:
    """Generate audio samples using both TTS backends."""
    from src.coach.tts import LightTTS, QwenTTS

    samples = {}  # {(character, tts_backend): audio_array}

    # Initialize TTS engines
    print("  Initializing LightTTS (edge-tts)...")
    light_tts = LightTTS()

    # Try to initialize QwenTTS, but it may not be available
    has_qwen = False
    qwen_tts = None
    try:
        print("  Initializing QwenTTS...")
        qwen_tts = QwenTTS()
        has_qwen = qwen_tts._available
        if has_qwen:
            print("  QwenTTS available!")
        else:
            print("  QwenTTS not available (module not found)")
    except Exception as e:
        print(f"  QwenTTS not available: {e}")

    for pair in pairs:
        for char, tone, meaning in pair.variants:
            # Generate with LightTTS
            try:
                audio = light_tts.speak(char, language="zh", speed="normal", play=False)
                if audio is not None and len(audio) > 0:
                    samples[(char, "light")] = audio
                    print(f"    {char} (T{tone}) - LightTTS: {len(audio)/16000:.2f}s")
            except Exception as e:
                print(f"    {char} - LightTTS failed: {e}")

            # Generate with QwenTTS if available
            if has_qwen:
                try:
                    audio = qwen_tts.speak(char, language="zh", speed="normal", play=False)
                    if audio is not None and len(audio) > 0:
                        samples[(char, "qwen")] = audio
                        print(f"    {char} (T{tone}) - QwenTTS: {len(audio)/16000:.2f}s")
                except Exception as e:
                    print(f"    {char} - QwenTTS failed: {e}")

    return samples


def run_classifier_only(audio: np.ndarray, evaluator) -> tuple[int, float]:
    """Run just the tone classifier without ASR."""
    # Extract MMS features
    features = evaluator.encoder.encode_audio(audio, sample_rate=16000)
    mms_features = features["pooled"]

    # Extract F0 features if available
    f0_features = None
    if evaluator.f0_extractor:
        f0_result = evaluator.f0_extractor.extract(audio)
        f0_features_raw = f0_result.get("features", np.zeros((1, 3)))
        if len(f0_features_raw) > 0:
            f0_summary = f0_features_raw.mean(axis=0)
            f0_features = torch.tensor(f0_summary).unsqueeze(0).to(evaluator.device)

    # Classify
    with torch.no_grad():
        if f0_features is not None and evaluator.classifier.use_f0:
            logits = evaluator.classifier(mms_features, f0_features)
        else:
            logits = evaluator.classifier(mms_features, None)

        probs = torch.softmax(logits, dim=-1)
        confidence, predicted_class = probs.max(dim=-1)

        # Convert class index (0-4) to tone number (1-5)
        tone_number = predicted_class.item() + 1

        return tone_number, confidence.item()


def run_full_evaluator(audio: np.ndarray, character: str, expected_tone: int,
                       evaluator) -> tuple[int, Optional[int], str, bool]:
    """Run full evaluator with ASR."""
    from pypinyin import pinyin, Style

    # Get pinyin for character
    char_pinyin_list = pinyin(character, style=Style.TONE3)
    char_pinyin = char_pinyin_list[0][0] if char_pinyin_list else character

    # Build expected phrase info - evaluator.evaluate needs different args
    result = evaluator.evaluate(
        audio=audio,
        expected_pinyin=[char_pinyin.rstrip("12345")],  # Remove tone number
        expected_tones=[expected_tone],
        characters=character,
    )

    if result.syllable_results:
        sr = result.syllable_results[0]
        return (
            sr.detected_tone,
            sr.asr_tone,
            result.transcription or "",
            sr.correct
        )

    return 0, None, "", False


def test_tone_classification_accuracy():
    """Main test: evaluate tone classification accuracy on minimal pairs."""
    from src.coach.evaluator import Evaluator

    print("\n" + "=" * 70)
    print("TONE CLASSIFICATION ACCURACY TEST")
    print("=" * 70)
    print("\nThis test evaluates how accurately the tone classifier detects tones")
    print("using minimal pairs (words that differ only in tone).\n")

    # Generate audio samples
    print("Generating audio samples...")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        samples = generate_audio_samples(MINIMAL_PAIRS, tmpdir)
        print(f"\nGenerated {len(samples)} audio samples.\n")

        if not samples:
            print("ERROR: No samples generated. Cannot run test.")
            return

        # Initialize evaluator (classifier-only first, then with ASR)
        print("Loading evaluator (classifier only, no ASR)...")
        print("-" * 40)
        evaluator = Evaluator(use_asr=False, lazy_load=False)

        # Run tests
        results: list[TestResult] = []

        print("\nRunning classification tests...")
        print("-" * 40)

        # Build lookup for expected tones
        tone_lookup = {}
        pinyin_lookup = {}
        for pair in MINIMAL_PAIRS:
            for char, tone, _ in pair.variants:
                tone_lookup[char] = tone
                pinyin_lookup[char] = pair.pinyin_base

        for (char, backend), audio in samples.items():
            expected_tone = tone_lookup[char]
            pinyin_base = pinyin_lookup[char]

            # Run classifier only
            classifier_tone, confidence = run_classifier_only(audio, evaluator)

            result = TestResult(
                character=char,
                expected_tone=expected_tone,
                classifier_tone=classifier_tone,
                classifier_confidence=confidence,
                asr_tone=None,
                asr_text="",
                final_correct=(classifier_tone == expected_tone),
                tts_backend=backend,
                pinyin_base=pinyin_base,
            )
            results.append(result)

            status = "+" if result.final_correct else "-"
            print(f"  {status} {char} (T{expected_tone}): "
                  f"classifier=T{classifier_tone} ({confidence:.1%}) [{backend}]")

        # Now test with ASR enabled
        print("\n" + "-" * 40)
        print("Loading evaluator with ASR (MMS)...")
        print("-" * 40)

        # Reset singleton
        Evaluator._instance = None
        evaluator_with_asr = Evaluator(use_asr=True, asr_backend="mms", lazy_load=False)

        results_with_asr: list[TestResult] = []

        print("\nRunning full evaluation (classifier + ASR)...")
        print("-" * 40)

        for (char, backend), audio in samples.items():
            expected_tone = tone_lookup[char]
            pinyin_base = pinyin_lookup[char]

            # Run classifier only (for comparison)
            classifier_tone, confidence = run_classifier_only(audio, evaluator_with_asr)

            # Run full evaluator with ASR
            detected_tone, asr_tone, asr_text, final_correct = run_full_evaluator(
                audio, char, expected_tone, evaluator_with_asr
            )

            result = TestResult(
                character=char,
                expected_tone=expected_tone,
                classifier_tone=classifier_tone,
                classifier_confidence=confidence,
                asr_tone=asr_tone,
                asr_text=asr_text,
                final_correct=final_correct,
                tts_backend=backend,
                pinyin_base=pinyin_base,
            )
            results_with_asr.append(result)

            status = "+" if final_correct else "-"
            asr_info = f"ASR='{asr_text}' T{asr_tone}" if asr_tone else f"ASR='{asr_text}'"
            print(f"  {status} {char} (T{expected_tone}): "
                  f"clf=T{classifier_tone}, {asr_info}, final={'OK' if final_correct else 'WRONG'} [{backend}]")

        # Print summary statistics
        print_summary(results, results_with_asr)


def print_summary(results: list[TestResult], results_with_asr: list[TestResult]):
    """Print detailed summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Overall accuracy
    classifier_correct = sum(1 for r in results if r.classifier_tone == r.expected_tone)
    classifier_total = len(results)
    classifier_acc = classifier_correct / classifier_total if classifier_total else 0

    final_correct = sum(1 for r in results_with_asr if r.final_correct)
    final_total = len(results_with_asr)
    final_acc = final_correct / final_total if final_total else 0

    print(f"\nOverall Accuracy:")
    print(f"  Classifier only: {classifier_correct}/{classifier_total} ({classifier_acc:.1%})")
    print(f"  With ASR logic:  {final_correct}/{final_total} ({final_acc:.1%})")

    # By TTS backend
    print(f"\nBy TTS Backend (classifier only):")
    for backend in ["light", "qwen"]:
        backend_results = [r for r in results if r.tts_backend == backend]
        if not backend_results:
            continue
        correct = sum(1 for r in backend_results if r.classifier_tone == r.expected_tone)
        total = len(backend_results)
        acc = correct / total if total else 0
        print(f"  {backend}: {correct}/{total} ({acc:.1%})")

    # By tone
    print(f"\nBy Expected Tone (classifier only):")
    for tone in [1, 2, 3, 4]:
        tone_results = [r for r in results if r.expected_tone == tone]
        if not tone_results:
            continue
        correct = sum(1 for r in tone_results if r.classifier_tone == r.expected_tone)
        total = len(tone_results)
        acc = correct / total if total else 0
        print(f"  Tone {tone}: {correct}/{total} ({acc:.1%})")

    # Confusion matrix
    print(f"\nConfusion Matrix (classifier only):")
    print("        " + " ".join(f"P{t}" for t in range(1, 5)))
    for expected in range(1, 5):
        row = []
        for predicted in range(1, 5):
            count = sum(1 for r in results
                       if r.expected_tone == expected and r.classifier_tone == predicted)
            row.append(f"{count:2d}")
        print(f"  T{expected}   " + "  ".join(row))
    print("  (Rows=Expected, Cols=Predicted)")

    # Cases where ASR helped
    print(f"\nASR Override Analysis:")
    classifier_wrong_asr_right = sum(
        1 for r in results_with_asr
        if r.classifier_tone != r.expected_tone and r.final_correct
    )
    classifier_right_asr_wrong = sum(
        1 for r in results_with_asr
        if r.classifier_tone == r.expected_tone and not r.final_correct
    )
    both_correct = sum(
        1 for r in results_with_asr
        if r.classifier_tone == r.expected_tone and r.final_correct
    )
    both_wrong = sum(
        1 for r in results_with_asr
        if r.classifier_tone != r.expected_tone and not r.final_correct
    )
    print(f"  Both correct:             {both_correct}")
    print(f"  Classifier wrong, ASR saved: {classifier_wrong_asr_right}")
    print(f"  Classifier right, ASR broke: {classifier_right_asr_wrong}")
    print(f"  Both wrong:               {both_wrong}")

    # By pinyin base (to see if certain sounds are harder)
    print(f"\nBy Pinyin Base (classifier only):")
    pinyin_bases = sorted(set(r.pinyin_base for r in results))
    for base in pinyin_bases:
        base_results = [r for r in results if r.pinyin_base == base]
        correct = sum(1 for r in base_results if r.classifier_tone == r.expected_tone)
        total = len(base_results)
        acc = correct / total if total else 0
        print(f"  {base:6s}: {correct}/{total} ({acc:.1%})")

    # Average confidence when correct vs incorrect
    correct_conf = [r.classifier_confidence for r in results if r.classifier_tone == r.expected_tone]
    wrong_conf = [r.classifier_confidence for r in results if r.classifier_tone != r.expected_tone]

    print(f"\nClassifier Confidence:")
    if correct_conf:
        print(f"  When correct: {np.mean(correct_conf):.1%} avg, {np.min(correct_conf):.1%}-{np.max(correct_conf):.1%} range")
    if wrong_conf:
        print(f"  When wrong:   {np.mean(wrong_conf):.1%} avg, {np.min(wrong_conf):.1%}-{np.max(wrong_conf):.1%} range")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_tone_classification_accuracy()
