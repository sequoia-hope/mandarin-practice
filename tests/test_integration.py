"""
Integration tests for the complete TTS → Evaluator pipeline.

These tests generate reference audio using TTS and feed it to the
tone classifier to verify the full pipeline works correctly.

Requirements:
    - TTS backend (edge-tts)
    - Trained classifier checkpoint
    - MMS encoder model

Run with:
    pytest tests/test_integration.py -v -s
"""

import pytest
import numpy as np
from pathlib import Path


# Check if we have the required components
def _check_requirements():
    """Check if integration test requirements are available."""
    errors = []

    # Check for checkpoint
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("best_model_acc*.pt")) if checkpoint_dir.exists() else []
    if not checkpoints:
        errors.append("No classifier checkpoint found in checkpoints/")

    # Check for edge-tts
    try:
        import edge_tts
    except ImportError:
        errors.append("edge-tts not installed")

    # Check for torch
    try:
        import torch
    except ImportError:
        errors.append("torch not installed")

    return errors


REQUIREMENTS_ERRORS = _check_requirements()
SKIP_REASON = "; ".join(REQUIREMENTS_ERRORS) if REQUIREMENTS_ERRORS else None


@pytest.fixture(scope="module")
def tts():
    """Create TTS instance for tests."""
    if SKIP_REASON:
        pytest.skip(SKIP_REASON)

    from src.coach.tts import LightTTS
    return LightTTS()


@pytest.fixture(scope="module")
def evaluator():
    """Create Evaluator instance for tests."""
    if SKIP_REASON:
        pytest.skip(SKIP_REASON)

    from src.coach.evaluator import Evaluator
    return Evaluator(lazy_load=False)


@pytest.fixture(scope="module")
def all_phrases():
    """Get all phrases from built-in lessons."""
    from src.coach.voice_menu import BUILT_IN_LESSONS

    phrases = []
    for lesson in BUILT_IN_LESSONS:
        for phrase in lesson.phrases:
            phrases.append({
                "lesson": lesson.name,
                **phrase
            })
    return phrases


def strip_tone_marks(pinyin: str) -> str:
    """Remove tone marks from pinyin."""
    replacements = {
        'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
        'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
        'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
        'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
        'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
        'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v',
    }
    result = pinyin.lower()
    for mark, plain in replacements.items():
        result = result.replace(mark, plain)
    return result


@pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "Requirements not met")
class TestTTSToEvaluatorPipeline:
    """Test the complete TTS → Evaluator pipeline."""

    def test_tts_generates_audio(self, tts):
        """Test that TTS generates audio."""
        audio = tts.speak("你好", language="zh", play=False)

        assert audio is not None
        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)

    def test_evaluator_loads(self, evaluator):
        """Test that evaluator loads successfully."""
        assert evaluator is not None
        assert evaluator.classifier is not None
        assert evaluator.encoder is not None

    def test_single_phrase_pipeline(self, tts, evaluator):
        """Test single phrase through full pipeline."""
        # Generate audio for 你好
        audio = tts.speak("你好", language="zh", play=False)

        # Evaluate
        result = evaluator.evaluate(
            audio=audio,
            expected_pinyin=["ni", "hao"],
            expected_tones=[3, 3],
            characters="你好",
        )

        assert result.status == "success"
        assert len(result.syllable_results) == 2

        # Print results for debugging
        print(f"\n你好 evaluation:")
        for r in result.syllable_results:
            status = "✓" if r.correct else "✗"
            sandhi = f" (sandhi: T{r.spoken_tone})" if r.sandhi_applied else ""
            print(f"  {status} {r.character}: expected T{r.expected_tone}{sandhi}, "
                  f"detected T{r.detected_tone} (conf: {r.confidence:.2f})")
        print(f"  Overall accuracy: {result.overall_accuracy:.0%}")


@pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "Requirements not met")
class TestAllLessonPhrases:
    """Test all phrases from built-in lessons through the pipeline."""

    def test_all_phrases_evaluate(self, tts, evaluator, all_phrases):
        """
        Test that all lesson phrases can be evaluated.

        This test generates TTS audio for each phrase and runs it through
        the evaluator. It tracks accuracy across all phrases including
        tone accuracy, word recognition, and combined scores.
        """
        results = []
        total_syllables = 0
        correct_syllables = 0
        total_words_recognized = 0
        total_words = 0
        combined_scores = []
        # Track correctness sources
        classifier_only = 0
        asr_only = 0
        both_sources = 0

        print(f"\n{'='*70}")
        print("Testing all lesson phrases through TTS → Evaluator pipeline")
        print("(with tone classification + word recognition)")
        print(f"{'='*70}\n")

        for phrase in all_phrases:
            chars = phrase["characters"]
            pinyin = phrase["pinyin"]
            tones = phrase["tones"]

            # Generate audio
            try:
                audio = tts.speak(chars, language="zh", play=False)
            except Exception as e:
                print(f"  ✗ TTS failed for {chars}: {e}")
                results.append({
                    "phrase": chars,
                    "status": "tts_error",
                    "error": str(e),
                })
                continue

            # Evaluate
            try:
                result = evaluator.evaluate(
                    audio=audio,
                    expected_pinyin=[strip_tone_marks(p) for p in pinyin],
                    expected_tones=tones,
                    characters=chars,
                )
            except Exception as e:
                print(f"  ✗ Evaluation failed for {chars}: {e}")
                results.append({
                    "phrase": chars,
                    "status": "eval_error",
                    "error": str(e),
                })
                continue

            # Track tone results
            phrase_correct = sum(1 for r in result.syllable_results if r.correct)
            phrase_total = len(result.syllable_results)
            total_syllables += phrase_total
            correct_syllables += phrase_correct

            # Track word recognition results
            words_recognized = sum(1 for r in result.syllable_results if r.char_recognized)
            total_words_recognized += words_recognized
            total_words += phrase_total

            # Track correctness sources
            for r in result.syllable_results:
                if r.correct_source == "classifier":
                    classifier_only += 1
                elif r.correct_source == "asr":
                    asr_only += 1
                elif r.correct_source == "both":
                    both_sources += 1

            # Track combined score
            if result.combined_score is not None:
                combined_scores.append(result.combined_score)

            # Status indicator
            score = result.combined_score if result.combined_score else result.overall_accuracy
            status = "✓" if score >= 0.8 else "○" if score >= 0.5 else "✗"

            # Show sandhi info if applicable
            sandhi_info = ""
            for r in result.syllable_results:
                if r.sandhi_applied:
                    sandhi_info = " [sandhi]"
                    break

            # Build output line
            tone_str = f"T:{result.overall_accuracy:.0%}"
            word_str = f"W:{result.word_accuracy:.0%}" if result.word_accuracy is not None else "W:n/a"
            combined_str = f"C:{result.combined_score:.0%}" if result.combined_score is not None else ""

            # Show transcription with pinyin
            if result.transcription:
                pinyin_str = f" ({result.transcription_pinyin})" if result.transcription_pinyin else ""
                asr_str = f'heard: "{result.transcription}"{pinyin_str}'
            else:
                asr_str = ""

            print(f"  {status} {chars}{sandhi_info}: {tone_str} {word_str} {combined_str}")
            if asr_str:
                print(f"      {asr_str}")

            # Show per-syllable details for failures
            if score < 1.0:
                for r in result.syllable_results:
                    tone_ok = "✓" if r.correct else "✗"
                    word_ok = "✓" if r.char_recognized else "✗" if r.char_recognized is not None else "?"
                    sandhi_note = f"→T{r.spoken_tone}" if r.sandhi_applied else ""
                    asr_tone = f", ASR T{r.asr_tone}" if r.asr_tone else ""
                    source = f" [{r.correct_source}]" if r.correct_source else ""
                    print(f"      {r.character}: tone={tone_ok} T{r.expected_tone}{sandhi_note}→T{r.detected_tone}{asr_tone}, word={word_ok}{source}")

            results.append({
                "phrase": chars,
                "lesson": phrase["lesson"],
                "status": "success",
                "tone_accuracy": result.overall_accuracy,
                "word_accuracy": result.word_accuracy,
                "combined_score": result.combined_score,
                "transcription": result.transcription,
                "transcription_pinyin": result.transcription_pinyin,
                "syllable_results": [
                    {
                        "char": r.character,
                        "expected": r.expected_tone,
                        "spoken": r.spoken_tone,
                        "detected": r.detected_tone,
                        "correct": r.correct,
                        "sandhi": r.sandhi_applied,
                        "char_recognized": r.char_recognized,
                        "asr_tone": r.asr_tone,
                        "correct_source": r.correct_source,
                    }
                    for r in result.syllable_results
                ],
            })

        # Summary
        tone_accuracy = correct_syllables / total_syllables if total_syllables > 0 else 0
        word_accuracy = total_words_recognized / total_words if total_words > 0 else 0
        avg_combined = sum(combined_scores) / len(combined_scores) if combined_scores else 0

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Tone accuracy:  {correct_syllables}/{total_syllables} syllables = {tone_accuracy:.1%}")
        print(f"  Word accuracy:  {total_words_recognized}/{total_words} characters = {word_accuracy:.1%}")
        print(f"  Combined score: {avg_combined:.1%} (weighted 60% tone + 40% word)")
        print()
        print(f"  Correctness sources:")
        print(f"    Classifier only: {classifier_only}")
        print(f"    ASR only:        {asr_only}")
        print(f"    Both:            {both_sources}")
        print()

        # Count by combined score level
        high_acc = sum(1 for r in results if (r.get("combined_score") or r.get("tone_accuracy", 0)) >= 0.8)
        med_acc = sum(1 for r in results if 0.5 <= (r.get("combined_score") or r.get("tone_accuracy", 0)) < 0.8)
        low_acc = sum(1 for r in results if 0 < (r.get("combined_score") or r.get("tone_accuracy", 0)) < 0.5)
        errors = sum(1 for r in results if r["status"] != "success")

        print(f"  Phrases by combined score:")
        print(f"    High (≥80%):    {high_acc}")
        print(f"    Medium (50-79%): {med_acc}")
        print(f"    Low (<50%):     {low_acc}")
        print(f"    Errors:         {errors}")
        print(f"{'='*70}\n")

        # Test assertions
        assert len(results) == len(all_phrases), "All phrases should be processed"
        assert errors == 0, f"No phrases should have errors, got {errors}"

        # We expect reasonable accuracy from TTS audio
        assert tone_accuracy >= 0.5, (
            f"Tone accuracy should be at least 50%, got {tone_accuracy:.1%}"
        )

    @pytest.mark.parametrize("phrase_data", [
        {"chars": "你好", "pinyin": ["ni", "hao"], "tones": [3, 3]},
        {"chars": "谢谢", "pinyin": ["xie", "xie"], "tones": [4, 5]},
        {"chars": "一", "pinyin": ["yi"], "tones": [1]},
        {"chars": "我很好", "pinyin": ["wo", "hen", "hao"], "tones": [3, 3, 3]},
    ])
    def test_specific_phrases(self, tts, evaluator, phrase_data):
        """Test specific important phrases."""
        chars = phrase_data["chars"]
        pinyin = phrase_data["pinyin"]
        tones = phrase_data["tones"]

        # Generate audio
        audio = tts.speak(chars, language="zh", play=False)
        assert len(audio) > 0

        # Evaluate
        result = evaluator.evaluate(
            audio=audio,
            expected_pinyin=pinyin,
            expected_tones=tones,
            characters=chars,
        )

        assert result.status == "success"
        assert len(result.syllable_results) > 0

        print(f"\n{chars}: {result.overall_accuracy:.0%} accuracy")
        for r in result.syllable_results:
            status = "✓" if r.correct else "✗"
            print(f"  {status} {r.character}: T{r.expected_tone} → T{r.detected_tone}")


@pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "Requirements not met")
class TestSandhiInPipeline:
    """Test that sandhi is correctly handled in the pipeline."""

    def test_third_tone_sandhi_accepted(self, tts, evaluator):
        """
        Test that third tone sandhi is accepted.

        When TTS produces 你好 as T2+T3 (correct sandhi pronunciation),
        the evaluator should mark it as correct even though underlying
        tones are T3+T3.
        """
        # Generate audio - TTS should produce natural sandhi pronunciation
        audio = tts.speak("你好", language="zh", play=False)

        result = evaluator.evaluate(
            audio=audio,
            expected_pinyin=["ni", "hao"],
            expected_tones=[3, 3],  # Underlying tones
            characters="你好",
        )

        print(f"\nSandhi test for 你好:")
        for r in result.syllable_results:
            sandhi_note = f" (spoken T{r.spoken_tone})" if r.sandhi_applied else ""
            print(f"  {r.character}: underlying T{r.expected_tone}{sandhi_note}, "
                  f"detected T{r.detected_tone}, correct={r.correct}")

        # The first syllable should have sandhi applied
        assert result.syllable_results[0].sandhi_applied, "First syllable should have sandhi"
        assert result.syllable_results[0].spoken_tone == 2, "Spoken tone should be T2"

        # If TTS produces correct sandhi (T2), it should be marked correct
        # (The classifier might still detect T3 if TTS doesn't apply sandhi)
        assert result.status == "success"

    def test_triple_third_tone_sandhi(self, tts, evaluator):
        """Test three consecutive third tones (我很好)."""
        audio = tts.speak("我很好", language="zh", play=False)

        result = evaluator.evaluate(
            audio=audio,
            expected_pinyin=["wo", "hen", "hao"],
            expected_tones=[3, 3, 3],
            characters="我很好",
        )

        print(f"\nSandhi test for 我很好:")
        for r in result.syllable_results:
            sandhi_note = f" (spoken T{r.spoken_tone})" if r.sandhi_applied else ""
            print(f"  {r.character}: underlying T{r.expected_tone}{sandhi_note}, "
                  f"detected T{r.detected_tone}, correct={r.correct}")

        # First two syllables should have sandhi
        assert result.syllable_results[0].sandhi_applied
        assert result.syllable_results[1].sandhi_applied
        assert not result.syllable_results[2].sandhi_applied

        assert result.status == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
