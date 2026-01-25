# Guided Practice Architecture

## Overview

User is shown a target phrase, speaks it, and gets per-syllable tone feedback.

```
┌─────────────────────────────────────────────────────────────────┐
│  TARGET: 你好吗  (nǐ hǎo ma)  Expected tones: [3, 3, 5]        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  USER SPEAKS: [audio waveform]                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ALIGN: Find syllable boundaries using known pinyin             │
│                                                                 │
│  Audio:    |----ni-----|----hao----|---ma---|                   │
│  Times:    0.0        0.35       0.72     1.05                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SEGMENT: Extract audio for each syllable                       │
│                                                                 │
│  segment_1 = audio[0.00:0.35]  # "ni"                          │
│  segment_2 = audio[0.35:0.72]  # "hao"                         │
│  segment_3 = audio[0.72:1.05]  # "ma"                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLASSIFY: Run tone classifier on each segment                  │
│                                                                 │
│  segment_1 → Tone 3 (92%)                                      │
│  segment_2 → Tone 2 (78%)  ← WRONG (expected 3)                │
│  segment_3 → Tone 5 (85%)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEEDBACK:                                                      │
│                                                                 │
│  你 (nǐ)  T3 ✓  [F0 contour looks good]                        │
│  好 (hǎo) T2 ✗  Expected T3 (dipping), you said T2 (rising)    │
│                 [Show F0 overlay: yours vs reference]           │
│  吗 (ma)  T5 ✓  [F0 contour looks good]                        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

```python
@dataclass
class PracticePhrase:
    """A phrase for guided practice."""
    characters: str          # "你好吗"
    pinyin: list[str]        # ["nǐ", "hǎo", "ma"]
    pinyin_plain: list[str]  # ["ni", "hao", "ma"] (for alignment)
    expected_tones: list[int]  # [3, 3, 5]
    english: str             # "How are you?"


@dataclass
class SyllableResult:
    """Result for a single syllable evaluation."""
    character: str                       # "好"
    pinyin: str                          # "hǎo"
    expected_tone: int                   # Underlying/lexical tone (3)
    detected_tone: int                   # Classifier detected tone (2)
    confidence: float                    # Classification confidence (0.78)
    correct: bool                        # Tone correct (considering all sources)
    f0_contour: Optional[np.ndarray]     # Extracted F0 contour
    # Sandhi-aware fields
    spoken_tone: Optional[int]           # Expected spoken tone after sandhi
    sandhi_applied: bool                 # Whether sandhi changed the expected tone
    # ASR-derived fields (dual ASR: MMS + Whisper)
    char_recognized: Optional[bool]      # ASR recognized this character
    asr_tone: Optional[int]              # Tone derived from ASR pinyin
    correct_source: Optional[str]        # What confirmed correctness:
                                         # "classifier", "asr_tone", "asr_char",
                                         # or combinations like "classifier+asr_char"


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    syllable_results: list[SyllableResult]
    overall_accuracy: float              # Tone accuracy (0-1)
    audio: np.ndarray
    duration_ms: int
    status: str                          # "success", "error", "timeout"
    error: Optional[str]                 # Error message if status != "success"
    # ASR results (from dual ASR: MMS + Whisper with OR logic)
    transcription: Optional[str]         # What ASR heard (Chinese characters)
    transcription_pinyin: Optional[str]  # Pinyin of transcription
    word_accuracy: Optional[float]       # Character recognition score (0-1)
    combined_score: Optional[float]      # Weighted: 60% tone + 40% word accuracy
```

### Validation Philosophy

The evaluator uses **flexible validation**: any indication of correctness counts as correct.

```python
# Tone is correct if ANY of these are true:
classifier_correct = detected_tone in acceptable_tones  # Including sandhi variants
asr_tone_correct = asr_tone in acceptable_tones         # ASR-derived tone matches
asr_char_correct = char_recognized is True              # ASR recognized the character

is_correct = classifier_correct or asr_tone_correct or asr_char_correct
```

This maximizes user success while still providing accurate feedback.

## Components

### 1. Phrase Database

```python
class PhraseDatabase:
    """Database of practice phrases organized by difficulty/topic."""

    def get_phrases_by_level(self, level: str) -> list[PracticePhrase]:
        """Get phrases for beginner/intermediate/advanced."""

    def get_phrases_by_tones(self, tones: list[int]) -> list[PracticePhrase]:
        """Get phrases containing specific tone patterns."""

    def get_minimal_pairs(self) -> list[tuple[PracticePhrase, PracticePhrase]]:
        """Get pairs that differ only in tone (e.g., 买/卖)."""
```

### 2. Aligner (Abstract Interface)

```python
class Aligner(ABC):
    """Abstract interface for audio-text alignment."""

    @abstractmethod
    def align(
        self,
        audio: np.ndarray,
        pinyin: list[str],
        sample_rate: int = 16000
    ) -> list[tuple[float, float]]:
        """
        Align audio to pinyin syllables.

        Returns:
            List of (start_time, end_time) for each syllable
        """

# Implementation options:
class MFAAligner(Aligner):
    """Montreal Forced Aligner - accurate but slow."""

class CTCAligner(Aligner):
    """Use CTC model outputs for alignment - fast."""

class EnergyAligner(Aligner):
    """Simple energy-based syllable detection - fastest but crude."""
```

### 3. Guided Practice Pipeline

```python
class GuidedPracticePipeline:
    """Main pipeline for guided practice."""

    def __init__(
        self,
        tone_classifier: ToneClassifier,
        encoder: MMSEncoder | OmnilingualEncoder,
        aligner: Aligner,
        f0_extractor: F0Extractor,
    ):
        self.classifier = tone_classifier
        self.encoder = encoder
        self.aligner = aligner
        self.f0 = f0_extractor

    def evaluate_attempt(
        self,
        audio: np.ndarray,
        phrase: PracticePhrase,
    ) -> PhraseResult:
        """
        Evaluate a user's attempt at speaking a phrase.

        1. Align audio to expected pinyin
        2. Segment audio by syllable
        3. Classify each segment
        4. Compare to expected tones
        5. Return detailed results
        """
        # Step 1: Align
        boundaries = self.aligner.align(audio, phrase.pinyin_plain)

        # Step 2-4: Segment and classify each syllable
        syllable_results = []
        for i, (start, end) in enumerate(boundaries):
            segment = self._extract_segment(audio, start, end)

            # Extract features and classify
            features = self.encoder.encode_audio(segment)
            f0_contour = self.f0.extract(segment)
            tone, confidence = self.classifier.predict(features, f0_contour)

            result = SyllableResult(
                syllable_idx=i,
                character=phrase.characters[i],
                pinyin=phrase.pinyin[i],
                expected_tone=phrase.expected_tones[i],
                detected_tone=tone,
                confidence=confidence,
                correct=(tone == phrase.expected_tones[i]),
                start_time=start,
                end_time=end,
                audio_segment=segment,
                f0_contour=f0_contour,
            )
            syllable_results.append(result)

        # Step 5: Aggregate results
        accuracy = sum(r.correct for r in syllable_results) / len(syllable_results)

        return PhraseResult(
            phrase=phrase,
            syllable_results=syllable_results,
            overall_accuracy=accuracy,
            audio=audio,
            alignment_method=self.aligner.__class__.__name__,
        )
```

### 4. Feedback Generator

```python
class FeedbackGenerator:
    """Generate human-readable feedback from results."""

    def generate_feedback(self, result: PhraseResult) -> str:
        """Generate text feedback for the attempt."""

    def generate_f0_comparison(
        self,
        syllable: SyllableResult
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (user_f0, reference_f0) for visualization.
        Reference F0 is a canonical contour for the expected tone.
        """

    def get_coaching_tip(self, syllable: SyllableResult) -> str:
        """
        Get specific coaching tip based on the error.

        e.g., "You said Tone 2 (rising) but this should be Tone 3 (dipping).
              Try starting lower and dipping down before rising."
        """
```

## Alignment Strategy Comparison

| Method | Speed | Accuracy | Dependencies |
|--------|-------|----------|--------------|
| **MFA** | ~2-5s | High | MFA installed, temp files |
| **CTC-based** | ~200ms | Medium-High | ASR model with CTC |
| **Energy-based** | ~50ms | Low-Medium | None |

### Recommended: Hybrid Approach

1. **First pass**: Energy-based for instant feedback
2. **Background**: Run MFA for accurate alignment
3. **Update UI**: Refine feedback when MFA completes

## Simple Energy-Based Aligner (MVP)

For an MVP, we can use energy-based segmentation:

```python
class EnergyAligner(Aligner):
    """Simple syllable boundary detection using energy."""

    def align(
        self,
        audio: np.ndarray,
        pinyin: list[str],
        sample_rate: int = 16000,
    ) -> list[tuple[float, float]]:
        # Compute short-time energy
        frame_length = int(0.025 * sample_rate)  # 25ms
        hop_length = int(0.010 * sample_rate)    # 10ms

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        energy = np.array(energy)

        # Smooth energy
        from scipy.ndimage import gaussian_filter1d
        energy_smooth = gaussian_filter1d(energy, sigma=3)

        # Find valleys (syllable boundaries)
        from scipy.signal import find_peaks
        valleys, _ = find_peaks(-energy_smooth, distance=10)

        # Convert to times
        valley_times = valleys * hop_length / sample_rate

        # Create boundaries for N syllables
        n_syllables = len(pinyin)
        boundaries = self._split_into_n_segments(
            valley_times, n_syllables, len(audio) / sample_rate
        )

        return boundaries
```

## UI Integration

```python
class GuidedPracticeUI:
    """Terminal UI for guided practice."""

    def run_session(self, phrases: list[PracticePhrase]):
        for phrase in phrases:
            # Display target
            self.display_target(phrase)

            # Record attempt
            print("Speak now...")
            audio = self.record_utterance()

            # Evaluate
            result = self.pipeline.evaluate_attempt(audio, phrase)

            # Show feedback
            self.display_results(result)

            # Show F0 visualization for errors
            for syl in result.syllable_results:
                if not syl.correct:
                    self.show_f0_comparison(syl)
```

## Implementation Status

The following components are **fully implemented**:

1. ✅ **EnergyAligner** - Energy-based syllable segmentation (`src/coach/aligner.py`)
2. ✅ **VowelAligner** - Enhanced aligner with spectral centroid analysis
3. ✅ **Unified Evaluator** - Complete evaluation pipeline (`src/coach/evaluator.py`)
   - MMS encoder for acoustic features
   - Tone classifier with F0 features
   - Sandhi-aware validation
   - Dual ASR (MMS + Whisper) with OR-based result combination
4. ✅ **F0 Visualization** - Cairo-based pitch contour display (`src/gtk_ui/widgets/pitch_visualizer.py`)
   - Reference F0 from TTS audio (gray line)
   - User's F0 contour (blue line)
   - Ideal tone patterns for all syllables in phrase
5. ✅ **GTK4 GUI** - Full graphical interface (`src/gtk_ui/`)
   - Practice view with recording and real-time feedback
   - Per-syllable tone results with visual pitch curves
   - Lesson navigation and progress display

## Future Enhancements

- **MFA integration** - Montreal Forced Aligner for higher accuracy alignment
- **Spaced repetition** - Focus on weak tones based on progress data
- **Minimal pairs practice** - Targeted tone contrast drills
