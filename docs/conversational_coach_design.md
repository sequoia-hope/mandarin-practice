# Conversational Tone Coach - System Design

## Overview

An interactive Mandarin tone learning system combining:
- **Dual TTS system** - LightTTS (edge-tts) for English, QwenTTS for high-quality Chinese
- **LLM backends** - Claude API, Ollama (local), or Transformers for orchestration
- **MMS Tone Classifier** - Per-syllable tone analysis with F0 features
- **Dual ASR** - MMS + Whisper for robust word recognition with OR-based validation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  RUNTIME: PRACTICE SESSION                      │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  LLM Coach  │───▶│   Practice  │───▶│  Evaluator  │         │
│  │ (Claude/    │    │    Loop     │    │  (unified)  │         │
│  │  Ollama)    │◀───│             │◀───│             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                                     │                 │
│         ▼                                     ├─────────────────┤
│  ┌─────────────┐                      ┌──────┴──────┐          │
│  │  Progress   │                      │ MMS Encoder │          │
│  │  Tracking   │                      │ + Classifier │          │
│  └─────────────┘                      │ + F0 Extract │          │
│         │                             │ + Aligner   │          │
│         ▼                             │ + Dual ASR  │          │
│  ┌─────────────┐                      └─────────────┘          │
│  │  Dual TTS   │                                               │
│  │ Light+Qwen  │                                               │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Built-in Lessons

Lessons are defined in `src/gtk_ui/views/lesson_list_view.py` with phrases including:
- Characters, pinyin (with tone marks), tone numbers, and English translation
- Sandhi rules are automatically applied during evaluation

## Components

### 1. LLM Backend Abstraction (`src/coach/llm.py`)

Supports multiple LLM backends with a unified interface:

```python
class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Send a chat request with tool support."""
        pass


# Available backends:
class ClaudeBackend(LLMBackend):
    """Claude API via Anthropic SDK."""
    # Model: claude-sonnet-4-20250514 (default)

class OllamaBackend(LLMBackend):
    """Local LLM via Ollama with native tool calling."""
    # Models: qwen2.5:7b (default), llama3.1:8b, etc.

class TransformersBackend(LLMBackend):
    """HuggingFace Transformers for local inference."""
    # Models: Qwen/Qwen2.5-7B-Instruct, etc.


# Factory function
llm = create_backend("ollama", model="qwen2.5:7b")
```

### 2. Unified Evaluator (`src/coach/evaluator.py`)

The evaluator combines all tone analysis components:

```python
class Evaluator:
    """Unified tone evaluation service."""

    def __init__(
        self,
        use_f0: bool = True,       # Include F0 features
        use_asr: bool = True,      # Enable ASR for word recognition
        asr_backend: str = "both", # "mms", "whisper", or "both"
    ):
        # Components (lazy loaded):
        # - MMS encoder (acoustic features)
        # - Tone classifier (with F0 support)
        # - F0 extractor (pYIN method)
        # - Energy aligner (syllable segmentation)
        # - Dual ASR (MMS + Whisper, combined with OR logic)

    def evaluate(
        self,
        audio: np.ndarray,
        expected_pinyin: list[str],
        expected_tones: list[int],
        characters: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate tone pronunciation with sandhi awareness.

        Applies sandhi rules to determine acceptable pronunciations.
        Uses flexible validation: classifier OR ASR = correct.
        """

    def evaluate_from_mic(
        self,
        expected_pinyin: list[str],
        expected_tones: list[int],
        characters: Optional[str] = None,
        timeout: float = 10.0,
    ) -> EvaluationResult:
        """Record from microphone and evaluate."""
```

### 3. Dual TTS System (`src/coach/tts.py`)

```python
class LightTTS(TTSBackend):
    """Fast TTS using edge-tts (free, no GPU required)."""
    # Voices: en-US-AriaNeural, zh-CN-XiaoxiaoNeural
    # Features: Audio caching, speed control

class QwenTTS(TTSBackend):
    """High-quality TTS using Qwen3-TTS (requires GPU)."""
    # Features: Voice cloning, voice design mode, caching

class DualTTS(TTSBackend):
    """Combines both: LightTTS for English, QwenTTS for Chinese."""

# Usage:
tts = create_tts("dual")  # or "light" or "qwen"
audio = tts.speak("你好", language="zh", play=True)
```

### 4. Conversational Coach (`src/coach/coach.py`)

The ToneCoach handles conversational interaction with tool calling:

```python
class ToneCoach:
    """Conversational tone coach with tool calling."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        tool_executor: ToolExecutor,
    ):
        self.llm = llm_backend
        self.tools = tool_executor
        self.conversation_history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """
        Process user message and return response.
        Handles tool calling loop internally.
        """
        # Add to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # LLM responds, potentially calling tools
        while True:
            response = self.llm.chat(
                messages=self.conversation_history,
                system=SYSTEM_PROMPT,
                tools=COACH_TOOLS,
            )

            if response.tool_calls:
                # Execute tools, add results to history
                for tool_call in response.tool_calls:
                    result = self.tools.execute(tool_call.name, tool_call.parameters)
                    self.llm.add_tool_result(self.conversation_history, tool_call.id, result)
                continue
            else:
                # Final text response
                self.conversation_history.append({"role": "assistant", "content": response.content})
                return response.content


# Available tools for the coach:
COACH_TOOLS = [
    "generate_audio",        # Create reference audio for Chinese text
    "record_and_evaluate",   # Record and check user's pronunciation
    "play_audio",            # Play audio files
    "show_pitch_comparison", # Visual pitch contour comparison
]
```

### 5. GTK Practice View (`src/gtk_ui/views/practice_view.py`)

The GUI implementation handles the practice loop:

```python
class PracticeView(Gtk.Box):
    """Main practice interface for learning tones."""

    def _do_record_and_evaluate(self):
        """Record and evaluate in background thread."""
        # Uses Evaluator.evaluate_from_mic()
        result = evaluator.evaluate_from_mic(
            expected_pinyin=phrase["pinyin"],
            expected_tones=phrase["tones"],
            characters=phrase["characters"],
            timeout=10.0,
        )

        # Update UI on main thread
        GLib.idle_add(self._on_evaluation_complete, result)

    def _on_play_clicked(self, button):
        """Play reference audio and extract F0 for visualization."""
        audio = tts.speak(phrase["characters"], language="zh", play=False)

        # Extract F0 from reference audio
        f0_extractor = F0Extractor(method="pyin")
        reference_f0 = f0_extractor.extract(audio)["f0"]
        self._pitch_viz.set_reference_f0(reference_f0)

        # Play the audio
        tts.speak(phrase["characters"], language="zh", play=True)
```

## Lesson Pack Format

```json
{
  "lesson_id": "beginner_greetings",
  "title": "Basic Greetings",
  "description": "Common greeting phrases",
  "difficulty": "beginner",
  "focus_tones": [1, 2, 3],
  "phrases": [
    {
      "id": "hello_001",
      "characters": "你好",
      "pinyin": ["nǐ", "hǎo"],
      "pinyin_plain": ["ni", "hao"],
      "tones": [3, 3],
      "english": "Hello",
      "audio_file": "hello_001.wav",
      "notes": "Common tone sandhi: 3-3 becomes 2-3"
    },
    {
      "id": "thankyou_001",
      "characters": "谢谢",
      "pinyin": ["xiè", "xie"],
      "tones": [4, 5],
      "english": "Thank you",
      "audio_file": "thankyou_001.wav"
    }
  ]
}
```

## Progress Tracking

```python
@dataclass
class UserProgress:
    """Track user learning progress."""

    # Per-tone accuracy
    tone_stats: dict[int, ToneStats]  # tone -> {attempts, correct, recent_accuracy}

    # Session history
    session_attempts: list[AttemptResult]
    recent_phrases: list[str]  # phrase IDs

    # Computed properties
    @property
    def weak_tones(self) -> list[int]:
        """Tones with accuracy below threshold."""
        return [t for t, stats in self.tone_stats.items() if stats.accuracy < 0.7]

    @property
    def current_level(self) -> str:
        """Infer difficulty level from overall performance."""
        overall = sum(s.accuracy for s in self.tone_stats.values()) / 5
        if overall < 0.6:
            return "beginner"
        elif overall < 0.8:
            return "intermediate"
        else:
            return "advanced"
```

## Benefits of This Design

1. **Flexible LLM backends**
   - Claude API for best quality feedback
   - Ollama for fully local operation
   - Transformers for custom models

2. **Robust tone detection**
   - Dual ASR (MMS + Whisper) with OR-based validation
   - Sandhi-aware evaluation
   - Multiple success signals maximize user experience

3. **High quality audio**
   - Dual TTS: fast edge-tts for prompts, Qwen3-TTS for Chinese
   - Audio caching reduces latency
   - Reference F0 extraction for visualization

4. **Offline-first capable**
   - Core evaluation runs entirely locally
   - Progress tracked in JSON files
   - Privacy-preserving (audio stays local)

## Implementation Status

### ✅ Completed

1. **Core Evaluation**
   - [x] Energy-based aligner (`EnergyAligner`, `VowelAligner`)
   - [x] MMS encoder integration
   - [x] Tone classifier with F0 features
   - [x] Sandhi rules (T3, 不, 一)
   - [x] Dual ASR (MMS + Whisper) with OR logic

2. **TTS System**
   - [x] LightTTS (edge-tts)
   - [x] QwenTTS (Qwen3-TTS)
   - [x] DualTTS combining both
   - [x] Audio caching

3. **LLM Integration**
   - [x] Claude backend
   - [x] Ollama backend with native tool calling
   - [x] Transformers backend
   - [x] Tool executor for coach actions

4. **Progress Tracking**
   - [x] Per-tone statistics
   - [x] Session history
   - [x] JSON persistence

5. **GTK4 GUI**
   - [x] Practice view with recording
   - [x] Pitch visualizer (reference + user F0)
   - [x] Per-syllable tone results
   - [x] Lesson navigation

### 🔮 Future Enhancements

- Spaced repetition targeting weak tones
- Minimal pairs practice mode
- Web-based interface alternative
- Voice cloning for personalized reference audio
