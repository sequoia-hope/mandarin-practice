# Conversational Tone Coach - Full Design

## User Experience Vision

```
User: How do I say "I am tired because I was up late fixing my bike"?

Coach: That's a great sentence! In Chinese it's:

  我很累因为昨晚熬夜修自行车
  wǒ hěn lèi yīnwèi zuówǎn áoyè xiū zìxíngchē

  That's a lot! Let me break it into chunks:

  1. 我很累 (wǒ hěn lèi) - "I am very tired"
  2. 因为 (yīnwèi) - "because"
  3. 昨晚熬夜 (zuówǎn áoyè) - "last night stayed up late"
  4. 修自行车 (xiū zìxíngchē) - "fixing bike"

  Let's start with chunk 1. Listen:
  [▶️ plays audio for 我很累]

  Now you try:

User: [speaks "我很累"]

Coach: Nice try! Here's how you did:

  我 (wǒ)  T3 ✓
  很 (hěn) T2 ✗ → should be T3 (dipping)
  累 (lèi) T4 ✓

  For 很, you went up (T2 rising) but it should dip down
  then rise (T3). Think "questioning disappointment."

  [Shows F0 comparison visual]

  Try again:

User: [speaks again]

Coach: Perfect! 很 was spot on that time. Ready for chunk 2?

User: Actually, can we try a simpler phrase first?

Coach: Of course! How about just "I'm tired" - 我累了 (wǒ lèi le)?
  That's only 3 syllables: T3, T4, T5 (neutral).

  Listen: [▶️ plays audio]

  Your turn:
```

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONVERSATIONAL LAYER                        │
│                                                                 │
│  LLM Backend (Claude/Ollama/Transformers) with Tool Calling    │
│    - Understands user intent                                    │
│    - Manages conversation flow                                  │
│    - Calls tools for audio/evaluation                           │
│    - Generates natural, encouraging responses                   │
└─────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  TRANSLATE   │    │  AUDIO GEN   │    │   EVALUATE   │
│    TOOL      │    │    TOOL      │    │    TOOL      │
│              │    │              │    │              │
│ LLM handles  │    │ Dual TTS:    │    │ Unified      │
│ translation  │    │ Light+Qwen   │    │ Evaluator:   │
│ inline       │    │              │    │ MMS+Classifier│
└──────────────┘    └──────────────┘    │ +F0+ASR      │
                                        └──────────────┘
                                               │
                                        ┌──────┴──────┐
                                        │  Sandhi     │
                                        │  + Dual ASR │
                                        │  (MMS+Whisper)│
                                        └─────────────┘
```

## Tool Definitions (Implemented in `src/coach/tools.py`)

```python
COACH_TOOLS = [
    {
        "name": "generate_audio",
        "description": "Generate native speaker audio for Chinese text",
        "parameters": {
            "text": "Chinese text to speak",
            "language": "Language: zh (Chinese) or en (English)",
            "speed": "Speaking speed: slow/normal/fast"
        },
        "returns": {
            "audio_path": "Path to generated audio file",
            "duration_ms": "Audio duration in milliseconds"
        }
    },
    {
        "name": "record_and_evaluate",
        "description": "Record user speech and evaluate tone accuracy",
        "parameters": {
            "expected_characters": "Chinese text user should say",
            "expected_pinyin": "List of pinyin syllables (without tone marks)",
            "expected_tones": "List of tone numbers (1-5)",
            "timeout_seconds": "Max recording time (default: 10)"
        },
        "returns": {
            "syllable_results": [
                {
                    "character": "Chinese character",
                    "expected_tone": "Underlying tone number",
                    "spoken_tone": "Expected spoken tone (after sandhi)",
                    "detected_tone": "Classifier detected tone",
                    "confidence": "Classification confidence (0-1)",
                    "correct": "Whether tone was correct",
                    "char_recognized": "ASR recognized this character",
                    "asr_tone": "Tone from ASR transcription",
                    "correct_source": "What confirmed correctness"
                }
            ],
            "overall_accuracy": "Tone accuracy (0-1)",
            "word_accuracy": "Character recognition accuracy (0-1)",
            "combined_score": "Weighted: 60% tone + 40% word",
            "transcription": "What ASR heard (Chinese)",
            "transcription_pinyin": "Pinyin of transcription"
        }
    },
    {
        "name": "play_audio",
        "description": "Play an audio file",
        "parameters": {
            "audio_path": "Path to audio file"
        },
        "returns": {
            "played": true,
            "duration_ms": "Playback duration"
        }
    },
    {
        "name": "show_pitch_comparison",
        "description": "Display pitch contour comparison",
        "parameters": {
            "user_f0": "User's F0 contour",
            "expected_tones": "List of expected tones"
        },
        "returns": {
            "displayed": true
        }
    }
]
```

### Evaluation Details

The `record_and_evaluate` tool uses **flexible validation**:

- **Sandhi awareness**: T3+T3 → T2+T3 is automatically accepted
- **Dual ASR**: MMS and Whisper results combined with OR logic
- **Multiple success signals**: Classifier, ASR tone, or ASR character recognition

```python
# Tone is correct if ANY of:
is_correct = (
    detected_tone in acceptable_tones  # Classifier (with sandhi)
    or asr_tone in acceptable_tones    # ASR-derived tone
    or char_recognized                  # ASR recognized character
)
```

## LLM System Prompt

```markdown
You are a friendly Mandarin Chinese tone coach. Your job is to help users
learn to pronounce Chinese tones correctly through interactive practice.

## Your Capabilities (via tools)

- **translate_to_chinese**: Convert English to Chinese with pinyin and tones
- **generate_reference_audio**: Create native speaker audio for any phrase
- **record_and_evaluate**: Record user speech and analyze their tones
- **show_f0_comparison**: Show pitch contour comparison for incorrect tones
- **play_audio**: Play reference or user audio
- **save_progress**: Track learning progress

## Conversation Guidelines

1. **When user asks "How do I say X"**:
   - Translate to Chinese
   - Break into manageable chunks (2-4 syllables) if long
   - Generate audio for first chunk
   - Guide through practice

2. **When evaluating attempts**:
   - Celebrate correct tones
   - For errors: explain what went wrong simply
   - Offer specific tips (e.g., "start higher", "dip down more")
   - Show F0 comparison for persistent errors
   - Offer to repeat

3. **Adaptive pacing**:
   - If user struggles, offer simpler phrases
   - If user succeeds easily, move faster
   - Respect user requests to skip or change

4. **Tone explanations** (use when helpful):
   - T1 (high level): "Like a flat, high note - māmāmā"
   - T2 (rising): "Like asking 'what?' - goes up"
   - T3 (dipping): "Dip down then up, like 'well...' when thinking"
   - T4 (falling): "Sharp drop, like a command"
   - T5 (neutral): "Light and quick, no strong pitch"

5. **Handle meta-requests naturally**:
   - "Let's try something else" → offer alternatives
   - "Can you repeat that?" → replay audio
   - "I don't understand" → explain differently
   - "What's my weakest tone?" → summarize progress
```

## Implementation Components

### 1. Tool Executor (`src/coach/tools.py`)

```python
class ToolExecutor:
    """Execute tools called by the LLM."""

    def __init__(
        self,
        tts: Optional[TTSBackend] = None,
        evaluator: Optional[Evaluator] = None,
        lazy_load: bool = True,
    ):
        self._tts = tts
        self._evaluator = evaluator
        self._lazy_load = lazy_load

    def execute(self, tool_name: str, params: dict) -> dict:
        """Execute a tool and return results."""
        if tool_name == "generate_audio":
            return self._generate_audio(params)
        elif tool_name == "record_and_evaluate":
            return self._record_and_evaluate(params)
        elif tool_name == "play_audio":
            return self._play_audio(params)
        elif tool_name == "show_pitch_comparison":
            return self._show_pitch_comparison(params)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _generate_audio(self, params: dict) -> dict:
        """Generate TTS audio using dual TTS system."""
        tts = self._get_tts()
        language = params.get("language", "zh")
        speed = params.get("speed", "normal")

        audio = tts.speak(params["text"], language=language, speed=speed, play=False)
        path = self._save_temp_audio(audio)

        return {"audio_path": path, "duration_ms": int(len(audio) / 16)}

    def _record_and_evaluate(self, params: dict) -> dict:
        """Record and evaluate using unified Evaluator."""
        evaluator = self._get_evaluator()

        # Uses Evaluator.evaluate_from_mic() which handles:
        # - Utterance detection
        # - Energy-based alignment
        # - MMS + classifier + F0 extraction
        # - Dual ASR (MMS + Whisper)
        # - Sandhi-aware validation
        result = evaluator.evaluate_from_mic(
            expected_pinyin=params["expected_pinyin"],
            expected_tones=params["expected_tones"],
            characters=params.get("expected_characters"),
            timeout=params.get("timeout_seconds", 10.0),
        )

        return evaluator.to_dict(result)
```

### 2. ToneCoach (`src/coach/coach.py`)

```python
class ToneCoach:
    """Conversational tone coach with tool calling."""

    def __init__(
        self,
        llm_backend: LLMBackend,
        tool_executor: ToolExecutor,
        use_local_prompt: bool = False,
    ):
        self.llm = llm_backend
        self.tools = tool_executor
        self.conversation_history: list[dict] = []
        # Simpler prompt for local LLMs (Ollama, Transformers)
        self.system_prompt = SYSTEM_PROMPT_LOCAL if use_local_prompt else SYSTEM_PROMPT

    def chat(self, user_message: str) -> str:
        """Process user message and return response (synchronous)."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        max_tool_rounds = 10  # Prevent infinite loops

        for _ in range(max_tool_rounds):
            try:
                response = self.llm.chat(
                    messages=self.conversation_history,
                    system=self.system_prompt,
                    tools=COACH_TOOLS,
                )
            except Exception as e:
                return f"Sorry, I encountered an error: {e}"

            if response.tool_calls:
                # Execute tools and add results to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "parameters": tc.parameters}
                        for tc in response.tool_calls
                    ]
                })

                for tool_call in response.tool_calls:
                    result = self.tools.execute(tool_call.name, tool_call.parameters)
                    self.llm.add_tool_result(
                        self.conversation_history,
                        tool_call.id,
                        result
                    )
                continue
            else:
                # Final text response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                return response.content

        return "I got confused. Let's start fresh!"


# Factory function
def create_coach(
    backend: str = "ollama",  # "claude", "ollama", or "transformers"
    model: Optional[str] = None,
    **kwargs,
) -> ToneCoach:
    """Create a ToneCoach with specified backend."""
    llm = create_backend(backend, model=model, **kwargs)
    tools = ToolExecutor(lazy_load=True)
    use_local = (backend != "claude")
    return ToneCoach(llm, tools, use_local_prompt=use_local)
```

### 3. Terminal Interface

```python
def run_interactive(coach: ToneCoach):
    """Run interactive terminal session."""
    print("=" * 60)
    print("  Mandarin Tone Coach")
    print("=" * 60)
    print()
    print("Ask me anything! Try:")
    print("  - 'How do I say hello?'")
    print("  - 'Teach me to count to five'")
    print()
    print("Type 'quit' to exit, 'reset' to start over.")
    print("=" * 60)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见! (Goodbye!)")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见! (Goodbye!)")
            break

        if user_input.lower() == 'reset':
            coach.reset()
            print("Conversation reset.")
            continue

        response = coach.chat(user_input)
        print(f"\nCoach: {response}\n")


# Entry point: python run_coach.py --backend ollama
```

## Comparison: GTK UI vs Conversational Mode

| Aspect | GTK UI (run_gtk.py) | Conversational (run_coach.py) |
|--------|---------------------|-------------------------------|
| **Interface** | Graphical | Terminal chat |
| **Phrase selection** | Manual from lessons | LLM-driven / user request |
| **Latency** | Low (cached audio) | Higher (runtime generation) |
| **LLM needs** | None (standalone) | Claude, Ollama, or Transformers |
| **Offline capable** | Yes (core features) | Yes (with Ollama) |
| **Visual feedback** | Pitch contours, tone curves | Text descriptions |

Both modes use the same underlying evaluation pipeline with dual ASR and sandhi support.

## Available Modes

The application supports multiple interaction modes:

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODE SELECTION                           │
├─────────────────────────────┬───────────────────────────────────┤
│   GTK GUI (run_gtk.py)      │   CONVERSATIONAL (run_coach.py)  │
│                             │                                   │
│   • Visual pitch feedback   │   • Natural language interaction │
│   • Built-in lesson packs   │   • "How do I say X?" queries    │
│   • Click to record/play    │   • LLM-generated feedback       │
│   • Works fully offline     │   • Works offline with Ollama    │
│                             │                                   │
├─────────────────────────────┴───────────────────────────────────┤
│   VOICE MENU (run_voice.py)                                     │
│                             │                                   │
│   • Voice-controlled navigation                                 │
│   • Hands-free practice                                         │
│   • Command ASR for menu control                                │
└─────────────────────────────────────────────────────────────────┘
```

All modes share:
- The same unified Evaluator (MMS + classifier + F0 + dual ASR)
- Sandhi-aware validation
- Progress tracking (JSON persistence)

## Implementation Status

All core components are implemented and working:

| Component | File | Status |
|-----------|------|--------|
| LLM Backends | `src/coach/llm.py` | ✅ Claude, Ollama, Transformers |
| Tool Executor | `src/coach/tools.py` | ✅ All coach tools |
| Conversational Coach | `src/coach/coach.py` | ✅ Tool calling loop |
| Dual TTS | `src/coach/tts.py` | ✅ Light + Qwen |
| Unified Evaluator | `src/coach/evaluator.py` | ✅ Full pipeline |
| Dual ASR | `src/coach/mms_asr.py` | ✅ MMS + Whisper |
| Progress Tracking | `src/coach/progress.py` | ✅ JSON persistence |
| GTK4 GUI | `src/gtk_ui/` | ✅ Full interface |

### Entry Points

- `python run_gtk.py` - GTK4 graphical interface
- `python run_coach.py` - Terminal conversational mode
- `python run_voice.py` - Voice-controlled menu mode

### Future Enhancements

- Spaced repetition targeting weak tones
- Web-based interface alternative
- Conversation history persistence
- Multi-user support with cloud sync
