# Mandarin Tone Coach

An interactive application for learning Mandarin Chinese tone pronunciation with real-time feedback.

## Features

- **Real-time tone evaluation** - Speak a phrase and get instant per-syllable feedback
- **Dual ASR system** - MMS + Whisper with contextual biasing for accurate recognition
- **Sandhi-aware validation** - Automatically handles tone sandhi rules (T3+T3, 不, 一)
- **Pitch visualization** - See your F0 contour compared to reference audio
- **Multiple interfaces** - GTK4 GUI, voice-controlled menu, or conversational mode
- **Offline capable** - Core evaluation runs entirely locally
- **Progress tracking** - Track your accuracy over time with per-tone statistics

## Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  Mandarin Tone Coach                               [─][□][×]│
├─────────────┬───────────────────────────────────────────────┤
│             │                                               │
│  Lessons    │   你好                                        │
│  ─────────  │   nǐ hǎo  (T3 T3)                            │
│             │   "hello"                                     │
│  ▶ Basics   │                                               │
│    Numbers  │   ┌─────────────────────────────────────┐     │
│    Food     │   │     F0 Pitch Visualization          │     │
│    ...      │   │  Reference (gray) vs You (blue)     │     │
│             │   └─────────────────────────────────────┘     │
│             │                                               │
│             │   [Record]  [Play Reference]                  │
│             │                                               │
│             │   Results: [你 T3 ✓] [好 T3 ✓]  Score: 100%  │
└─────────────┴───────────────────────────────────────────────┘
```

## Installation

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1
```

**Fedora:**
```bash
sudo dnf install python3-gobject gtk4 libadwaita
```

**Arch Linux:**
```bash
sudo pacman -S python-gobject gtk4 libadwaita
```

### Python Environment

```bash
# Create venv with system site packages (for GTK bindings)
python -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install pyctcdecode for improved ASR
pip install pyctcdecode
```

## Usage

### GTK4 GUI (Recommended)

```bash
python run_gtk.py
```

Options:
- `--asr mms` - Use only MMS ASR (lower GPU memory)
- `--asr whisper` - Use only Whisper ASR
- `--asr both` - Use dual ASR (default, best accuracy)
- `--asr none` - Disable ASR (tone classifier only)

### Voice-Controlled Menu

Hands-free operation, designed for Bluetooth headphones:

```bash
python run_voice.py
```

### Conversational Mode

Chat with an AI coach that guides your practice:

```bash
python run_coach.py --backend ollama  # Local LLM
python run_coach.py --backend claude  # Claude API
```

## Lesson Packs

| Lesson | Phrases | Description |
|--------|---------|-------------|
| Starter Words | 5 | Essential beginner phrases (你好, 谢谢, 再见) |
| Going to the Park | 6 | Outdoor activity vocabulary |
| At the Restaurant | 6 | Ordering food and drinks |
| Numbers | 10 | Counting 1-10 |
| Feelings | 7 | Express emotions |
| Daily Conversations | 18 | Common questions and responses |
| Shopping & Bargaining | 9 | Market and store phrases |
| Asking Directions | 10 | Navigation vocabulary |

## How It Works

### Evaluation Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Record     │───▶│   Segment    │───▶│  Classify    │
│   Audio      │    │  by Syllable │    │   Tones      │
└──────────────┘    └──────────────┘    └──────────────┘
                                               │
┌──────────────┐    ┌──────────────┐           ▼
│   Display    │◀───│   Combine    │◀───┌──────────────┐
│   Results    │    │   Signals    │    │   Dual ASR   │
└──────────────┘    └──────────────┘    │  MMS+Whisper │
                                        └──────────────┘
```

### Flexible Validation

The system uses multiple signals to determine correctness:

```python
is_correct = (
    classifier_detected_correct_tone  # Direct match
    or asr_detected_correct_tone      # ASR-derived tone matches
    or asr_recognized_character       # ASR heard the right word
)
```

This maximizes user success while providing accurate feedback.

### Tone Sandhi Support

Automatically handles Chinese tone sandhi rules:

- **Third tone sandhi**: T3 + T3 → T2 + T3 (你好 → ní hǎo)
- **不 sandhi**: T4 → T2 before T4 (不对 → bú duì)
- **一 sandhi**: T1 → T4 before T1/2/3, T1 → T2 before T4

## Architecture

```
src/
├── coach/
│   ├── evaluator.py      # Unified evaluation pipeline
│   ├── aligner.py        # Syllable boundary detection
│   ├── mms_asr.py        # MMS + Whisper ASR with contextual biasing
│   ├── sandhi.py         # Tone sandhi rules
│   ├── tts.py            # Dual TTS (edge-tts + Qwen3-TTS)
│   ├── llm.py            # LLM backends (Claude, Ollama, Transformers)
│   ├── coach.py          # Conversational coach
│   ├── progress.py       # Progress tracking
│   └── voice_menu.py     # Voice-controlled menu + lesson definitions
├── gtk_ui/
│   ├── views/
│   │   └── practice_view.py    # Main practice interface
│   └── widgets/
│       ├── pitch_visualizer.py # F0 contour visualization
│       └── tone_result_card.py # Per-syllable results
└── features/
    └── f0_extract.py     # F0 extraction (pYIN)
```

## Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Acoustic Features | facebook/mms-1b-all | Encode audio for classification |
| Tone Classifier | Custom MLP | Classify tones 1-5 |
| ASR (primary) | facebook/mms-1b-all | Chinese speech recognition |
| ASR (secondary) | openai/whisper-base | Backup recognition with prompting |
| TTS (fast) | edge-tts | English prompts, Chinese fallback |
| TTS (quality) | Qwen3-TTS | High-quality Chinese audio |

### Contextual ASR Biasing

Both ASR models use the expected phrase to improve recognition:

- **Whisper**: Uses `prompt_ids` to hint expected vocabulary
- **MMS**: Uses pyctcdecode with hotwords for beam search biasing

This significantly improves recognition of short phrases like "你好".

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Project Structure

- `run_gtk.py` - GTK4 GUI entry point
- `run_voice.py` - Voice menu entry point
- `run_coach.py` - Conversational mode entry point
- `src/coach/` - Core evaluation and coaching logic
- `src/gtk_ui/` - GTK4 user interface
- `docs/` - Architecture documentation

## License

MIT License

## Acknowledgments

- [Facebook MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) for multilingual speech models
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [edge-tts](https://github.com/rany2/edge-tts) for fast TTS
- [Qwen3-TTS](https://github.com/QwenLM/Qwen) for high-quality Chinese TTS
