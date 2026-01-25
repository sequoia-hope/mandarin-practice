"""Dual TTS system: LightTTS for English menus, QwenTTS for Chinese reference audio."""

import asyncio
import hashlib
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import numpy as np


class TTSBackend(ABC):
    """Abstract TTS backend interface."""

    @abstractmethod
    def speak(
        self,
        text: str,
        language: Literal["en", "zh"] = "en",
        speed: str = "normal",
        play: bool = True,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            language: Language code ("en" or "zh")
            speed: Speaking speed ("slow", "normal", "fast")
            play: Whether to play audio immediately

        Returns:
            Audio samples as numpy array (16kHz, mono)
        """
        pass


class LightTTS(TTSBackend):
    """
    Lightweight TTS using edge-tts for English prompts.

    Fast, free, and requires no GPU.
    """

    # Voice selections by language
    VOICES = {
        "en": "en-US-AriaNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
    }

    # Speed adjustments for edge-tts rate parameter
    SPEED_MAP = {
        "teacher": "-15%",  # Slightly slower for clarity, not unnaturally slow
        "slow": "-30%",
        "normal": "+0%",
        "fast": "+20%",
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_voice_en: str = "en-US-AriaNeural",
        default_voice_zh: str = "zh-CN-XiaoxiaoNeural",
    ):
        """
        Initialize LightTTS.

        Args:
            cache_dir: Directory for caching generated audio
            default_voice_en: Default voice for English
            default_voice_zh: Default voice for Chinese
        """
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="tts_cache_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.voices = {
            "en": default_voice_en,
            "zh": default_voice_zh,
        }

        # Check edge-tts availability
        try:
            import edge_tts
            self._available = True
        except ImportError:
            self._available = False
            print("[LightTTS] edge-tts not installed. Install with: pip install edge-tts")

    def _get_cache_path(self, text: str, voice: str, speed: str) -> Path:
        """Get cache file path for given text/voice/speed combination."""
        key = f"{text}_{voice}_{speed}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hash_key}.wav"

    async def _synthesize_async(
        self,
        text: str,
        voice: str,
        speed: str,
        output_path: Path,
    ):
        """Async synthesis using edge-tts."""
        import edge_tts

        rate = self.SPEED_MAP.get(speed, "+0%")

        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(output_path))

    def speak(
        self,
        text: str,
        language: Literal["en", "zh"] = "en",
        speed: str = "normal",
        play: bool = True,
    ) -> np.ndarray:
        """Synthesize and optionally play audio."""
        if not self._available:
            print(f"[LightTTS] Would say: {text}")
            return np.zeros(16000, dtype=np.float32)

        voice = self.voices.get(language, self.voices["en"])
        cache_path = self._get_cache_path(text, voice, speed)

        # Check cache
        if not cache_path.exists():
            # Generate audio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                self._synthesize_async(text, voice, speed, cache_path)
            )

        # Load audio
        try:
            import soundfile as sf
            audio, sr = sf.read(str(cache_path))

            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            audio = audio.astype(np.float32)

            if play:
                import sounddevice as sd
                sd.play(audio, samplerate=16000)
                sd.wait()

            return audio

        except Exception as e:
            print(f"[LightTTS] Error loading audio: {e}")
            return np.zeros(16000, dtype=np.float32)


class QwenTTS(TTSBackend):
    """
    High-quality TTS using Qwen3-TTS for Chinese audio.

    Requires GPU and the Qwen3-TTS model to be available.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "cuda:0",
        qwen_tts_dir: str = "/home/sequoia/Software/Qwen3-TTS",
        cache_dir: Optional[Path] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
    ):
        """
        Initialize QwenTTS.

        Args:
            model_path: HuggingFace model path or local path
            device: Device to run on
            qwen_tts_dir: Path to Qwen3-TTS repository
            cache_dir: Directory for caching generated audio
            ref_audio_path: Path to reference audio for voice cloning
            ref_text: Text spoken in reference audio
        """
        self.model_path = model_path
        self.device = device
        self.qwen_tts_dir = Path(qwen_tts_dir)
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="qwen_tts_cache_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text

        self._model = None
        self._lock = threading.Lock()

        # Check if Qwen3-TTS is available
        import sys
        if str(self.qwen_tts_dir) not in sys.path:
            sys.path.insert(0, str(self.qwen_tts_dir))

        try:
            from qwen_tts import Qwen3TTSModel
            self._available = True
        except ImportError:
            self._available = False
            print(f"[QwenTTS] qwen_tts not found. Make sure {self.qwen_tts_dir} is correct.")

    def _load_model(self):
        """Load model lazily."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            print("[QwenTTS] Loading model...")
            import torch
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print("[QwenTTS] Model loaded!")

    def _get_cache_path(self, text: str, speed: str) -> Path:
        """Get cache file path."""
        key = f"{text}_{speed}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"qwen_{hash_key}.wav"

    def speak(
        self,
        text: str,
        language: Literal["en", "zh"] = "zh",
        speed: str = "normal",
        play: bool = True,
    ) -> np.ndarray:
        """Synthesize and optionally play audio."""
        if not self._available:
            print(f"[QwenTTS] Would say: {text}")
            return np.zeros(16000, dtype=np.float32)

        cache_path = self._get_cache_path(text, speed)

        # Check cache
        if cache_path.exists():
            try:
                import soundfile as sf
                audio, sr = sf.read(str(cache_path))
                if sr != 16000:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio = audio.astype(np.float32)

                if play:
                    import sounddevice as sd
                    sd.play(audio, samplerate=16000)
                    sd.wait()

                return audio
            except Exception as e:
                print(f"[QwenTTS] Cache read error: {e}")

        # Load model
        self._load_model()

        # Generate audio
        try:
            import torch

            # Map language to Qwen format
            lang_map = {"en": "English", "zh": "Chinese"}
            qwen_lang = lang_map.get(language, "Auto")

            # Speed adjustment (lower temperature = more consistent pronunciation)
            speed_temps = {
                "teacher": 0.8,  # Slightly more deliberate for clarity
                "slow": 0.7,
                "normal": 0.9,
                "fast": 1.0,
            }
            temperature = speed_temps.get(speed, 0.9)

            gen_kwargs = dict(
                max_new_tokens=2048,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=temperature,
                repetition_penalty=1.05,
            )

            # Use voice cloning if reference audio is provided
            if self.ref_audio_path and self.ref_text:
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language=qwen_lang,
                    ref_audio=self.ref_audio_path,
                    ref_text=self.ref_text,
                    **gen_kwargs,
                )
            else:
                # Use voice design mode
                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    language=qwen_lang,
                    gender="female",
                    age="young_adult",
                    **gen_kwargs,
                )

            audio = wavs[0] if isinstance(wavs, list) else wavs

            # Save to cache
            import soundfile as sf
            sf.write(str(cache_path), audio, sr)

            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            audio = audio.astype(np.float32)

            if play:
                import sounddevice as sd
                sd.play(audio, samplerate=16000)
                sd.wait()

            return audio

        except Exception as e:
            print(f"[QwenTTS] Generation error: {e}")
            return np.zeros(16000, dtype=np.float32)


class DualTTS(TTSBackend):
    """
    Dual TTS system that uses LightTTS for English and QwenTTS for Chinese.

    With --tts qwen flag, uses QwenTTS for everything.
    """

    def __init__(
        self,
        use_qwen_for_all: bool = False,
        qwen_kwargs: Optional[dict] = None,
        light_kwargs: Optional[dict] = None,
    ):
        """
        Initialize DualTTS.

        Args:
            use_qwen_for_all: If True, use QwenTTS for all languages
            qwen_kwargs: Keyword arguments for QwenTTS
            light_kwargs: Keyword arguments for LightTTS
        """
        self.use_qwen_for_all = use_qwen_for_all

        # Initialize backends
        self.light_tts = LightTTS(**(light_kwargs or {}))
        self._qwen_tts = None
        self._qwen_kwargs = qwen_kwargs or {}

    @property
    def qwen_tts(self) -> QwenTTS:
        """Get QwenTTS instance, creating lazily."""
        if self._qwen_tts is None:
            self._qwen_tts = QwenTTS(**self._qwen_kwargs)
        return self._qwen_tts

    def speak(
        self,
        text: str,
        language: Literal["en", "zh"] = "en",
        speed: str = "normal",
        play: bool = True,
    ) -> np.ndarray:
        """
        Synthesize speech using appropriate backend.

        - Chinese text: Always uses QwenTTS (high quality)
        - English text: Uses LightTTS by default, QwenTTS if use_qwen_for_all=True
        """
        if language == "zh" or self.use_qwen_for_all:
            return self.qwen_tts.speak(text, language, speed, play)
        else:
            return self.light_tts.speak(text, language, speed, play)


# Convenience factory
def create_tts(
    backend: Literal["light", "qwen", "dual"] = "dual",
    use_qwen_for_all: bool = False,
    **kwargs,
) -> TTSBackend:
    """
    Create a TTS backend.

    Args:
        backend: "light" for edge-tts only, "qwen" for Qwen3-TTS only, "dual" for both
        use_qwen_for_all: If True and backend="dual", use Qwen for all languages
        **kwargs: Additional arguments for the backend

    Returns:
        TTS backend instance
    """
    if backend == "light":
        return LightTTS(**kwargs)
    elif backend == "qwen":
        return QwenTTS(**kwargs)
    elif backend == "dual":
        return DualTTS(use_qwen_for_all=use_qwen_for_all, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test TTS backends")
    parser.add_argument("--backend", choices=["light", "qwen", "dual"], default="light")
    parser.add_argument("--text", type=str, default="Hello, welcome to the lesson!")
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--speed", choices=["teacher", "slow", "normal", "fast"], default="teacher")
    args = parser.parse_args()

    print(f"Testing {args.backend} TTS...")
    tts = create_tts(args.backend)

    print(f"Speaking: '{args.text}' in {args.lang}")
    audio = tts.speak(args.text, args.lang, args.speed, play=True)
    print(f"Generated {len(audio)} samples ({len(audio)/16000:.2f}s)")
