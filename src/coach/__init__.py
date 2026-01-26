# Mandarin Tone Coach
#
# A hands-free Mandarin Chinese tone learning application.
#
# Main components:
# - Evaluator: Unified tone evaluation service
# - VoiceMenu: Phone tree menu system for hands-free operation
# - ToneCoach: LLM-driven conversational learning
# - TTS: Dual text-to-speech (edge-tts + Qwen3-TTS)
# - ProgressTracker: Learning progress persistence

from .evaluator import Evaluator, get_evaluator
from .coach import ToneCoach, create_coach
from .voice_menu import VoiceMenu, InputMode, BUILT_IN_LESSONS
from .audio_feedback import AudioFeedback
from .progress import ProgressTracker

__all__ = [
    "Evaluator",
    "get_evaluator",
    "ToneCoach",
    "create_coach",
    "VoiceMenu",
    "InputMode",
    "BUILT_IN_LESSONS",
    "AudioFeedback",
    "ProgressTracker",
]
