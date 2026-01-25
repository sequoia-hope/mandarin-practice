"""Voice-controlled menu system for hands-free operation."""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio_feedback import AudioFeedback
from .progress import ProgressTracker


class MenuState(Enum):
    MAIN = "main"
    LESSON = "lesson"
    PRACTICE = "practice"
    CONVERSATION = "conversation"


class InputMode(Enum):
    VOICE = "voice"
    TEXT = "text"


@dataclass
class LessonPack:
    """A themed lesson pack."""
    id: str
    name: str
    description: str
    phrases: list[dict]  # [{characters, pinyin, tones, english, sandhi_note?}, ...]


# Built-in lesson packs
# Note: "tones" are the underlying/lexical tones. The evaluator automatically
# applies tone sandhi rules, so users can correctly pronounce with sandhi
# (e.g., 你好 as T2+T3) and still be marked correct.
BUILT_IN_LESSONS = [
    LessonPack(
        id="starter",
        name="Starter Words",
        description="Essential beginner phrases",
        phrases=[
            # 你好: T3+T3 → spoken as T2+T3 (third tone sandhi)
            {"characters": "你好", "pinyin": ["nǐ", "hǎo"], "tones": [3, 3], "english": "Hello",
             "sandhi_note": "Spoken as ní hǎo (T2+T3) due to third tone sandhi"},
            {"characters": "谢谢", "pinyin": ["xiè", "xie"], "tones": [4, 5], "english": "Thank you"},
            {"characters": "再见", "pinyin": ["zài", "jiàn"], "tones": [4, 4], "english": "Goodbye"},
            {"characters": "对不起", "pinyin": ["duì", "bu", "qǐ"], "tones": [4, 5, 3], "english": "Sorry"},
            {"characters": "没关系", "pinyin": ["méi", "guān", "xi"], "tones": [2, 1, 5], "english": "No problem"},
        ]
    ),
    LessonPack(
        id="park",
        name="Going to the Park",
        description="Vocabulary for outdoor activities",
        phrases=[
            {"characters": "公园", "pinyin": ["gōng", "yuán"], "tones": [1, 2], "english": "Park"},
            {"characters": "走路", "pinyin": ["zǒu", "lù"], "tones": [3, 4], "english": "Walk"},
            {"characters": "跑步", "pinyin": ["pǎo", "bù"], "tones": [3, 4], "english": "Run/jog"},
            {"characters": "树", "pinyin": ["shù"], "tones": [4], "english": "Tree"},
            {"characters": "花", "pinyin": ["huā"], "tones": [1], "english": "Flower"},
            # 很好: T3+T3 → spoken as T2+T3 (third tone sandhi)
            {"characters": "天气很好", "pinyin": ["tiān", "qì", "hěn", "hǎo"], "tones": [1, 4, 3, 3], "english": "The weather is nice",
             "sandhi_note": "很好 spoken as hén hǎo (T2+T3)"},
        ]
    ),
    LessonPack(
        id="restaurant",
        name="At the Restaurant",
        description="Ordering food and drinks",
        phrases=[
            {"characters": "菜单", "pinyin": ["cài", "dān"], "tones": [4, 1], "english": "Menu"},
            {"characters": "水", "pinyin": ["shuǐ"], "tones": [3], "english": "Water"},
            {"characters": "米饭", "pinyin": ["mǐ", "fàn"], "tones": [3, 4], "english": "Rice"},
            {"characters": "好吃", "pinyin": ["hǎo", "chī"], "tones": [3, 1], "english": "Delicious"},
            {"characters": "买单", "pinyin": ["mǎi", "dān"], "tones": [3, 1], "english": "Check please"},
            {"characters": "我要这个", "pinyin": ["wǒ", "yào", "zhè", "ge"], "tones": [3, 4, 4, 5], "english": "I want this"},
        ]
    ),
    LessonPack(
        id="numbers",
        name="Numbers",
        description="Counting from 1 to 10",
        phrases=[
            # Note: 一 has special sandhi rules when not in isolation (T1→T4 before T1/2/3, T1→T2 before T4)
            # When counting, 一 keeps its original T1
            {"characters": "一", "pinyin": ["yī"], "tones": [1], "english": "One"},
            {"characters": "二", "pinyin": ["èr"], "tones": [4], "english": "Two"},
            {"characters": "三", "pinyin": ["sān"], "tones": [1], "english": "Three"},
            {"characters": "四", "pinyin": ["sì"], "tones": [4], "english": "Four"},
            {"characters": "五", "pinyin": ["wǔ"], "tones": [3], "english": "Five"},
            {"characters": "六", "pinyin": ["liù"], "tones": [4], "english": "Six"},
            {"characters": "七", "pinyin": ["qī"], "tones": [1], "english": "Seven"},
            {"characters": "八", "pinyin": ["bā"], "tones": [1], "english": "Eight"},
            {"characters": "九", "pinyin": ["jiǔ"], "tones": [3], "english": "Nine"},
            {"characters": "十", "pinyin": ["shí"], "tones": [2], "english": "Ten"},
        ]
    ),
    LessonPack(
        id="feelings",
        name="Feelings",
        description="Express how you feel",
        phrases=[
            {"characters": "高兴", "pinyin": ["gāo", "xìng"], "tones": [1, 4], "english": "Happy"},
            {"characters": "累", "pinyin": ["lèi"], "tones": [4], "english": "Tired"},
            {"characters": "饿", "pinyin": ["è"], "tones": [4], "english": "Hungry"},
            {"characters": "渴", "pinyin": ["kě"], "tones": [3], "english": "Thirsty"},
            {"characters": "冷", "pinyin": ["lěng"], "tones": [3], "english": "Cold"},
            {"characters": "热", "pinyin": ["rè"], "tones": [4], "english": "Hot"},
            # 我很好: T3+T3+T3 → spoken as T2+T2+T3 (third tone sandhi chain)
            {"characters": "我很好", "pinyin": ["wǒ", "hěn", "hǎo"], "tones": [3, 3, 3], "english": "I'm fine",
             "sandhi_note": "Spoken as wó hén hǎo (T2+T2+T3) due to third tone sandhi"},
        ]
    ),
    LessonPack(
        id="conversations",
        name="Daily Conversations",
        description="Common phrases for everyday situations",
        phrases=[
            # Greetings and introductions
            {"characters": "你好吗", "pinyin": ["nǐ", "hǎo", "ma"], "tones": [3, 3, 5], "english": "How are you?",
             "sandhi_note": "你好 spoken as ní hǎo (T2+T3)"},
            {"characters": "我叫什么名字", "pinyin": ["wǒ", "jiào", "shén", "me", "míng", "zi"], "tones": [3, 4, 2, 5, 2, 5], "english": "What is my name?"},
            {"characters": "很高兴认识你", "pinyin": ["hěn", "gāo", "xìng", "rèn", "shi", "nǐ"], "tones": [3, 1, 4, 4, 5, 3], "english": "Nice to meet you"},
            {"characters": "你是哪里人", "pinyin": ["nǐ", "shì", "nǎ", "lǐ", "rén"], "tones": [3, 4, 3, 3, 2], "english": "Where are you from?",
             "sandhi_note": "哪里 spoken as ná lǐ (T2+T3)"},
            {"characters": "我是美国人", "pinyin": ["wǒ", "shì", "měi", "guó", "rén"], "tones": [3, 4, 3, 2, 2], "english": "I am American"},

            # Common questions
            {"characters": "这个多少钱", "pinyin": ["zhè", "ge", "duō", "shǎo", "qián"], "tones": [4, 5, 1, 3, 2], "english": "How much is this?"},
            {"characters": "洗手间在哪里", "pinyin": ["xǐ", "shǒu", "jiān", "zài", "nǎ", "lǐ"], "tones": [3, 3, 1, 4, 3, 3], "english": "Where is the bathroom?",
             "sandhi_note": "洗手 spoken as xí shǒu, 哪里 as ná lǐ"},
            {"characters": "你会说英语吗", "pinyin": ["nǐ", "huì", "shuō", "yīng", "yǔ", "ma"], "tones": [3, 4, 1, 1, 3, 5], "english": "Do you speak English?"},
            {"characters": "我听不懂", "pinyin": ["wǒ", "tīng", "bù", "dǒng"], "tones": [3, 1, 4, 3], "english": "I don't understand"},
            {"characters": "请再说一遍", "pinyin": ["qǐng", "zài", "shuō", "yī", "biàn"], "tones": [3, 4, 1, 1, 4], "english": "Please say it again",
             "sandhi_note": "一遍 spoken as yí biàn (T2+T4) due to 一 sandhi before T4"},

            # Polite expressions
            {"characters": "不客气", "pinyin": ["bù", "kè", "qi"], "tones": [4, 4, 5], "english": "You're welcome",
             "sandhi_note": "不客 spoken as bú kè (T2+T4) due to 不 sandhi"},
            {"characters": "请问一下", "pinyin": ["qǐng", "wèn", "yī", "xià"], "tones": [3, 4, 1, 4], "english": "Excuse me (to ask)",
             "sandhi_note": "一下 spoken as yí xià (T2+T4)"},
            {"characters": "麻烦你了", "pinyin": ["má", "fan", "nǐ", "le"], "tones": [2, 5, 3, 5], "english": "Sorry to trouble you"},
            {"characters": "没问题", "pinyin": ["méi", "wèn", "tí"], "tones": [2, 4, 2], "english": "No problem"},

            # Useful responses
            {"characters": "好的没问题", "pinyin": ["hǎo", "de", "méi", "wèn", "tí"], "tones": [3, 5, 2, 4, 2], "english": "OK, no problem"},
            {"characters": "我知道了", "pinyin": ["wǒ", "zhī", "dào", "le"], "tones": [3, 1, 4, 5], "english": "I understand / Got it"},
            {"characters": "太好了", "pinyin": ["tài", "hǎo", "le"], "tones": [4, 3, 5], "english": "Great! / Awesome!"},
            {"characters": "对对对", "pinyin": ["duì", "duì", "duì"], "tones": [4, 4, 4], "english": "Yes yes yes / Exactly"},
        ]
    ),
    LessonPack(
        id="shopping",
        name="Shopping & Bargaining",
        description="Useful phrases for markets and stores",
        phrases=[
            {"characters": "我想买这个", "pinyin": ["wǒ", "xiǎng", "mǎi", "zhè", "ge"], "tones": [3, 3, 3, 4, 5], "english": "I want to buy this",
             "sandhi_note": "我想买 spoken as wó xiáng mǎi (T2+T2+T3)"},
            {"characters": "太贵了", "pinyin": ["tài", "guì", "le"], "tones": [4, 4, 5], "english": "Too expensive"},
            {"characters": "便宜一点", "pinyin": ["pián", "yi", "yī", "diǎn"], "tones": [2, 5, 1, 3], "english": "A little cheaper",
             "sandhi_note": "一点 spoken as yì diǎn (T4+T3) due to 一 sandhi before T3"},
            {"characters": "可以便宜吗", "pinyin": ["kě", "yǐ", "pián", "yi", "ma"], "tones": [3, 3, 2, 5, 5], "english": "Can you make it cheaper?",
             "sandhi_note": "可以 spoken as ké yǐ (T2+T3)"},
            {"characters": "我只是看看", "pinyin": ["wǒ", "zhǐ", "shì", "kàn", "kan"], "tones": [3, 3, 4, 4, 5], "english": "I'm just looking",
             "sandhi_note": "我只 spoken as wó zhǐ (T2+T3)"},
            {"characters": "有没有别的颜色", "pinyin": ["yǒu", "méi", "yǒu", "bié", "de", "yán", "sè"], "tones": [3, 2, 3, 2, 5, 2, 4], "english": "Do you have other colors?"},
            {"characters": "可以试一下吗", "pinyin": ["kě", "yǐ", "shì", "yī", "xià", "ma"], "tones": [3, 3, 4, 1, 4, 5], "english": "Can I try it?",
             "sandhi_note": "可以 spoken as ké yǐ, 一下 as yí xià"},
            {"characters": "我要付现金", "pinyin": ["wǒ", "yào", "fù", "xiàn", "jīn"], "tones": [3, 4, 4, 4, 1], "english": "I'll pay cash"},
            {"characters": "可以刷卡吗", "pinyin": ["kě", "yǐ", "shuā", "kǎ", "ma"], "tones": [3, 3, 1, 3, 5], "english": "Can I pay by card?",
             "sandhi_note": "可以 spoken as ké yǐ"},
        ]
    ),
    LessonPack(
        id="directions",
        name="Asking Directions",
        description="Navigate like a local",
        phrases=[
            {"characters": "请问怎么走", "pinyin": ["qǐng", "wèn", "zěn", "me", "zǒu"], "tones": [3, 4, 3, 5, 3], "english": "How do I get there?",
             "sandhi_note": "怎么走 with T3 sandhi"},
            {"characters": "往左走", "pinyin": ["wǎng", "zuǒ", "zǒu"], "tones": [3, 3, 3], "english": "Go left",
             "sandhi_note": "Spoken as wáng zuó zǒu (T2+T2+T3)"},
            {"characters": "往右走", "pinyin": ["wǎng", "yòu", "zǒu"], "tones": [3, 4, 3], "english": "Go right"},
            {"characters": "一直走", "pinyin": ["yī", "zhí", "zǒu"], "tones": [1, 2, 3], "english": "Go straight",
             "sandhi_note": "一直 spoken as yì zhí (T4+T2) due to 一 sandhi before T2"},
            {"characters": "在前面", "pinyin": ["zài", "qián", "miàn"], "tones": [4, 2, 4], "english": "In front / Ahead"},
            {"characters": "在后面", "pinyin": ["zài", "hòu", "miàn"], "tones": [4, 4, 4], "english": "Behind / In back"},
            {"characters": "在旁边", "pinyin": ["zài", "páng", "biān"], "tones": [4, 2, 1], "english": "Next to / Beside"},
            {"characters": "离这里远吗", "pinyin": ["lí", "zhè", "lǐ", "yuǎn", "ma"], "tones": [2, 4, 3, 3, 5], "english": "Is it far from here?",
             "sandhi_note": "里远 with potential T3 sandhi"},
            {"characters": "走路要多久", "pinyin": ["zǒu", "lù", "yào", "duō", "jiǔ"], "tones": [3, 4, 4, 1, 3], "english": "How long to walk?"},
            {"characters": "我迷路了", "pinyin": ["wǒ", "mí", "lù", "le"], "tones": [3, 2, 4, 5], "english": "I'm lost"},
        ]
    ),
]


# Voice command mappings
MENU_COMMANDS = {
    # Numbers for menu selection
    "one": 1, "1": 1, "first": 1,
    "two": 2, "2": 2, "second": 2,
    "three": 3, "3": 3, "third": 3,
    "four": 4, "4": 4, "fourth": 4,
    "five": 5, "5": 5, "fifth": 5,
    "six": 6, "6": 6, "sixth": 6,
    "seven": 7, "7": 7, "seventh": 7,
    "eight": 8, "8": 8, "eighth": 8,
    "nine": 9, "9": 9, "ninth": 9,

    # Navigation
    "menu": "menu", "main menu": "menu", "go back": "menu", "back": "menu",
    "quit": "quit", "exit": "quit", "stop": "quit",
    "skip": "skip", "next": "skip",
    "repeat": "repeat", "again": "repeat", "replay": "repeat",
    "help": "help",

    # Conversation mode
    "conversation": "conversation", "free": "conversation", "chat": "conversation",
}


class VoiceMenu:
    """
    Voice-controlled phone tree menu system.

    Designed for hands-free operation with Bluetooth headphones.
    """

    def __init__(
        self,
        tts=None,  # Text-to-speech for prompts
        asr=None,  # Speech recognition for commands
        evaluator=None,  # Tone evaluator
        practice_callback: Optional[Callable] = None,  # Called for practice (legacy)
        conversation_callback: Optional[Callable] = None,  # Called for conversation mode
        input_mode: InputMode = InputMode.TEXT,  # Default input mode
        lesson_audio_dir: Optional[Path] = None,  # Pre-generated lesson audio
    ):
        self.tts = tts
        self.asr = asr
        self._evaluator = evaluator
        self.practice_callback = practice_callback
        self.conversation_callback = conversation_callback
        self.input_mode = input_mode
        self.lesson_audio_dir = lesson_audio_dir or (
            Path(__file__).parent.parent.parent / "data" / "lessons" / "audio"
        )

        self.feedback = AudioFeedback()
        self.lessons = {l.id: l for l in BUILT_IN_LESSONS}
        self.state = MenuState.MAIN
        self.current_lesson: Optional[LessonPack] = None
        self.current_phrase_idx = 0

        # Session stats
        self.session_correct = 0
        self.session_total = 0

        # Progress tracking
        self.progress = ProgressTracker()

        # Load lesson audio manifest if available
        self.lesson_audio_manifest = self._load_audio_manifest()

    @property
    def evaluator(self):
        """Get evaluator, loading lazily if needed."""
        if self._evaluator is None:
            try:
                from .evaluator import Evaluator
                print("[VoiceMenu] Loading evaluator...")
                self._evaluator = Evaluator(lazy_load=True)
            except Exception as e:
                print(f"[VoiceMenu] Failed to load evaluator: {e}")
        return self._evaluator

    def _load_audio_manifest(self) -> dict:
        """Load pre-generated audio manifest if available."""
        manifest_path = self.lesson_audio_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _get_phrase_audio_path(self, lesson_id: str, phrase_idx: int) -> Optional[Path]:
        """Get pre-generated audio path for a phrase."""
        if lesson_id in self.lesson_audio_manifest:
            for entry in self.lesson_audio_manifest[lesson_id]:
                if entry.get("index") == phrase_idx:
                    return self.lesson_audio_dir / entry["audio_path"]
        return None

    def speak(self, text: str, language: str = "en", wait: bool = True, speed: str = "normal"):
        """Speak text using TTS."""
        print(f"  {text}")
        if self.tts:
            try:
                audio = self.tts.speak(text, language=language, speed=speed, play=True)
                if wait:
                    time.sleep(0.1)  # Small pause after speaking
            except Exception as e:
                print(f"  (TTS error: {e})")
        else:
            if wait:
                time.sleep(0.3)

    def play_audio_file(self, audio_path: Path) -> bool:
        """Play a pre-generated audio file."""
        if not audio_path.exists():
            return False

        try:
            import soundfile as sf
            import sounddevice as sd

            audio, sr = sf.read(str(audio_path))
            sd.play(audio, samplerate=sr)
            sd.wait()
            return True
        except Exception as e:
            print(f"  (Audio playback error: {e})")
            return False

    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for voice command or get text input."""
        if self.input_mode == InputMode.TEXT:
            # Text mode - keyboard input
            try:
                text = input("  > ").strip().lower()
                return text if text else None
            except (KeyboardInterrupt, EOFError):
                return "quit"

        # Voice mode
        self.feedback.listening()

        if self.asr:
            try:
                from ..realtime.audio_capture import UtteranceDetector
                import threading

                utterance_event = threading.Event()
                captured_audio = [None]

                def on_utterance(audio):
                    captured_audio[0] = audio
                    utterance_event.set()

                detector = UtteranceDetector(
                    on_utterance=on_utterance,
                    min_speech_duration_ms=100,
                    max_speech_duration_ms=3000,
                )

                detector.start()
                got_utterance = utterance_event.wait(timeout=timeout)
                detector.stop()

                if got_utterance and captured_audio[0] is not None:
                    result = self.asr.transcribe(captured_audio[0])
                    text = result.get("text", "").lower().strip()
                    print(f"  Heard: '{text}'")
                    return text
                return None

            except Exception as e:
                print(f"  (ASR error: {e})")
                return None
        else:
            # No ASR - fall back to text input
            try:
                text = input("  > ").strip().lower()
                return text if text else None
            except (KeyboardInterrupt, EOFError):
                return "quit"

    def parse_command(self, text: str) -> Optional[str | int]:
        """Parse voice input to command."""
        if not text:
            return None

        text = text.lower().strip()

        # Check direct mappings
        if text in MENU_COMMANDS:
            return MENU_COMMANDS[text]

        # Check partial matches
        for cmd, value in MENU_COMMANDS.items():
            if cmd in text:
                return value

        return None

    def show_main_menu(self):
        """Display and speak main menu."""
        self.feedback.menu()
        self.speak("Main Menu")
        time.sleep(0.3)

        menu_items = [
            "1. Starter Words",
            "2. Going to the Park",
            "3. At the Restaurant",
            "4. Numbers",
            "5. Feelings",
            "6. Daily Conversations",
            "7. Shopping & Bargaining",
            "8. Asking Directions",
            "9. Conversation Mode (free chat)",
            "",
            "Say 'quit' to exit",
        ]

        for item in menu_items:
            print(f"  {item}")

        # Speak condensed version
        if self.tts:
            self.speak(
                "One: Starter Words. Two: Park. Three: Restaurant. "
                "Four: Numbers. Five: Feelings. Six: Daily Conversations. "
                "Seven: Shopping. Eight: Directions. Nine: Free Conversation. "
                "Or say quit to exit.",
                wait=True
            )

    def start_lesson(self, lesson_id: str):
        """Start a lesson pack."""
        if lesson_id not in self.lessons:
            self.speak("Lesson not found")
            return

        self.current_lesson = self.lessons[lesson_id]
        self.current_phrase_idx = 0
        self.state = MenuState.LESSON

        self.feedback.menu()
        self.speak(f"Starting {self.current_lesson.name}")
        self.speak(self.current_lesson.description)
        time.sleep(0.3)
        self.speak("Say 'menu' anytime to go back. Let's begin!")

    def practice_current_phrase(self):
        """Practice the current phrase in the lesson."""
        if not self.current_lesson:
            return

        if self.current_phrase_idx >= len(self.current_lesson.phrases):
            # Lesson complete
            self.speak("Great job! You've completed this lesson.")
            accuracy = self.session_correct / self.session_total if self.session_total > 0 else 0
            self.speak(
                f"You got {self.session_correct} out of {self.session_total} correct. "
                f"That's {int(accuracy * 100)} percent."
            )
            self.feedback.perfect() if accuracy >= 0.8 else self.feedback.success()
            self.state = MenuState.MAIN
            return

        phrase = self.current_lesson.phrases[self.current_phrase_idx]

        # Announce phrase
        print(f"\n  {phrase['characters']} ({' '.join(phrase['pinyin'])})")
        print(f"  {phrase['english']}")

        self.speak(phrase['english'])
        time.sleep(0.2)

        # Play reference audio (pre-generated or TTS)
        audio_path = self._get_phrase_audio_path(
            self.current_lesson.id,
            self.current_phrase_idx
        )

        if audio_path and audio_path.exists():
            print("  [Playing reference audio...]")
            self.play_audio_file(audio_path)
        elif self.tts:
            # Use teacher speed: slow and deliberate for language learning
            self.speak(phrase['characters'], language="zh", wait=True, speed="teacher")
        else:
            print(f"  [Reference: {phrase['characters']}]")

        time.sleep(0.3)
        self.speak("Your turn")

        # Wait for attempt and evaluate
        result = self._evaluate_attempt(phrase)

        self.session_total += 1

        # Record progress
        if result and "syllable_results" in result:
            self.progress.record_evaluation(
                result["syllable_results"],
                phrase=phrase["characters"]
            )

        if result and result.get("overall_accuracy", 0) >= 0.8:
            self.session_correct += 1
            self.feedback.success()
            self.speak("Great job!")
            self.current_phrase_idx += 1
        else:
            self.feedback.try_again()
            self.speak("Let's try again")
            # Show what went wrong
            if result and "syllable_results" in result:
                for syl in result["syllable_results"]:
                    if not syl.get("correct"):
                        char = syl.get('character', '?')
                        expected = syl.get('expected_tone', '?')
                        detected = syl.get('detected_tone', '?')
                        self.speak(f"{char} should be tone {expected}, you said tone {detected}")

    def _evaluate_attempt(self, phrase: dict) -> Optional[dict]:
        """Evaluate a practice attempt."""
        # Use evaluator if available
        if self.evaluator is not None:
            try:
                result = self.evaluator.evaluate_from_mic(
                    expected_pinyin=[self._strip_tone_marks(p) for p in phrase["pinyin"]],
                    expected_tones=phrase["tones"],
                    characters=phrase["characters"],
                    timeout=10.0,
                )

                # Convert to dict format
                return {
                    "syllable_results": [
                        {
                            "character": r.character,
                            "pinyin": r.pinyin,
                            "expected_tone": r.expected_tone,
                            "detected_tone": r.detected_tone,
                            "confidence": r.confidence,
                            "correct": r.correct,
                        }
                        for r in result.syllable_results
                    ],
                    "overall_accuracy": result.overall_accuracy,
                    "status": result.status,
                }

            except Exception as e:
                print(f"  [Evaluation error: {e}]")

        # Fall back to legacy practice callback
        if self.practice_callback:
            return self.practice_callback(phrase)

        # No evaluator - just listen and move on
        self.feedback.listening()
        self.listen(timeout=8.0)
        self.feedback.success()
        return {"overall_accuracy": 1.0, "syllable_results": [], "status": "mock"}

    def _strip_tone_marks(self, pinyin: str) -> str:
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

    def run(self):
        """Main voice menu loop."""
        print("=" * 60)
        print("  Mandarin Tone Coach - Voice Mode")
        print("=" * 60)
        print()

        mode_str = "VOICE" if self.input_mode == InputMode.VOICE else "TEXT"
        print(f"  Input mode: {mode_str}")
        print("  Designed for hands-free operation with Bluetooth headphones.")
        print()

        # Start progress session
        self.progress.start_session()

        self.speak("Welcome to Mandarin Tone Coach")
        time.sleep(0.3)

        while True:
            try:
                if self.state == MenuState.MAIN:
                    self.show_main_menu()
                    command = self.listen(timeout=10.0)
                    parsed = self.parse_command(command) if command else None

                    if parsed == "quit":
                        self.speak("Goodbye!")
                        self.speak("再见", language="zh")
                        break
                    elif parsed == 1:
                        self.start_lesson("starter")
                    elif parsed == 2:
                        self.start_lesson("park")
                    elif parsed == 3:
                        self.start_lesson("restaurant")
                    elif parsed == 4:
                        self.start_lesson("numbers")
                    elif parsed == 5:
                        self.start_lesson("feelings")
                    elif parsed == 6:
                        self.start_lesson("conversations")
                    elif parsed == 7:
                        self.start_lesson("shopping")
                    elif parsed == 8:
                        self.start_lesson("directions")
                    elif parsed == 9 or parsed == "conversation":
                        self.state = MenuState.CONVERSATION
                        self.speak("Entering conversation mode. Say 'menu' to go back.")
                    elif parsed == "help":
                        self.speak("Say a number to select a lesson, or say quit to exit.")
                    else:
                        self.feedback.error()
                        self.speak("I didn't understand. Please try again.")

                elif self.state == MenuState.LESSON:
                    self.practice_current_phrase()

                    # Check for navigation commands between phrases
                    if self.current_phrase_idx < len(self.current_lesson.phrases):
                        print("  (Say 'skip' to skip, 'menu' to go back)")

                        # Listen briefly for navigation command
                        nav_cmd = self.listen(timeout=2.0)
                        if nav_cmd:
                            parsed = self.parse_command(nav_cmd)
                            if parsed == "menu":
                                self.state = MenuState.MAIN
                                self.speak("Returning to menu")
                            elif parsed == "skip":
                                self.current_phrase_idx += 1
                                self.speak("Skipping to next phrase")
                            elif parsed == "quit":
                                self.speak("Goodbye!")
                                break

                elif self.state == MenuState.CONVERSATION:
                    if self.conversation_callback:
                        result = self.conversation_callback()
                        if result == "menu":
                            self.state = MenuState.MAIN
                    else:
                        self.speak("Conversation mode not configured. Going back to menu.")
                        self.state = MenuState.MAIN

            except KeyboardInterrupt:
                print("\n")
                self.speak("Goodbye!")
                break

        # End session and save progress
        self.progress.end_session()

        print("\nSession ended.")
        if self.session_total > 0:
            accuracy = self.session_correct / self.session_total
            print(f"Session stats: {self.session_correct}/{self.session_total} ({accuracy:.0%})")

        # Show overall progress
        self.progress.print_summary()


if __name__ == "__main__":
    # Test voice menu without real TTS/ASR
    menu = VoiceMenu()
    menu.run()
