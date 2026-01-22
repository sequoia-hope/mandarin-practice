"""Interactive Mandarin tone coaching interface."""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from ..model.classifier import TONE_NAMES
from ..realtime.pipeline import RealtimePipeline, ToneResult
from ..realtime.visualizer import F0Visualizer


# Practice words for each tone
PRACTICE_WORDS = {
    0: [  # Tone 1
        ("妈", "mā", "mother"),
        ("高", "gāo", "tall"),
        ("天", "tiān", "sky"),
        ("书", "shū", "book"),
        ("飞", "fēi", "fly"),
    ],
    1: [  # Tone 2
        ("麻", "má", "hemp"),
        ("人", "rén", "person"),
        ("来", "lái", "come"),
        ("学", "xué", "learn"),
        ("茶", "chá", "tea"),
    ],
    2: [  # Tone 3
        ("马", "mǎ", "horse"),
        ("好", "hǎo", "good"),
        ("小", "xiǎo", "small"),
        ("你", "nǐ", "you"),
        ("水", "shuǐ", "water"),
    ],
    3: [  # Tone 4
        ("骂", "mà", "scold"),
        ("大", "dà", "big"),
        ("去", "qù", "go"),
        ("是", "shì", "is"),
        ("对", "duì", "correct"),
    ],
    4: [  # Tone 5 (neutral)
        ("吗", "ma", "question particle"),
        ("的", "de", "possessive"),
        ("了", "le", "completion"),
        ("呢", "ne", "question particle"),
        ("吧", "ba", "suggestion"),
    ],
}


@dataclass
class SessionStats:
    """Statistics for a coaching session."""

    total_attempts: int = 0
    correct: int = 0
    by_tone: dict = field(default_factory=lambda: {i: {"attempts": 0, "correct": 0} for i in range(5)})
    recent_results: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.correct / self.total_attempts

    def record(self, expected_tone: int, predicted_tone: int, correct: bool):
        self.total_attempts += 1
        self.by_tone[expected_tone]["attempts"] += 1

        if correct:
            self.correct += 1
            self.by_tone[expected_tone]["correct"] += 1

        self.recent_results.append({
            "expected": expected_tone,
            "predicted": predicted_tone,
            "correct": correct,
        })

        # Keep only last 20
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)


class ToneCoach:
    """
    Interactive coaching interface for Mandarin tones.

    Provides:
    - Guided practice with feedback
    - Free practice mode
    - Progress tracking
    - F0 visualization
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize tone coach.

        Args:
            checkpoint_path: Path to trained classifier checkpoint
            device: Torch device
        """
        self.console = Console()
        self.visualizer = F0Visualizer()
        self.stats = SessionStats()

        self.console.print("\n[bold blue]Initializing Mandarin Tone Coach...[/bold blue]\n")

        self.pipeline = RealtimePipeline(
            classifier_checkpoint=checkpoint_path,
            use_f0=True,
            use_light_encoder=True,
            device=device,
        )

    def print_welcome(self):
        """Print welcome message."""
        welcome_text = """
[bold cyan]Welcome to the Mandarin Tone Coach![/bold cyan]

This tool will help you practice and improve your Mandarin tones.

[bold]The Four Tones (and Neutral):[/bold]
  1. [red]First Tone (ˉ)[/red] - High and level: mā (妈)
  2. [yellow]Second Tone (ˊ)[/yellow] - Rising: má (麻)
  3. [green]Third Tone (ˇ)[/green] - Dipping: mǎ (马)
  4. [blue]Fourth Tone (ˋ)[/blue] - Falling: mà (骂)
  5. [magenta]Neutral Tone[/magenta] - Light and short: ma (吗)

[bold]Practice Modes:[/bold]
  • [cyan]Guided Practice[/cyan] - Practice specific tones with prompts
  • [cyan]Free Practice[/cyan] - Speak freely and get tone feedback
  • [cyan]Quiz Mode[/cyan] - Test your tone accuracy
"""
        self.console.print(Panel(welcome_text, title="Mandarin Tone Coach"))

    def print_stats(self):
        """Print current session statistics."""
        table = Table(title="Session Statistics")
        table.add_column("Tone", style="cyan")
        table.add_column("Attempts", justify="right")
        table.add_column("Correct", justify="right")
        table.add_column("Accuracy", justify="right")

        for tone in range(5):
            attempts = self.stats.by_tone[tone]["attempts"]
            correct = self.stats.by_tone[tone]["correct"]
            accuracy = correct / attempts if attempts > 0 else 0

            table.add_row(
                TONE_NAMES[tone].split("(")[0].strip(),
                str(attempts),
                str(correct),
                f"{accuracy:.0%}",
            )

        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{self.stats.total_attempts}[/bold]",
            f"[bold]{self.stats.correct}[/bold]",
            f"[bold]{self.stats.accuracy:.0%}[/bold]",
        )

        self.console.print(table)

    def display_result(self, result: ToneResult, expected_tone: Optional[int] = None):
        """Display classification result with visualization."""
        # Tone colors
        tone_styles = {
            0: "red",
            1: "yellow",
            2: "green",
            3: "blue",
            4: "magenta",
        }

        style = tone_styles.get(result.tone, "white")

        # Result panel
        result_text = Text()
        result_text.append(f"Detected: ", style="bold")
        result_text.append(f"{result.tone_name}\n", style=f"bold {style}")
        result_text.append(f"Confidence: {result.confidence:.0%}\n")
        result_text.append(f"Processing: {result.processing_time_ms:.0f}ms")

        if expected_tone is not None:
            if result.tone == expected_tone:
                result_text.append("\n\n")
                result_text.append("Correct!", style="bold green")
            else:
                result_text.append("\n\n")
                result_text.append(f"Expected: {TONE_NAMES[expected_tone]}", style="bold red")

        self.console.print(Panel(result_text, title="Result"))

        # F0 visualization
        if result.f0_contour is not None:
            self.console.print("\n[bold]Pitch Contour:[/bold]")
            self.console.print(
                self.visualizer.render_terminal(
                    result.f0_contour,
                    tone=result.tone,
                    show_reference=True,
                )
            )

            # Feedback
            feedback = self.visualizer.generate_feedback(
                result.f0_contour,
                result.tone,
                expected_tone,
            )
            self.console.print(f"\n[italic]{feedback}[/italic]\n")

    def guided_practice(self, tone: Optional[int] = None):
        """
        Guided practice mode for specific tones.

        Args:
            tone: Specific tone to practice (0-4), or None for random
        """
        if tone is None:
            tone = random.randint(0, 4)

        # Get practice word
        word_list = PRACTICE_WORDS[tone]
        char, pinyin, meaning = random.choice(word_list)

        # Display prompt
        prompt = f"""
[bold]Practice: {TONE_NAMES[tone]}[/bold]

Character: [bold cyan]{char}[/bold cyan]
Pinyin: [bold]{pinyin}[/bold]
Meaning: {meaning}

Say this syllable when ready...
"""
        self.console.print(Panel(prompt, title="Guided Practice"))

        # Wait for speech and classify
        result = self._capture_and_classify()

        if result:
            correct = result.tone == tone
            self.stats.record(tone, result.tone, correct)
            self.display_result(result, expected_tone=tone)

    def free_practice(self):
        """Free practice mode - speak any syllable."""
        self.console.print(
            Panel(
                "Speak any Mandarin syllable...\n\n"
                "[dim]Press Ctrl+C to return to menu[/dim]",
                title="Free Practice",
            )
        )

        result = self._capture_and_classify()

        if result:
            self.display_result(result)

    def quiz_mode(self, num_questions: int = 10):
        """
        Quiz mode - test tone accuracy.

        Args:
            num_questions: Number of questions in quiz
        """
        self.console.print(
            Panel(
                f"[bold]Tone Quiz[/bold]\n\n"
                f"You will be asked to produce {num_questions} tones.\n"
                "Try to get as many correct as possible!",
                title="Quiz Mode",
            )
        )

        quiz_stats = SessionStats()

        for i in range(num_questions):
            tone = random.randint(0, 4)
            word_list = PRACTICE_WORDS[tone]
            char, pinyin, meaning = random.choice(word_list)

            self.console.print(f"\n[bold]Question {i + 1}/{num_questions}[/bold]")
            self.console.print(f"Say: [bold cyan]{char}[/bold cyan] ({pinyin}) - {meaning}")

            result = self._capture_and_classify()

            if result:
                correct = result.tone == tone
                quiz_stats.record(tone, result.tone, correct)
                self.stats.record(tone, result.tone, correct)

                if correct:
                    self.console.print("[bold green]Correct![/bold green]")
                else:
                    self.console.print(
                        f"[bold red]Incorrect.[/bold red] "
                        f"Expected {TONE_NAMES[tone]}, got {result.tone_name}"
                    )

        # Quiz summary
        self.console.print(
            Panel(
                f"[bold]Quiz Complete![/bold]\n\n"
                f"Score: {quiz_stats.correct}/{quiz_stats.total_attempts} "
                f"({quiz_stats.accuracy:.0%})",
                title="Quiz Results",
            )
        )

    def _capture_and_classify(self) -> Optional[ToneResult]:
        """Capture audio and classify tone."""
        from ..realtime.audio_capture import UtteranceDetector

        result_holder = []
        detected = False

        def on_utterance(audio):
            nonlocal detected
            result = self.pipeline.classify(audio)
            result_holder.append(result)
            detected = True

        def on_speech_start():
            self.console.print("[dim]Listening...[/dim]")

        detector = UtteranceDetector(
            on_utterance=on_utterance,
            on_speech_start=on_speech_start,
            vad_threshold=0.35,  # Lower threshold for quieter mics
            min_speech_duration_ms=100,  # Shorter trigger time
            max_speech_duration_ms=3000,
        )

        try:
            with detector:
                # Wait for detection with timeout
                timeout = 10.0
                start = time.time()
                while not detected and (time.time() - start) < timeout:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            return None

        return result_holder[0] if result_holder else None

    def run(self):
        """Run the interactive coaching session."""
        self.print_welcome()

        while True:
            # Menu
            menu = """
[bold]Choose a mode:[/bold]
  1. Guided Practice (specific tone)
  2. Random Practice
  3. Free Practice
  4. Quiz Mode (10 questions)
  5. View Statistics
  6. Exit
"""
            self.console.print(Panel(menu, title="Menu"))

            try:
                choice = input("Enter choice (1-6): ").strip()

                if choice == "1":
                    self.console.print("\nWhich tone? (1-5, or 0 for random): ", end="")
                    tone_choice = input().strip()
                    if tone_choice.isdigit():
                        tone = int(tone_choice) - 1 if int(tone_choice) > 0 else None
                        self.guided_practice(tone=tone if tone in range(5) else None)
                    else:
                        self.guided_practice()

                elif choice == "2":
                    self.guided_practice(tone=None)

                elif choice == "3":
                    while True:
                        try:
                            self.free_practice()
                        except KeyboardInterrupt:
                            break

                elif choice == "4":
                    self.quiz_mode(num_questions=10)

                elif choice == "5":
                    self.print_stats()

                elif choice == "6":
                    self.console.print("\n[bold]Thanks for practicing! 再见![/bold]\n")
                    break

                else:
                    self.console.print("[red]Invalid choice. Please enter 1-6.[/red]")

            except KeyboardInterrupt:
                self.console.print("\n\n[bold]Thanks for practicing! 再见![/bold]\n")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mandarin Tone Coach")
    parser.add_argument("--checkpoint", type=str, default=None, help="Classifier checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    args = parser.parse_args()

    coach = ToneCoach(checkpoint_path=args.checkpoint, device=args.device)
    coach.run()
