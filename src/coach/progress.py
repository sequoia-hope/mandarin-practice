"""Progress persistence for tracking learning over time."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ToneStats:
    """Statistics for a single tone."""
    attempts: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        """Get accuracy as a percentage."""
        return self.correct / self.attempts if self.attempts > 0 else 0.0


@dataclass
class SessionStats:
    """Statistics for a single practice session."""
    start_time: str = ""
    end_time: str = ""
    total_attempts: int = 0
    total_correct: int = 0
    tone_stats: dict = field(default_factory=lambda: {
        "1": {"attempts": 0, "correct": 0},
        "2": {"attempts": 0, "correct": 0},
        "3": {"attempts": 0, "correct": 0},
        "4": {"attempts": 0, "correct": 0},
        "5": {"attempts": 0, "correct": 0},
    })
    phrases_practiced: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Get overall accuracy."""
        return self.total_correct / self.total_attempts if self.total_attempts > 0 else 0.0


@dataclass
class UserProgress:
    """User's overall learning progress."""
    user_id: str = "default"
    created_at: str = ""
    updated_at: str = ""
    total_sessions: int = 0
    total_practice_time_minutes: int = 0
    lifetime_attempts: int = 0
    lifetime_correct: int = 0
    lifetime_tone_stats: dict = field(default_factory=lambda: {
        "1": {"attempts": 0, "correct": 0},
        "2": {"attempts": 0, "correct": 0},
        "3": {"attempts": 0, "correct": 0},
        "4": {"attempts": 0, "correct": 0},
        "5": {"attempts": 0, "correct": 0},
    })
    lessons_completed: list = field(default_factory=list)
    phrases_mastered: list = field(default_factory=list)  # >80% accuracy
    recent_sessions: list = field(default_factory=list)  # Last 10 sessions

    @property
    def lifetime_accuracy(self) -> float:
        """Get lifetime accuracy."""
        return self.lifetime_correct / self.lifetime_attempts if self.lifetime_attempts > 0 else 0.0


class ProgressTracker:
    """
    Track and persist user learning progress.

    Stores data in JSON format in the user's data directory.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        user_id: str = "default",
    ):
        """
        Initialize progress tracker.

        Args:
            data_dir: Directory to store progress data
            user_id: User identifier for multi-user support
        """
        self.data_dir = data_dir or (
            Path(__file__).parent.parent.parent / "data" / "progress"
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.user_id = user_id
        self.progress_file = self.data_dir / f"{user_id}_progress.json"

        self.progress = self._load_progress()
        self.current_session: Optional[SessionStats] = None

    def _load_progress(self) -> UserProgress:
        """Load progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    progress = UserProgress(**data)
                    return progress
            except Exception as e:
                print(f"[Progress] Error loading: {e}")

        # Create new progress
        return UserProgress(
            user_id=self.user_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

    def save(self):
        """Save progress to file."""
        self.progress.updated_at = datetime.now().isoformat()

        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.progress), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Progress] Error saving: {e}")

    def start_session(self):
        """Start a new practice session."""
        self.current_session = SessionStats(
            start_time=datetime.now().isoformat(),
        )

    def end_session(self):
        """End current session and update progress."""
        if self.current_session is None:
            return

        self.current_session.end_time = datetime.now().isoformat()

        # Calculate session duration
        try:
            start = datetime.fromisoformat(self.current_session.start_time)
            end = datetime.fromisoformat(self.current_session.end_time)
            duration_minutes = int((end - start).total_seconds() / 60)
            self.progress.total_practice_time_minutes += duration_minutes
        except Exception:
            pass

        # Update lifetime stats
        self.progress.total_sessions += 1
        self.progress.lifetime_attempts += self.current_session.total_attempts
        self.progress.lifetime_correct += self.current_session.total_correct

        # Update tone stats
        for tone, stats in self.current_session.tone_stats.items():
            self.progress.lifetime_tone_stats[tone]["attempts"] += stats["attempts"]
            self.progress.lifetime_tone_stats[tone]["correct"] += stats["correct"]

        # Add to recent sessions (keep last 10)
        session_summary = {
            "date": self.current_session.start_time,
            "attempts": self.current_session.total_attempts,
            "correct": self.current_session.total_correct,
            "accuracy": self.current_session.accuracy,
        }
        self.progress.recent_sessions.insert(0, session_summary)
        self.progress.recent_sessions = self.progress.recent_sessions[:10]

        self.save()
        self.current_session = None

    def record_attempt(
        self,
        expected_tone: int,
        detected_tone: int,
        phrase: Optional[str] = None,
    ):
        """
        Record a single tone attempt.

        Args:
            expected_tone: Expected tone number (1-5)
            detected_tone: Detected tone number (1-5)
            phrase: Optional phrase being practiced
        """
        if self.current_session is None:
            self.start_session()

        correct = (expected_tone == detected_tone)
        tone_key = str(expected_tone)

        self.current_session.total_attempts += 1
        if correct:
            self.current_session.total_correct += 1

        if tone_key in self.current_session.tone_stats:
            self.current_session.tone_stats[tone_key]["attempts"] += 1
            if correct:
                self.current_session.tone_stats[tone_key]["correct"] += 1

        if phrase and phrase not in self.current_session.phrases_practiced:
            self.current_session.phrases_practiced.append(phrase)

    def record_evaluation(
        self,
        syllable_results: list[dict],
        phrase: Optional[str] = None,
    ):
        """
        Record results from a full phrase evaluation.

        Args:
            syllable_results: List of {expected_tone, detected_tone, correct, ...}
            phrase: The phrase being practiced
        """
        for result in syllable_results:
            self.record_attempt(
                expected_tone=result.get("expected_tone", 0),
                detected_tone=result.get("detected_tone", 0),
                phrase=phrase,
            )

    def mark_phrase_mastered(self, phrase: str):
        """Mark a phrase as mastered (>80% accuracy consistently)."""
        if phrase not in self.progress.phrases_mastered:
            self.progress.phrases_mastered.append(phrase)
            self.save()

    def mark_lesson_completed(self, lesson_id: str):
        """Mark a lesson as completed."""
        if lesson_id not in self.progress.lessons_completed:
            self.progress.lessons_completed.append(lesson_id)
            self.save()

    def get_weakest_tones(self, n: int = 2) -> list[int]:
        """
        Get the N tones with lowest accuracy.

        Useful for spaced repetition targeting.
        """
        tone_accuracies = []

        for tone, stats in self.progress.lifetime_tone_stats.items():
            if stats["attempts"] >= 5:  # Need minimum attempts
                accuracy = stats["correct"] / stats["attempts"]
                tone_accuracies.append((int(tone), accuracy))

        # Sort by accuracy (ascending)
        tone_accuracies.sort(key=lambda x: x[1])

        return [tone for tone, _ in tone_accuracies[:n]]

    def get_summary(self) -> dict:
        """Get a summary of user progress."""
        return {
            "total_sessions": self.progress.total_sessions,
            "practice_time_hours": self.progress.total_practice_time_minutes / 60,
            "lifetime_accuracy": self.progress.lifetime_accuracy,
            "lifetime_attempts": self.progress.lifetime_attempts,
            "lessons_completed": len(self.progress.lessons_completed),
            "phrases_mastered": len(self.progress.phrases_mastered),
            "weakest_tones": self.get_weakest_tones(2),
            "tone_accuracies": {
                tone: stats["correct"] / stats["attempts"] if stats["attempts"] > 0 else 0
                for tone, stats in self.progress.lifetime_tone_stats.items()
            },
        }

    def print_summary(self):
        """Print a formatted progress summary."""
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("  Your Learning Progress")
        print("=" * 50)

        print(f"\n  Sessions: {summary['total_sessions']}")
        print(f"  Practice time: {summary['practice_time_hours']:.1f} hours")
        print(f"  Total attempts: {summary['lifetime_attempts']}")
        print(f"  Overall accuracy: {summary['lifetime_accuracy']:.0%}")
        print(f"  Lessons completed: {summary['lessons_completed']}")
        print(f"  Phrases mastered: {summary['phrases_mastered']}")

        print("\n  Accuracy by Tone:")
        tone_names = {
            "1": "T1 (high)",
            "2": "T2 (rising)",
            "3": "T3 (dipping)",
            "4": "T4 (falling)",
            "5": "T5 (neutral)",
        }
        for tone, accuracy in summary["tone_accuracies"].items():
            bar = "=" * int(accuracy * 20)
            print(f"    {tone_names[tone]:15} {accuracy:5.0%} [{bar:20}]")

        if summary["weakest_tones"]:
            weak = ", ".join(f"T{t}" for t in summary["weakest_tones"])
            print(f"\n  Focus on: {weak}")

        print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test progress tracking
    tracker = ProgressTracker()

    print("Testing progress tracker...")

    tracker.start_session()

    # Simulate some practice
    tracker.record_attempt(3, 3, "你好")  # Correct
    tracker.record_attempt(3, 2, "你好")  # Wrong - said rising instead of dipping
    tracker.record_attempt(4, 4, "谢谢")  # Correct
    tracker.record_attempt(5, 5, "谢谢")  # Correct
    tracker.record_attempt(1, 1, "公园")  # Correct
    tracker.record_attempt(2, 4, "公园")  # Wrong - said falling instead of rising

    tracker.end_session()

    tracker.print_summary()
