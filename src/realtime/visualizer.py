"""F0 contour visualization for tone coaching."""

from typing import Optional

import numpy as np


# Reference F0 contours for each tone (normalized)
# These represent idealized tone patterns
REFERENCE_CONTOURS = {
    0: np.array([5, 5, 5, 5, 5]),  # Tone 1: High level (55)
    1: np.array([3, 3.5, 4, 4.5, 5]),  # Tone 2: Rising (35)
    2: np.array([2, 1.5, 1, 1.5, 2.5]),  # Tone 3: Dipping (214)
    3: np.array([5, 4, 3, 2, 1]),  # Tone 4: Falling (51)
    4: np.array([3, 3, 3, 3, 3]),  # Tone 5: Neutral (varies)
}

TONE_COLORS = {
    0: "\033[91m",  # Red - Tone 1
    1: "\033[93m",  # Yellow - Tone 2
    2: "\033[92m",  # Green - Tone 3
    3: "\033[94m",  # Blue - Tone 4
    4: "\033[95m",  # Magenta - Tone 5
}

RESET_COLOR = "\033[0m"


class F0Visualizer:
    """
    Visualize F0 contours for tone feedback.

    Supports both terminal (ASCII art) and matplotlib visualization.
    """

    def __init__(
        self,
        height: int = 10,
        width: int = 40,
        min_f0: float = 80,
        max_f0: float = 400,
    ):
        """
        Initialize visualizer.

        Args:
            height: Terminal display height in lines
            width: Terminal display width in characters
            min_f0: Minimum F0 for display range
            max_f0: Maximum F0 for display range
        """
        self.height = height
        self.width = width
        self.min_f0 = min_f0
        self.max_f0 = max_f0

    def normalize_f0(self, f0: np.ndarray) -> np.ndarray:
        """Normalize F0 to 0-1 range."""
        f0_clipped = np.clip(f0, self.min_f0, self.max_f0)
        return (f0_clipped - self.min_f0) / (self.max_f0 - self.min_f0)

    def resample(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """Resample F0 contour to target length."""
        if len(f0) == target_length:
            return f0

        indices = np.linspace(0, len(f0) - 1, target_length)
        return np.interp(indices, np.arange(len(f0)), f0)

    def render_terminal(
        self,
        f0: np.ndarray,
        tone: Optional[int] = None,
        show_reference: bool = True,
    ) -> str:
        """
        Render F0 contour as ASCII art for terminal display.

        Args:
            f0: F0 contour (Hz)
            tone: Predicted tone (0-4) for coloring
            show_reference: Whether to show reference contour

        Returns:
            ASCII art string
        """
        # Handle unvoiced frames
        voiced_mask = f0 > 0
        if not voiced_mask.any():
            return "No voiced frames detected"

        # Interpolate unvoiced regions
        f0_interp = f0.copy()
        if not voiced_mask.all():
            voiced_indices = np.where(voiced_mask)[0]
            unvoiced_indices = np.where(~voiced_mask)[0]
            f0_interp[unvoiced_indices] = np.interp(
                unvoiced_indices,
                voiced_indices,
                f0[voiced_indices],
            )

        # Resample to display width
        f0_resampled = self.resample(f0_interp, self.width)
        f0_normalized = self.normalize_f0(f0_resampled)

        # Get reference contour if showing
        if show_reference and tone is not None and tone in REFERENCE_CONTOURS:
            ref_contour = REFERENCE_CONTOURS[tone]
            ref_resampled = self.resample(ref_contour / 5.0, self.width)
        else:
            ref_resampled = None

        # Build ASCII grid
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Plot reference contour (dots)
        if ref_resampled is not None:
            for x, val in enumerate(ref_resampled):
                y = int((1 - val) * (self.height - 1))
                y = max(0, min(self.height - 1, y))
                grid[y][x] = "·"

        # Plot actual F0 contour (asterisks)
        color = TONE_COLORS.get(tone, "") if tone is not None else ""

        for x, val in enumerate(f0_normalized):
            y = int((1 - val) * (self.height - 1))
            y = max(0, min(self.height - 1, y))
            grid[y][x] = f"{color}*{RESET_COLOR}" if color else "*"

        # Build output string
        lines = []

        # Y-axis labels
        lines.append(f"{self.max_f0:>4.0f} Hz ┤" + "".join(grid[0]))

        for row in range(1, self.height - 1):
            lines.append("        │" + "".join(grid[row]))

        lines.append(f"{self.min_f0:>4.0f} Hz ┤" + "".join(grid[-1]))
        lines.append("        └" + "─" * self.width)

        return "\n".join(lines)

    def plot_matplotlib(
        self,
        f0: np.ndarray,
        tone: Optional[int] = None,
        confidence: Optional[float] = None,
        show_reference: bool = True,
        output_path: Optional[str] = None,
    ):
        """
        Plot F0 contour using matplotlib.

        Args:
            f0: F0 contour (Hz)
            tone: Predicted tone
            confidence: Prediction confidence
            show_reference: Show reference contour
            output_path: Path to save figure (shows if None)
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))

        # Time axis (assuming 10ms frames)
        time = np.arange(len(f0)) * 0.01

        # Plot voiced regions only
        voiced_mask = f0 > 0
        f0_plot = f0.copy()
        f0_plot[~voiced_mask] = np.nan

        # Plot F0 contour
        tone_colors = ["#E63946", "#F4A261", "#2A9D8F", "#264653", "#9B5DE5"]
        color = tone_colors[tone] if tone is not None and tone < len(tone_colors) else "blue"

        ax.plot(time, f0_plot, color=color, linewidth=2, label="Your pitch")

        # Plot reference contour
        if show_reference and tone is not None and tone in REFERENCE_CONTOURS:
            ref = REFERENCE_CONTOURS[tone]
            # Scale reference to actual F0 range
            voiced_f0 = f0[voiced_mask]
            if len(voiced_f0) > 0:
                f0_mean = np.mean(voiced_f0)
                ref_scaled = (ref - 3) * (f0_mean * 0.3) + f0_mean
                ref_time = np.linspace(time[0], time[-1], len(ref))
                ax.plot(
                    ref_time, ref_scaled,
                    color=color, linewidth=1, linestyle="--", alpha=0.5,
                    label="Target contour",
                )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("F0 (Hz)")

        title = "F0 Contour"
        if tone is not None:
            from ..model.classifier import TONE_NAMES
            title += f" - {TONE_NAMES.get(tone, f'Tone {tone + 1}')}"
        if confidence is not None:
            title += f" ({confidence:.0%})"

        ax.set_title(title)
        ax.set_ylim(self.min_f0, self.max_f0)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_feedback(
        self,
        f0: np.ndarray,
        predicted_tone: int,
        expected_tone: Optional[int] = None,
    ) -> str:
        """
        Generate coaching feedback based on F0 analysis.

        Args:
            f0: F0 contour
            predicted_tone: Model's prediction
            expected_tone: Expected tone (if known)

        Returns:
            Feedback string
        """
        voiced_mask = f0 > 0
        if not voiced_mask.any():
            return "Could not detect pitch. Try speaking louder or closer to the microphone."

        voiced_f0 = f0[voiced_mask]

        # Analyze F0 characteristics
        f0_start = voiced_f0[:len(voiced_f0)//3].mean()
        f0_end = voiced_f0[-len(voiced_f0)//3:].mean()
        f0_diff = f0_end - f0_start
        f0_mean = voiced_f0.mean()
        f0_std = voiced_f0.std()

        feedback = []

        # Check if tone matches expectation
        if expected_tone is not None and predicted_tone != expected_tone:
            feedback.append(
                f"Expected Tone {expected_tone + 1}, but detected Tone {predicted_tone + 1}."
            )

        # Tone-specific feedback
        if predicted_tone == 0:  # Tone 1 - High level
            if f0_std > f0_mean * 0.1:
                feedback.append("Tone 1 should be steady. Try to maintain a constant pitch.")
            if f0_mean < 200:
                feedback.append("Try raising your pitch higher for Tone 1.")

        elif predicted_tone == 1:  # Tone 2 - Rising
            if f0_diff < 0:
                feedback.append("Tone 2 should rise. Your pitch went down instead.")
            elif abs(f0_diff) < f0_mean * 0.1:
                feedback.append("Try making the rise more pronounced for Tone 2.")

        elif predicted_tone == 2:  # Tone 3 - Dipping
            # Check for dip
            f0_mid = voiced_f0[len(voiced_f0)//3:-len(voiced_f0)//3].mean()
            if f0_mid >= min(f0_start, f0_end):
                feedback.append("Tone 3 should dip in the middle. Try lowering then raising.")

        elif predicted_tone == 3:  # Tone 4 - Falling
            if f0_diff > 0:
                feedback.append("Tone 4 should fall. Your pitch went up instead.")
            elif abs(f0_diff) < f0_mean * 0.15:
                feedback.append("Try making the fall more dramatic for Tone 4.")

        elif predicted_tone == 4:  # Tone 5 - Neutral
            feedback.append("Neutral tone detected. Duration and context matter most.")

        if not feedback:
            feedback.append("Good job! Your tone pattern looks correct.")

        return " ".join(feedback)


if __name__ == "__main__":
    # Demo visualization
    print("F0 Visualizer Demo\n")

    visualizer = F0Visualizer()

    # Generate sample F0 contours for each tone
    for tone in range(5):
        print(f"\n{'='*50}")
        print(f"Tone {tone + 1} Example:")
        print("=" * 50)

        # Generate synthetic F0 based on tone pattern
        ref = REFERENCE_CONTOURS[tone]
        # Scale to realistic Hz range
        f0 = np.interp(
            np.linspace(0, 1, 50),
            np.linspace(0, 1, len(ref)),
            ref * 40 + 150,  # Scale to ~150-350 Hz range
        )

        # Add some noise
        f0 += np.random.randn(len(f0)) * 5

        print(visualizer.render_terminal(f0, tone))

        feedback = visualizer.generate_feedback(f0, tone)
        print(f"\nFeedback: {feedback}")
