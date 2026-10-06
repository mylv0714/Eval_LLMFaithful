"""Chen-style CoT faithfulness pipeline. Generators are local; the judge is gpt-oss-20b."""

__version__ = "0.1.0"

HINT_TYPES = (
    "sycophancy",
    "consistency",
    "visual_pattern",
    "metadata",
    "grader",
    "unethical",
)

BASELINE = "baseline"
N_CHOICES = 4
LETTERS = ("A", "B", "C", "D")
