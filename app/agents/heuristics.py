"""Light heuristics for PubMedQA maybe-risk abstracts and early-exit vetoes."""

from __future__ import annotations

# Keep phrases SPECIFIC. Broad tokens like "limitation" / "preliminary" match nearly
# every abstract and previously caused false maybe / noisy early-exit behavior.
INCONCLUSIVE_ABSTRACT_PHRASES: tuple[str, ...] = (
    "small sample size",
    "further research is needed",
    "further studies are needed",
    "further study is needed",
    "no statistically significant difference",
    "not statistically significant",
    "did not reach statistical significance",
    "borderline significant",
    "pilot study",
    "inconclusive",
    "remains unclear",
    "remain unclear",
    "mixed results",
    "conflicting results",
    "cannot be determined",
    "unable to conclude",
    "underpowered",
)


def abstract_suggests_inconclusive(
    patient_case: str,
    *,
    phrases: tuple[str, ...] = INCONCLUSIVE_ABSTRACT_PHRASES,
) -> bool:
    """Return True if the case/abstract contains strong inconclusiveness cue phrases."""
    text = (patient_case or "").lower()
    if not text:
        return False
    return any(phrase in text for phrase in phrases)
