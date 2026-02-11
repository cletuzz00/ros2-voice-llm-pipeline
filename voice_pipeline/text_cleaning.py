import re
import string


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip ends."""
    return " ".join((text or "").split())


def strip_punctuation(text: str) -> str:
    """Remove ASCII punctuation characters."""
    return (text or "").translate(str.maketrans("", "", string.punctuation))


def to_lower(text: str) -> str:
    """Lowercase helper that is robust to None."""
    return (text or "").lower()


def clean_transcript(text: str) -> str:
    """
    Basic transcript cleaning used across the project.

    You can extend this (e.g. for key-phrase handling, regex normalization)
    without having to touch STT or LLM code.
    """
    return normalize_whitespace(text)


