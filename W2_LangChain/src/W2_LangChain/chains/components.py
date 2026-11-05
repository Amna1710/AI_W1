
def count_words(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


def count_characters(text: str) -> int:
    """Count all characters, including spaces."""
    return len(text)


def clean_text(text: str) -> str:
    """Remove unnecessary whitespace."""
    return " ".join(text.split())


def get_text_statistics(text: str) -> dict:
    """Return basic statistics about a text."""

    return {
        "word_count": count_words(text),
        "character_count": count_characters(text),
    }
