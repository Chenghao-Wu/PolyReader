"""Shared constants and text utilities."""

import re
import unicodedata


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "or", "but", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "as", "until", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their",
}


def normalize_text(text: str) -> str:
    """Lowercase, Unicode-normalize, strip HTML tags, collapse whitespace, strip."""
    text = re.sub(r'<[^>]+>', '', text)
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize_query(query: str) -> set[str]:
    """Extract lowercase keywords from query, filtering stopwords."""
    words = set(re.findall(r'[a-z0-9]+', query.lower()))
    return words - STOPWORDS
