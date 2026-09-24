"""
Text Normalization for Indic Languages.
Handles Unicode NFC normalization, number-to-words expansion, punctuation sanitization.
"""

import re
import unicodedata

# Basic Hindi digit to word mapping for Devanagari and Latin numbers
HINDI_DIGITS = {
    '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
    '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
    '०': 'शून्य', '१': 'एक', '२': 'दो', '३': 'तीन', '४': 'चार',
    '५': 'पांच', '६': 'छह', '७': 'सात', '८': 'आठ', '९': 'नौ'
}

PUNCTUATION_MAP = {
    '।': '.',
    '॥': '.',
    '|': '.',
    ';': ',',
    ':': ',',
    '-': ' ',
    '—': ' ',
    '–': ' ',
    '"': '',
    "'": '',
    '`': '',
    '(': '',
    ')': '',
    '[': '',
    ']': '',
    '{': '',
    '}': ''
}

def normalize_unicode(text: str) -> str:
    """Normalize text to Unicode NFC format."""
    return unicodedata.normalize('NFC', text)

def clean_whitespace(text: str) -> str:
    """Remove redundant spaces, tabs, and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def expand_numbers(text: str, lang: str = "hi") -> str:
    """Expand digits into spoken words."""
    result = []
    for char in text:
        if char in HINDI_DIGITS:
            result.append(" " + HINDI_DIGITS[char] + " ")
        else:
            result.append(char)
    return "".join(result)

def sanitize_punctuation(text: str) -> str:
    """Standardize punctuation marks."""
    for p_in, p_out in PUNCTUATION_MAP.items():
        text = text.replace(p_in, p_out)
    # Ensure punctuation has appropriate space
    text = re.sub(r'([.,?!])', r' \1 ', text)
    return clean_whitespace(text)

def normalize_indic_text(text: str, lang: str = "hi") -> str:
    """
    Complete pipeline to normalize Indic text:
    1. Unicode NFC normalization
    2. Expand numbers to words
    3. Clean punctuation
    4. Clean whitespace
    """
    text = normalize_unicode(text)
    text = expand_numbers(text, lang=lang)
    text = sanitize_punctuation(text)
    return text
