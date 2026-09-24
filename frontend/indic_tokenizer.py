"""
Portable Indic Tokenizer.
Provides character/akshara-level phonetic tokenization with zero native C++ dependencies.
Cross-platform compatible across Python, Node.js, and Android Kotlin.
"""

import json
import os
from typing import List, Dict, Tuple
from .normalizer import normalize_indic_text

# Default Hindi / Devanagari characters
DEFAULT_HINDI_VOCAB = [
    # Special tokens
    "_",  # 0: Padding / Blank
    "^",  # 1: Beginning of Sentence (BOS)
    "$",  # 2: End of Sentence (EOS)
    " ",  # 3: Word boundary / Space
    
    # Punctuation
    ".", ",", "!", "?",
    
    # Devanagari Independent Vowels
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "अं", "अः",
    
    # Devanagari Consonants
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    
    # Nukta variations (Perso-Arabic / loan phonemes)
    "क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़",
    
    # Dependent Vowels (Matras)
    "ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ",
    
    # Signs and Modifiers
    "्",  # Virama / Halant
    "ं",  # Anusvara
    "ँ",  # Chandrabindu
    "ः",  # Visarga
    "़",  # Nukta
    "ऽ"   # Avagraha
]

class IndicTokenizer:
    """
    Tokenizer for Indic scripts that parses characters, matras, and modifiers.
    Works natively across platforms without espeak or external C++ libraries.
    """
    def __init__(self, vocab_path: str = None, lang: str = "hi"):
        self.lang = lang
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.char_to_id = data.get("char_to_id", {})
                self.id_to_char = {int(k): v for k, v in data.get("id_to_char", {}).items()}
        else:
            # Build default vocab
            self.char_to_id = {char: idx for idx, char in enumerate(DEFAULT_HINDI_VOCAB)}
            self.id_to_char = {idx: char for idx, char in enumerate(DEFAULT_HINDI_VOCAB)}

        self.pad_id = self.char_to_id.get("_", 0)
        self.bos_id = self.char_to_id.get("^", 1)
        self.eos_id = self.char_to_id.get("$", 2)
        self.space_id = self.char_to_id.get(" ", 3)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def save_vocab(self, filepath: str):
        """Save vocabulary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "lang": self.lang,
            "vocab_size": len(self.char_to_id),
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def text_to_tokens(self, text: str) -> List[str]:
        """Convert normalized text into list of valid vocabulary tokens."""
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            # Check 2-character tokens (e.g. consonant + nukta: क + ़ = क़)
            if i + 1 < n and text[i:i+2] in self.char_to_id:
                tokens.append(text[i:i+2])
                i += 2
            elif text[i] in self.char_to_id:
                tokens.append(text[i])
                i += 1
            else:
                # If unknown char, ignore or map to space
                if text[i].isspace():
                    tokens.append(" ")
                i += 1
        return tokens

    def encode(self, text: str, add_bos_eos: bool = True, add_blank: bool = True) -> List[int]:
        """
        Encode text to integer IDs.
        If add_blank is True, inserts blank (pad_id) between tokens (standard VITS format: _ t1 _ t2 _).
        """
        norm_text = normalize_indic_text(text, lang=self.lang)
        tokens = self.text_to_tokens(norm_text)
        
        token_ids = [self.char_to_id[t] for t in tokens if t in self.char_to_id]
        
        if add_blank:
            # Interleave with blank token (0)
            interleaved = [self.pad_id]
            for tid in token_ids:
                interleaved.append(tid)
                interleaved.append(self.pad_id)
            token_ids = interleaved
            
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
            
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """Decode integer IDs back to string."""
        chars = []
        for tid in ids:
            if tid in self.id_to_char:
                char = self.id_to_char[tid]
                if char not in ("_", "^", "$"):
                    chars.append(char)
        return "".join(chars)
