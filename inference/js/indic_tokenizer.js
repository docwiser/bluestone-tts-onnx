/**
 * Pure JavaScript Indic Tokenizer.
 * Matches Python tokenizer output with 100% numerical parity.
 * Zero external native C++ dependencies.
 */

const fs = require('fs');

const HINDI_DIGITS = {
  '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
  '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
  '०': 'शून्य', '१': 'एक', '२': 'दो', '३': 'तीन', '४': 'चार',
  '५': 'पांच', '६': 'छह', '७': 'सात', '८': 'आठ', '९': 'नौ'
};

const PUNCTUATION_MAP = {
  '।': '.', '॥': '.', '|': '.', ';': ',', ':': ',',
  '-': ' ', '—': ' ', '–': ' ', '"': '', "'": '',
  '(': '', ')': '', '[': '', ']': '', '{': '', '}': ''
};

class IndicTokenizer {
  constructor(vocabPath) {
    if (!fs.existsSync(vocabPath)) {
      throw new Error(`Vocab file not found: ${vocabPath}`);
    }
    const data = JSON.parse(fs.readFileSync(vocabPath, 'utf8'));
    this.charToId = data.char_to_id;
    this.padId = this.charToId['_'] ?? 0;
    this.bosId = this.charToId['^'] ?? 1;
    this.eosId = this.charToId['$'] ?? 2;
  }

  normalize(text) {
    // 1. Unicode NFC
    let normalized = text.normalize('NFC');
    
    // 2. Expand digits
    let expanded = '';
    for (const char of normalized) {
      if (HINDI_DIGITS[char]) {
        expanded += ` ${HINDI_DIGITS[char]} `;
      } else {
        expanded += char;
      }
    }

    // 3. Punctuation cleanup
    for (const [pIn, pOut] of Object.entries(PUNCTUATION_MAP)) {
      expanded = expanded.split(pIn).join(pOut);
    }

    // 4. Whitespace cleanup
    return expanded.replace(/\s+/g, ' ').trim();
  }

  textToTokens(text) {
    const tokens = [];
    let i = 0;
    const n = text.length;

    while (i < n) {
      const twoChar = text.substring(i, i + 2);
      if (this.charToId[twoChar] !== undefined) {
        tokens.push(twoChar);
        i += 2;
      } else if (this.charToId[text[i]] !== undefined) {
        tokens.push(text[i]);
        i += 1;
      } else {
        if (/\s/.test(text[i])) {
          tokens.push(' ');
        }
        i += 1;
      }
    }
    return tokens;
  }

  encode(text, addBosEos = true, addBlank = true) {
    const normalized = this.normalize(text);
    const tokens = this.textToTokens(normalized);
    let tokenIds = tokens
      .filter(t => this.charToId[t] !== undefined)
      .map(t => this.charToId[t]);

    if (addBlank) {
      const interleaved = [this.padId];
      for (const id of tokenIds) {
        interleaved.push(id);
        interleaved.push(this.padId);
      }
      tokenIds = interleaved;
    }

    if (addBosEos) {
      tokenIds = [this.bosId, ...tokenIds, this.eosId];
    }

    return tokenIds;
  }
}

module.exports = { IndicTokenizer };
