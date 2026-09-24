package com.bluestone.tts

import org.json.JSONObject
import java.text.Normalizer

/**
 * Pure Kotlin Indic Tokenizer for Android.
 * Zero native C++ or JNI dependencies. Runs in microseconds.
 */
class IndicTokenizer(vocabJsonString: String) {

    private val charToId = mutableMapOf<String, Long>()
    private val padId: Long
    private val bosId: Long
    private val eosId: Long

    private val hindiDigits = mapOf(
        '0' to "शून्य", '1' to "एक", '2' to "दो", '3' to "तीन", '4' to "चार",
        '5' to "पांच", '6' to "छह", '7' to "सात", '8' to "आठ", '9' to "नौ",
        '०' to "शून्य", '१' to "एक", '२' to "दो", '३' to "तीन", '४' to "चार",
        '५' to "पांच", '६' to "छह", '७' to "सात", '८' to "आठ", '९' to "नौ"
    )

    private val punctuationMap = mapOf(
        '।' to '.', '॥' to '.', '|' to '.', ';' to ',', ':' to ',',
        '-' to ' ', '—' to ' ', '–' to ' ', '"' to ' ', '\'' to ' '
    )

    init {
        val root = JSONObject(vocabJsonString)
        val charMapObj = root.getJSONObject("char_to_id")
        val keys = charMapObj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            charToId[key] = charMapObj.getLong(key)
        }
        padId = charToId["_"] ?: 0L
        bosId = charToId["^"] ?: 1L
        eosId = charToId["$"] ?: 2L
    }

    fun normalize(text: String): String {
        // 1. Unicode NFC
        val nfc = Normalizer.normalize(text, Normalizer.Form.NFC)

        // 2. Expand digits
        val sb = StringBuilder()
        for (ch in nfc) {
            val digitWord = hindiDigits[ch]
            if (digitWord != null) {
                sb.append(" ").append(digitWord).append(" ")
            } else {
                val punct = punctuationMap[ch]
                if (punct != null) {
                    sb.append(punct)
                } else {
                    sb.append(ch)
                }
            }
        }

        // 3. Remove multiple spaces
        return sb.toString().replace(Regex("\\s+"), " ").trim()
    }

    fun encode(text: String, addBosEos: Boolean = true, addBlank: Boolean = true): LongArray {
        val normalized = normalize(text)
        val tokens = mutableListOf<String>()
        var i = 0
        val n = normalized.length

        while (i < n) {
            val twoChar = if (i + 2 <= n) normalized.substring(i, i + 2) else null
            if (twoChar != null && charToId.containsKey(twoChar)) {
                tokens.add(twoChar)
                i += 2
            } else {
                val oneChar = normalized.substring(i, i + 1)
                if (charToId.containsKey(oneChar)) {
                    tokens.add(oneChar)
                } else if (normalized[i].isWhitespace()) {
                    tokens.add(" ")
                }
                i += 1
            }
        }

        val tokenIds = tokens.mapNotNull { charToId[it] }.toMutableList()
        val result = mutableListOf<Long>()

        if (addBlank) {
            result.add(padId)
            for (id in tokenIds) {
                result.add(id)
                result.add(padId)
            }
        } else {
            result.addAll(tokenIds)
        }

        if (addBosEos) {
            result.add(0, bosId)
            result.add(eosId)
        }

        return result.toLongArray()
    }
}
