# Android ONNX Runtime Indic TTS Integration

This module enables high-fidelity, real-time, offline Indic Text-to-Speech directly on Android mobile devices with **zero external C++ dependencies**.

## 1. Add Dependencies (`app/build.gradle.kts`)

```kotlin
dependencies {
    // Microsoft ONNX Runtime for Android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.0")
    
    // Kotlin Coroutines for non-blocking background speech synthesis
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

## 2. Place Assets

Place the exported ONNX model and vocabulary into your app's `assets/` directory:
- `app/src/main/assets/bluestone_tts_hi_quant.onnx` (~22 MB)
- `app/src/main/assets/vocab.json` (~2 KB)

## 3. Usage in Android (Kotlin)

```kotlin
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.bluestone.tts.IndicTTS

class MainActivity : AppCompatActivity() {

    private lateinit var tts: IndicTTS

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize TTS engine from assets
        val modelBytes = assets.open("bluestone_tts_hi_quant.onnx").readBytes()
        val vocabJson = assets.open("vocab.json").bufferedReader().use { it.readText() }

        tts = IndicTTS(
            context = this,
            modelBytes = modelBytes,
            vocabJsonString = vocabJson,
            numThreads = 4 // CPU threads
        )

        findViewById<Button>(R.id.btnSpeak).setOnClickListener {
            speakText("नमस्ते, यह ऑन-डिवाइस हिंदी टेक्स्ट टू स्पीच इंजन है।")
        }
    }

    private fun speakText(text: String) {
        lifecycleScope.launch(Dispatchers.Default) {
            // Synthesize and play on device speaker
            tts.speak(text, speed = 1.0f)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.close()
    }
}
```

## Performance on Android
- **Model Footprint in RAM**: ~30 MB
- **Real-Time Factor (RTF)**: ~0.08 - 0.12 (Generates 10 seconds of speech in ~1 second on Snapdragon / MediaTek CPUs)
- **Architecture Support**: `arm64-v8a`, `armeabi-v7a`, `x86_64`
