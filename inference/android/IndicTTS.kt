package com.bluestone.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Ultra-compact Android ONNX Runtime TTS Engine.
 * CPU-only, real-time audio playback on Android devices.
 */
class IndicTTS(
    private val context: Context,
    modelBytes: ByteArray,
    vocabJsonString: String,
    private val numThreads: Int = 4
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: IndicTokenizer = IndicTokenizer(vocabJsonString)
    val sampleRate: Int = 22050

    init {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(numThreads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, sessionOptions)
    }

    /**
     * Synthesizes audio samples (float32 [-1.0, 1.0]) from Indic text.
     */
    fun synthesize(text: String, speed: Float = 1.0f): FloatArray {
        val tokens = tokenizer.encode(text)
        val lengthScale = 1.0f / speed.coerceAtLeast(0.2f)

        // 1. Prepare input_ids tensor: [1, seq_len]
        val shapeIds = longArrayOf(1, tokens.size.toLong())
        val idsBuffer = LongBuffer.wrap(tokens)
        val inputIdsTensor = OnnxTensor.createTensor(env, idsBuffer, shapeIds)

        // 2. Prepare input_lengths tensor: [1]
        val shapeLens = longArrayOf(1)
        val lensBuffer = LongBuffer.wrap(longArrayOf(tokens.size.toLong()))
        val inputLengthsTensor = OnnxTensor.createTensor(env, lensBuffer, shapeLens)

        // 3. Prepare scales tensor: [3] (noise_scale, length_scale, noise_scale_w)
        val shapeScales = longArrayOf(3)
        val scalesBuffer = FloatBuffer.wrap(floatArrayOf(0.667f, lengthScale, 0.8f))
        val scalesTensor = OnnxTensor.createTensor(env, scalesBuffer, shapeScales)

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "input_lengths" to inputLengthsTensor,
            "scales" to scalesTensor
        )

        // 4. Run inference
        val results = session.run(inputs)
        val audioTensor = results[0] as OnnxTensor
        val floatBuffer = audioTensor.floatBuffer
        val audioSamples = FloatArray(floatBuffer.remaining())
        floatBuffer.get(audioSamples)

        // Clean up tensors
        inputIdsTensor.close()
        inputLengthsTensor.close()
        scalesTensor.close()
        results.close()

        return audioSamples
    }

    /**
     * Synthesizes and immediately plays audio on the Android device speaker.
     */
    fun speak(text: String, speed: Float = 1.0f) {
        val samples = synthesize(text, speed)
        val pcmData = floatToPcm16(samples)

        val minBufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcmData.size.coerceAtLeast(minBufferSize))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack.play()
        audioTrack.write(pcmData, 0, pcmData.size)
        audioTrack.stop()
        audioTrack.release()
    }

    /**
     * Converts float32 audio [-1.0, 1.0] to 16-bit PCM bytes.
     */
    private fun floatToPcm16(samples: FloatArray): ByteArray {
        val byteBuffer = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) {
            val clamped = sample.coerceIn(-1.0f, 1.0f)
            val pcmVal = if (clamped < 0) (clamped * 32768.0f).toInt() else (clamped * 32767.0f).toInt()
            byteBuffer.putShort(pcmVal.toShort())
        }
        return byteBuffer.array()
    }

    override fun close() {
        session.close()
        env.close()
    }
}
