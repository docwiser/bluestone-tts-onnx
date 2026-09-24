/**
 * Node.js Inference Runner for Bluestone Indic TTS.
 * Generates spoken audio and saves to WAV format directly.
 */

const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const { IndicTokenizer } = require('./indic_tokenizer');

function writeWavFile(filepath, samples, sampleRate = 22050) {
  // Convert float32 [-1.0, 1.0] to 16-bit PCM
  const buffer = Buffer.alloc(44 + samples.length * 2);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + samples.length * 2, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);          // Subchunk1Size (16 for PCM)
  buffer.writeUInt16LE(1, 20);           // AudioFormat (1 = PCM)
  buffer.writeUInt16LE(1, 22);           // NumChannels (1 = Mono)
  buffer.writeUInt32LE(sampleRate, 24);  // SampleRate
  buffer.writeUInt32LE(sampleRate * 2, 28); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
  buffer.writeUInt16LE(2, 32);           // BlockAlign (NumChannels * BitsPerSample/8)
  buffer.writeUInt16LE(16, 34);          // BitsPerSample (16 bits)

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(samples.length * 2, 40);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1.0, Math.min(1.0, samples[i]));
    let val = s < 0 ? s * 32768 : s * 32767;
    buffer.writeInt16LE(Math.floor(val), offset);
    offset += 2;
  }

  fs.writeFileSync(filepath, buffer);
}

async function runTTS(modelPath, vocabPath, text, outputPath = 'output_node.wav', speed = 1.0) {
  console.log(`[*] Loading ONNX model: ${modelPath}`);
  const tokenizer = new IndicTokenizer(vocabPath);

  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all'
  });

  const tokens = tokenizer.encode(text, true, true);
  console.log(`[*] Encoded ${tokens.length} phonetic tokens`);

  const lengthScale = 1.0 / Math.max(0.2, speed);
  const inputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(tokens.map(t => BigInt(t))), [1, tokens.length]);
  const inputLengthsTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(tokens.length)]), [1]);
  const scalesTensor = new ort.Tensor('float32', Float32Array.from([0.667, lengthScale, 0.8]), [3]);

  const feeds = {
    input_ids: inputIdsTensor,
    input_lengths: inputLengthsTensor,
    scales: scalesTensor
  };

  const startTime = Date.now();
  const results = await session.run(feeds);
  const latencyMs = Date.now() - startTime;

  const audioTensor = results.audio;
  const audioData = audioTensor.data; // Float32Array
  const sampleRate = 22050;
  const durationSec = audioData.length / sampleRate;
  const rtf = (latencyMs / 1000) / durationSec;

  writeWavFile(outputPath, audioData, sampleRate);

  console.log(`[✓] Audio generated successfully!`);
  console.log(`    Output:      ${outputPath}`);
  console.log(`    Duration:    ${durationSec.toFixed(2)}s`);
  console.log(`    Latency:     ${latencyMs}ms`);
  console.log(`    RTF on CPU:  ${rtf.toFixed(3)} (${(1 / rtf).toFixed(1)}x real-time speed)`);
}

// Example execution
if (require.main === module) {
  const modelPath = process.argv[2] || path.join(__dirname, '../../exported/bluestone_tts_hi.onnx');
  const vocabPath = process.argv[3] || path.join(__dirname, '../../frontend/vocab.json');
  const text = process.argv[4] || 'नमस्ते! यह नोड जेएस में चल रहा ब्लूस्टोन टेक्स्ट टू स्पीच इंजन है।';
  const outPath = process.argv[5] || 'output_node.wav';

  runTTS(modelPath, vocabPath, text, outPath).catch(err => {
    console.error('[!] Error running TTS in Node.js:', err);
  });
}

module.exports = { runTTS };
