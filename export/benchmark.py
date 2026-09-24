"""
Benchmarking script for CPU inference speed, RTF (Real-Time Factor), and memory footprint.
"""

import time
import argparse
import numpy as np
import onnxruntime as ort
from frontend.indic_tokenizer import IndicTokenizer

TEST_SENTENCES = [
    "नमस्ते दुनिया।",
    "कृत्रिम बुद्धिमत्ता भविष्य की सबसे बड़ी तकनीक है।",
    "भारत एक बहुत बड़ा देश है जहां कई भाषाएं बोली जाती हैं और सभी लोग मिलजुल कर रहते हैं।"
]

def benchmark_onnx_model(onnx_path: str, vocab_path: str = None, num_runs: int = 5):
    print(f"[*] Benchmarking ONNX model: {onnx_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # Typical mobile/desktop core allocation

    session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    tokenizer = IndicTokenizer(vocab_path=vocab_path, lang="hi")

    # Read metadata
    meta = session.get_modelmeta().custom_metadata_map
    sample_rate = int(meta.get("sample_rate", 22050))
    print(f"[*] Detected Sample Rate: {sample_rate} Hz")

    print(f"\n{'='*70}")
    print(f"{'Sentence':<45} | {'Audio(s)':<8} | {'Lat(ms)':<8} | {'RTF':<6}")
    print(f"{'='*70}")

    rtfs = []

    for text in TEST_SENTENCES:
        tokens = tokenizer.encode(text, add_bos_eos=True, add_blank=True)
        input_ids = np.array([tokens], dtype=np.int64)
        input_lengths = np.array([len(tokens)], dtype=np.int64)
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

        inputs = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "scales": scales
        }

        # Warmup
        session.run(None, inputs)

        latencies = []
        audio_durations = []

        for _ in range(num_runs):
            t0 = time.perf_counter()
            outputs = session.run(None, inputs)
            lat = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(lat)

            audio = outputs[0]
            audio_samples = audio.shape[-1]
            dur = audio_samples / sample_rate
            audio_durations.append(dur)

        avg_lat = np.mean(latencies)
        avg_dur = np.mean(audio_durations)
        rtf = (avg_lat / 1000.0) / avg_dur if avg_dur > 0 else 0
        rtfs.append(rtf)

        disp_text = text if len(text) < 40 else text[:37] + "..."
        print(f"{disp_text:<45} | {avg_dur:<8.2f} | {avg_lat:<8.1f} | {rtf:<6.3f}")

    print(f"{'='*70}")
    mean_rtf = np.mean(rtfs)
    speedup = 1.0 / mean_rtf if mean_rtf > 0 else 0
    print(f"[✓] Overall Mean RTF: {mean_rtf:.4f} (Generates speech {speedup:.1f}x faster than real-time on CPU)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX TTS on CPU")
    parser.add_argument("--model", type=str, default="exported/bluestone_tts_hi.onnx", help="Path to ONNX model")
    parser.add_argument("--vocab", type=str, default="frontend/vocab.json", help="Path to vocab.json")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark iterations")
    args = parser.parse_args()

    benchmark_onnx_model(args.model, args.vocab, args.runs)
