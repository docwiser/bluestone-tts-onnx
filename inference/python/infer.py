"""
Python ONNX Runtime Inference Engine.
Synthesizes speech from text using exported Bluestone TTS ONNX models.
"""

import os
import argparse
import numpy as np
import onnxruntime as ort
import soundfile as sf
from frontend.indic_tokenizer import IndicTokenizer

class BluestoneTTS:
    def __init__(self, model_path: str, vocab_path: str = None, num_threads: int = 4):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads

        self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        
        # Read model metadata
        meta = self.session.get_modelmeta().custom_metadata_map
        self.sample_rate = int(meta.get("sample_rate", 22050))
        self.lang = meta.get("language", "hi")

        self.tokenizer = IndicTokenizer(vocab_path=vocab_path, lang=self.lang)

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8
    ) -> np.ndarray:
        """
        Synthesize speech from Indic text.
        Args:
            text: Input text in Indic script (e.g. Hindi Devanagari)
            speed: Speaking speed multiplier (1.0 = normal, 1.2 = faster, 0.8 = slower)
            noise_scale: Phoneme variation noise scale
            noise_scale_w: Duration variation noise scale
        Returns:
            audio_array: 1D numpy float32 array
        """
        # Convert speed to length_scale (speed 1.25 -> length_scale 0.8)
        length_scale = 1.0 / max(0.2, speed)

        tokens = self.tokenizer.encode(text, add_bos_eos=True, add_blank=True)
        input_ids = np.array([tokens], dtype=np.int64)
        input_lengths = np.array([len(tokens)], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_scale_w], dtype=np.float32)

        inputs = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "scales": scales
        }

        outputs = self.session.run(None, inputs)
        audio = outputs[0]  # [1, 1, num_samples]
        audio = audio.squeeze()
        return audio

    def save_wav(self, audio: np.ndarray, output_path: str):
        """Save synthesized audio array to WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Normalize and prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.99
        sf.write(output_path, audio, self.sample_rate)
        print(f"[✓] Audio saved to: {output_path} ({len(audio)/self.sample_rate:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bluestone TTS inference")
    parser.add_argument("--model", type=str, default="exported/bluestone_tts_hi.onnx", help="Path to ONNX model")
    parser.add_argument("--vocab", type=str, default="frontend/vocab.json", help="Path to vocab.json")
    parser.add_argument("--text", type=str, default="नमस्ते, यह ब्लूस्टोन टेक्स्ट टू स्पीच का परीक्षण है।", help="Text to speak")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to output WAV file")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (e.g. 1.0, 1.2)")
    args = parser.parse_args()

    engine = BluestoneTTS(model_path=args.model, vocab_path=args.vocab)
    audio = engine.synthesize(args.text, speed=args.speed)
    engine.save_wav(audio, args.output)
