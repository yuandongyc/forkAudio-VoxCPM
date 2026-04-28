#!/usr/bin/env python3
"""Test VoxCPM-0.5B TTS synthesis with local model."""

import os

os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import sys
import soundfile as sf
from voxcpm import VoxCPM


def main():
    model_path = (
        r"D:\MyProjects\c_projects\forkAudio-VoxCPM\models\OpenBMB\VoxCPM-0___5B"
    )

    print(f"[Test] Loading VoxCPM-0.5B model from: {model_path}", file=sys.stderr)

    model = VoxCPM(
        voxcpm_model_path=model_path,
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=True,
    )

    print(f"[Test] Using device: {model.tts_model.device}", file=sys.stderr)

    text = "Hello, this is a test of VoxCPM text to speech synthesis."
    print(f"[Test] Synthesizing: {text}", file=sys.stderr)

    audio = model.generate(
        text=text,
        max_len=600,
    )

    output_path = "test_output.wav"
    sf.write(output_path, audio, model.tts_model.sample_rate)

    duration = len(audio) / model.tts_model.sample_rate
    print(f"[Test] Saved to: {output_path}, duration: {duration:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
