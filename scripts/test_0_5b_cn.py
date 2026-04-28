#!/usr/bin/env python3
"""Test VoxCPM-0.5B 中文男声 TTS synthesis."""

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

    # 使用 Voice Design 功能 - 中文男声描述
    # 可以尝试: (male), (男性), (adult male voice), (deep voice) 等
    text = "(男性)你好，这是一个中文语音合成测试，我希望生成一个男性的声音。"
    print(f"[Test] Synthesizing: {text}", file=sys.stderr)

    audio = model.generate(
        text=text,
        max_len=600,
        normalize=True,
    )

    output_path = "test_output_male.wav"
    sf.write(output_path, audio, model.tts_model.sample_rate)

    duration = len(audio) / model.tts_model.sample_rate
    print(f"[Test] Saved to: {output_path}, duration: {duration:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
