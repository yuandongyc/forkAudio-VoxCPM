#!/usr/bin/env python3
"""Test VoxCPM-0.5B TTS synthesis with local model."""

import os

os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import sys
import soundfile as sf
import torch
import torchaudio

original_torchaudio_load = torchaudio.load


def patched_torchaudio_load(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio).unsqueeze(0)
    return audio, sr


torchaudio.load = patched_torchaudio_load

original_resample = torchaudio.functional.resample


def patched_resample(waveform, orig_freq, new_freq):
    if orig_freq == new_freq:
        return waveform
    ratio = new_freq / orig_freq
    new_len = int(waveform.shape[-1] * ratio)
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)


torchaudio.functional.resample = patched_resample

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
        device="auto",
        optimize=True,
    )

    print(f"[Test] Using device: {model.tts_model.device}", file=sys.stderr)

    text = "Hello, this is a test of VoxCPM text to speech synthesis."
    print(f"[Test] Synthesizing: {text}", file=sys.stderr)

    audio = model.generate(
        text=text,
        max_len=600,
    )

    output_path = "my_output/test_0_5b_gpu_clone_output.wav"
    sf.write(output_path, audio, model.tts_model.sample_rate)

    duration = len(audio) / model.tts_model.sample_rate
    print(f"[Test] Saved to: {output_path}, duration: {duration:.2f}s", file=sys.stderr)

    prompt_wav_path = r"D:\MyProjects\c_projects\forkAudio-VoxCPM\my_voice\录音.wav"
    prompt_text = "这是我的声音录音。"

    print(f"[Test] Cloning voice from: {prompt_wav_path}", file=sys.stderr)

    clone_audio = model.generate(
        text=text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        max_len=600,
    )

    clone_output_path = "my_output/test_0_5b_gpu_clone_output.wav"
    sf.write(clone_output_path, clone_audio, model.tts_model.sample_rate)

    clone_duration = len(clone_audio) / model.tts_model.sample_rate
    print(
        f"[Test] Clone saved to: {clone_output_path}, duration: {clone_duration:.2f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
