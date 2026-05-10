"""Smoke test — verifies all dependencies and internal imports resolve."""
import sys


def test_dependencies():
    import librosa
    import cv2
    import torch
    import transformers
    import whisper
    from facenet_pytorch import MTCNN
    import pandas as pd
    import numpy as np
    print("all third-party dependencies ok")


def test_internal_imports():
    from preprocessing import config
    from preprocessing.utils import audio, text, visual, progress
    from preprocessing.datasets import cremad, meld, savee
    print("all internal imports ok")


def test_config_values():
    from preprocessing.config import (
        SAMPLE_RATE, N_MELS, WIN_LENGTH, HOP_LENGTH, N_FFT,
        MAX_TOKEN_LEN, IMG_SIZE, TOP_K_FRAMES,
    )
    assert SAMPLE_RATE == 16_000
    assert N_MELS == 80
    assert WIN_LENGTH == 400
    assert HOP_LENGTH == 160
    assert N_FFT == 1024
    assert MAX_TOKEN_LEN == 128
    assert IMG_SIZE == 224
    assert TOP_K_FRAMES == 8
    print("config values ok")


if __name__ == "__main__":
    test_dependencies()
    test_internal_imports()
    test_config_values()
    print("\nall checks passed")
