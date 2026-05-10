import os
import numpy as np
import librosa
from preprocessing.config import SAMPLE_RATE, N_MELS, WIN_LENGTH, HOP_LENGTH, N_FFT


def extract_audio(video_path: str, out_wav_path: str) -> bool:
    """Extract mono 16 kHz WAV from video via ffmpeg. Returns True on success."""
    ret = os.system(
        f'ffmpeg -y -i "{video_path}" -ar {SAMPLE_RATE} -ac 1 -vn "{out_wav_path}" -loglevel quiet'
    )
    return ret == 0 and os.path.exists(out_wav_path)


def compute_log_mel(wav_path: str) -> np.ndarray:
    """Return 80-band log-Mel spectrogram (F x T) from a WAV file."""
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max)


def save_audio(log_mel: np.ndarray, out_dir: str, file_id: str) -> None:
    np.save(os.path.join(out_dir, "audio", f"{file_id}_melspec.npy"), log_mel)
