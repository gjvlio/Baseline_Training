import os
import re
import numpy as np
import whisper
from transformers import BertTokenizer
from preprocessing.config import MAX_TOKEN_LEN, ASR_MODEL, BERT_MODEL


def load_asr() -> whisper.Whisper:
    return whisper.load_model(ASR_MODEL)


def load_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained(BERT_MODEL)


def transcribe(wav_path: str, asr_model: whisper.Whisper) -> str:
    result = asr_model.transcribe(wav_path)
    text = result["text"].lower().strip()
    return re.sub(r"[^\w\s]", "", text)


def tokenize(transcript: str, tokenizer: BertTokenizer) -> dict:
    return tokenizer(
        transcript,
        max_length=MAX_TOKEN_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )


def save_text(tokens: dict, out_dir: str, file_id: str) -> None:
    np.save(os.path.join(out_dir, "text", f"{file_id}_input_ids.npy"),
            tokens["input_ids"])
    np.save(os.path.join(out_dir, "text", f"{file_id}_attention_mask.npy"),
            tokens["attention_mask"])
