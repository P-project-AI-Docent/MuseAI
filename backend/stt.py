# backend/stt.py

import json
import os
import wave
import subprocess
from vosk import Model, KaldiRecognizer

# ----------------------------------------------------
# 모델 로딩
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stt", "model", "vosk-model-small-ko-0.22")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"STT 모델이 존재하지 않습니다: {MODEL_PATH}")

model = Model(MODEL_PATH)


# ----------------------------------------------------
# 1) ffmpeg로 wav(16kHz mono)으로 변환
# ----------------------------------------------------
def convert_to_wav(input_path: str, output_path: str):
    """
    모든 형식(m4a, mp3, webm 등)을 wav 16kHz mono로 변환
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",        # mono
        "-ar", "16000",    # 16kHz
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ----------------------------------------------------
# 2) STT 처리
# ----------------------------------------------------
def speech_to_text(audio_path: str) -> str:
    """
    다양한 형식의 오디오 → wav 변환 → 텍스트
    """

    # 임시 변환된 wav 파일
    wav_path = "temp_stt.wav"

    # 음성이 이미 wav가 아닐 수도 있으므로 ffmpeg 변환
    convert_to_wav(audio_path, wav_path)

    wf = wave.open(wav_path, "rb")

    # wav 조건 검사
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        return "지원되지 않는 오디오 형식입니다."

    rec = KaldiRecognizer(model, wf.getframerate())

    final_text = ""

    while True:
        data = wf.readframes(4096)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result())["text"]
            final_text += part + " "

    # 마지막 결과
    final = json.loads(rec.FinalResult())["text"]
    final_text += final

    return final_text.strip()
