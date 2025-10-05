import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np


from faster_whisper import WhisperModel
from .settings import settings


# Utilities


def pcm16_bytes_to_float32(pcm: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


@dataclass
class DecodeResult:
    text: str
    start: float
    end: float
    is_final: bool


class StreamingWhisper:
    def __init__(self):
        self.model = WhisperModel(
        settings.model_name,
        device=("cuda" if settings.device == "auto" else settings.device),
        compute_type=settings.compute_type,
        )
        self.sample_rate = settings.sample_rate
        self.buffer = np.zeros(0, dtype=np.float32)
        self.last_decode_t = 0.0
        self.tail_keep = int(settings.preroll_s * self.sample_rate)
        self.offset_s = 0.0 # running timeline offset for emitted timestamps


    def append_audio(self, pcm16: bytes):
        f32 = pcm16_bytes_to_float32(pcm16)
        self.buffer = np.concatenate([self.buffer, f32])


    async def maybe_decode(self) -> Optional[list[DecodeResult]]:
        now = time.monotonic()
        if now - self.last_decode_t < settings.max_segment_s:
            return None
        if len(self.buffer) < int(0.3 * self.sample_rate):
            return None


        audio = self.buffer
        if len(audio) > self.tail_keep:
            # keep a small tail for context next round
            head = audio[:-self.tail_keep]
            tail = audio[-self.tail_keep:]
        else:
            head = audio
            tail = np.zeros(0, dtype=np.float32)


        segments, _ = self.model.transcribe(
            head,
            language=settings.lang,
            beam_size=settings.beam_size,
            vad_filter=False, # we already gate with VAD
            word_timestamps=False,
            condition_on_previous_text=True,
        )


        out: list[DecodeResult] = []
        for seg in segments:
            out.append(DecodeResult(
                text=seg.text.strip(),
                start=self.offset_s + float(seg.start),
                end=self.offset_s + float(seg.end),
                is_final=True,
            ))


        # advance timeline and keep tail for stability
        self.offset_s += max(0.0, (len(head) / self.sample_rate))
        self.buffer = np.concatenate([tail])
        self.last_decode_t = now
        return out if out else None