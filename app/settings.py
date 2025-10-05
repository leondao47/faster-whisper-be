from dotenv import load_dotenv
load_dotenv()  # loads .env before reading Settings()

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "distil-small.en" # good latency; try "small" or multilingual models
    # model_name: str = "tiny"
    device: str = "cpu" # "cuda" | "cpu" | "auto"
    compute_type: str = "int8" # good speed/quality trade-off; use "int8" on CPU
    beam_size: int = 1
    vad_frame_ms: int = 20 # 10 | 20 | 30 ms
    vad_aggressiveness: int = 2 # 0-3 (3 is most aggressive)
    sample_rate: int = 16000 # 16kHz PCM16 expected from client
    whisper_lang: str | None = None # e.g. "en", "vi", or None for auto
    max_segment_s: float = 2.0 # decode cadence for streaming
    preroll_s: float = 0.8 # context to keep for stability


settings = Settings()