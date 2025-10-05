import webrtcvad
from collections import deque


class VADGate:
    """Simple voice-activity gate using WebRTC VAD to decide when to trigger decode."""
    def __init__(self, sample_rate: int, frame_ms: int = 20, aggressiveness: int = 2,
        start_padding_ms: int = 200, stop_padding_ms: int = 500):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2 # 16-bit PCM
        self.start_padding_frames = start_padding_ms // frame_ms
        self.stop_padding_frames = stop_padding_ms // frame_ms
        self.ring = deque(maxlen=self.stop_padding_frames)
        self.triggered = False


    def is_speech(self, pcm16: bytes) -> bool:
        if len(pcm16) != self.frame_bytes:
            return False
        return self.vad.is_speech(pcm16, self.sample_rate)


    def update(self, pcm16: bytes) -> tuple[bool, bool]:
        """Feed one frame; returns (just_started, just_stopped)."""
        is_speech = self.is_speech(pcm16)
        just_started = False
        just_stopped = False


        if not self.triggered:
            self.ring.append(is_speech)
            if sum(self.ring) >= self.start_padding_frames:
                self.triggered = True
                self.ring.clear()
                just_started = True
        else:
            self.ring.append(is_speech)
            if sum(self.ring) == 0:
                self.triggered = False
                self.ring.clear()
                just_stopped = True
        return just_started, just_stopped