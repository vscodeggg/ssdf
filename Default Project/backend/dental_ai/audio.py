import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

# A peak below this (out of 32767) is treated as silence: the mic heard
# nothing usable, so there is no point sending the audio to Whisper, which
# would otherwise hallucinate words out of room noise. This mic's ambient
# floor measures ~800, so the threshold sits just above it.
SILENCE_PEAK_THRESHOLD = 1000

# Leading/trailing samples quieter than this are trimmed before transcription.
_TRIM_FLOOR = 150
# Keep a little padding around the trimmed speech so word onsets survive.
_PAD_SAMPLES = 2400  # 0.15 s at 16 kHz
# Cap the gain boost so pure noise is not blown up into fake speech.
_MAX_GAIN = 20.0
# Target peak (~70% of int16 full scale) after normalization.
_TARGET_PEAK = 0.7 * 32767


def wav_peak_amplitude(path: Path) -> int:
    """Returns the peak absolute sample value of a 16-bit WAV file."""
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    if not frames:
        return 0
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    return max((abs(s) for s in samples), default=0)


class MicrophoneRecorder:
    """Record short mono WAV chunks using sounddevice.

    Recordings are normalized (quiet mics boosted) and leading/trailing
    silence trimmed so Whisper receives clean speech instead of near-silence
    it can hallucinate words from.
    """

    def __init__(self, sample_rate: int = 16_000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        # Peak of the most recent recording BEFORE normalization - the honest
        # level the mic actually captured (ambient noise or true silence).
        self.last_raw_peak = 0

    def record_chunk(self, seconds: int = 6, output_path: Optional[Path] = None) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install sounddevice and numpy to record from a microphone.") from exc

        if output_path is not None:
            output = Path(output_path)
        else:
            # mkstemp() returns an OPEN file descriptor; on Windows that handle
            # keeps the file locked (WinError 32) and wave.open() below cannot
            # write to it. Create the name and close the descriptor at once.
            fd, name = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output = Path(name)
        frames = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
        )
        sd.wait()

        samples = frames.reshape(-1, self.channels).astype("int32")  # int32 headroom for gain
        raw_peak = int(np.abs(samples).max()) if samples.size else 0
        self.last_raw_peak = raw_peak
        peak = raw_peak

        if peak >= SILENCE_PEAK_THRESHOLD:
            # Trim quiet lead-in/tail so Whisper only hears actual speech.
            loud = np.abs(samples).max(axis=1) > _TRIM_FLOOR
            if loud.any():
                first = max(int(np.argmax(loud)) - _PAD_SAMPLES, 0)
                last = min(int(np.argwhere(loud)[-1][0]) + _PAD_SAMPLES, samples.shape[0])
                if last - first >= int(0.3 * self.sample_rate):
                    samples = samples[first:last]

            # Normalize: boost a quiet mic to a healthy level, capped so noise
            # is never amplified into fake speech.
            peak = int(np.abs(samples).max()) if samples.size else 0
            if peak > 0:
                gain = min(_TARGET_PEAK / peak, _MAX_GAIN)
                samples = np.clip(samples * gain, -32768, 32767).astype("int16")
            else:
                samples = samples.astype("int16")
        else:
            samples = samples.astype("int16")

        with wave.open(str(output), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(samples.tobytes())
        return output
