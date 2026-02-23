"""
BlackRoad Audio Engine
Real-time audio synthesis and processing engine.
"""

import math
import struct
import io
import wave
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class WaveType(str, Enum):
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    NOISE = "noise"


class ScaleType(str, Enum):
    MAJOR = "major"
    MINOR = "minor"
    PENTATONIC = "pentatonic"


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALE_INTERVALS = {
    ScaleType.MAJOR:      [0, 2, 4, 5, 7, 9, 11, 12],
    ScaleType.MINOR:      [0, 2, 3, 5, 7, 8, 10, 12],
    ScaleType.PENTATONIC: [0, 2, 4, 7, 9, 12],
}

CHORD_INTERVALS = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dim":   [0, 3, 6],
    "aug":   [0, 4, 8],
}


# ---------------------------------------------------------------------------
# AudioBuffer
# ---------------------------------------------------------------------------

@dataclass
class AudioBuffer:
    """Container for PCM audio samples."""
    samples: List[float]
    sample_rate: int = 44100

    def __len__(self) -> int:
        return len(self.samples)

    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(list(self.samples), self.sample_rate)

    def __add__(self, other: "AudioBuffer") -> "AudioBuffer":
        assert self.sample_rate == other.sample_rate, "Sample rates must match"
        length = max(len(self.samples), len(other.samples))
        s1 = self.samples + [0.0] * (length - len(self.samples))
        s2 = other.samples + [0.0] * (length - len(other.samples))
        return AudioBuffer([a + b for a, b in zip(s1, s2)], self.sample_rate)


def mix(buffers: List[AudioBuffer], gains: Optional[List[float]] = None) -> AudioBuffer:
    """Mix multiple AudioBuffers together with optional per-buffer gains."""
    if not buffers:
        return AudioBuffer([], 44100)
    if gains is None:
        gains = [1.0] * len(buffers)
    assert len(gains) == len(buffers), "gains length must match buffers length"

    sample_rate = buffers[0].sample_rate
    length = max(len(b.samples) for b in buffers)
    result = [0.0] * length

    for buf, gain in zip(buffers, gains):
        for i, s in enumerate(buf.samples):
            result[i] += s * gain

    return AudioBuffer(result, sample_rate)


def clip(buffer: AudioBuffer, threshold: float = 1.0) -> AudioBuffer:
    """Hard-clip samples to [-threshold, threshold]."""
    clipped = [max(-threshold, min(threshold, s)) for s in buffer.samples]
    return AudioBuffer(clipped, buffer.sample_rate)


def normalize(buffer: AudioBuffer, target_peak: float = 0.9) -> AudioBuffer:
    """Normalize buffer so peak amplitude equals target_peak."""
    if not buffer.samples:
        return buffer.copy()
    peak = max(abs(s) for s in buffer.samples)
    if peak == 0:
        return buffer.copy()
    factor = target_peak / peak
    return AudioBuffer([s * factor for s in buffer.samples], buffer.sample_rate)


def fade_in(buffer: AudioBuffer, duration_s: float) -> AudioBuffer:
    """Apply linear fade-in over duration_s seconds."""
    samples = list(buffer.samples)
    fade_samples = int(duration_s * buffer.sample_rate)
    fade_samples = min(fade_samples, len(samples))
    for i in range(fade_samples):
        samples[i] *= i / fade_samples
    return AudioBuffer(samples, buffer.sample_rate)


def fade_out(buffer: AudioBuffer, duration_s: float) -> AudioBuffer:
    """Apply linear fade-out over duration_s seconds."""
    samples = list(buffer.samples)
    fade_samples = int(duration_s * buffer.sample_rate)
    fade_samples = min(fade_samples, len(samples))
    n = len(samples)
    for i in range(fade_samples):
        idx = n - fade_samples + i
        samples[idx] *= (fade_samples - i) / fade_samples
    return AudioBuffer(samples, buffer.sample_rate)


def apply_reverb(buffer: AudioBuffer, decay: float = 0.5, delay_ms: float = 50.0) -> AudioBuffer:
    """Simple comb-filter reverb."""
    delay_samples = int(delay_ms * buffer.sample_rate / 1000)
    samples = list(buffer.samples)
    result = samples[:]
    for i in range(delay_samples, len(result)):
        result[i] += decay * result[i - delay_samples]
    # Normalize to avoid clipping
    peak = max(abs(s) for s in result) if result else 1.0
    if peak > 1.0:
        result = [s / peak for s in result]
    return AudioBuffer(result, buffer.sample_rate)


def apply_lowpass(buffer: AudioBuffer, cutoff_hz: float) -> AudioBuffer:
    """Simple single-pole IIR low-pass filter."""
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    dt = 1.0 / buffer.sample_rate
    alpha = dt / (rc + dt)
    samples = list(buffer.samples)
    result = [0.0] * len(samples)
    prev = 0.0
    for i, s in enumerate(samples):
        prev = prev + alpha * (s - prev)
        result[i] = prev
    return AudioBuffer(result, buffer.sample_rate)


def apply_highpass(buffer: AudioBuffer, cutoff_hz: float) -> AudioBuffer:
    """Simple single-pole IIR high-pass filter."""
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    dt = 1.0 / buffer.sample_rate
    alpha = rc / (rc + dt)
    samples = list(buffer.samples)
    result = [0.0] * len(samples)
    prev_in = 0.0
    prev_out = 0.0
    for i, s in enumerate(samples):
        result[i] = alpha * (prev_out + s - prev_in)
        prev_in = s
        prev_out = result[i]
    return AudioBuffer(result, buffer.sample_rate)


def export_wav_bytes(buffer: AudioBuffer) -> bytes:
    """Export AudioBuffer as raw WAV bytes (PCM 16-bit)."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(buffer.sample_rate)
        for s in buffer.samples:
            clamped = max(-1.0, min(1.0, s))
            pcm = int(clamped * 32767)
            wf.writeframes(struct.pack("<h", pcm))
    return out.getvalue()


def import_wav_bytes(data: bytes) -> AudioBuffer:
    """Import WAV bytes into an AudioBuffer."""
    buf = io.BytesIO(data)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = [struct.unpack_from("<h", raw, i * 2)[0] / 32768.0 for i in range(n)]
    return AudioBuffer(samples, sr)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

@dataclass
class Envelope:
    """ADSR Envelope."""
    attack_ms: float = 10.0
    decay_ms: float = 50.0
    sustain: float = 0.7      # 0.0 – 1.0 amplitude level
    release_ms: float = 100.0

    def build(self, total_samples: int, sample_rate: int) -> List[float]:
        """Return per-sample gain curve of length total_samples."""
        a = int(self.attack_ms * sample_rate / 1000)
        d = int(self.decay_ms * sample_rate / 1000)
        r = int(self.release_ms * sample_rate / 1000)
        s_len = max(0, total_samples - a - d - r)

        env = []
        # Attack
        for i in range(min(a, total_samples)):
            env.append(i / a if a > 0 else 1.0)
        # Decay
        for i in range(min(d, total_samples - len(env))):
            env.append(1.0 - (1.0 - self.sustain) * i / d if d > 0 else self.sustain)
        # Sustain
        env.extend([self.sustain] * min(s_len, total_samples - len(env)))
        # Release
        current = env[-1] if env else self.sustain
        for i in range(min(r, total_samples - len(env))):
            env.append(current * (1.0 - i / r) if r > 0 else 0.0)
        # Pad remainder
        while len(env) < total_samples:
            env.append(0.0)
        return env[:total_samples]


def apply_envelope(buffer: AudioBuffer, envelope: Envelope) -> AudioBuffer:
    """Multiply buffer samples by the ADSR envelope curve."""
    env = envelope.build(len(buffer.samples), buffer.sample_rate)
    return AudioBuffer([s * e for s, e in zip(buffer.samples, env)], buffer.sample_rate)


# ---------------------------------------------------------------------------
# Oscillator
# ---------------------------------------------------------------------------

@dataclass
class Oscillator:
    """Single oscillator that generates waveforms."""
    wave_type: WaveType = WaveType.SINE
    frequency: float = 440.0     # Hz
    amplitude: float = 0.5       # 0.0 – 1.0
    phase: float = 0.0           # radians

    def generate(self, duration_s: float, sample_rate: int = 44100) -> List[float]:
        """Generate samples for duration_s seconds."""
        n_samples = int(duration_s * sample_rate)
        samples = []
        wt = self.wave_type
        freq = self.frequency
        amp = self.amplitude
        phase = self.phase
        sr = sample_rate

        for i in range(n_samples):
            t = i / sr
            angle = 2 * math.pi * freq * t + phase
            if wt == WaveType.SINE:
                s = amp * math.sin(angle)
            elif wt == WaveType.SQUARE:
                s = amp * (1.0 if math.sin(angle) >= 0 else -1.0)
            elif wt == WaveType.SAWTOOTH:
                s = amp * (2 * (freq * t % 1.0) - 1.0)
            elif wt == WaveType.TRIANGLE:
                saw = 2 * (freq * t % 1.0) - 1.0
                s = amp * (2 * abs(saw) - 1.0)
            else:  # NOISE
                s = amp * (random.random() * 2 - 1)
            samples.append(s)
        return samples

    def generate_buffer(self, duration_s: float, sample_rate: int = 44100) -> AudioBuffer:
        return AudioBuffer(self.generate(duration_s, sample_rate), sample_rate)


class MultiOscillator:
    """Multiple detuned oscillators for a richer sound."""

    def __init__(self, oscillators: List[Oscillator]):
        self.oscillators = oscillators

    def generate_buffer(self, duration_s: float, sample_rate: int = 44100) -> AudioBuffer:
        buffers = [osc.generate_buffer(duration_s, sample_rate) for osc in self.oscillators]
        n = 1.0 / len(buffers)
        return mix(buffers, [n] * len(buffers))


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class Synthesizer:
    """High-level synthesizer combining oscillators and effects."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._played: List[AudioBuffer] = []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def note_to_freq(note: str) -> float:
        """Convert note name (e.g. 'A4', 'C#5') to frequency in Hz."""
        if len(note) < 2:
            raise ValueError(f"Invalid note: {note}")
        if note[-2] in ("#", "b"):
            name = note[:-2]
            octave_str = note[-1]
        else:
            name = note[:-1]
            octave_str = note[-1]
        octave = int(octave_str)
        if name not in NOTES:
            raise ValueError(f"Unknown note name: {name}")
        semitone = NOTES.index(name) + (octave + 1) * 12
        a4_semitone = NOTES.index("A") + 5 * 12   # A4
        return 440.0 * (2.0 ** ((semitone - a4_semitone) / 12.0))

    @staticmethod
    def freq_to_note(freq: float) -> str:
        """Convert frequency to nearest note name."""
        semitones_from_a4 = round(12 * math.log2(freq / 440.0))
        a4_semitone = NOTES.index("A") + 5 * 12
        idx = a4_semitone + semitones_from_a4
        octave = idx // 12 - 1
        note = NOTES[idx % 12]
        return f"{note}{octave}"

    # ------------------------------------------------------------------
    # Note & Chord generation
    # ------------------------------------------------------------------

    def play_note(
        self,
        note: str,
        duration_s: float = 0.5,
        wave_type: WaveType = WaveType.SINE,
        envelope: Optional[Envelope] = None,
    ) -> AudioBuffer:
        """Synthesize a single note."""
        freq = self.note_to_freq(note)
        osc = Oscillator(wave_type=wave_type, frequency=freq, amplitude=0.6)
        buf = osc.generate_buffer(duration_s, self.sample_rate)
        if envelope:
            buf = apply_envelope(buf, envelope)
        return buf

    def play_chord(
        self,
        root: str,
        chord_type: str = "major",
        duration_s: float = 1.0,
        wave_type: WaveType = WaveType.SINE,
    ) -> AudioBuffer:
        """Synthesize a chord."""
        intervals = CHORD_INTERVALS.get(chord_type, CHORD_INTERVALS["major"])
        root_freq = self.note_to_freq(root)
        buffers = []
        for interval in intervals:
            freq = root_freq * (2 ** (interval / 12.0))
            osc = Oscillator(wave_type=wave_type, frequency=freq, amplitude=0.4)
            buffers.append(osc.generate_buffer(duration_s, self.sample_rate))
        return mix(buffers, [1.0 / len(buffers)] * len(buffers))

    def play_scale(
        self,
        root: str,
        scale_type: ScaleType = ScaleType.MAJOR,
        note_duration_s: float = 0.3,
        wave_type: WaveType = WaveType.SINE,
    ) -> AudioBuffer:
        """Synthesize a scale (ascending)."""
        intervals = SCALE_INTERVALS[scale_type]
        root_freq = self.note_to_freq(root)
        env = Envelope(attack_ms=10, decay_ms=20, sustain=0.7, release_ms=50)
        buffers = []
        for interval in intervals:
            freq = root_freq * (2 ** (interval / 12.0))
            osc = Oscillator(wave_type=wave_type, frequency=freq, amplitude=0.6)
            buf = apply_envelope(osc.generate_buffer(note_duration_s, self.sample_rate), env)
            buffers.append(buf)
        # Concatenate into a single buffer
        all_samples: List[float] = []
        for b in buffers:
            all_samples.extend(b.samples)
        return AudioBuffer(all_samples, self.sample_rate)

    def arpeggio(
        self,
        chord: str,
        chord_type: str = "major",
        bpm: float = 120.0,
        bars: int = 2,
        wave_type: WaveType = WaveType.SQUARE,
    ) -> AudioBuffer:
        """Synthesize an arpeggio pattern."""
        beat_duration = 60.0 / bpm
        note_duration = beat_duration / 2
        intervals = CHORD_INTERVALS.get(chord_type, CHORD_INTERVALS["major"])
        root_freq = self.note_to_freq(chord)
        env = Envelope(attack_ms=5, decay_ms=10, sustain=0.6, release_ms=30)

        all_samples: List[float] = []
        pattern = intervals + intervals[::-1][1:-1]   # up-down pattern
        total_notes = bars * 4 * 2

        for i in range(total_notes):
            interval = pattern[i % len(pattern)]
            freq = root_freq * (2 ** (interval / 12.0))
            osc = Oscillator(wave_type=wave_type, frequency=freq, amplitude=0.5)
            buf = apply_envelope(osc.generate_buffer(note_duration, self.sample_rate), env)
            all_samples.extend(buf.samples)

        return AudioBuffer(all_samples, self.sample_rate)

    def drum_beat(
        self,
        bpm: float = 120.0,
        bars: int = 1,
        pattern: Optional[List[int]] = None,
    ) -> AudioBuffer:
        """Simple drum beat using white noise bursts."""
        if pattern is None:
            pattern = [1, 0, 0, 0, 1, 0, 0, 0]   # kick on beats 1 & 3

        beat_duration = 60.0 / bpm
        note_duration = beat_duration / 2
        env = Envelope(attack_ms=1, decay_ms=30, sustain=0.0, release_ms=20)

        all_samples: List[float] = []
        total_steps = len(pattern) * bars
        for i in range(total_steps):
            hit = pattern[i % len(pattern)]
            if hit:
                osc = Oscillator(wave_type=WaveType.NOISE, frequency=80.0, amplitude=0.8)
                buf = apply_envelope(osc.generate_buffer(note_duration, self.sample_rate), env)
            else:
                buf = AudioBuffer([0.0] * int(note_duration * self.sample_rate), self.sample_rate)
            all_samples.extend(buf.samples)

        return AudioBuffer(all_samples, self.sample_rate)

    # ------------------------------------------------------------------
    # Effects chain
    # ------------------------------------------------------------------

    def fx_chain(
        self,
        buffer: AudioBuffer,
        reverb: bool = False,
        reverb_decay: float = 0.4,
        lowpass_hz: Optional[float] = None,
        highpass_hz: Optional[float] = None,
        normalize_output: bool = True,
    ) -> AudioBuffer:
        """Apply a chain of effects to a buffer."""
        out = buffer
        if lowpass_hz:
            out = apply_lowpass(out, lowpass_hz)
        if highpass_hz:
            out = apply_highpass(out, highpass_hz)
        if reverb:
            out = apply_reverb(out, decay=reverb_decay)
        if normalize_output:
            out = normalize(out)
        return out

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "wave_types": [w.value for w in WaveType],
            "scale_types": [s.value for s in ScaleType],
            "notes": NOTES,
        }


# ---------------------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------------------

@dataclass
class Step:
    note: Optional[str]
    duration_s: float = 0.25
    velocity: float = 1.0
    wave_type: WaveType = WaveType.SINE


class Sequencer:
    """Step-based sequencer."""

    def __init__(self, bpm: float = 120.0, sample_rate: int = 44100):
        self.bpm = bpm
        self.sample_rate = sample_rate
        self.steps: List[Step] = []
        self.synth = Synthesizer(sample_rate)

    def add_step(self, note: Optional[str], velocity: float = 1.0, wave_type: WaveType = WaveType.SINE):
        beat = 60.0 / self.bpm
        self.steps.append(Step(note=note, duration_s=beat, velocity=velocity, wave_type=wave_type))

    def render(self, loops: int = 1) -> AudioBuffer:
        """Render all steps into an AudioBuffer."""
        env = Envelope(attack_ms=5, decay_ms=20, sustain=0.7, release_ms=40)
        all_samples: List[float] = []
        for _ in range(loops):
            for step in self.steps:
                if step.note:
                    freq = self.synth.note_to_freq(step.note)
                    osc = Oscillator(wave_type=step.wave_type, frequency=freq, amplitude=0.6 * step.velocity)
                    buf = apply_envelope(osc.generate_buffer(step.duration_s, self.sample_rate), env)
                else:
                    buf = AudioBuffer([0.0] * int(step.duration_s * self.sample_rate), self.sample_rate)
                all_samples.extend(buf.samples)
        return AudioBuffer(all_samples, self.sample_rate)


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def demo():
    """Quick demonstration of the audio engine."""
    synth = Synthesizer(sample_rate=44100)

    print("=== BlackRoad Audio Engine Demo ===")
    print(f"Engine info: {synth.info()}")

    # Single note
    print("\n[1] Generating A4 sine note (0.5s)...")
    note_buf = synth.play_note("A4", duration_s=0.5, wave_type=WaveType.SINE)
    print(f"  Samples: {len(note_buf)}, Duration: {note_buf.duration_s():.2f}s")

    # Scale
    print("\n[2] Playing C4 major scale...")
    scale_buf = synth.play_scale("C4", ScaleType.MAJOR)
    print(f"  Samples: {len(scale_buf)}, Duration: {scale_buf.duration_s():.2f}s")

    # Chord
    print("\n[3] Playing C major chord...")
    chord_buf = synth.play_chord("C4", "major", duration_s=1.0)
    print(f"  Samples: {len(chord_buf)}, Duration: {chord_buf.duration_s():.2f}s")

    # Arpeggio
    print("\n[4] Generating Am arpeggio at 120bpm...")
    arp_buf = synth.arpeggio("A4", "minor", bpm=120, bars=2, wave_type=WaveType.SQUARE)
    print(f"  Samples: {len(arp_buf)}, Duration: {arp_buf.duration_s():.2f}s")

    # Reverb
    print("\n[5] Applying reverb to note...")
    rev_buf = apply_reverb(note_buf, decay=0.5, delay_ms=50)
    print(f"  Reverb applied. Peak: {max(abs(s) for s in rev_buf.samples):.4f}")

    # Fade
    print("\n[6] Fade in/out...")
    faded = fade_in(fade_out(scale_buf, 0.2), 0.1)
    print(f"  Faded buffer duration: {faded.duration_s():.2f}s")

    # Envelope
    print("\n[7] ADSR envelope...")
    env = Envelope(attack_ms=20, decay_ms=50, sustain=0.6, release_ms=100)
    osc = Oscillator(WaveType.SAWTOOTH, frequency=220.0, amplitude=0.7)
    raw = osc.generate_buffer(1.0)
    env_buf = apply_envelope(raw, env)
    print(f"  Envelope buffer: {len(env_buf)} samples")

    # Sequencer
    print("\n[8] Sequencer...")
    seq = Sequencer(bpm=140)
    for note in ["C4", "E4", "G4", "E4", "C4", None, "G3", None]:
        seq.add_step(note)
    seq_buf = seq.render(loops=2)
    print(f"  Sequence: {len(seq_buf)} samples, {seq_buf.duration_s():.2f}s")

    # WAV export
    print("\n[9] Exporting to WAV bytes...")
    wav_data = export_wav_bytes(note_buf)
    print(f"  WAV size: {len(wav_data)} bytes")

    # note_to_freq
    print("\n[10] Note-to-frequency table:")
    for note in ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]:
        print(f"  {note}: {synth.note_to_freq(note):.2f} Hz")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
