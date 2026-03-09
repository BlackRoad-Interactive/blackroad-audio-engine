"""Comprehensive tests for BlackRoad Audio Engine."""
import pytest
import math
from audio_engine import (
    AudioBuffer, Oscillator, MultiOscillator, Synthesizer, Sequencer,
    Envelope, WaveType, ScaleType, Step,
    mix, clip, normalize, fade_in, fade_out,
    apply_reverb, apply_lowpass, apply_highpass, apply_envelope,
    export_wav_bytes, import_wav_bytes,
)


class TestAudioBuffer:
    def test_create(self):
        buf = AudioBuffer([0.0, 0.5, 1.0], 44100)
        assert len(buf) == 3
        assert buf.sample_rate == 44100

    def test_duration(self):
        buf = AudioBuffer([0.0] * 44100, 44100)
        assert abs(buf.duration_s() - 1.0) < 0.001

    def test_copy(self):
        buf = AudioBuffer([1.0, 2.0], 44100)
        c = buf.copy()
        assert c.samples == buf.samples
        c.samples[0] = 99.0
        assert buf.samples[0] == 1.0

    def test_add(self):
        a = AudioBuffer([1.0, 0.5], 44100)
        b = AudioBuffer([0.5, 0.5, 1.0], 44100)
        c = a + b
        assert len(c) == 3
        assert c.samples[0] == 1.5
        assert c.samples[2] == 1.0


class TestMix:
    def test_empty(self):
        assert len(mix([])) == 0

    def test_single(self):
        buf = AudioBuffer([1.0, 0.5], 44100)
        result = mix([buf])
        assert result.samples == [1.0, 0.5]

    def test_with_gains(self):
        a = AudioBuffer([1.0], 44100)
        b = AudioBuffer([1.0], 44100)
        result = mix([a, b], [0.5, 0.25])
        assert abs(result.samples[0] - 0.75) < 1e-9


class TestClipNormalize:
    def test_clip(self):
        buf = AudioBuffer([2.0, -2.0, 0.5], 44100)
        c = clip(buf, 1.0)
        assert c.samples == [1.0, -1.0, 0.5]

    def test_normalize(self):
        buf = AudioBuffer([0.5, -0.5, 0.25], 44100)
        n = normalize(buf, target_peak=1.0)
        assert abs(max(abs(s) for s in n.samples) - 1.0) < 1e-9

    def test_normalize_empty(self):
        buf = AudioBuffer([], 44100)
        n = normalize(buf)
        assert len(n) == 0

    def test_normalize_silence(self):
        buf = AudioBuffer([0.0, 0.0], 44100)
        n = normalize(buf)
        assert n.samples == [0.0, 0.0]


class TestFades:
    def test_fade_in(self):
        buf = AudioBuffer([1.0] * 1000, 1000)
        faded = fade_in(buf, 0.5)
        assert faded.samples[0] == 0.0
        assert faded.samples[-1] == 1.0

    def test_fade_out(self):
        buf = AudioBuffer([1.0] * 1000, 1000)
        faded = fade_out(buf, 0.5)
        assert faded.samples[0] == 1.0
        assert abs(faded.samples[-1]) < 0.01


class TestFilters:
    def test_reverb(self):
        buf = AudioBuffer([1.0] + [0.0] * 999, 44100)
        rev = apply_reverb(buf, decay=0.5, delay_ms=10)
        assert len(rev) == 1000

    def test_lowpass(self):
        buf = AudioBuffer([(-1.0) ** i for i in range(1000)], 44100)
        lp = apply_lowpass(buf, 100.0)
        assert max(abs(s) for s in lp.samples[-100:]) < 0.5

    def test_highpass(self):
        buf = AudioBuffer([1.0] * 1000, 44100)
        hp = apply_highpass(buf, 100.0)
        assert abs(hp.samples[-1]) < 0.1


class TestEnvelope:
    def test_build(self):
        env = Envelope(attack_ms=10, decay_ms=10, sustain=0.5, release_ms=10)
        curve = env.build(44100, 44100)
        assert len(curve) == 44100
        assert curve[0] == 0.0
        assert curve[-1] == 0.0

    def test_apply_envelope(self):
        buf = AudioBuffer([1.0] * 4410, 44100)
        env = Envelope(attack_ms=10, decay_ms=10, sustain=0.7, release_ms=10)
        result = apply_envelope(buf, env)
        assert len(result) == 4410
        assert result.samples[0] == 0.0


class TestOscillator:
    def test_sine(self):
        osc = Oscillator(WaveType.SINE, 440.0, 1.0)
        samples = osc.generate(0.01, 44100)
        assert len(samples) == 441
        assert all(-1.0 <= s <= 1.0 for s in samples)

    def test_square(self):
        osc = Oscillator(WaveType.SQUARE, 440.0, 1.0)
        samples = osc.generate(0.01, 44100)
        assert all(abs(s) == 1.0 for s in samples)

    def test_generate_buffer(self):
        osc = Oscillator(WaveType.SINE, 440.0, 0.5)
        buf = osc.generate_buffer(0.1, 44100)
        assert isinstance(buf, AudioBuffer)
        assert len(buf) == 4410

    def test_all_wave_types(self):
        for wt in WaveType:
            osc = Oscillator(wt, 440.0, 0.5)
            buf = osc.generate_buffer(0.01)
            assert len(buf) > 0


class TestMultiOscillator:
    def test_generate(self):
        oscs = [
            Oscillator(WaveType.SINE, 440.0, 0.5),
            Oscillator(WaveType.SINE, 442.0, 0.5),
        ]
        multi = MultiOscillator(oscs)
        buf = multi.generate_buffer(0.1)
        assert len(buf) > 0


class TestSynthesizer:
    def test_note_to_freq_a4(self):
        assert abs(Synthesizer.note_to_freq("A4") - 440.0) < 0.01

    def test_note_to_freq_c4(self):
        assert abs(Synthesizer.note_to_freq("C4") - 261.63) < 0.5

    def test_freq_to_note(self):
        assert Synthesizer.freq_to_note(440.0) == "A4"

    def test_play_note(self):
        synth = Synthesizer(44100)
        buf = synth.play_note("A4", 0.1)
        assert buf.duration_s() > 0

    def test_play_chord(self):
        synth = Synthesizer(44100)
        buf = synth.play_chord("C4", "major", 0.5)
        assert buf.duration_s() > 0

    def test_play_scale(self):
        synth = Synthesizer(44100)
        buf = synth.play_scale("C4", ScaleType.MAJOR, 0.1)
        assert buf.duration_s() > 0.5

    def test_arpeggio(self):
        synth = Synthesizer(44100)
        buf = synth.arpeggio("A4", "minor", bpm=120, bars=1)
        assert buf.duration_s() > 0

    def test_drum_beat(self):
        synth = Synthesizer(44100)
        buf = synth.drum_beat(bpm=120, bars=1)
        assert buf.duration_s() > 0

    def test_fx_chain(self):
        synth = Synthesizer(44100)
        buf = synth.play_note("A4", 0.2)
        out = synth.fx_chain(buf, reverb=True, lowpass_hz=5000.0)
        assert len(out) > 0

    def test_info(self):
        synth = Synthesizer()
        info = synth.info()
        assert info["sample_rate"] == 44100
        assert "sine" in info["wave_types"]


class TestSequencer:
    def test_render(self):
        seq = Sequencer(bpm=120)
        seq.add_step("C4")
        seq.add_step("E4")
        seq.add_step(None)
        buf = seq.render()
        assert buf.duration_s() > 0

    def test_render_loops(self):
        seq = Sequencer(bpm=120)
        seq.add_step("A4")
        buf1 = seq.render(loops=1)
        buf2 = seq.render(loops=2)
        assert abs(buf2.duration_s() / buf1.duration_s() - 2.0) < 0.1


class TestWavExport:
    def test_roundtrip(self):
        osc = Oscillator(WaveType.SINE, 440.0, 0.5)
        buf = osc.generate_buffer(0.1, 44100)
        wav = export_wav_bytes(buf)
        assert len(wav) > 0
        imported = import_wav_bytes(wav)
        assert abs(imported.duration_s() - buf.duration_s()) < 0.01
        assert imported.sample_rate == 44100
