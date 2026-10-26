import shutil
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import lhotse
from lhotse import AudioSource, Recording
from lhotse.audio import AudioLoadingError, suppress_audio_loading_errors
from lhotse.audio.backend import (
    info,
    read_audio,
    read_opus_ffmpeg,
    read_opus_torchaudio,
    read_sph,
    save_audio,
    torchaudio_info,
    torchaudio_load,
)
from lhotse.serialization import IOBackend, io_backend
from lhotse.utils import is_torchaudio_available


@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.wav",
        "test/fixtures/mono_c1.wav",
        "test/fixtures/stereo.wav",
        "test/fixtures/libri/libri-1088-134315-0000.wav",
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
        "test/fixtures/stereo.sph",
        "test/fixtures/common_voice_en_651325.mp3",
    ],
)
def test_info_and_read_audio_consistency(path):
    recording = Recording.from_file(path)
    audio = recording.load_audio()
    assert audio.shape[1] == recording.num_samples


@pytest.mark.skipif(not is_torchaudio_available(), reason="Requires torchaudio")
@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.wav",
        "test/fixtures/mono_c1.wav",
        "test/fixtures/stereo.wav",
        "test/fixtures/libri/libri-1088-134315-0000.wav",
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
        "test/fixtures/stereo.mp3",
        "test/fixtures/common_voice_en_651325.mp3",
    ],
)
@pytest.mark.parametrize("offset", [0, 0.1])
@pytest.mark.parametrize("duration", [None, 0.1])
def test_torchaudio_load_with_offset_duration_works(path, offset, duration):
    audio, sr = torchaudio_load(path, offset=offset, duration=duration)
    # just test that it runs -- the assertions are inside the function


@pytest.mark.skipif(not is_torchaudio_available(), reason="Requires torchaudio")
@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
    ],
)
def test_opus_torchaudio_vs_ffmpeg(path):
    audio_ta, sr_ta = read_opus_torchaudio(path)
    audio_ff, sr_ff = read_opus_ffmpeg(path)
    assert sr_ta == sr_ff
    assert audio_ta.shape == audio_ff.shape
    # Apparently FFMPEG and SOX (torchaudio) apply different decoders
    # and/or resampling algorithms for reading OPUS, so they yield
    # different results up to 3rd decimal place (for 16bit PCM,
    # this is affecting around 6 least significant bits)
    np.testing.assert_almost_equal(audio_ta, audio_ff, decimal=3)


def test_resample_opus():
    # Test that reading OPUS files after resampling
    # does not raise an exception.
    r = Recording.from_file("test/fixtures/mono_c0.opus")
    r.load_audio()
    r1 = r.resample(24000)
    r1.load_audio()


def test_opus_name_with_whitespaces():
    with TemporaryDirectory() as d:
        path_with_ws = Path(d) / "white space.opus"
        shutil.copy("test/fixtures/mono_c0.opus", path_with_ws)
        r = Recording.from_file(path_with_ws)
        r.load_audio()  # does not raise


def test_sph_name_with_whitespaces():
    with TemporaryDirectory() as d:
        path_with_ws = Path(d) / "white space.sph"
        shutil.copy("test/fixtures/stereo.sph", path_with_ws)
        audio = Recording.from_file(path_with_ws).load_audio()
        expected = Recording.from_file("test/fixtures/stereo.sph").load_audio()
        np.testing.assert_array_equal(audio, expected)


def test_opus_ffmpeg_name_with_single_quote():
    with TemporaryDirectory() as d:
        path_with_quote = Path(d) / "it's.opus"
        shutil.copy("test/fixtures/mono_c0.opus", path_with_quote)
        audio, sr = read_opus_ffmpeg(path_with_quote)
        expected, expected_sr = read_opus_ffmpeg("test/fixtures/mono_c0.opus")
        assert sr == expected_sr
        np.testing.assert_array_equal(audio, expected)


def test_opus_ffmpeg_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(AudioLoadingError, match="'ffmpeg' binary is not installed"):
        read_opus_ffmpeg("test/fixtures/mono_c0.opus")


def test_sph_missing_sph2pipe(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="'sph2pipe' binary is not installed"):
        read_sph("test/fixtures/stereo.sph")


@pytest.mark.skipif(not is_torchaudio_available(), reason="Requires torchaudio")
@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
    ],
)
@pytest.mark.parametrize(
    "force_opus_sampling_rate",
    [
        pytest.param(
            8000, marks=pytest.mark.xfail(reason="Mismatch in shape by one sample.")
        ),
        16000,
        pytest.param(
            22050, marks=pytest.mark.xfail(reason="Mismatch in shape by one sample.")
        ),
        24000,
        pytest.param(
            32000, marks=pytest.mark.xfail(reason="Mismatch in shape by one sample.")
        ),
        44100,
        48000,
    ],
)
def test_opus_torchaudio_vs_ffmpeg_with_resampling(path, force_opus_sampling_rate):
    audio_ta, sr_ta = read_opus_torchaudio(
        path, force_opus_sampling_rate=force_opus_sampling_rate
    )
    audio_ff, sr_ff = read_opus_ffmpeg(
        path, force_opus_sampling_rate=force_opus_sampling_rate
    )
    assert sr_ta == sr_ff
    # Note: for some resampling rates, there will be mismatch by one
    # sample. Recording.load_audio() will fix these cases.
    assert audio_ta.shape == audio_ff.shape
    # Note: when we resample, for a very small number of samples,
    # there are typically discrepancies up to second decimal place
    # between the two implementations.
    # I won't fight the audio codec world -- it is what it is.
    np.testing.assert_almost_equal(audio_ta, audio_ff, decimal=1)


def test_audio_caching_disabled_works():
    lhotse.set_caching_enabled(False)  # Disable caching.

    np.random.seed(89)  # Reproducibility.

    # Prepare two different waveforms.
    noise1 = np.random.rand(1, 32000).astype(np.float32)
    noise2 = np.random.rand(1, 32000).astype(np.float32)
    # Sanity check -- the noises are different
    assert np.abs(noise1 - noise2).sum() != 0

    # Save the first waveform in a file.
    with NamedTemporaryFile(suffix=".wav") as f:
        save_audio(f.name, noise1, sampling_rate=16000)
        recording = Recording.from_file(f.name)

        # Read the audio -- should be equal to noise1.
        audio = recording.load_audio()
        np.testing.assert_allclose(audio, noise1, atol=3e-4)

        # Save noise2 to the same location.
        save_audio(f.name, noise2, sampling_rate=16000)

        # Read the audio -- should be equal to noise2,
        # and the caching is ignored (doesn't happen).
        audio = recording.load_audio()
        np.testing.assert_allclose(audio, noise2, atol=3e-4)


def test_command_audio_caching_enabled_works():
    lhotse.set_caching_enabled(True)  # Enable caching.

    np.random.seed(89)  # Reproducibility.

    # Prepare two different waveforms.
    noise1 = np.random.rand(1, 32000).astype(np.float32)
    noise2 = np.random.rand(1, 32000).astype(np.float32)
    # Sanity check -- the noises are different
    assert np.abs(noise1 - noise2).sum() != 0

    # Save the first waveform in a file.
    with NamedTemporaryFile(suffix=".wav") as f:
        save_audio(f.name, noise1, sampling_rate=16000)

        audio_source = AudioSource("command", list([1]), f"cat {f.name}")

        # Read the audio -- should be equal to noise1.
        audio = audio_source.load_audio()
        audio = np.atleast_2d(audio)
        np.testing.assert_allclose(audio, noise1, atol=3e-4)

        # Save noise2 to the same location.
        save_audio(f.name, noise2, sampling_rate=16000)

        # Read the audio -- should *still* be equal to noise1,
        # because reading from this path was cached before.
        audio = audio_source.load_audio()
        audio = np.atleast_2d(audio)
        np.testing.assert_allclose(audio, noise1, atol=3e-4)


def test_command_audio_caching_disabled_works():
    lhotse.set_caching_enabled(False)  # Disable caching.

    np.random.seed(89)  # Reproducibility.

    # Prepare two different waveforms.
    noise1 = np.random.rand(1, 32000).astype(np.float32)
    noise2 = np.random.rand(1, 32000).astype(np.float32)
    # Sanity check -- the noises are different
    assert np.abs(noise1 - noise2).sum() != 0

    # Save the first waveform in a file.
    with NamedTemporaryFile(suffix=".wav") as f:
        save_audio(f.name, noise1, sampling_rate=16000)

        audio_source = AudioSource("command", list([1]), f"cat {f.name}")

        # Read the audio -- should be equal to noise1.
        audio = audio_source.load_audio()
        audio = np.atleast_2d(audio)
        np.testing.assert_allclose(audio, noise1, atol=3e-4)

        # Save noise2 to the same location.
        save_audio(f.name, noise2, sampling_rate=16000)

        # Read the audio -- should be equal to noise2,
        # and the caching is ignored (doesn't happen).
        audio = audio_source.load_audio()
        audio = np.atleast_2d(audio)
        np.testing.assert_allclose(audio, noise2, atol=3e-4)


def test_audio_loading_optimization_returns_expected_num_samples():
    # This is a test for audio loading optimization
    # that kicks in when cut is very minimally shorter than the recording
    cut = Recording.from_file("test/fixtures/mono_c0.opus").to_cut()
    orig_num_samples = cut.num_samples
    reduced_num_samples = orig_num_samples - 1
    cut.duration = reduced_num_samples / cut.sampling_rate
    audio = cut.load_audio()
    assert audio.shape[1] == reduced_num_samples


@pytest.mark.skipif(not is_torchaudio_available(), reason="Requires torchaudio")
def test_torchaudio_info_from_bytes_io():
    audio_filelike = BytesIO(open("test/fixtures/mono_c0.wav", "rb").read())

    meta = torchaudio_info(audio_filelike)
    assert meta.duration == 0.5
    assert meta.frames == 4000
    assert meta.samplerate == 8000
    assert meta.channels == 1


def test_set_audio_backend():
    recording = Recording.from_file("test/fixtures/mono_c0.wav")
    lhotse.audio.set_current_audio_backend(lhotse.audio.backend.LibsndfileBackend())
    audio1 = recording.load_audio()
    lhotse.audio.set_current_audio_backend(
        lhotse.audio.backend.get_default_audio_backend()
    )
    audio2 = recording.load_audio()
    np.testing.assert_array_almost_equal(audio1, audio2)


def test_audio_source_url_uses_current_io_backend():
    class DummyUrlIOBackend(IOBackend):
        def open(self, identifier, mode):
            assert identifier == "mock://mono_c0.wav"
            return open("test/fixtures/mono_c0.wav", mode)

    expected, _ = read_audio("test/fixtures/mono_c0.wav")
    source = AudioSource(type="url", channels=[0], source="mock://mono_c0.wav")

    with io_backend(DummyUrlIOBackend()):
        restored = source.load_audio()

    np.testing.assert_allclose(np.atleast_2d(restored), np.atleast_2d(expected))


def test_fault_tolerant_audio_network_exception():
    def _mock_load_audio(*args, **kwargs):
        raise ConnectionResetError()

    source = Mock()
    source.load_audio = _mock_load_audio
    source.has_video = False

    recording = Recording(
        id="irrelevant",
        sources=[source],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
        channel_ids=[0],
    )

    with pytest.raises(ConnectionResetError):
        recording.load_audio()  # does raise

    with suppress_audio_loading_errors(True):
        recording.load_audio()  # is silently caught
