import io
import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import soundfile as sf
import torch
from loguru import logger
from numpy import ndarray
from torch import Tensor
from torchaudio.functional import resample as ta_resample

from df.logger import warn_once
from df.utils import download_file, get_cache_dir, get_git_root

TA_RESAMPLE_SINC = "sinc_interp_hann"
TA_RESAMPLE_KAISER = "sinc_interp_kaiser"

# libsndfile subtype -> (bits_per_sample, torchaudio encoding name, raw dtype)
# The raw dtype is the smallest dtype that soundfile can read without a change of the sample
# format. libsndfile has no 8 bit read type, so 8 bit PCM uses int16.
_FLOAT_DTYPE = "float32"
_SUBTYPES = {
    "PCM_S8": (8, "PCM_S", "int16"),
    "PCM_U8": (8, "PCM_U", "int16"),
    "PCM_16": (16, "PCM_S", "int16"),
    "PCM_24": (24, "PCM_S", "int32"),
    "PCM_32": (32, "PCM_S", "int32"),
    "FLOAT": (32, "PCM_F", "float32"),
    "DOUBLE": (64, "PCM_F", "float64"),
    "ULAW": (8, "ULAW", _FLOAT_DTYPE),
    "ALAW": (8, "ALAW", _FLOAT_DTYPE),
    "VORBIS": (0, "VORBIS", _FLOAT_DTYPE),
    "OPUS": (0, "OPUS", _FLOAT_DTYPE),
    "MPEG_LAYER_III": (0, "MP3", _FLOAT_DTYPE),
}
# torch dtype -> libsndfile subtype used by save_audio. All other dtypes are saved as FLOAT.
_SAVE_SUBTYPES = {torch.int16: "PCM_16", torch.int32: "PCM_32"}


@dataclass
class AudioMetaData:
    """Replacement for torchaudio.AudioMetaData (removed in torchaudio 2.9)."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def audio_info(file: str) -> AudioMetaData:
    sf_info = sf.info(file)
    bits, encoding, _ = _SUBTYPES.get(sf_info.subtype, (0, "UNKNOWN", _FLOAT_DTYPE))
    if sf_info.format == "FLAC":
        encoding = "FLAC"  # torchaudio reports the codec for FLAC, not the PCM sample format
    return AudioMetaData(
        sample_rate=sf_info.samplerate,
        num_frames=sf_info.frames,
        num_channels=sf_info.channels,
        bits_per_sample=bits,
        encoding=encoding,
    )


def read_audio(
    file: Union[str, BinaryIO], num_frames: int = -1, frame_offset: int = 0, raw: bool = False
) -> Tuple[Tensor, int]:
    """Reads an audio file or an encoded in-memory stream using soundfile.

    Args:
        file (str, BinaryIO): Path to an audio file or a file-like object.
        num_frames (int): Number of frames to read. Defaults to -1, which reads all frames.
        frame_offset (int): Number of frames to skip at the start.
        raw (bool): If True, keep the stored sample format of PCM WAV files, like
            `normalize=False` of torchaudio. Otherwise, return normalized float32 samples.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T].
        sr (int): Sampling rate of the audio.
    """
    dtype = _FLOAT_DTYPE
    if raw:
        sf_info = sf.info(file)
        if sf_info.format == "WAV":
            dtype = _SUBTYPES.get(sf_info.subtype, (0, "UNKNOWN", _FLOAT_DTYPE))[2]
    data, sr = sf.read(file, frames=num_frames, start=frame_offset, dtype=dtype, always_2d=True)
    return torch.from_numpy(data.T.copy()), sr  # [T, C] -> [C, T]


def saturate_int16(audio: Tensor) -> Tensor:
    """Converts audio in the 16 bit value range to int16 and clamps values out of the range."""
    return audio.clamp(-(1 << 15), (1 << 15) - 1).to(torch.int16)


def load_audio(
    file: str,
    sr: Optional[int] = None,
    verbose=True,
    num_frames: int = -1,
    frame_offset: int = 0,
    channels_first: bool = True,
    method: str = "sinc_fast",
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using soundfile.

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        verbose (bool): Print a warning if the audio is resampled.
        num_frames (int): Number of frames to load. Defaults to -1, which loads all frames.
        frame_offset (int): Number of frames to skip at the start. If `sr` is given,
            `num_frames` and `frame_offset` are in samples at `sr`, else in samples at the
            file's sampling rate.
        channels_first (bool): Return audio of shape [C, T] if True, else [T, C].
        method (str): Resample method passed to `resample()`.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    info = audio_info(file)
    if sr is not None and info.sample_rate != sr:
        # num_frames and frame_offset are given at the target sr; convert to the file's sr
        if num_frames > 0:
            num_frames = round(num_frames * info.sample_rate / sr)
        frame_offset = round(frame_offset * info.sample_rate / sr)

    audio, orig_sr = read_audio(file, num_frames=num_frames, frame_offset=frame_offset)

    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, method=method)
    if not channels_first:
        audio = audio.T
    return audio.contiguous(), info


def save_audio(
    file: str,
    audio: Union[Tensor, ndarray],
    sr: int,
    output_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    log: bool = False,
    dtype=torch.int16,
):
    outpath = file
    if suffix is not None:
        file, ext = os.path.splitext(file)
        outpath = file + f"_{suffix}" + ext
    if output_dir is not None:
        outpath = os.path.join(output_dir, os.path.basename(outpath))
    if log:
        logger.info(f"Saving audio file '{outpath}'")
    audio = torch.as_tensor(audio).detach().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if dtype == torch.int16 and audio.dtype != torch.int16:
        audio = saturate_int16(audio * (1 << 15))
    if dtype == torch.float32 and audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)

    subtype = _SAVE_SUBTYPES.get(audio.dtype)
    if subtype is None:
        audio = audio.to(torch.float32)
        subtype = "FLOAT"
    # Container formats such as FLAC or OGG do not support every subtype; use their default then.
    fmt = os.path.splitext(outpath)[1].lstrip(".").upper()
    if fmt in sf.available_formats() and not sf.check_format(fmt, subtype):
        subtype = sf.default_subtype(fmt)
    sr = int(sr)  # May be a 1-element tensor, e.g. when collated by a torch DataLoader
    sf.write(outpath, audio.T.numpy(), sr, subtype=subtype)  # [C, T] -> [T, C]


def load_encoded(buffer: Union[ndarray, bytes]) -> Tensor:
    """Decodes an encoded in-memory audio buffer, e.g. a vorbis or flac sample from an HDF5 dataset.

    libsndfile detects the container format from the stream header.

    Args:
        buffer (ndarray, bytes): Encoded bytes or a uint8 array (or h5py dataset) containing them.

    Returns:
        audio (Tensor): float32 audio tensor of shape [C, T].
    """
    if not isinstance(buffer, (bytes, bytearray)):
        buffer = buffer[...].tobytes()
    return read_audio(io.BytesIO(buffer))[0]


def get_resample_params(method: str) -> Dict[str, Any]:
    params = {
        "sinc_fast": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": TA_RESAMPLE_SINC, "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": TA_RESAMPLE_KAISER,
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio: Tensor, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def get_test_sample(sr: int = 48000) -> Tensor:
    dir = get_git_root()
    file_path = os.path.join("assets", "clean_freesound_33711.wav")
    if dir is None:
        url = "https://github.com/Rikorose/DeepFilterNet/raw/main/" + file_path
        save_dir = get_cache_dir()
        path = download_file(url, save_dir)
    else:
        path = os.path.join(dir, file_path)
    sample, _ = load_audio(path, sr=sr)
    return sample
