import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

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

# libsndfile subtype -> (bits_per_sample, torchaudio encoding name)
_SUBTYPE_META = {
    "PCM_S8": (8, "PCM_S"),
    "PCM_U8": (8, "PCM_U"),
    "PCM_16": (16, "PCM_S"),
    "PCM_24": (24, "PCM_S"),
    "PCM_32": (32, "PCM_S"),
    "FLOAT": (32, "PCM_F"),
    "DOUBLE": (64, "PCM_F"),
    "ULAW": (8, "ULAW"),
    "ALAW": (8, "ALAW"),
    "VORBIS": (0, "VORBIS"),
    "OPUS": (0, "OPUS"),
    "MPEG_LAYER_III": (0, "MP3"),
}


@dataclass
class AudioMetaData:
    """Replacement for torchaudio.AudioMetaData (removed in torchaudio 2.9)."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def audio_info(file: str) -> AudioMetaData:
    i = sf.info(file)
    bits, encoding = _SUBTYPE_META.get(i.subtype, (0, "UNKNOWN"))
    if i.format == "FLAC":
        encoding = "FLAC"  # torchaudio reports the codec for FLAC, not the PCM sample format
    return AudioMetaData(
        sample_rate=i.samplerate,
        num_frames=i.frames,
        num_channels=i.channels,
        bits_per_sample=bits,
        encoding=encoding,
    )


def load_audio(
    file: str, sr: Optional[int] = None, verbose=True, **kwargs
) -> Tuple[Tensor, AudioMetaData]:
    """Loads an audio file using soundfile.

    Args:
        file (str): Path to an audio file.
        sr (int): Optionally resample audio to specified target sampling rate.
        **kwargs: Supported: `num_frames`, `frame_offset`, `channels_first`, `method`
            (resample method passed to `resample()`). If `sr` is given, `num_frames` and
            `frame_offset` are in samples at `sr`, else in samples at the file's sampling rate.
            `format` is ignored, because libsndfile detects the format from the file header.

    Returns:
        audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
        info (AudioMetaData): Meta data of the original audio file. Contains the original sr.
    """
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")
    num_frames = kwargs.pop("num_frames", -1)
    frame_offset = kwargs.pop("frame_offset", 0)
    channels_first = kwargs.pop("channels_first", True)
    if kwargs.pop("format", None) is not None:
        warn_once(
            "load_audio: ignoring `format`, libsndfile detects the format from the file header"
        )
    if kwargs:
        warn_once(f"load_audio: ignoring unsupported arguments {list(kwargs)}")

    info = audio_info(file)
    if sr is not None and info.sample_rate != sr:
        # num_frames and frame_offset are given at the target sr; convert to the file's sr
        if num_frames > 0:
            num_frames = round(num_frames * info.sample_rate / sr)
        frame_offset = round(frame_offset * info.sample_rate / sr)

    data, orig_sr = sf.read(
        file, frames=num_frames, start=frame_offset, dtype="float32", always_2d=True
    )
    audio = torch.from_numpy(data.T.copy())  # [T, C] -> [C, T]

    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
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
        audio = (audio * (1 << 15)).clamp(-(1 << 15), (1 << 15) - 1).to(torch.int16)
    if dtype == torch.float32 and audio.dtype != torch.float32:
        audio = audio.to(torch.float32) / (1 << 15)

    if audio.dtype == torch.int16:
        subtype = "PCM_16"
    elif audio.dtype == torch.int32:
        subtype = "PCM_32"
    else:
        audio = audio.to(torch.float32)
        subtype = "FLOAT"
    # Container formats such as FLAC or OGG do not support every subtype; use their default then.
    fmt = os.path.splitext(outpath)[1].lstrip(".").upper()
    if fmt in sf.available_formats() and not sf.check_format(fmt, subtype):
        subtype = sf.default_subtype(fmt)
    sr = int(sr)  # May be a 1-element tensor, e.g. when collated by a torch DataLoader
    sf.write(outpath, audio.T.numpy(), sr, subtype=subtype)  # [C, T] -> [T, C]


def load_encoded(buffer: Union[ndarray, bytes], codec: Optional[str] = None) -> Tensor:
    """Decodes an encoded in-memory audio buffer, e.g. a vorbis or flac sample from an HDF5 dataset.

    Args:
        buffer (ndarray, bytes): Encoded bytes or a uint8 array (or h5py dataset) containing them.
        codec (str): Codec of the buffer. Only for reference, libsndfile detects the container
            format from the stream header.

    Returns:
        audio (Tensor): float32 audio tensor of shape [C, T].
    """
    if not isinstance(buffer, (bytes, bytearray)):
        buffer = buffer[...].tobytes()
    data, _ = sf.read(io.BytesIO(buffer), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy())  # [T, C] -> [C, T]


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
