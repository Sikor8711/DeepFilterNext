#!/usr/bin/env python3

import argparse
import io
import os
import time
import warnings
from multiprocessing import Pool
from typing import List, Optional

import h5py as h5
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from df.io import load_audio
from df.logger import init_logger


def write_to_h5(
    file_name: str,
    data: dict,
    sr: int,
    max_freq: int = -1,
    dtype: str = "float32",
    codec: str = "pcm",
    mono: bool = False,
    compression: Optional[int] = None,
    num_workers: int = 4,
):
    """Creates a HDF5 dataset based on the provided dict.

    Args:
        file_name: Full path of the HDF5 dataset file.
        data: Dictionary containing for each key (dataset group) a list of file names
            to pre-process and store.
    """
    if max_freq <= 0:
        max_freq = sr // 2
    compression_factor = None
    if codec is not None:
        compression_factor = 8
    with h5.File(file_name, "a", libver="latest", swmr=True) as f, torch.no_grad():
        # Document attributes first
        f.attrs["db_id"] = int(time.time())
        f.attrs["db_name"] = os.path.basename(file_name)
        f.attrs["max_freq"] = max_freq
        f.attrs["dtype"] = dtype
        f.attrs["sr"] = sr
        f.attrs["codec"] = codec
        # Write encoded/decoded samples
        for key, data_dict in data.items():
            try:
                grp = f.create_group(key)
            except ValueError:
                logger.info(f"Found existing group {key}")
                grp = f[key]
            dataset = PreProcessingDataset(
                sr,
                file_names=data_dict["files"],
                dtype=dtype,
                codec=codec,
                mono=mono,
                compression=compression_factor,
            )
            loader = DataLoader(dataset, num_workers=num_workers, batch_size=1, shuffle=False)
            # Computes the samples in several worker processes
            n_samples = len(dataset)
            for i, sample in enumerate(loader):
                # Sample is a dict containing a list
                fn = os.path.relpath(sample["file_name"][0], data_dict["working_dir"])
                audio: np.ndarray = sample["data"][0].numpy()
                if codec in ("flac", "vorbis"):
                    audio = audio.squeeze()
                if sample["n_samples"] < sr / 100:  # Should be at least 100 ms
                    logger.warning(f"Short audio {fn}: {audio.shape}.")
                progress = i / n_samples * 100
                logger.info(f"{progress:2.0f}% | Writing file {fn} to the {key} dataset.")
                if sample["n_samples"] == 0:
                    continue
                ds_key = fn.replace("/", "_")
                if ds_key in grp:
                    logger.info(f"Found dataset {ds_key}. Replacing.")
                    del grp[ds_key]
                ds = grp.create_dataset(ds_key, data=audio, compression=compression)
                ds.attrs["n_samples"] = sample["n_samples"]
                del audio, sample
            logger.info("Added {} samples to the group {}.".format(n_samples, key))


class PreProcessingDataset(Dataset):
    def __init__(
        self,
        sr: int,
        file_names: List[str] = None,
        dtype="float32",
        codec="pcm",
        mono=False,
        compression: Optional[int] = None,
    ):
        self.file_names = file_names or []
        self.sr = sr
        if dtype == "float32":
            self.dtype = np.float32
        elif dtype == "int16":
            self.dtype = np.int16
        else:
            raise ValueError("Unkown dtype")
        self.codec = codec.lower()
        if self.codec == "vorbis":
            self.dtype = np.float32
        self.mono = mono
        self.compression = compression  # -1 - 10 for vorbis, 0-8 for flac

    def read(self, file: str) -> Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta = sf.info(file)
            if meta.samplerate != self.sr:
                # Load as normalized float32 and resample
                x, _ = load_audio(file, sr=self.sr, verbose=False, method="kaiser_best")
            else:
                # Keep the stored sample format of PCM WAV files (like torchaudio's normalize=False)
                dtype = "float32"
                if meta.format == "WAV":
                    dtype = _RAW_DTYPES.get(meta.subtype, "float32")
                x, _ = sf.read(file, dtype=dtype, always_2d=True)
                x = torch.from_numpy(x.T.copy())  # [T, C] -> [C, T]
            if self.mono and x.shape[0] > 1:
                x = x.mean(0, keepdim=True)
            if x.dim() == 1:
                x = x.reshape(1, -1)
        return x

    def __getitem__(self, index):
        fn = self.file_names[index]
        logger.debug(f"Reading audio file {fn}")
        x = self.read(fn)
        assert x.dim() == 2 and x.shape[0] <= 16, f"Got sample {fn} with unexpected shape {x.shape}"
        n_samples = x.shape[1]
        x = encode(x, self.sr, self.codec, self.compression)
        return {"file_name": fn, "data": x, "n_samples": n_samples}

    def __len__(self):
        return len(self.file_names)


# Sample formats that are kept when reading PCM WAV files without resampling
_RAW_DTYPES = {
    "PCM_16": "int16",
    "PCM_24": "int32",
    "PCM_32": "int32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}


def _encode_to_buffer(x: Tensor, sr: int, **kwargs) -> np.ndarray:
    """Encodes x of shape [C, T] with soundfile and returns the binary buffer as uint8 array."""
    buf = io.BytesIO()
    sf.write(buf, x.numpy().T, sr, **kwargs)  # [C, T] -> [T, C]
    return np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()


def encode(x: Tensor, sr: int, codec: str, compression: Optional[int] = None) -> np.ndarray:
    if codec == "vorbis":
        # `compression` is the vorbis quality in [-1, 10]. libsndfile uses a compression level in
        # [0, 1] instead, where 0 is the highest quality. It maps to quality [0, 10] only, so
        # negative qualities are clipped to 0.
        level = None
        if compression is not None:
            if compression < 0:
                logger.warning(f"Vorbis quality {compression} is not supported, using 0 instead")
            level = min(max(1.0 - compression / 10.0, 0.0), 1.0)
        x = _encode_to_buffer(x, sr, format="OGG", subtype="VORBIS", compression_level=level)
    elif codec == "flac":
        # `compression` is the FLAC compression level in [0, 8]. libsndfile uses [0, 1] instead.
        level = None
        if compression is not None:
            level = min(max(compression / 8.0, 0.0), 1.0)
        if x.dtype == torch.int32:  # 24 or 32 bit PCM
            x = x.to(torch.float32) / (1 << 31)
        if x.dtype != torch.int16:
            # Convert to 16 bit with saturation (libsndfile would not clip)
            x = (x * (1 << 15)).round().clamp(-(1 << 15), (1 << 15) - 1).to(torch.int16)
        x = _encode_to_buffer(x, sr, format="FLAC", subtype="PCM_16", compression_level=level)
    elif codec == "pcm":
        x = x.numpy()
    else:
        raise NotImplementedError(f"Codec '{codec}' not supported.")
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, help="Either 'speech', 'noise' or 'noisy'.")
    parser.add_argument(
        "audio_files",
        type=str,
        help="Text file containing speech or noise files separated by a new line.",
    )
    parser.add_argument(
        "hdf5_db", type=str, help="HDF5 file name where the data will be stored in."
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_freq",
        type=int,
        default=-1,
        help="Only frequencies below the specified frequency will be considered during loss computation.\n"
        "This is useful for upsampled signals, that contain no information in higher frequencies.",
    )
    parser.add_argument("--sr", type=int, default=48_000, help="Sample rate to resample to.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="int16",
        help="Dtype that will be used to store the audio files. Can be float32 or int16.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="pcm",
        help="codec that will be stored in the database. Can be PCM or Vorbis. Defaults to PCM.",
    )
    parser.add_argument("--mono", action="store_true")
    parser.add_argument(
        "--compression", type=str, default=None, help="HDF5 dataset compression (e.g. `gzip`)."
    )
    args = parser.parse_args()

    if not args.hdf5_db.endswith(".hdf5"):
        args.hdf5_db += ".hdf5"

    init_logger("/tmp/prepare_data.log")
    valid_types = ("speech", "noise", "noisy", "rir")
    if args.type not in valid_types:
        raise ValueError(f"Dataset type must be one of {valid_types}, but got {args.type}")

    data = {
        args.type: {"working_dir": None, "files": []},
    }
    with open(args.audio_files) as f:
        working_dir = os.path.dirname(args.audio_files)
        data[args.type]["working_dir"] = working_dir
        logger.info(f"Using speech working directory {working_dir}")

        def _check_file(file: str):
            file = os.path.join(working_dir, file.strip())
            if not os.path.isfile(file):
                raise FileNotFoundError(f"file {file} not found")
            return file

        with Pool(max(args.num_workers, 1)) as p:
            res = p.imap(_check_file, f, 100)
            data[args.type]["files"] = list(res)
        logger.info("Checking all audio files complete")
    write_to_h5(
        file_name=args.hdf5_db,
        data=data,
        sr=args.sr,
        max_freq=args.max_freq,
        dtype=args.dtype,
        codec=args.codec.lower(),
        mono=args.mono,
        compression=args.compression,
        num_workers=args.num_workers,
    )
