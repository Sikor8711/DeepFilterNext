import os
import sys

from df.io import load_audio
from df.visualization import plt, spec_figure


def main(path: str):
    audio, info = load_audio(path)
    sr = info.sample_rate
    f = plt.figure(figsize=(10, 6))
    f.set_tight_layout(True)
    spec_figure(
        audio,
        sr,
        colorbar=True,
        from_audio=True,
        figure=f,
        labels=True,
        colorbar_format="%+2.0f dB",
        cmap="inferno",
        n_fft=2048,
        hop=512,
        kHz=True,
    )
    plt.savefig(os.path.splitext(path)[0] + ".pdf")


if __name__ == "__main__":
    assert len(sys.argv) == 2
    assert os.path.isfile(sys.argv[1])
    main(sys.argv[1])
