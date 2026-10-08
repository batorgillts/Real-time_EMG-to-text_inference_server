r"""Hands-on tour of the Decoder: run it and read each printed line.

    & "$env:USERPROFILE\.venvs\emgserve\Scripts\python.exe" scripts\explore_decoder.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from emgserve.decoder import GreedyCTCStream, load_decoder  # noqa: E402

d = load_decoder("real")
print("1. device:", d.device)

# Window geometry: how many raw samples one inference call needs, and how far we slide.
print("2. window samples:", d.window_samples(20), "| stride samples:", d.stride_samples(20))

# A fake batch: 3 windows, each 2352 time steps x 2 wrists x 16 electrodes.
x = np.random.randn(3, 2352, 2, 16).astype(np.float32)
print("3. input shape: ", x.shape)

out = d.infer(x)
print("4. output shape:", out.shape, "-> (windows, frames per window, class scores)")

# Each frame's 99 numbers are log-probabilities; exp() turns them into probabilities.
print("5. probabilities in one frame sum to:", round(float(np.exp(out[0, 0]).sum()), 4))

winner = int(out[0, 0].argmax())
print("6. winning class in frame 0:", winner, "| blank class is:", d.blank)

# CTC: merge repeats, drop blanks. Random noise in -> mostly blanks out.
labels = GreedyCTCStream(d.blank).step(out[0])
print("7. characters decoded from noise:", repr(d.labels_to_text(labels)))
