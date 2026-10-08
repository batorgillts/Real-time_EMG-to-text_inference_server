"""Model interface for the serving layer.

The server never imports torch model code directly. It only talks to a
``Decoder``:

    window samples (B, S, 2, 16) float32  ->  log-probs (B, K, num_classes)

where S = decoder.window_samples(K). Both the real emg2qwerty checkpoint and
the synthetic stand-in implement exactly this contract, so swapping them is a
one-line config change (``EMG_DECODER=real|synthetic``).

Streaming model (why stateless sliding windows are exact):
  * EMG is 2 kHz x 2 wrists x 16 electrodes.
  * Log-spectrogram front-end, n_fft=64, hop=16 -> 125 frames/s (8 ms/frame),
    computed with center=False so a frame only depends on its own 64 samples.
  * The TDS encoder uses unpadded convs: 4 blocks x (kernel 32 - 1) = 124
    frames are consumed, so each output frame sees exactly 125 input frames
    (~1 s). Everything else (eval-mode BatchNorm, LayerNorm, MLPs) is per-frame.
  => One output frame depends on a fixed 2048-sample (1.024 s) span of raw EMG.
     Feeding windows of (2048 + (K-1)*16) samples with stride K*16 yields K new
     frames per window that are bit-for-bit what an offline full-session pass
     would produce (verified in scripts/smoke_test_decoder.py).
"""

from __future__ import annotations

import abc
import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMG2QWERTY_ROOT = PROJECT_ROOT / "third_party" / "emg2qwerty"
if str(EMG2QWERTY_ROOT) not in sys.path:
    sys.path.insert(0, str(EMG2QWERTY_ROOT))

# Only these two modules are imported from Meta's repo: both are plain torch /
# stdlib (+unidecode). We deliberately avoid emg2qwerty.lightning (pins
# pytorch-lightning 1.8) and emg2qwerty.decoder (needs kenlm, a C++ build that
# is painful on Windows).
from emg2qwerty.charset import charset  # noqa: E402
from emg2qwerty.modules import (  # noqa: E402
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)

SAMPLE_RATE = 2000
NUM_BANDS = 2
ELECTRODE_CHANNELS = 16
N_FFT = 64
HOP_LENGTH = 16
FREQ_BINS = N_FFT // 2 + 1  # 33


class LogSpectrogram(nn.Module):
    """GPU log-spectrogram identical to emg2qwerty.transforms.LogSpectrogram
    (torchaudio Spectrogram, normalized=True, center=False, power=2, hann),
    reimplemented with torch.stft so the whole pipeline runs on the GPU and
    batches with the model.

    (B, S, bands, C) raw EMG -> (T, B, bands, C, freq) log10 power.
    """

    def __init__(self) -> None:
        super().__init__()
        window = torch.hann_window(N_FFT)
        self.register_buffer("window", window, persistent=False)
        self.norm = float(window.pow(2).sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, bands, C = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(B * bands * C, S)
        spec = torch.stft(
            flat,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window=self.window,
            center=False,
            return_complex=True,
        )  # (B*bands*C, freq, T)
        power = spec.real.square() + spec.imag.square()
        logspec = torch.log10(power / self.norm + 1e-6)
        T = logspec.shape[-1]
        return logspec.reshape(B, bands, C, FREQ_BINS, T).permute(4, 0, 1, 2, 3)


class Decoder(abc.ABC):
    """Batch inference contract used by the server.

    Subclasses define ``receptive_field_frames`` (spectrogram frames that one
    output frame depends on) and ``_emissions`` (the network).
    """

    name: str = "abstract"
    receptive_field_frames: int

    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # TF32 (default for cuDNN convs on Ampere/Ada GPUs) makes results depend
        # on input shape: streamed vs offline log-probs differed by up to 0.18
        # and ~0.01% of greedy argmaxes flipped. Full fp32 keeps the output
        # independent of window length and batch size (diff ~2e-4, 0 flips), so
        # batching can't change what the user sees.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.charset = charset()
        self.blank = self.charset.null_class
        self.num_classes = self.charset.num_classes

    # --- window geometry -------------------------------------------------
    @property
    def context_samples(self) -> int:
        """Raw samples one output frame depends on (2048 for emg2qwerty)."""
        return (self.receptive_field_frames - 1) * HOP_LENGTH + N_FFT

    def window_samples(self, frames_per_step: int) -> int:
        """Window length that yields exactly ``frames_per_step`` output frames."""
        return self.context_samples + (frames_per_step - 1) * HOP_LENGTH

    @staticmethod
    def stride_samples(frames_per_step: int) -> int:
        return frames_per_step * HOP_LENGTH

    # --- inference -------------------------------------------------------
    @abc.abstractmethod
    def _emissions(self, windows: torch.Tensor) -> torch.Tensor:
        """(B, S, 2, 16) on device -> (T_out, B, num_classes) log-probs."""

    @torch.inference_mode()
    def infer(self, windows: np.ndarray) -> np.ndarray:
        """(B, S, 2, 16) float32 numpy -> (B, K, num_classes) float32 numpy.

        Blocking call (GPU work + device sync). The server runs it in a worker
        thread so the asyncio event loop keeps serving sockets meanwhile.
        """
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(self.device, non_blocking=True)
        out = self._emissions(x).transpose(0, 1)  # (B, K, C)
        return out.float().cpu().numpy()

    def warmup(self, frames_per_step: int, batch_sizes: Sequence[int] = (1, 8, 32)) -> None:
        """Trigger CUDA context init + cuDNN autotune before serving traffic,
        so the first real request doesn't eat a ~1 s one-time cost."""
        S = self.window_samples(frames_per_step)
        for b in batch_sizes:
            self.infer(np.zeros((b, S, NUM_BANDS, ELECTRODE_CHANNELS), np.float32))

    def labels_to_text(self, labels: Sequence[int]) -> str:
        return "".join(self.charset.label_to_char(int(l)) for l in labels)


class EMG2QwertyDecoder(Decoder):
    """Meta's pretrained TDS-Conv-CTC model, rebuilt in plain torch and loaded
    from the Lightning checkpoint's state_dict (no Lightning dependency)."""

    name = "emg2qwerty"

    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        super().__init__(device)
        ckpt = _load_lightning_checkpoint(Path(checkpoint))
        hp = ckpt.get("hyper_parameters", {})
        in_features = int(hp.get("in_features", 528))
        mlp_features = list(hp.get("mlp_features", [384]))
        block_channels = list(hp.get("block_channels", [24, 24, 24, 24]))
        kernel_width = int(hp.get("kernel_width", 32))
        num_features = NUM_BANDS * mlp_features[-1]

        # Same layer order/indices as TDSConvCTCModule.model, so state_dict
        # keys ("model.0.batch_norm.weight", ...) line up exactly.
        self.model = nn.Sequential(
            SpectrogramNorm(channels=NUM_BANDS * ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features, mlp_features=mlp_features, num_bands=NUM_BANDS
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(num_features=num_features, block_channels=block_channels, kernel_width=kernel_width),
            nn.Linear(num_features, self.num_classes),
            nn.LogSoftmax(dim=-1),
        )
        state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.spec = LogSpectrogram().to(self.device)
        self.receptive_field_frames = len(block_channels) * (kernel_width - 1) + 1  # 125

    def _emissions(self, windows: torch.Tensor) -> torch.Tensor:
        return self.model(self.spec(windows))


class SyntheticDecoder(Decoder):
    """Fallback model with the same I/O contract and receptive field but random
    weights. Output text is meaningless; it exists so the serving stack can be
    exercised without the real checkpoint. Results using it must be labelled
    synthetic."""

    name = "synthetic"
    receptive_field_frames = 125

    def __init__(self, device: str | None = None, hidden: int = 256, seed: int = 0) -> None:
        super().__init__(device)
        torch.manual_seed(seed)
        in_feats = NUM_BANDS * ELECTRODE_CHANNELS * FREQ_BINS
        self.proj = nn.Linear(in_feats, hidden)
        # depthwise temporal conv spanning the full 125-frame receptive field
        self.temporal = nn.Conv1d(hidden, hidden, kernel_size=self.receptive_field_frames, groups=hidden)
        self.head = nn.Linear(hidden, self.num_classes)
        self.spec = LogSpectrogram()
        for m in (self.proj, self.temporal, self.head, self.spec):
            m.eval().to(self.device)

    def _emissions(self, windows: torch.Tensor) -> torch.Tensor:
        x = self.spec(windows).flatten(start_dim=2)  # (T, B, 1056)
        x = torch.relu(self.proj(x))  # (T, B, H)
        x = self.temporal(x.permute(1, 2, 0))  # (B, H, T_out)
        x = self.head(torch.relu(x).permute(2, 0, 1))  # (T_out, B, C)
        return x.log_softmax(dim=-1)


class GreedyCTCStream:
    """Per-connection streaming greedy CTC decoder.

    CTC collapses repeats and drops blanks: "hh_e_ll_l_o" -> "hello". In a
    stream, a repeated label can straddle two inference steps, so we carry the
    previous frame's label across calls; otherwise "...h | h..." at a window
    boundary would wrongly emit "hh".
    """

    def __init__(self, blank: int) -> None:
        self.blank = blank
        self.prev = blank

    def step(self, log_probs: np.ndarray) -> list[int]:
        """(K, num_classes) -> newly emitted labels."""
        out: list[int] = []
        for label in log_probs.argmax(axis=-1).tolist():
            if label != self.blank and label != self.prev:
                out.append(label)
            self.prev = label
        return out


def load_decoder(kind: str | None = None, device: str | None = None) -> Decoder:
    """Factory used by the server. ``kind`` defaults to $EMG_DECODER or 'real'."""
    kind = (kind or os.environ.get("EMG_DECODER", "real")).lower()
    if kind == "real":
        ckpt = os.environ.get("EMG_CHECKPOINT", str(EMG2QWERTY_ROOT / "models" / "generic.ckpt"))
        return EMG2QwertyDecoder(ckpt, device=device)
    if kind == "synthetic":
        return SyntheticDecoder(device=device)
    raise ValueError(f"unknown decoder kind: {kind!r} (expected 'real' or 'synthetic')")


# --- checkpoint loading ---------------------------------------------------

class _Stub(dict):
    """Placeholder for classes we don't have installed (Lightning, omegaconf,
    ...). We only need the tensors + plain hyperparameters from the pickle.
    Subclasses dict so dict-like pickles (Lightning's AttributeDict) still
    rebuild their items."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def __setstate__(self, state: Any) -> None:
        self.__dict__["_state"] = state


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


class _stub_pickle_module:  # torch.load(pickle_module=...) expects a module-like object
    Unpickler = _StubUnpickler
    load = pickle.load


def _load_lightning_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. Is git-lfs installed? Run `git lfs pull` in third_party/emg2qwerty"
        )
    # Lightning checkpoints pickle hyperparameters as omegaconf/Lightning
    # objects, which weights_only=True rejects. This file comes from Meta's
    # repo (trusted source), so a full unpickle with unknown classes stubbed
    # out is acceptable here.
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_stub_pickle_module)
    hp = ckpt.get("hyper_parameters")
    if hp is not None and not isinstance(hp, dict):
        # omegaconf DictConfig / AttributeDict stubbed -> fall back to defaults
        state = getattr(hp, "_state", None)
        ckpt["hyper_parameters"] = state if isinstance(state, dict) else {}
    return ckpt
