"""Phase 1 verification: does the model load, is streaming exact, how fast is it?

1. GPU log-spectrogram == emg2qwerty's torchaudio LogSpectrogram.
2. Streaming (sliding windows, stride K frames, carried CTC state) produces the
   same emissions and the same text as one offline pass over the same audio.
3. Greedy-CTC character error rate over a full real session.
4. Raw inference latency per call at a few batch sizes (no server involved).
5. Synthetic decoder honours the same I/O contract.

Usage:
    python scripts/smoke_test_decoder.py --session $env:USERPROFILE\\emg_data\\<file>.hdf5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emgserve.decoder import (  # noqa: E402
    EMG2QWERTY_ROOT,
    EMG2QwertyDecoder,
    GreedyCTCStream,
    SyntheticDecoder,
)
from emg2qwerty.data import EMGSessionData  # noqa: E402
from emg2qwerty.transforms import LogSpectrogram as RefLogSpectrogram  # noqa: E402


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def load_emg(session: EMGSessionData) -> np.ndarray:
    ts = session[:]  # structured array, loads whole session
    return np.stack([ts["emg_left"], ts["emg_right"]], axis=1).astype(np.float32)  # (T, 2, 16)


def stream(decoder, emg: np.ndarray, frames_per_step: int, batch: int = 64):
    """Cut emg into the exact windows the server will use and decode them in order."""
    S = decoder.window_samples(frames_per_step)
    stride = decoder.stride_samples(frames_per_step)
    starts = list(range(0, len(emg) - S + 1, stride))
    ctc = GreedyCTCStream(decoder.blank)
    all_lp, labels = [], []
    for i in range(0, len(starts), batch):
        wins = np.stack([emg[s : s + S] for s in starts[i : i + batch]])
        lp = decoder.infer(wins)  # (b, K, C)
        for w in lp:
            all_lp.append(w)
            labels += ctc.step(w)
    return np.concatenate(all_lp), labels, len(starts)


def latency(decoder, frames_per_step: int, batch: int, iters: int = 200) -> tuple[float, float]:
    S = decoder.window_samples(frames_per_step)
    x = np.random.randn(batch, S, 2, 16).astype(np.float32)
    for _ in range(10):
        decoder.infer(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        decoder.infer(x)  # .cpu() inside forces a GPU sync, so this is wall time
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=EMG2QWERTY_ROOT / "models" / "generic.ckpt")
    p.add_argument("--frames-per-step", type=int, default=20)
    p.add_argument("--equiv-seconds", type=float, default=60.0)
    args = p.parse_args()
    K = args.frames_per_step

    dec = EMG2QwertyDecoder(args.checkpoint)
    n_params = sum(p.numel() for p in dec.model.parameters())
    print(f"[load] {args.checkpoint.name}: {n_params / 1e6:.2f}M params on {dec.device}, "
          f"receptive field {dec.receptive_field_frames} frames = {dec.context_samples} samples "
          f"({dec.context_samples / 2000 * 1e3:.0f} ms)")

    session = EMGSessionData(args.session)
    emg = load_emg(session)
    print(f"[data] {args.session.name}: user={session.user} condition={session.condition} "
          f"{len(emg) / 2000:.0f}s of EMG, shape {emg.shape}")

    # 1. spectrogram parity on 5 s of real data
    clip = torch.from_numpy(emg[:10000])
    ref = RefLogSpectrogram(n_fft=64, hop_length=16)(clip)  # (T, 2, 16, 33)
    ours = dec.spec(clip[None].to(dec.device))[:, 0].cpu()
    print(f"[spec] max |ours - torchaudio| = {(ours - ref).abs().max().item():.2e}")

    # 2. streaming == offline on the first N seconds
    n = int(args.equiv_seconds * 2000)
    seg = emg[:n]
    offline = dec.infer(seg[None])[0]  # one big window -> (T_out, C)
    streamed, stream_labels, n_win = stream(dec, seg, K)
    m = len(streamed)
    diff = np.abs(offline[:m] - streamed).max()
    off_labels = GreedyCTCStream(dec.blank).step(offline[:m])
    print(f"[stream] {n_win} windows of {dec.window_samples(K)} samples, stride {dec.stride_samples(K)} "
          f"-> {m} frames; offline {len(offline)} frames")
    print(f"[stream] max |offline - streamed| log-prob = {diff:.2e}; "
          f"greedy text identical: {off_labels == stream_labels}")

    # 3. full-session CER
    t0 = time.perf_counter()
    _, labels, n_win = stream(dec, emg, K)
    pred = dec.labels_to_text(labels)
    truth = session.ground_truth().text
    cer = edit_distance(pred, truth) / max(len(truth), 1)
    print(f"[cer] {n_win} windows in {time.perf_counter() - t0:.1f}s; greedy CER (no LM) = {cer * 100:.1f}% "
          f"({len(pred)} predicted vs {len(truth)} true chars)")
    print(f"[cer] truth[:120]: {truth[:120]!r}")
    print(f"[cer] pred [:120]: {pred[:120]!r}")

    # 4. latency
    for b in (1, 8, 32):
        p50, p99 = latency(dec, K, b)
        print(f"[latency] real  batch={b:>2}: p50 {p50:6.2f} ms  p99 {p99:6.2f} ms  "
              f"({p50 / b:.2f} ms/window)")

    # 5. synthetic contract
    syn = SyntheticDecoder()
    out = syn.infer(np.zeros((4, syn.window_samples(K), 2, 16), np.float32))
    assert out.shape == (4, K, syn.num_classes), out.shape
    assert syn.window_samples(K) == dec.window_samples(K)
    p50, p99 = latency(syn, K, 1)
    print(f"[synthetic] contract OK, output {out.shape}; batch=1 p50 {p50:.2f} ms p99 {p99:.2f} ms")
    print(f"[gpu] peak memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
