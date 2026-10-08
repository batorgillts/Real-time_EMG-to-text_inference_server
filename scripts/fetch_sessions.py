"""Stream the first N session files out of the 308 GB emg2qwerty tarball.

The dataset is a single .tar.gz. Gzip has no random access, so we can't seek
to a specific session -- but tar members are laid out sequentially, so we can
stream-decompress from the start and stop as soon as we have N .hdf5 files.
That costs ~N * 300 MB of download instead of 308 GB.

Stdlib only, so it runs before the ML environment is installed.

Usage (PowerShell):
    python scripts/fetch_sessions.py --n 3 --out $env:USERPROFILE\\emg_data
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="number of .hdf5 sessions to keep")
    parser.add_argument("--out", type=Path, default=Path.home() / "emg_data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kept = 0
    with urllib.request.urlopen(URL) as resp:
        # "r|gz" = streaming mode: reads members strictly in order, never seeks.
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile():
                    continue
                if name.endswith(".hdf5") or name == "metadata.csv":
                    target = args.out / name
                    src = tar.extractfile(member)
                    assert src is not None
                    with open(target, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                    print(f"[{time.time() - t0:6.1f}s] saved {name} ({member.size / 1e6:.0f} MB)", flush=True)
                    if name.endswith(".hdf5"):
                        kept += 1
                        if kept >= args.n:
                            break
    print(f"done: {kept} sessions in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
