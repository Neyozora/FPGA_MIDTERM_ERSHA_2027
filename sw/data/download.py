"""
Download the official Fashion-MNIST IDX files and verify their MD5 checksums.

Source: https://github.com/zalandoresearch/fashion-mnist
The MD5 values below are the ones published in the official repository README.

Usage (from the repository root):
    python -m sw.data.download            # download missing files
    python -m sw.data.download --force    # re-download everything
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sw.config import RAW_DIR

# file name -> official MD5
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

# Tried in order. The first is the official GitHub repository itself,
# the second is the official mirror linked from that repository.
MIRRORS = [
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    """Download to a temporary .part file, then rename (no half-written files)."""
    tmp = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fpga-cnn-project/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dst)


def fetch_all(raw_dir: Path = RAW_DIR, force: bool = False) -> dict:
    """Ensure all four files exist in raw_dir with correct MD5. Returns {name: path}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, expected_md5 in FILES.items():
        dst = raw_dir / name
        if dst.exists() and not force:
            if md5sum(dst) == expected_md5:
                print(f"[ok]       {name} (cached, MD5 verified)")
                paths[name] = dst
                continue
            print(f"[mismatch] {name}: cached file has wrong MD5, re-downloading")

        last_err = None
        for base in MIRRORS:
            url = base + name
            try:
                print(f"[download] {url}")
                _download(url, dst)
            except (urllib.error.URLError, OSError) as e:
                last_err = e
                print(f"           failed: {e}")
                continue
            got = md5sum(dst)
            if got == expected_md5:
                print(f"[ok]       {name} (MD5 {got})")
                break
            last_err = RuntimeError(f"MD5 mismatch for {name}: got {got}, expected {expected_md5}")
            print(f"           {last_err}")
            dst.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Could not obtain {name} from any mirror. Last error: {last_err}")
        paths[name] = dst
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args(argv)
    fetch_all(args.raw_dir, args.force)
    print(f"All Fashion-MNIST files present in {args.raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
