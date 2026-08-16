"""Runs ONCE per issue via `colab exec -f colab_job/setup.py`.

Builds the environment and downloads model weights -- the ~500-900s fixed
cost that used to be re-paid on every single-shot run. Chunked synthesis
(colab_job/synth_chunk.py) reuses this same warm session, so a long request
pays this once instead of once per chunk.

Expects:
  /content/_common.py    -- shared run() helper (via `colab upload`)
  /content/hf_token      -- optional HuggingFace token (via `colab upload`),
                             improves model-download stability. Read once,
                             then deleted.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, "/content")
from _common import run  # noqa: E402

WORK = Path("/content")
REPO = WORK / "index-tts"
MODEL_DIR = REPO / "checkpoints_2"
HF_TOKEN_FILE = WORK / "hf_token"


def setup_hf_token():
    if not HF_TOKEN_FILE.is_file():
        print(">> no /content/hf_token uploaded; HF downloads will be unauthenticated "
              "(slower/rate-limited on the free tier).")
        return
    token = HF_TOKEN_FILE.read_text().strip()
    HF_TOKEN_FILE.unlink()  # don't leave it sitting on the VM disk
    if token:
        os.environ["HF_TOKEN"] = token
        print(">> HF_TOKEN set from uploaded credential.")


def main():
    setup_hf_token()

    if not REPO.is_dir():
        run(["git", "clone", "--depth", "1",
             "https://github.com/index-tts/index-tts.git", str(REPO)], timeout=180)

    run(["uv", "sync"], cwd=str(REPO), timeout=600)
    # Per-attempt timeout kept short (rather than one long attempt) so a
    # stalled download is detected and retried sooner -- see the 2026-08-16
    # incident where an unauthenticated download sat silent for 28 minutes.
    run(["uv", "run", "indextts2", "download", "--model-dir", str(MODEL_DIR)],
        cwd=str(REPO), timeout=600, retries=2, retry_backoff=30)
    run(["uv", "run", "indextts2", "check", "--model-dir", str(MODEL_DIR),
         "--device", "cuda"], cwd=str(REPO), timeout=90)

    print("\n>> setup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
