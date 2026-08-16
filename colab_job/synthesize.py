"""Runs on the Colab VM via `colab exec -f colab_job/synthesize.py`.

Expects, already placed on the VM before this script runs:
  /content/ref.wav      -- reference voice clip (via `colab upload`)
  /content/batch.jsonl  -- indextts2 batch tasks (via `colab upload`)

Produces:
  /content/output.wav   -- concatenated synthesis result
"""
import subprocess
import sys
from pathlib import Path

WORK = Path("/content")
REPO = WORK / "index-tts"
MODEL_DIR = REPO / "checkpoints_2"
BATCH_FILE = WORK / "batch.jsonl"
VOICE_FILE = WORK / "ref.wav"
OUTPUT_FILE = WORK / "output.wav"


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def main():
    assert BATCH_FILE.is_file(), f"missing {BATCH_FILE}, did the upload step run?"
    assert VOICE_FILE.is_file(), f"missing {VOICE_FILE}, did the upload step run?"

    if not REPO.is_dir():
        run(["git", "clone", "--depth", "1",
             "https://github.com/index-tts/index-tts.git", str(REPO)])

    run(["uv", "sync"], cwd=str(REPO))
    run(["uv", "run", "indextts2", "download", "--model-dir", str(MODEL_DIR)],
        cwd=str(REPO))
    run(["uv", "run", "indextts2", "check", "--model-dir", str(MODEL_DIR),
         "--device", "cuda"], cwd=str(REPO))
    run(["uv", "run", "indextts2", "batch",
         "--batch-file", str(BATCH_FILE),
         "--model-dir", str(MODEL_DIR),
         "--concat", "--output", str(OUTPUT_FILE),
         "--no-cuda-kernel", "--force", "--verbose"],
        cwd=str(REPO))

    assert OUTPUT_FILE.is_file(), "synthesis finished but output.wav was not created"
    size = OUTPUT_FILE.stat().st_size
    print(f"\n>> done: {OUTPUT_FILE} ({size} bytes)")
    if size < 1000:
        raise RuntimeError(f"output.wav suspiciously small ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
