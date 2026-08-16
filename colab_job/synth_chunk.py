"""Runs once PER CHUNK via `colab exec -f colab_job/synth_chunk.py`, reusing
the session colab_job/setup.py already warmed up (env built, models on
disk). Each call is a fresh `indextts2` subprocess, so it re-pays loading
the model into GPU memory (~131s measured), but not the env build or
model download.

Expects, uploaded fresh before each call:
  /content/_common.py         -- shared run() helper
  /content/chunk_index.txt    -- which chunk this call should process, e.g. "2"
  /content/batch_chunk_N.jsonl -- that chunk's indextts2 batch tasks
  /content/ref.wav            -- reference voice clip (uploaded once, reused)

Produces:
  /content/chunk_N.wav
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/content")
from _common import run  # noqa: E402

WORK = Path("/content")
REPO = WORK / "index-tts"
MODEL_DIR = REPO / "checkpoints_2"
CHUNK_INDEX_FILE = WORK / "chunk_index.txt"
VOICE_FILE = WORK / "ref.wav"


def count_chars(batch_file):
    total = 0
    with open(batch_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += len(json.loads(line).get("text", ""))
    return total


def main():
    assert VOICE_FILE.is_file(), f"missing {VOICE_FILE}"
    assert CHUNK_INDEX_FILE.is_file(), f"missing {CHUNK_INDEX_FILE}"
    idx = CHUNK_INDEX_FILE.read_text().strip()

    batch_file = WORK / f"batch_chunk_{idx}.jsonl"
    output_file = WORK / f"chunk_{idx}.wav"
    assert batch_file.is_file(), f"missing {batch_file}"

    # Regressed from two real runs (36 chars->173s, 93 chars->240s on T4):
    # batch_time ≈ 131s fixed (model load) + 1.18s/char. ~2x safety margin
    # over the measured slope/intercept since it's only two data points.
    # scripts/parse_issue.py caps each chunk at CHUNK_MAX_CHARS=700 chars.
    total_chars = count_chars(batch_file)
    batch_timeout = 260 + 2 * total_chars

    run(["uv", "run", "indextts2", "batch",
         "--batch-file", str(batch_file),
         "--model-dir", str(MODEL_DIR),
         "--concat", "--output", str(output_file),
         "--no-cuda-kernel", "--force", "--verbose"],
        cwd=str(REPO), timeout=batch_timeout)

    assert output_file.is_file(), f"synthesis finished but {output_file} was not created"
    size = output_file.stat().st_size
    print(f"\n>> chunk {idx} done: {output_file} ({size} bytes)")
    if size < 1000:
        raise RuntimeError(f"{output_file} suspiciously small ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
