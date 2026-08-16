"""Runs once PER CHUNK via `colab exec -f colab_job/synth_chunk.py`, reusing
the session colab_job/setup.py already warmed up (env built, models on
disk). Each call is a fresh `uv run python _synth_inner_25.py` subprocess,
so it re-pays loading the model into GPU memory, but not the env build or
model download.

Expects, uploaded fresh before each call:
  /content/_common.py          -- shared run() + concat_wavs_with_fade() helpers
  /content/_synth_inner_25.py  -- the actual IndexTTS-2.5 call (no official
                                   CLI wraps infer_v2_5.py, so we invoke it
                                   ourselves inside the uv venv)
  /content/chunk_index.txt     -- which chunk this call should process, e.g. "2"
  /content/batch_chunk_N.jsonl -- that chunk's tasks (text/voice/emotion_vector/
                                   emotion_weight/silence_after_ms)
  /content/ref.wav             -- reference voice clip (uploaded once, reused)

Produces:
  /content/chunk_N.wav

Synthesizes each line to its own file, then stitches them ourselves via
concat_wavs_with_fade -- see that function's docstring for why: raw digital
silence butted directly against full-amplitude audio reads as an abrupt
"click" at every line boundary (2026-08-16 user feedback: the cut points
sounded wrong). A short fade envelope on each segment fixes that without
changing the pause durations parse_issue.py already computed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/content")
from _common import run, concat_wavs_with_fade  # noqa: E402

WORK = Path("/content")
REPO = WORK / "index-tts"
MODEL_DIR = REPO / "checkpoints_25"
CHUNK_INDEX_FILE = WORK / "chunk_index.txt"
VOICE_FILE = WORK / "ref.wav"
INNER_SCRIPT = WORK / "_synth_inner_25.py"
SEG_PREFIX = "seg"


def load_tasks(batch_file):
    tasks = []
    with open(batch_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def main():
    assert VOICE_FILE.is_file(), f"missing {VOICE_FILE}"
    assert INNER_SCRIPT.is_file(), f"missing {INNER_SCRIPT}"
    assert CHUNK_INDEX_FILE.is_file(), f"missing {CHUNK_INDEX_FILE}"
    idx = CHUNK_INDEX_FILE.read_text().strip()

    batch_file = WORK / f"batch_chunk_{idx}.jsonl"
    output_file = WORK / f"chunk_{idx}.wav"
    assert batch_file.is_file(), f"missing {batch_file}"

    tasks = load_tasks(batch_file)
    total_chars = sum(len(t.get("text", "")) for t in tasks)
    # Regressed from IndexTTS-2 real runs (36 chars->173s, 93 chars->240s on
    # T4): batch_time ≈ 131s fixed (model load) + 1.18s/char, ~2x safety
    # margin. IndexTTS-2.5 is claimed faster, not slower, so this generous
    # a ceiling should still hold -- will recalibrate from real 2.5 numbers
    # once we have them. scripts/parse_issue.py caps each chunk at
    # CHUNK_MAX_CHARS=700 chars.
    batch_timeout = 260 + 2 * total_chars

    seg_dir = WORK / f"segs_{idx}"
    seg_dir.mkdir(exist_ok=True)
    run(["uv", "run", "python", str(INNER_SCRIPT),
         str(batch_file), str(seg_dir), str(MODEL_DIR), SEG_PREFIX],
        cwd=str(REPO), timeout=batch_timeout)

    segments = []
    for i, task in enumerate(tasks, start=1):
        seg_path = seg_dir / f"{SEG_PREFIX}-{i:04d}.wav"
        assert seg_path.is_file(), f"missing expected segment {seg_path}"
        segments.append((seg_path, task.get("silence_after_ms", 0)))

    concat_wavs_with_fade(segments, output_file, fade_ms=20)

    assert output_file.is_file(), f"stitched output {output_file} was not created"
    size = output_file.stat().st_size
    print(f"\n>> chunk {idx} done: {output_file} ({size} bytes, {len(segments)} segments)")
    if size < 1000:
        raise RuntimeError(f"{output_file} suspiciously small ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
