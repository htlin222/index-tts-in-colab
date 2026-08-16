"""Runs once PER CHUNK via `colab exec -f colab_job/synth_chunk.py`, reusing
the session colab_job/setup.py already warmed up (env built, models on
disk). Each call is a fresh subprocess, so it re-pays loading the model
into GPU memory, but not the env build or model download.

Supports both IndexTTS-2 and IndexTTS-2.5 via the same
/content/model_version.txt marker setup.py reads (default "2.0" -- see
that file's docstring for why 2.0 is the default: 2.5's required
`lang="zh"` conditioning token leaned the output toward a Mainland accent
for a Taiwanese reference voice, 2.0 has no such token).

- 2.0: `indextts2 batch --output-dir` (official CLI). Its batch-file
  validator rejects `silence_after_ms` unless --concat is passed, so we
  write indextts2 a filtered copy without that field and keep the real
  values ourselves for concat_wavs_with_fade.
- 2.5: no official CLI wraps infer_v2_5.py, so /content/_synth_inner_25.py
  (uploaded alongside this script) calls indextts.infer_v2_5.IndexTTS2
  directly inside the uv venv.

Expects, uploaded fresh before each call:
  /content/_common.py          -- shared run() + concat_wavs_with_fade() helpers
  /content/_synth_inner_25.py  -- only needed when model_version=2.5
  /content/model_version.txt   -- "2.0" or "2.5" (uploaded once, same as setup.py)
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
MODEL_VERSION_FILE = WORK / "model_version.txt"
CHUNK_INDEX_FILE = WORK / "chunk_index.txt"
VOICE_FILE = WORK / "ref.wav"
INNER_SCRIPT_25 = WORK / "_synth_inner_25.py"
SEG_PREFIX = "seg"


def read_model_version():
    if not MODEL_VERSION_FILE.is_file():
        return "2.0"
    v = MODEL_VERSION_FILE.read_text().strip()
    return v if v in ("2.0", "2.5") else "2.0"


def model_dir_for(version):
    return REPO / ("checkpoints_25" if version == "2.5" else "checkpoints_2")


def load_tasks(batch_file):
    tasks = []
    with open(batch_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def synth_chunk_20(batch_file, seg_dir, model_dir, batch_timeout, tasks):
    row_batch_file = batch_file.with_name(batch_file.stem + "_row.jsonl")
    with open(row_batch_file, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps({k: v for k, v in t.items() if k != "silence_after_ms"},
                                ensure_ascii=False) + "\n")
    run(["uv", "run", "indextts2", "batch",
         "--batch-file", str(row_batch_file),
         "--model-dir", str(model_dir),
         "--output-dir", str(seg_dir), "--output-prefix", SEG_PREFIX,
         "--no-cuda-kernel", "--force", "--verbose"],
        cwd=str(REPO), timeout=batch_timeout)


def synth_chunk_25(batch_file, seg_dir, model_dir, batch_timeout):
    assert INNER_SCRIPT_25.is_file(), f"missing {INNER_SCRIPT_25}"
    run(["uv", "run", "python", str(INNER_SCRIPT_25),
         str(batch_file), str(seg_dir), str(model_dir), SEG_PREFIX],
        cwd=str(REPO), timeout=batch_timeout)


def main():
    assert VOICE_FILE.is_file(), f"missing {VOICE_FILE}"
    assert CHUNK_INDEX_FILE.is_file(), f"missing {CHUNK_INDEX_FILE}"
    idx = CHUNK_INDEX_FILE.read_text().strip()
    version = read_model_version()
    model_dir = model_dir_for(version)
    print(f">> model version: {version} -> {model_dir}")

    batch_file = WORK / f"batch_chunk_{idx}.jsonl"
    output_file = WORK / f"chunk_{idx}.wav"
    assert batch_file.is_file(), f"missing {batch_file}"

    tasks = load_tasks(batch_file)
    total_chars = sum(len(t.get("text", "")) for t in tasks)
    # Regressed from IndexTTS-2 real runs (36 chars->173s, 93 chars->240s on
    # T4): batch_time ≈ 131s fixed (model load) + 1.18s/char, ~2x safety
    # margin. Reused as-is for 2.5 (measured comparably fast in practice);
    # will split the formula per-version if real numbers ever diverge
    # enough to matter. scripts/parse_issue.py caps each chunk at
    # CHUNK_MAX_CHARS=700 chars.
    batch_timeout = 260 + 2 * total_chars

    seg_dir = WORK / f"segs_{idx}"
    seg_dir.mkdir(exist_ok=True)
    if version == "2.5":
        synth_chunk_25(batch_file, seg_dir, model_dir, batch_timeout)
    else:
        synth_chunk_20(batch_file, seg_dir, model_dir, batch_timeout, tasks)

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
