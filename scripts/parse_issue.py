#!/usr/bin/env python3
"""Parse a GitHub issue-form body into chunked indextts2 batch JSONL files.

Reads the raw issue body from the ISSUE_BODY env var (never interpolate
untrusted issue content directly into a shell `run:` block — pass it via
`env:` instead, which is what the calling workflow does).

Emotion vector order (IndexTTS-2, 8-dim): [happy, angry, sad, afraid,
disgusted, melancholic, surprised, calm]. Each preset's sum is kept <= 0.8,
the hard limit enforced by the `indextts2` CLI's batch-file validator.

Output is CHUNKED rather than a single batch.jsonl: colab_job/setup.py pays
the env-build + model-download cost once per issue, then colab_job/
synth_chunk.py runs once per chunk file against the same warm Colab
session, so a long request costs (setup once) + (chunks * per-chunk
synth), not (chunks * full pipeline). See README for the throughput math
behind CHUNK_MAX_CHARS.

Writes:
  batch_chunk_0.jsonl, batch_chunk_1.jsonl, ...  -- one indextts2 batch file
                                                     per chunk
  chunk_count.txt                                -- integer, for the
                                                     workflow's loop
"""
import json
import os
import re
import sys

# Measured on 2026-08-16 (T4, HF_TOKEN set): batch synthesis costs roughly
# 131s fixed (model load) + 1.18s/char (regressed from two real runs: 36
# chars->173s, 93 chars->240s). CHUNK_MAX_CHARS keeps each individual
# colab-exec call's synthesis time predictable and comfortably inside its
# own timeout; MAX_TOTAL_CHARS bounds how many chunks (and therefore how
# much total wall-clock and Colab compute quota) a single issue can cost.
# See README for the full breakdown.
MAX_LINES = 150
MAX_TOTAL_CHARS = 3000
CHUNK_MAX_CHARS = 700
MAX_LINE_CHARS = 200  # a single absurdly long line would blow a chunk's budget alone
VOICE_PATH = "/content/ref.wav"
DEFAULT_WEIGHT = 0.85

EMOTION_PRESETS = {
    "自然": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60],
    "高興": [0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15],
    "憤怒": [0.0, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10],
    "悲傷": [0.0, 0.0, 0.55, 0.0, 0.0, 0.20, 0.0, 0.0],
    "低落": [0.0, 0.0, 0.25, 0.0, 0.0, 0.50, 0.0, 0.0],
    "驚訝": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.20],
}


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_form_sections(body):
    """Split a GitHub issue-form body into {header_text: value_text}."""
    sections = {}
    parts = re.split(r"^### (.+)$", body, flags=re.MULTILINE)
    # parts = [preamble, header1, value1, header2, value2, ...]
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[header] = value
    return sections


def pick_emotion(raw):
    raw = (raw or "").strip()
    if not raw or raw == "_No response_":
        return "自然"
    for name in EMOTION_PRESETS:
        if name in raw:
            return name
    die(f"unrecognized emotion option: {raw!r}")


def pick_weight(raw):
    raw = (raw or "").strip()
    if not raw or raw == "_No response_":
        return DEFAULT_WEIGHT
    try:
        w = float(raw)
    except ValueError:
        die(f"emotion weight is not a number: {raw!r}")
    if not (0.1 <= w <= 1.0):
        die(f"emotion weight must be between 0.1 and 1.0, got {w}")
    return w


def silence_after(line):
    if line.endswith(("。", "！", "？", "…")):
        return 550
    if line.endswith(("，", "、", "；")):
        return 250
    return 350


def build_tasks(lines, vector, weight):
    tasks = []
    for i, line in enumerate(lines):
        tasks.append({
            "text": line,
            "voice": VOICE_PATH,
            "emotion_vector": vector,
            "emotion_weight": weight,
            "silence_after_ms": 0 if i == len(lines) - 1 else silence_after(line),
        })
    return tasks


def chunk_tasks(tasks, chunk_max_chars):
    """Group tasks (in order) into chunks, each staying under chunk_max_chars.

    A single task is never split. Each task's own silence_after_ms is left
    untouched (computed once over the full, un-chunked line sequence), so
    the pause at a chunk boundary is whatever the punctuation there implied
    -- chunking doesn't create or remove pauses, just where the underlying
    indextts2 batch calls are split.
    """
    chunks = []
    current = []
    current_chars = 0
    for task in tasks:
        n = len(task["text"])
        if current and current_chars + n > chunk_max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(task)
        current_chars += n
    if current:
        chunks.append(current)
    return chunks


def main():
    body = os.environ.get("ISSUE_BODY", "")
    if not body.strip():
        die("ISSUE_BODY is empty")

    sections = parse_form_sections(body)

    text_raw = sections.get("要朗讀的文字", "")
    if not text_raw or text_raw == "_No response_":
        die("missing required field: 要朗讀的文字")

    lines = [ln.strip() for ln in text_raw.splitlines() if ln.strip()]
    if not lines:
        die("no non-empty lines found in 要朗讀的文字")
    if len(lines) > MAX_LINES:
        die(f"too many lines: {len(lines)} > {MAX_LINES} limit")
    too_long = [ln for ln in lines if len(ln) > MAX_LINE_CHARS]
    if too_long:
        die(f"a single line exceeds {MAX_LINE_CHARS} chars ({len(too_long[0])} chars); "
            f"break it into shorter lines: {too_long[0][:40]}...")
    total_chars = sum(len(ln) for ln in lines)
    if total_chars > MAX_TOTAL_CHARS:
        die(f"text too long: {total_chars} chars > {MAX_TOTAL_CHARS} limit")

    emotion_name = pick_emotion(sections.get("情緒"))
    weight = pick_weight(sections.get("情緒強度 (0.1 - 1.0)"))
    vector = EMOTION_PRESETS[emotion_name]

    tasks = build_tasks(lines, vector, weight)
    chunks = chunk_tasks(tasks, CHUNK_MAX_CHARS)

    for i, chunk in enumerate(chunks):
        with open(f"batch_chunk_{i}.jsonl", "w", encoding="utf-8") as f:
            for t in chunk:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    with open("chunk_count.txt", "w", encoding="utf-8") as f:
        f.write(str(len(chunks)))

    print(f"parsed {len(tasks)} lines / {total_chars} chars into {len(chunks)} chunk(s), "
          f"emotion={emotion_name}, weight={weight}", file=sys.stderr)


if __name__ == "__main__":
    main()
