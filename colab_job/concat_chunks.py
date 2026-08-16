"""Runs once, after all chunks are synthesized, via
`colab exec -f colab_job/concat_chunks.py`.

Concatenates /content/chunk_0.wav .. chunk_{N-1}.wav into /content/output.wav
by raw PCM frame concatenation (same approach indextts2's own --concat uses
internally). No extra silence is inserted between chunks: each chunk's own
indextts2 batch --concat call already wrote its last task's silence_after_ms
as trailing silence, since indextts2's _write_concat_wav writes that pause
after every task unconditionally, including the last one in a call -- so the
punctuation-based pause at a chunk boundary already lives inside the
preceding chunk's own wav file. See scripts/parse_issue.py's chunk_tasks().

Expects:
  /content/chunk_count.txt  -- how many chunk_N.wav files to expect (via
                                `colab upload`)
Produces:
  /content/output.wav
"""
import sys
import wave
from pathlib import Path

WORK = Path("/content")
CHUNK_COUNT_FILE = WORK / "chunk_count.txt"
OUTPUT_FILE = WORK / "output.wav"


def main():
    assert CHUNK_COUNT_FILE.is_file(), f"missing {CHUNK_COUNT_FILE}"
    n = int(CHUNK_COUNT_FILE.read_text().strip())
    assert n >= 1, f"chunk_count.txt has non-positive value: {n}"

    chunk_paths = [WORK / f"chunk_{i}.wav" for i in range(n)]
    missing = [p for p in chunk_paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"missing chunk wav(s): {missing}")

    if len(chunk_paths) == 1:
        chunk_paths[0].rename(OUTPUT_FILE)
        print(f">> single chunk, renamed directly to {OUTPUT_FILE}")
        return

    with wave.open(str(chunk_paths[0]), "rb") as first:
        params = first.getparams()

    with wave.open(str(OUTPUT_FILE), "wb") as out:
        out.setparams(params)
        for p in chunk_paths:
            with wave.open(str(p), "rb") as w:
                fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
                expected = (params.framerate, params.nchannels, params.sampwidth)
                if fmt != expected:
                    raise RuntimeError(f"{p} format {fmt} != expected {expected}")
                out.writeframes(w.readframes(w.getnframes()))

    size = OUTPUT_FILE.stat().st_size
    print(f">> concatenated {n} chunks -> {OUTPUT_FILE} ({size} bytes)")
    if size < 1000:
        raise RuntimeError(f"output.wav suspiciously small ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
