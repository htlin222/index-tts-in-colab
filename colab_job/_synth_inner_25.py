"""Runs inside the project's uv-managed venv via
`uv run python _synth_inner_25.py <batch_file> <seg_dir> <model_dir> <prefix>`.

There is no official CLI wrapping indextts.infer_v2_5 (only infer_v2 has
cli_v2.py), so this calls it directly. One line per file, matching the
naming indextts2's own CLI uses for --output-dir mode (<prefix>-0001.wav,
...) so downstream code doesn't need to know which version produced them.

batch_file is a JSONL of tasks (same shape scripts/parse_issue.py writes:
text/voice/emotion_vector/emotion_weight/silence_after_ms) -- this script
only reads the fields it needs and ignores silence_after_ms (that's
handled later by _common.py's concat_wavs_with_fade), so no filtered copy
is needed the way cli_v2.py's stricter batch-file validator required.
"""
import json
import sys
from pathlib import Path


def main():
    batch_file, seg_dir, model_dir, prefix = sys.argv[1:5]
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=str(Path(model_dir) / "config.yaml"),
        model_dir=model_dir,
        use_bf16=False,       # T4 (Turing) has no real bf16 acceleration
        use_cuda_kernel=False,  # avoid a runtime CUDA-kernel compile step
        use_qwen_emo=False,   # we only ever pass emo_vector, never emo_text
    )

    with open(batch_file, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    for i, task in enumerate(tasks, start=1):
        out_path = seg_dir / f"{prefix}-{i:04d}.wav"
        print(f">>> line {i}/{len(tasks)}: {task['text'][:30]}", flush=True)
        tts.infer(
            spk_audio_prompt=task["voice"],
            text=task["text"],
            output_path=str(out_path),
            lang="ZH",
            emo_vector=task.get("emotion_vector"),
            emo_alpha=task.get("emotion_weight", 0.85),
            verbose=True,
        )
        if not out_path.is_file():
            raise RuntimeError(f"line {i} did not produce {out_path}")
        print(f"Generated: {out_path}", flush=True)

    print(f">> all {len(tasks)} lines done", flush=True)


if __name__ == "__main__":
    main()
