"""Runs ONCE per issue via `colab exec -f colab_job/setup.py`.

Builds the environment and downloads model weights -- the ~500-900s fixed
cost that used to be re-paid on every single-shot run. Chunked synthesis
(colab_job/synth_chunk.py) reuses this same warm session, so a long request
pays this once instead of once per chunk.

Supports both IndexTTS-2 and IndexTTS-2.5, chosen by the uploaded
/content/model_version.txt marker ("2.0" or "2.5", default "2.0" if the
file is missing). 2.5 sounded like it added a Mainland-Mandarin-leaning
accent for a Taiwanese reference voice (2026-08-16 user feedback, traced
to the `lang="zh"` language-conditioning token infer_v2_5.py requires --
infer_v2.py/2.0 has no such token at all and just clones whatever accent
is in the reference clip), so 2.0 is the default; 2.5 stays available for
its speed/feature advantages when accent fidelity isn't the priority.

- 2.0: downloaded via the official `indextts2 download` CLI (hardcoded to
  IndexTeam/IndexTTS-2, no way to skip the unused QwenEmotion folder).
- 2.5: no official CLI wraps infer_v2_5.py, so this downloads
  IndexTeam/IndexTTS-2.5 directly via index-tts's own snapshot_download/
  ensure_models_available helpers, skipping the QwenEmotion folder since
  we only ever pass emo_vector, never emo_text.

Expects:
  /content/_common.py       -- shared run() helper (via `colab upload`)
  /content/model_version.txt -- "2.0" or "2.5" (via `colab upload`)
  /content/hf_token          -- optional HuggingFace token (via `colab upload`),
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
MODEL_VERSION_FILE = WORK / "model_version.txt"
HF_TOKEN_FILE = WORK / "hf_token"
MODEL_REPO_25 = "IndexTeam/IndexTTS-2.5"

DOWNLOAD_SCRIPT_25 = WORK / "_download_25.py"


def read_model_version():
    if not MODEL_VERSION_FILE.is_file():
        return "2.0"
    v = MODEL_VERSION_FILE.read_text().strip()
    return v if v in ("2.0", "2.5") else "2.0"


def model_dir_for(version):
    return REPO / ("checkpoints_25" if version == "2.5" else "checkpoints_2")


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


def download_20(model_dir):
    # Per-attempt timeout kept short (rather than one long attempt) so a
    # stalled download is detected and retried sooner -- see the 2026-08-16
    # incident where an unauthenticated download sat silent for 28 minutes.
    run(["uv", "run", "indextts2", "download", "--model-dir", str(model_dir)],
        cwd=str(REPO), timeout=600, retries=2, retry_backoff=30)
    run(["uv", "run", "indextts2", "check", "--model-dir", str(model_dir),
         "--device", "cuda"], cwd=str(REPO), timeout=90)


def download_25(model_dir):
    DOWNLOAD_SCRIPT_25.write_text(f'''
import sys
sys.path.insert(0, {str(REPO)!r})
from indextts.utils.model_download import snapshot_download, ensure_models_available

print(">> snapshot_download({MODEL_REPO_25!r}) ...", flush=True)
snapshot_download(
    {MODEL_REPO_25!r},
    local_dir={str(model_dir)!r},
    ignore_patterns=["qwen0.6bemo4-merge/*"],
)
print(">> ensure_models_available (aux: w2v-bert-2.0/campplus/bigvgan/semantic_codec) ...", flush=True)
ensure_models_available({str(model_dir)!r})
print(">> download complete", flush=True)
''')
    run(["uv", "run", "python", str(DOWNLOAD_SCRIPT_25)],
        cwd=str(REPO), timeout=600, retries=2, retry_backoff=30)

    required = ["config.yaml", "gpt.pth", "s2mel.pth", "codec.pth"]
    missing = [f for f in required if not (model_dir / f).is_file()]
    if missing:
        raise RuntimeError(f"missing expected model files after download: {missing}")
    aux_dir = model_dir / "hf_cache"
    for name in ("w2v-bert-2.0", "bigvgan"):
        if not (aux_dir / name).is_dir():
            raise RuntimeError(f"missing expected aux resource dir: {aux_dir / name}")
    for name in ("campplus_cn_common.bin", "semantic_codec_model.safetensors"):
        if not (aux_dir / name).is_file():
            raise RuntimeError(f"missing expected aux resource file: {aux_dir / name}")


def main():
    setup_hf_token()
    version = read_model_version()
    model_dir = model_dir_for(version)
    print(f">> model version: {version} -> {model_dir}")

    if not REPO.is_dir():
        run(["git", "clone", "--depth", "1",
             "https://github.com/index-tts/index-tts.git", str(REPO)], timeout=180)

    run(["uv", "sync"], cwd=str(REPO), timeout=600)

    if version == "2.5":
        download_25(model_dir)
    else:
        download_20(model_dir)

    print("\n>> setup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
