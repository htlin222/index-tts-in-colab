"""Runs ONCE per issue via `colab exec -f colab_job/setup.py`.

Builds the environment and downloads model weights -- the ~500-900s fixed
cost that used to be re-paid on every single-shot run. Chunked synthesis
(colab_job/synth_chunk.py) reuses this same warm session, so a long request
pays this once instead of once per chunk.

Uses IndexTTS-2.5 (IndexTeam/IndexTTS-2.5), not 2.0. There is no official
`indextts2 download`-equivalent CLI for 2.5 (indextts2 CLI's `download`
subcommand has IndexTeam/IndexTTS-2 hardcoded as MODEL_REPO_ID), so this
downloads the 2.5 repo directly via index-tts's own snapshot_download
helper (same HF/ModelScope auto-switch logic the CLI uses internally),
skipping the ~1.2GB qwen0.6bemo4-merge/ folder since we only ever pass
emo_vector, never emo_text (which is the only thing that needs it).
Auxiliary resources (w2v-bert-2.0/campplus/bigvgan/semantic_codec) are
downloaded via ensure_models_available(), which is version-agnostic --
same function IndexTTS-2's infer_v2.py uses internally.

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
MODEL_DIR = REPO / "checkpoints_25"
MODEL_REPO_ID = "IndexTeam/IndexTTS-2.5"
HF_TOKEN_FILE = WORK / "hf_token"

DOWNLOAD_SCRIPT = WORK / "_download_25.py"
DOWNLOAD_SCRIPT_SOURCE = f'''
import sys
sys.path.insert(0, {str(REPO)!r})
from indextts.utils.model_download import snapshot_download, ensure_models_available

print(">> snapshot_download({MODEL_REPO_ID!r}) ...", flush=True)
snapshot_download(
    {MODEL_REPO_ID!r},
    local_dir={str(MODEL_DIR)!r},
    ignore_patterns=["qwen0.6bemo4-merge/*"],
)
print(">> ensure_models_available (aux: w2v-bert-2.0/campplus/bigvgan/semantic_codec) ...", flush=True)
ensure_models_available({str(MODEL_DIR)!r})
print(">> download complete", flush=True)
'''


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


def verify_download():
    required = ["config.yaml", "gpt.pth", "s2mel.pth", "codec.pth"]
    missing = [f for f in required if not (MODEL_DIR / f).is_file()]
    if missing:
        raise RuntimeError(f"missing expected model files after download: {missing}")
    aux_dir = MODEL_DIR / "hf_cache"
    for name in ("w2v-bert-2.0", "bigvgan"):
        if not (aux_dir / name).is_dir():
            raise RuntimeError(f"missing expected aux resource dir: {aux_dir / name}")
    for name in ("campplus_cn_common.bin", "semantic_codec_model.safetensors"):
        if not (aux_dir / name).is_file():
            raise RuntimeError(f"missing expected aux resource file: {aux_dir / name}")


def main():
    setup_hf_token()

    if not REPO.is_dir():
        run(["git", "clone", "--depth", "1",
             "https://github.com/index-tts/index-tts.git", str(REPO)], timeout=180)

    run(["uv", "sync"], cwd=str(REPO), timeout=600)

    DOWNLOAD_SCRIPT.write_text(DOWNLOAD_SCRIPT_SOURCE)
    # Per-attempt timeout kept short (rather than one long attempt) so a
    # stalled download is detected and retried sooner -- see the 2026-08-16
    # incident where an unauthenticated download sat silent for 28 minutes.
    run(["uv", "run", "python", str(DOWNLOAD_SCRIPT)],
        cwd=str(REPO), timeout=600, retries=2, retry_backoff=30)

    verify_download()
    print("\n>> setup complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
