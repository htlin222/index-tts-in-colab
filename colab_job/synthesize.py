"""Runs on the Colab VM via `colab exec -f colab_job/synthesize.py`.

Expects, already placed on the VM before this script runs:
  /content/ref.wav      -- reference voice clip (via `colab upload`)
  /content/batch.jsonl  -- indextts2 batch tasks (via `colab upload`)
  /content/hf_token     -- optional HuggingFace token (via `colab upload`),
                           improves model-download stability/speed over
                           anonymous requests. Read once, then deleted.

Produces:
  /content/output.wav   -- concatenated synthesis result

Each step gets its own timeout enforced by *this* script (not just the
outer `colab exec --timeout`), and prints a heartbeat line periodically
so a stuck step is diagnosable instead of going silent for the whole
outer deadline. A run on 2026-08-16 spent 28 minutes with zero log
output before hitting the outer timeout during model download --
these heartbeats plus per-step timeouts are the direct fix for that.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content")
REPO = WORK / "index-tts"
MODEL_DIR = REPO / "checkpoints_2"
BATCH_FILE = WORK / "batch.jsonl"
VOICE_FILE = WORK / "ref.wav"
OUTPUT_FILE = WORK / "output.wav"
HF_TOKEN_FILE = WORK / "hf_token"
STEP_LOG = WORK / "_step.log"

HEARTBEAT_SECONDS = 60
STALL_HEARTBEATS_BEFORE_WARNING = 5  # 5 min of zero new output -> call it out


def run(cmd, cwd=None, timeout=600, retries=0, retry_backoff=30):
    """Run cmd with a heartbeat + hard timeout. On failure (timeout or non-zero
    exit), retry up to `retries` more times with a fixed backoff. Safe to
    retry for the HF download step specifically: huggingface_hub resumes
    partially-downloaded blobs from its local cache instead of restarting.
    """
    attempt = 0
    while True:
        attempt += 1
        label = f"  (attempt {attempt}/{retries + 1})" if retries else ""
        print(f"\n>>> {' '.join(cmd)}  (timeout={timeout}s){label}", flush=True)
        start = time.monotonic()
        stall_count = 0
        last_size = -1
        with open(STEP_LOG, "wb") as logf:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
            timed_out = False
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    break
                time.sleep(HEARTBEAT_SECONDS)
                size = STEP_LOG.stat().st_size
                grew = size != last_size
                stall_count = 0 if grew else stall_count + 1
                note = "" if grew else f"  [no new output for {stall_count * HEARTBEAT_SECONDS}s]"
                print(f"    ... still running ({elapsed:.0f}s elapsed, {size} bytes so far){note}", flush=True)
                if stall_count >= STALL_HEARTBEATS_BEFORE_WARNING:
                    print(f"    ⚠ possible stall: no output growth for "
                          f"{stall_count * HEARTBEAT_SECONDS}s (will still respect the {timeout}s timeout)", flush=True)
                last_size = size

        _print_tail()
        if timed_out:
            err = RuntimeError(f"command exceeded {timeout}s timeout: {' '.join(cmd)}")
        elif proc.returncode != 0:
            err = RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
        else:
            return  # success

        if attempt > retries:
            raise err
        print(f"    retrying in {retry_backoff}s after: {err}", flush=True)
        time.sleep(retry_backoff)


def _print_tail(max_bytes=20000):
    data = STEP_LOG.read_bytes()
    if len(data) > max_bytes:
        print(f"[... truncated, showing last {max_bytes} bytes of {len(data)} ...]")
        data = data[-max_bytes:]
    sys.stdout.write(data.decode(errors="replace"))
    sys.stdout.flush()


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


def count_batch_tasks():
    with open(BATCH_FILE, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    assert BATCH_FILE.is_file(), f"missing {BATCH_FILE}, did the upload step run?"
    assert VOICE_FILE.is_file(), f"missing {VOICE_FILE}, did the upload step run?"

    setup_hf_token()

    if not REPO.is_dir():
        run(["git", "clone", "--depth", "1",
             "https://github.com/index-tts/index-tts.git", str(REPO)], timeout=180)

    run(["uv", "sync"], cwd=str(REPO), timeout=600)
    run(["uv", "run", "indextts2", "download", "--model-dir", str(MODEL_DIR)],
        cwd=str(REPO), timeout=1200, retries=2, retry_backoff=30)
    run(["uv", "run", "indextts2", "check", "--model-dir", str(MODEL_DIR),
         "--device", "cuda"], cwd=str(REPO), timeout=90)

    num_tasks = count_batch_tasks()
    batch_timeout = 120 + 90 * num_tasks
    run(["uv", "run", "indextts2", "batch",
         "--batch-file", str(BATCH_FILE),
         "--model-dir", str(MODEL_DIR),
         "--concat", "--output", str(OUTPUT_FILE),
         "--no-cuda-kernel", "--force", "--verbose"],
        cwd=str(REPO), timeout=batch_timeout)

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
