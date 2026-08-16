"""Shared helper for the Colab-side scripts. Uploaded once via `colab upload`
to /content/_common.py so setup.py and synth_chunk.py can both import it
(colab exec sends a script's *content* to the kernel directly, it doesn't
place the file on the VM's disk -- only `colab upload` does that -- so a
plain sibling-module import wouldn't otherwise be reliable across separate
colab exec calls).
"""
import subprocess
import sys
import time
from pathlib import Path

STEP_LOG = Path("/content/_step.log")
HEARTBEAT_SECONDS = 60
STALL_HEARTBEATS_BEFORE_WARNING = 5  # 5 min of zero new output -> call it out


def run(cmd, cwd=None, timeout=600, retries=0, retry_backoff=30):
    """Run cmd with a heartbeat + hard timeout. On failure (timeout or non-zero
    exit), retry up to `retries` more times with a fixed backoff. Safe to
    retry for network-bound steps like model download: huggingface_hub
    resumes partially-downloaded blobs from its local cache instead of
    restarting from zero.
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
                    # SIGTERM first, not SIGKILL: a killed huggingface_hub
                    # download can leave stale .lock files in its cache dir,
                    # which then makes the *next* retry attempt hang waiting
                    # on a lock nobody holds -- plausibly what happened on
                    # 2026-08-16 (2 of 3 retries stalled at exactly 58 bytes
                    # for the full timeout with zero growth). Give it a
                    # chance to clean up before escalating.
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
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
