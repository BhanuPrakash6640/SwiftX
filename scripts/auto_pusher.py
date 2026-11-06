"""
Auto-pusher daemon: watches for newly completed experiment results,
commits them, and pushes them to GitHub automatically.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

print("[Auto-Pusher] Watching for newly completed experiment results...", flush=True)

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def check_and_push():
    # Stage newly created summary, metrics, and plot files
    run_cmd("git add experiments/results/*/summary.json experiments/results/*/metrics.jsonl experiments/results/research/ experiments/results/plots/ docs/ README.md frontend/")

    # Check if anything is staged
    staged = run_cmd("git diff --name-only --cached")
    if staged.stdout.strip():
        files = [f.strip() for f in staged.stdout.strip().split("\n") if f.strip()]
        summary_files = [f for f in files if "summary.json" in f]
        if summary_files:
            models = [os.path.basename(os.path.dirname(f)) for f in summary_files]
            msg = f"feat(results): add completed run(s) for {', '.join(models)}"
        else:
            msg = "chore(sync): update research artifacts and metrics"

        print(f"[Auto-Pusher] Staged files: {files}", flush=True)
        print(f"[Auto-Pusher] Committing with: '{msg}'", flush=True)
        c_res = run_cmd(f'git commit -m "{msg}"')
        if c_res.returncode == 0:
            print("[Auto-Pusher] Pushing to origin main...", flush=True)
            p_res = run_cmd("git push origin main")
            print(f"[Auto-Pusher] Push output:\n{p_res.stdout}{p_res.stderr}", flush=True)

def main():
    while True:
        try:
            check_and_push()
        except Exception as e:
            print(f"[Auto-Pusher Error]: {e}", flush=True)
        time.sleep(30)

if __name__ == "__main__":
    main()

