#!/usr/bin/env bash
# Reproduce every experiment in the README. Run from the repo root.
set -e
for cfg in copy adding shakespeare; do
  python -m training.train configs/sift_${cfg}.yaml
  python -m training.train configs/transformer_${cfg}.yaml
done
python -m benchmarks.latency
python -m benchmarks.memory
python -m benchmarks.sequence_scaling
python -m benchmarks.plots --device cpu
