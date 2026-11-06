# Status

Everything is driven by one script now:

    scripts\run_everything.bat      (Windows, RTX GPU; resumable - double-click it again after any stop)

It trains the Part 1 models (Sift vs Transformer), runs the benchmarks, then the SiftX research
suite, the analysis and the plots. Results land in `experiments/results/` and the log in
`experiments/logs/everything.log`. See README.md for what each result means.
