@echo off
REM ==========================================================
REM  Part 1: Sift vs Transformer on GPU
REM  Runs Shakespeare, WikiText-2, benchmarks, plots, and samples.
REM  Stops cleanly when Part 1 is complete before SiftX (Part 2).
REM  Progress: experiments\logs\everything.log
REM ==========================================================
cd /d "%~dp0\.."
if not exist experiments\logs mkdir experiments\logs
set LOG=experiments\logs\everything.log
set PY=.venv\Scripts\python.exe
if not exist %PY% (
  echo Python environment missing - run scripts\start_gpu.bat once first.
  exit /b 1
)
echo [%date% %time%] START PART 1 >> %LOG%
%PY% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo CUDA GPU not available.
  exit /b 1
)

echo [1/3] Tests...
%PY% -m research.make_configs >> %LOG% 2>&1
%PY% -m pytest -q >> %LOG% 2>&1

echo [2/3] Part 1: Sift vs Transformer (copy, adding, Shakespeare, WikiText-2)...
for %%c in (copy adding shakespeare wikitext2) do (
  call :run gpu sift_%%c_gpu sift_%%c
  call :run gpu transformer_%%c_gpu transformer_%%c
)

echo [3/3] Part 1 benchmarks and plots...
if not exist experiments\results\benchmarks\latency_cuda.json %PY% -m benchmarks.latency > experiments\logs\bench_latency_gpu.log 2>&1
if not exist experiments\results\benchmarks\memory_cuda.json %PY% -m benchmarks.memory > experiments\logs\bench_memory_gpu.log 2>&1
if not exist experiments\results\benchmarks\sequence_scaling_cuda.json %PY% -m benchmarks.sequence_scaling > experiments\logs\bench_scaling_gpu.log 2>&1
%PY% -m benchmarks.plots --device cuda >> %LOG% 2>&1
if not exist experiments\results\sample_sift_gpu.txt %PY% -m training.evaluate experiments\results\sift_shakespeare_gpu\ckpt.pt --prompt "ROMEO:" > experiments\results\sample_sift_gpu.txt 2>&1
if not exist experiments\results\sample_transformer_gpu.txt %PY% -m training.evaluate experiments\results\transformer_shakespeare_gpu\ckpt.pt --prompt "ROMEO:" > experiments\results\sample_transformer_gpu.txt 2>&1
echo [%date% %time%] PART 1 DONE >> %LOG%
echo Part 1 finished!
exit /b 0

REM ---- :run <config folder> <run name> <config file name> ----
:run
if exist experiments\results\%2\summary.json (
  echo     done already: %2
  exit /b 0
)
if exist experiments\results\%2\ckpt.pt (
  echo     resuming: %2
  echo [%date% %time%] RESUME %2 >> %LOG%
  %PY% -m training.train configs\%1\%3.yaml --resume experiments\results\%2\ckpt.pt >> experiments\logs\%2.log 2>&1
) else (
  echo     training: %2
  echo [%date% %time%] TRAIN %2 >> %LOG%
  %PY% -m training.train configs\%1\%3.yaml > experiments\logs\%2.log 2>&1
)
exit /b 0

