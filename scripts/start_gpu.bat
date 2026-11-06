@echo off
REM ==========================================================
REM  Sift: full GPU run. Double-click this file.
REM  Installs CUDA PyTorch in a local .venv, runs the tests,
REM  trains every model, runs the benchmarks, makes the plots.
REM  Progress: experiments\logs\gpu_run.log
REM ==========================================================
cd /d "%~dp0\.."
if not exist experiments\logs mkdir experiments\logs
set LOG=experiments\logs\gpu_run.log
echo [%date% %time%] START > %LOG%

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or 3.12 from python.org, tick "Add to PATH", then run this again.
  echo ERROR: python not found >> %LOG%
  pause
  exit /b 1
)

echo [1/5] Setting up Python environment (first time takes a few minutes)...
set PY=python
if not exist .venv\Scripts\python.exe python -m venv .venv >> %LOG% 2>&1
.venv\Scripts\python.exe -m pip --version >nul 2>nul
if not errorlevel 1 set PY=.venv\Scripts\python.exe
echo Using Python: %PY% >> %LOG%
%PY% -m pip install --upgrade pip >> %LOG% 2>&1
%PY% -c "import torch; assert torch.cuda.is_available()" >nul 2>nul
if errorlevel 1 %PY% -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 >> %LOG% 2>&1
%PY% -m pip install numpy pyyaml matplotlib pytest pandas pyarrow >> %LOG% 2>&1
%PY% -c "import torch; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); print('torch', torch.__version__, 'bf16', ok and torch.cuda.is_bf16_supported())" >> %LOG% 2>&1
%PY% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo CUDA GPU not available to PyTorch. See %LOG%
  echo ERROR: CUDA not available >> %LOG%
  pause
  exit /b 1
)

echo [2/5] Running tests...
echo [%date% %time%] TESTS >> %LOG%
%PY% -m pytest -q >> %LOG% 2>&1

echo [3/5] Training (copy, adding, Tiny Shakespeare, WikiText-2) - this is the long part...
for %%c in (copy adding shakespeare wikitext2) do (
  echo     sift %%c
  echo [%date% %time%] TRAIN sift_%%c >> %LOG%
  %PY% -m training.train configs\gpu\sift_%%c.yaml > experiments\logs\sift_%%c_gpu.log 2>&1
  echo     transformer %%c
  echo [%date% %time%] TRAIN transformer_%%c >> %LOG%
  %PY% -m training.train configs\gpu\transformer_%%c.yaml > experiments\logs\transformer_%%c_gpu.log 2>&1
)

echo [4/5] Benchmarks...
echo [%date% %time%] BENCH >> %LOG%
%PY% -m benchmarks.latency > experiments\logs\bench_latency_gpu.log 2>&1
%PY% -m benchmarks.memory > experiments\logs\bench_memory_gpu.log 2>&1
%PY% -m benchmarks.sequence_scaling > experiments\logs\bench_scaling_gpu.log 2>&1

echo [5/5] Plots and text samples...
%PY% -m benchmarks.plots --device cuda >> %LOG% 2>&1
%PY% -m training.evaluate experiments\results\sift_shakespeare_gpu\ckpt.pt --prompt "ROMEO:" > experiments\results\sample_sift_gpu.txt 2>&1
%PY% -m training.evaluate experiments\results\transformer_shakespeare_gpu\ckpt.pt --prompt "ROMEO:" > experiments\results\sample_transformer_gpu.txt 2>&1

echo [%date% %time%] ALL DONE >> %LOG%
echo.
echo Done! Tell Claude it finished.
pause
