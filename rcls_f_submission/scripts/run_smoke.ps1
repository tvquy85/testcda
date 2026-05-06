$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SubmissionRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $SubmissionRoot
$LogDir = Join-Path $SubmissionRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $SubmissionRoot "results") | Out-Null
$env:PYTHONDONTWRITEBYTECODE = "1"

$Train = Join-Path $SubmissionRoot "code/src/train.py"
$Log = Join-Path $LogDir "smoke_rcls_f_k3_SP500_seed0.log"

& python $Train `
  --dataset SP500 `
  --model rcls_f_k3 `
  --seed 0 `
  --numpy-seed 123456789 `
  --torch-seed 12345678 `
  --epochs 2 `
  --patience 1 `
  --require-gpu 3090 `
  --dataset-root (Join-Path $RepoRoot "dataset") `
  --output-root $SubmissionRoot `
  --activation hardswish `
  --main-mixer-activation hardswish `
  --scale-mixer-activation gelu `
  --stock-activation hardswish `
  2>&1 | Tee-Object -FilePath $Log

& python (Join-Path $ScriptDir "summarize_results.py") --output-root $SubmissionRoot
& python (Join-Path $ScriptDir "evaluate_stress.py") --output-root $SubmissionRoot
& python (Join-Path $ScriptDir "evaluate_selective.py") --output-root $SubmissionRoot
& python (Join-Path $ScriptDir "profile_efficiency.py") --output-root $SubmissionRoot
