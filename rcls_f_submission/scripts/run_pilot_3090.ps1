$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SubmissionRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $SubmissionRoot
$LogDir = Join-Path $SubmissionRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $SubmissionRoot "results") | Out-Null
$env:PYTHONDONTWRITEBYTECODE = "1"

$Train = Join-Path $SubmissionRoot "code/src/train.py"
$Datasets = @("NASDAQ", "SP500")
$Models = @("stockmixer", "rcls_f_k1", "rcls_f_k3")

foreach ($Dataset in $Datasets) {
  foreach ($Model in $Models) {
    $Log = Join-Path $LogDir ("{0}_{1}_seed0.log" -f $Model, $Dataset)
    Write-Host "Running $Model on $Dataset"
    & python $Train `
      --dataset $Dataset `
      --model $Model `
      --seed 0 `
      --numpy-seed 123456789 `
      --torch-seed 12345678 `
      --epochs 60 `
      --patience 8 `
      --require-gpu 3090 `
      --dataset-root (Join-Path $RepoRoot "dataset") `
      --output-root $SubmissionRoot `
      --activation hardswish `
      --main-mixer-activation hardswish `
      --scale-mixer-activation gelu `
      --stock-activation hardswish `
      2>&1 | Tee-Object -FilePath $Log
  }
}

& python (Join-Path $ScriptDir "summarize_results.py") --output-root $SubmissionRoot
& python (Join-Path $ScriptDir "evaluate_stress.py") --output-root $SubmissionRoot
& python (Join-Path $ScriptDir "evaluate_selective.py") --output-root $SubmissionRoot
& python (Join-Path $ScriptDir "profile_efficiency.py") --output-root $SubmissionRoot
