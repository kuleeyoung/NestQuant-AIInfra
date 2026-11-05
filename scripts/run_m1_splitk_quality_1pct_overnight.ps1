$ErrorActionPreference = "Stop"

$workspace = "D:\code\NestQuant-AIInfra"
$logDirectory = Join-Path $workspace "results\raw\m1_splitk_quality_1pct_overnight"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "run.log"
$startedPath = Join-Path $logDirectory "started_at.txt"
Get-Date -Format o | Set-Content -Encoding utf8 $startedPath

$command = @"
cd /mnt/d/code/NestQuant-AIInfra && \
source ~/.venvs/nestquant/bin/activate && \
export CUDA_HOME=/usr/local/cuda && \
export PYTHONPATH=. && \
python scripts/preflight_m1_splitk_quality.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --verify-checksum \
  --output results/raw/m1_splitk_teacher_forced_2048x32_1pct_v1.json \
  --progress results/raw/m1_splitk_teacher_forced_2048x32_1pct_v1.progress.json && \
python scripts/evaluate_m1_splitk_teacher_forced.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --samples 2048 \
  --prompt-tokens 32 \
  --target-tokens 32 \
  --capacity 4 \
  --max-cache-len 1024 \
  --offline \
  --resume \
  --output results/raw/m1_splitk_teacher_forced_2048x32_1pct_v1.json \
  --progress results/raw/m1_splitk_teacher_forced_2048x32_1pct_v1.progress.json
"@

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& wsl.exe -e bash -lc $command *>> $logPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($exitCode -eq 0) {
    $analysisCommand = @"
cd /mnt/d/code/NestQuant-AIInfra && \
source ~/.venvs/nestquant/bin/activate && \
export PYTHONPATH=. && \
python scripts/analyze_m1_splitk_teacher_forced.py \
  results/raw/m1_splitk_teacher_forced_2048x32_1pct_v1.json \
  --bootstrap-repetitions 20000 \
  --output results/raw/m1_splitk_teacher_forced_2048x32_1pct_analysis_v1.json && \
python scripts/verify_m1_splitk_quality_gate.py \
  results/raw/m1_splitk_teacher_forced_2048x32_1pct_analysis_v1.json \
  --margin 0.01 \
  --min-contexts 2048 \
  --output results/raw/m1_splitk_teacher_forced_2048x32_1pct_gate_v1.json
"@
    $ErrorActionPreference = "Continue"
    & wsl.exe -e bash -lc $analysisCommand *>> $logPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
}
Get-Date -Format o | Set-Content -Encoding utf8 (Join-Path $logDirectory "finished_at.txt")
$exitCode | Set-Content -Encoding ascii (Join-Path $logDirectory "exit_code.txt")
exit $exitCode
