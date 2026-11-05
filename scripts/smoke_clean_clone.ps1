param(
    [string]$Python = $(if ($env:NESTQUANT_PYTHON) { $env:NESTQUANT_PYTHON } else { "python" })
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("nestquant-clean-clone-" + [guid]::NewGuid().ToString("N"))
$sourceDir = Join-Path $tempRoot "source"
$cloneDir = Join-Path $tempRoot "clone"
$wheelDir = Join-Path $tempRoot "wheel"
$installDir = Join-Path $tempRoot "install"
$outputDir = Join-Path $tempRoot "output"
New-Item -ItemType Directory -Force -Path $sourceDir, $wheelDir, $installDir, $outputDir | Out-Null

Push-Location $repoRoot
try {
    $publishableFiles = @(git -c core.quotepath=false ls-files --cached --others --exclude-standard)
    if ($LASTEXITCODE -ne 0 -or $publishableFiles.Count -eq 0) { throw "Git returned no publishable files" }
    foreach ($relativePath in $publishableFiles) {
        $sourcePath = Join-Path $repoRoot $relativePath
        $targetPath = Join-Path $sourceDir $relativePath
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath
    }
}
finally { Pop-Location }

Push-Location $sourceDir
try {
    git init -q
    git config user.name "NestQuant clean-clone validation"
    git config user.email "clean-clone@local.invalid"
    git add -A
    git commit -q -m "test: clean-clone validation snapshot"
    if ($LASTEXITCODE -ne 0) { throw "temporary Git commit failed" }
    $sourceCommit = git rev-parse HEAD
}
finally { Pop-Location }

git clone -q --no-local $sourceDir $cloneDir
if ($LASTEXITCODE -ne 0) { throw "Git clone failed" }

Push-Location $cloneDir
try {
    $dirtyBefore = git status --porcelain --untracked-files=all
    if ($dirtyBefore) { throw "fresh clone is unexpectedly dirty: $dirtyBefore" }
    $auditPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $cloneDir
        & $Python -m nestquant.cli.audit_repository --json-output (Join-Path $outputDir "repository_audit.json")
        if ($LASTEXITCODE -ne 0) { throw "repository audit failed" }
    }
    finally { $env:PYTHONPATH = $auditPythonPath }
    & $Python -m pip wheel --no-deps --wheel-dir $wheelDir .
    if ($LASTEXITCODE -ne 0) { throw "wheel build failed" }
    $wheelPath = (Get-ChildItem $wheelDir -Filter *.whl | Select-Object -First 1).FullName
    if (-not $wheelPath) { throw "wheel build produced no wheel" }
    & $Python -m pip install --no-deps --target $installDir $wheelPath
    if ($LASTEXITCODE -ne 0) { throw "independent wheel install failed" }

    $oldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $installDir
        Push-Location $tempRoot
        try {
            & $Python -c "from pathlib import Path; import nestquant; p=Path(nestquant.__file__).resolve(); expected=Path(r'$installDir').resolve(); assert expected in p.parents, (p, expected); print(f'installed package import passed: {p}')"
            if ($LASTEXITCODE -ne 0) { throw "installed-package import failed" }
        }
        finally { Pop-Location }
    }
    finally { $env:PYTHONPATH = $oldPythonPath }

    & $Python scripts/demo_e8.py --device cpu
    if ($LASTEXITCODE -ne 0) { throw "E8 CPU smoke failed" }
    & $Python scripts/demo_beta_search.py --device cpu
    if ($LASTEXITCODE -ne 0) { throw "beta-search CPU smoke failed" }
    & $Python scripts/run_synthetic_matrix_product.py --device cpu --output (Join-Path $outputDir "synthetic_matrix_product.json")
    if ($LASTEXITCODE -ne 0) { throw "synthetic matrix-product CPU smoke failed" }

    $commands = @(
        "nestquant-audit-repository", "nestquant-validate-service-benchmark",
        "nestquant-promote-service-benchmark", "nestquant-verify-service-evidence",
        "nestquant-verify-splitk-quality", "nestquant-validate-hybrid-pareto",
        "nestquant-promote-hybrid-pareto"
    )
    $entryPointDir = if (Test-Path (Join-Path $installDir "bin")) {
        Join-Path $installDir "bin"
    }
    else {
        Join-Path $installDir "Scripts"
    }
    foreach ($command in $commands) {
        $commandPath = Join-Path $entryPointDir ($command + ".exe")
        if (-not (Test-Path $commandPath)) { throw "missing console entry point: $command" }
        $entryPointPythonPath = $env:PYTHONPATH
        try {
            $env:PYTHONPATH = $installDir
            & $commandPath --help | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "console entry point failed: $command" }
        }
        finally { $env:PYTHONPATH = $entryPointPythonPath }
    }

    $dirtyAfter = git status --porcelain --untracked-files=all
    if ($dirtyAfter) { throw "clean clone became dirty during validation: $dirtyAfter" }
}
finally { Pop-Location }

Write-Host "clean-clone validation passed"
Write-Host "source_commit=$sourceCommit"
Write-Host "clone_dir=$cloneDir"
Write-Host "wheel_path=$wheelPath"
Write-Host "install_dir=$installDir"
Write-Host "smoke_output_dir=$outputDir"
