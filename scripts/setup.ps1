$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating the repository environment with Python 3.12..."
    py -3.12 -m venv (Join-Path $repoRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv with Python 3.12." }
}

$version = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { throw "Could not run .venv Python. Inspect .venv before rebuilding it." }
if ($version -ne "3.12") { throw ".venv uses Python $version; this project setup expects Python 3.12. Rename the existing .venv if you want to preserve it, then rerun this script." }

$venvConfig = Get-Content (Join-Path $repoRoot ".venv\pyvenv.cfg") -Raw
if ($venvConfig -match "(?im)^include-system-site-packages\s*=\s*true") {
    throw ".venv can see system site packages. Preserve it for inspection, rename it, then rerun this script to create an isolated environment."
}

$lockFile = Join-Path $repoRoot "requirements-lock.txt"
$dependencyFile = if (Test-Path $lockFile) { $lockFile } else { Join-Path $repoRoot "requirements.txt" }
Write-Host "Installing dependencies from $([System.IO.Path]::GetFileName($dependencyFile))..."
& $venvPython -m pip install -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "The environment has incompatible dependencies." }

& $venvPython -m ipykernel install --prefix (Join-Path $repoRoot ".venv") --name "data-driven-dynamics-and-control" --display-name "Python (data-driven-dynamics-and-control)"
if ($LASTEXITCODE -ne 0) { throw "Could not register the project Jupyter kernel." }

Write-Host "Environment ready. Interpreter: $venvPython"
