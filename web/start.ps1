# DeepSentinel launcher
# Run from repo root: .\web\start.ps1
# Or from web/ dir:   .\start.ps1

$root = Split-Path -Parent $PSScriptRoot
$py   = "$root\.venv\Scripts\python.exe"
$web  = "$root\web"

Write-Host "[ACE-Net] Installing web dependencies..."
& $py -m pip install -q -r "$web\requirements.txt"

Write-Host "[ACE-Net] Starting server on http://localhost:8000"
$env:PYTHONPATH = $root
Set-Location $web
& $py -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
