# Murshid demo launcher (Windows PowerShell):  .\run_demo.ps1
# UTF-8 -> venv -> preflight checks (also warms the answer cache) -> server on http://localhost:8000
# If PowerShell blocks scripts:  powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "No .venv found. Run: py -3.11 -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
. .\.venv\Scripts\Activate.ps1

python -m scripts.preflight
if ($LASTEXITCODE -ne 0) {
    Write-Host "Preflight has red items. The app still works (fallback mode quotes the source)." -ForegroundColor Yellow
    $ans = Read-Host "Start the server anyway? (y/n)"
    if ($ans -ne "y") { exit 1 }
}

Write-Host "`nStarting http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Green
python -m uvicorn app.server:app --port 8000
