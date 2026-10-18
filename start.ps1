[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$DebugMode,
    [switch]$Chain,
    [switch]$Help,
    [int]$MaxBuilds = 0,
    [int]$MaxSwings = 0,
    [string]$LogFile = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Passthrough
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $root

if ($Help) {
    Write-Host @"
Usage: .\start.bat [-DryRun] [-DebugMode] [-Chain] [-MaxBuilds N] [-MaxSwings N] [-LogFile PATH]

Examples:
  .\start.bat
  .\start.bat -DryRun
  .\start.bat -DebugMode -MaxBuilds 10
  .\start.bat -Chain -MaxSwings 30
  .\start.bat -DebugMode -MaxSwings 30 -LogFile logs\debug_run30.log
"@
    exit 0
}

$python = Join-Path $root ".venv\Scripts\python.exe"
$requirements = Join-Path $root "requirements.txt"

if (-not (Test-Path -LiteralPath $python)) {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $systemPython) {
        throw "Python was not found in PATH. Install Python 3.10+ first."
    }
    Write-Host "[setup] Creating .venv ..." -ForegroundColor Cyan
    & $systemPython.Source -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv (exit code $LASTEXITCODE)."
    }
}

& $python -c "import cv2, mss, numpy, yaml, win32gui" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup] Installing dependencies ..." -ForegroundColor Cyan
    & $python -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install requirements (exit code $LASTEXITCODE)."
    }
}

$arguments = @()
if ($DryRun) { $arguments += "--dry-run" }
if ($DebugMode) { $arguments += "--debug" }
if ($Chain) { $arguments += "--chain" }
if ($MaxBuilds -gt 0) { $arguments += @("--max-builds", "$MaxBuilds") }
if ($MaxSwings -gt 0) { $arguments += @("--max-swings", "$MaxSwings") }
if ($LogFile) { $arguments += @("--log-file", $LogFile) }
if ($Passthrough) { $arguments += $Passthrough }

Write-Host "[run] $python auto_build.py $($arguments -join ' ')" -ForegroundColor Cyan
& $python (Join-Path $root "auto_build.py") @arguments
exit $LASTEXITCODE
