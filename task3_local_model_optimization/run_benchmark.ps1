# run_benchmark.ps1 – PowerShell benchmark runner for Windows
# Usage:
#   .\run_benchmark.ps1
#   .\run_benchmark.ps1 -Model1 "llama3.2:1b" -Model2 "llama3.2:1b:q4_0"

param(
    [string]$Model1       = "llama3.2:1b",
    [string]$Model2       = "llama3.2:1b:q4_0",
    [string]$ResultsFile  = "benchmark_results.json",
    [string]$ReportFile   = "analysis_report.md",
    [switch]$SkipPull,
    [switch]$SkipAnalysis
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ── Colour helpers ────────────────────────────────────────────────────────────
function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green  }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red    }

Write-Host ""
Write-Host "╔════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Task 3 – Local Model Optimisation Benchmark   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Resolve working directory ─────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

# ── 1. Locate Python ──────────────────────────────────────────────────────────
$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $PythonCmd = $cmd
        break
    }
}
if (-not $PythonCmd) {
    Write-Err "Python not found. Install Python 3.9+ and add it to PATH."
    exit 1
}
$PythonVer = & $PythonCmd --version 2>&1
Write-Info "Python: $PythonVer"

# ── 2. Install dependencies ───────────────────────────────────────────────────
Write-Info "Installing Python dependencies from requirements.txt…"
& $PythonCmd -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Err "pip install failed. Check requirements.txt and your Python environment."
    exit 1
}
Write-Info "Dependencies OK."

# ── 3. Check Ollama installed ─────────────────────────────────────────────────
$OllamaExe = Get-Command "ollama" -ErrorAction SilentlyContinue
if (-not $OllamaExe) {
    Write-Warn "ollama not found on PATH. Download from: https://ollama.com/download"
    Write-Warn "Proceeding; Ollama server may already be running."
}

# ── 4. Check / start Ollama server ────────────────────────────────────────────
$OllamaURL = "http://localhost:11434"

function Test-OllamaRunning {
    try {
        $resp = Invoke-WebRequest -Uri "$OllamaURL/api/tags" -UseBasicParsing -TimeoutSec 5
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (Test-OllamaRunning) {
    Write-Info "Ollama server already running at $OllamaURL."
} else {
    if ($OllamaExe) {
        Write-Info "Starting Ollama server in background…"
        $OllamaLog = "$env:TEMP\ollama_serve.log"
        $OllamaProc = Start-Process -FilePath "ollama" -ArgumentList "serve" `
            -RedirectStandardOutput $OllamaLog `
            -RedirectStandardError  $OllamaLog `
            -WindowStyle Hidden -PassThru
        Write-Info "Ollama PID: $($OllamaProc.Id)  (log: $OllamaLog)"

        $ready = $false
        for ($i = 1; $i -le 30; $i++) {
            Start-Sleep -Seconds 1
            if (Test-OllamaRunning) {
                Write-Info "Ollama ready after ${i}s."
                $ready = $true
                break
            }
        }
        if (-not $ready) {
            Write-Err "Ollama did not start within 30 s. See: $OllamaLog"
            exit 1
        }
    } else {
        Write-Err "Ollama not installed and not running. Please install and start it."
        exit 1
    }
}

# ── 5. Pull models if needed ──────────────────────────────────────────────────
function Invoke-PullModel {
    param([string]$Model)
    if ($OllamaExe) {
        $listOutput = & ollama list 2>&1
        $baseName   = $Model.Split(":")[0]
        if ($listOutput -match [regex]::Escape($baseName)) {
            Write-Info "Model '$Model' appears to be present locally. Skipping pull."
        } else {
            Write-Info "Pulling model '$Model'…"
            & ollama pull $Model
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Pull returned non-zero exit for '$Model'. Will retry via benchmark.py."
            } else {
                Write-Info "Model '$Model' ready."
            }
        }
    } else {
        Write-Warn "Cannot pull '$Model' (ollama CLI missing). benchmark.py will attempt pull."
    }
}

if (-not $SkipPull) {
    Invoke-PullModel -Model $Model1
    Invoke-PullModel -Model $Model2
}

# ── 6. Run the benchmark ──────────────────────────────────────────────────────
Write-Info "Starting benchmark: $Model1  vs  $Model2"
Write-Host ""
& $PythonCmd benchmark.py `
    --model1 $Model1 `
    --model2 $Model2 `
    --out    $ResultsFile

if ($LASTEXITCODE -ne 0) {
    Write-Err "benchmark.py exited with error code $LASTEXITCODE."
    exit $LASTEXITCODE
}

# ── 7. Run analysis ───────────────────────────────────────────────────────────
if (-not $SkipAnalysis) {
    if (Test-Path $ResultsFile) {
        Write-Host ""
        Write-Info "Running analysis…"
        & $PythonCmd analysis.py `
            --results $ResultsFile `
            --out     $ReportFile

        if ($LASTEXITCODE -ne 0) {
            Write-Warn "analysis.py exited with error code $LASTEXITCODE (non-fatal)."
        }
    } else {
        Write-Warn "Results file '$ResultsFile' not found – skipping analysis."
    }
}

# ── 8. Done ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Info "All done."
Write-Info "Results : $ResultsFile"
if (-not $SkipAnalysis) {
    Write-Info "Report  : $ReportFile"
}

Pop-Location
