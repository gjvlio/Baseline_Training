# Sequential capped training of all 5 ACE-Net runs to fit an ~8h window.
# Schedule (epochs/patience) is shortened from the paper's 50/25 to meet a
# deadline; architecture is unchanged. Documented in FIDELITY.md.
$ErrorActionPreference = "Continue"
$py = "D:\Documents\Programming\Baseline_Training\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\Documents\Programming\Baseline_Training"
Set-Location "D:\Documents\Programming\Baseline_Training"

# Self-log via transcript so the launcher does not need an external redirect
# (avoids file-lock contention on the log).
$logfile = "D:\Documents\Programming\Baseline_Training\checkpoints\run_all.log"
Start-Transcript -Path $logfile -Force | Out-Null

function Run($name, $argline) {
    Write-Output "================================================================"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] START $name"
    Write-Output "  args: $argline"
    Write-Output "================================================================"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $parts = $argline -split '\s+' | Where-Object { $_ -ne "" }
    try {
        & $py -m @parts
        $code = $LASTEXITCODE
    } catch {
        $code = -1
        Write-Output "EXCEPTION: $_"
    }
    $sw.Stop()
    if ($code -eq 0) {
        Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] DONE  $name  ($([int]$sw.Elapsed.TotalMinutes) min)"
    } else {
        # A crashed run (OOM / pagefile) must not abort the chain. Log and move on.
        Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] FAILED $name  exit=$code  ($([int]$sw.Elapsed.TotalMinutes) min) -- continuing"
    }
    Write-Output ""
}

$overall = [Diagnostics.Stopwatch]::StartNew()

# FULL DATA (CREMA genuine 7438 incl. FirstHalf, MELD 10907).
# Two-layer 8h guarantee: epoch caps sized to estimate + a hard per-run
# --max-minutes wall-clock cap that stops mid-run if the estimate is wrong.
# Per-run budget sums to ~430 min (<480 = 8h), leaving buffer for eval.
# 1. CREMA visual (feeds Stage-2)
Run "CREMA visual"      "src.train_stage1 --branch visual      --dataset crema --batch-size 16 --epochs 12 --early-stop 5 --max-minutes 80  --num-workers 0"
# 2. CREMA speech (BERT heavy, small batch)
Run "CREMA speech_text" "src.train_stage1 --branch speech_text --dataset crema --batch-size 8  --epochs 6  --early-stop 3 --max-minutes 95  --num-workers 0"
# 3. MELD visual
Run "MELD visual"       "src.train_stage1 --branch visual      --dataset meld  --batch-size 16 --epochs 8  --early-stop 4 --max-minutes 75  --num-workers 0"
# 4. MELD speech
Run "MELD speech_text"  "src.train_stage1 --branch speech_text --dataset meld  --batch-size 8  --epochs 4  --early-stop 2 --max-minutes 90  --num-workers 0"
# 5. Stage-2 (needs CREMA stage-1 above; full genuine now)
Run "Stage-2 ACE-Net"   "src.train_stage2 --batch-size 8 --epochs 8  --early-stop 4 --max-minutes 90  --num-workers 0"

$overall.Stop()
Write-Output "ALL RUNS COMPLETE in $([int]$overall.Elapsed.TotalMinutes) min"
Stop-Transcript | Out-Null
