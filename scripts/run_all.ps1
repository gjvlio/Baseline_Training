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
    & $py -m @parts
    $sw.Stop()
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] DONE  $name  ($([int]$sw.Elapsed.TotalMinutes) min)"
    Write-Output ""
}

$overall = [Diagnostics.Stopwatch]::StartNew()

# Dedup fix: CREMA genuine = 3771 (was double-counted 7542), MELD = 10907.
# Smaller real sets -> can afford more epochs within the ~8h budget.
# 1. CREMA visual (feeds Stage-2) - 3771 samples, fast
Run "CREMA visual"      "src.train_stage1 --branch visual      --dataset crema --batch-size 16 --epochs 15 --early-stop 6 --num-workers 2"
# 2. CREMA speech (feeds Stage-2) - BERT heavy, small batch
Run "CREMA speech_text" "src.train_stage1 --branch speech_text --dataset crema --batch-size 8  --epochs 12 --early-stop 5 --num-workers 2"
# 3. MELD visual
Run "MELD visual"       "src.train_stage1 --branch visual      --dataset meld  --batch-size 16 --epochs 10 --early-stop 5 --num-workers 2"
# 4. MELD speech
Run "MELD speech_text"  "src.train_stage1 --branch speech_text --dataset meld  --batch-size 8  --epochs 8 --early-stop 4 --num-workers 2"
# 5. Stage-2 (needs CREMA stage-1 done above)
Run "Stage-2 ACE-Net"   "src.train_stage2 --batch-size 8 --epochs 20 --early-stop 6 --num-workers 2"

$overall.Stop()
Write-Output "ALL RUNS COMPLETE in $([int]$overall.Elapsed.TotalMinutes) min"
Stop-Transcript | Out-Null
