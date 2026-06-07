# Run all ACE-Net evaluations after training completes.
# Reproduces paper Tables 2/3 (emotion) and Table 4 (forgery detection).
# Each eval re-derives the identical seeded 80/10/10 split and scores the
# held-out test portion only. Results print to console and mirror to
# checkpoints/eval_*.log.
$ErrorActionPreference = "Continue"
$py = "D:\Documents\Programming\Baseline_Training\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\Documents\Programming\Baseline_Training"
Set-Location "D:\Documents\Programming\Baseline_Training"

function Eval($name, $argline) {
    Write-Output "================================================================"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] EVAL $name"
    Write-Output "================================================================"
    $parts = $argline -split '\s+' | Where-Object { $_ -ne "" }
    & $py -m @parts
    Write-Output ""
}

# Table 2 - speech-text emotion recognition
Eval "Table2 speech-text CREMA" "src.eval_stage1 --branch speech_text --dataset crema"
Eval "Table2 speech-text MELD"  "src.eval_stage1 --branch speech_text --dataset meld"

# Table 3 - facial emotion recognition
Eval "Table3 visual CREMA"      "src.eval_stage1 --branch visual --dataset crema"
Eval "Table3 visual MELD"       "src.eval_stage1 --branch visual --dataset meld"

# Table 4 - forgery detection by pairing type (headline result)
Eval "Table4 Stage-2"           "src.eval_stage2"

Write-Output "ALL EVALUATIONS COMPLETE."
Write-Output "Visualize: jupyter notebook notebooks/results.ipynb"
