# Convert Paradigm-2 test FLVs to MP4
# Creates two sibling folders: paradigm2_test_FLV  (moved originals)
#                              paradigm2_test_MP4  (converted output)

$src   = "D:\Documents\Programming\Baseline_Training\testing videos\paradigm2_test"
$flvDir = "D:\Documents\Programming\Baseline_Training\testing videos\paradigm2_test_FLV"
$mp4Dir = "D:\Documents\Programming\Baseline_Training\testing videos\paradigm2_test_MP4"

New-Item -ItemType Directory -Force -Path $flvDir | Out-Null
New-Item -ItemType Directory -Force -Path $mp4Dir | Out-Null

$files = Get-ChildItem -Path $src -Filter "*.flv"
$total = $files.Count
$i     = 0

foreach ($f in $files) {
    $i++
    $stem   = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    $mp4Out = Join-Path $mp4Dir "$stem.mp4"

    Write-Host "[$i/$total] $($f.Name)"

    ffmpeg -y -i $f.FullName `
        -c:v libx264 -crf 18 -preset fast `
        -c:a aac -b:a 192k `
        $mp4Out 2>$null

    if ($LASTEXITCODE -eq 0) {
        Move-Item -Path $f.FullName -Destination (Join-Path $flvDir $f.Name)
        Write-Host "         -> OK"
    } else {
        Write-Host "         -> FAILED (ffmpeg exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done. FLV originals -> $flvDir"
Write-Host "      MP4 output    -> $mp4Dir"
