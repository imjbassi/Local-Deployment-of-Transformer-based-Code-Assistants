param(
    [string]$Tectonic = "tectonic"
)

$ErrorActionPreference = "Stop"
$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $paperDir "output\pdf"
$buildDir = Join-Path $paperDir "build"
$finalPdf = Join-Path $outputDir "Do_Published_HumanEval_Rankings_Survive_Local_Deployment.pdf"

New-Item -ItemType Directory -Force -Path $outputDir, $buildDir | Out-Null

Push-Location $paperDir
try {
    & $Tectonic -X compile main.tex --outdir $buildDir --keep-logs
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic failed with exit code $LASTEXITCODE"
    }
    Copy-Item -LiteralPath (Join-Path $buildDir "main.pdf") -Destination $finalPdf -Force
}
finally {
    Pop-Location
}

Write-Output $finalPdf
