# Optional: Pfad zum Zielordner
$targetDir = "$HOME\oct-analysis"
$repoUrl = "https://github.com/AndreasNetsch/oct_analysis.git"
$packageName = "oct-analysis"

# github repo oct-analysis klonen
if (-not (Test-Path $targetDir)) {
    Write-Host "⬇️ Klone Repository $packageName ..."
    git clone $repoUrl $targetDir
} else {
    Write-Host "📁 Ordner '$targetDir' existiert bereits. Skipping clone."
}
Set-Location $targetDir

# venv erstellen
uv venv
Write-Host "`n Virtual environment was created.`n"

# Sync environment
Write-Host " Syncing dev environment via uv.lock..."
.\.venv\Scripts\uv.exe sync

Write-Host "`n Done! Ready to work on the project now.`n"

# venv aktivieren
Write-Host "`n Opening new terminal inside virtual environment ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", ". .\.venv\Scripts\activate"
Write-Host "`n You can close this terminal now."
Write-Host "Don't forget to create a branch when making changes ;)."
Write-Host "Look for guidance in oct-analysis/docs/collab_workflow.md"