# Optional: Pfad zum Zielordner
$targetDir = "$HOME\napari"
$packageName = "oct-analysis"

# Ordner erstellen
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Set-Location $targetDir

# venv erstellen
uv venv

# Aktivieren (manuell durch User, Anleitung folgt)
Write-Host "`n Virtual environment was created.`n"

# Paket installieren
Write-Host " Installing package: '$packageName' ..."
.\.venv\Scripts\uv.exe pip install $packageName

Write-Host "`n Done!`n"

# venv aktivieren
Write-Host "`n Opening new terminal inside virtual environment ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", ". .\.venv\Scripts\activate"
Write-Host "`n You can close this terminal now."