# Optional: Pfad zum Zielordner
$targetDir = "$HOME\napari"
$packageName = "ebi_oct"

# Ordner erstellen
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Set-Location $targetDir

# venv erstellen
uv venv

# Aktivieren (manuell durch User, Anleitung folgt)
Write-Host "`n🌀 Virtuelle Umgebung wurde erstellt. Aktiviere sie mit:"
Write-Host "    .\.venv\Scripts\Activate.ps1`n"

# Paket installieren
Write-Host "📦 Installiere Paket '$packageName' ..."
.\.venv\Scripts\uv.exe pip install $packageName

Write-Host "`n✅ Fertig! Starte napari mit:"
Write-Host "    napari"
