$pythonAvailable = $false
try {
    $output = python --version 2>&1
    $pythonAvailable = $LASTEXITCODE -eq 0
} finally {}

if (!$pythonAvailable){
    Write-Error "Python not installed: install python and add it to PATH"
    Read-Host -Prompt "Press Enter to exit"
    Exit
}

Set-Location $PSScriptRoot

if (!(Test-Path -Path ".venv")){
    Write-Host "Creating virtual environtment .venv"
    python -m venv .venv
}
Write-Host "Upgrading pip"
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Write-Host "Installing requirements"
pip install -r requirements.txt

if (!(Test-Path -Path "gemini.key")){
    write-host "`n`n"
    Write-Host "Creating empty gemini.key file. Populate it with your gemini key"
    New-Item "./gemini.key" -type file | Out-Null
    write-host "`n"
}
Read-Host -Prompt "Press Enter to exit"