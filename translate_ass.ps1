$py_version = python -V
if (!$py_version.StartsWith("Python")){
    Write-Error "Python not installed"
    Read-Host -Prompt "Press Enter to exit"
    Exit
}

Set-Location $PSScriptRoot

$venv_path = Join-Path $PSScriptRoot ".venv"
if (!(Test-Path -Path $venv_path)){
    Write-Host "Creating virtual environtment .venv"
    python -m venv .venv
    Write-Host "Upgrading pip"
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    Write-Host "Installing requirements"
    pip install -t requirements.txt
}
else {
    .\.venv\Scripts\Activate.ps1
}

Add-Type -AssemblyName System.Windows.Forms
$FileBrowser = New-Object System.Windows.Forms.OpenFileDialog -Property @{
    Title = "Select subtitles"
    InitialDirectory = $pwd
    Multiselect = $true # Multiple files can be chosen
}

$result = $FileBrowser.ShowDialog()

if ($result -ne "Cancel") {
    python ./translate_ass.py @($FileBrowser.FileNames)
}

Read-Host -Prompt "Press Enter to exit"