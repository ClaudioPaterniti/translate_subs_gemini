$script_path = Join-Path $PSScriptRoot "./translate_subs.py"
$env_path = Join-Path $PSScriptRoot ".\.venv\Scripts\Activate.ps1"
& $env_path

Add-Type -AssemblyName System.Windows.Forms
$FileBrowser = New-Object System.Windows.Forms.OpenFileDialog -Property @{
    Title = "Select subtitles"
    InitialDirectory = $pwd
    Multiselect = $true # Multiple files can be chosen
}

$result = $FileBrowser.ShowDialog()

if ($result -ne "Cancel") {
    python $script_path @($FileBrowser.FileNames)
}

Read-Host -Prompt "Press Enter to exit"