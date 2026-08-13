# Setup virtual environment and install dependencies for Beats Video Editor
# Run this from the repository root (beats-video-editor)

# Create venv
python -m venv .venv

# Allow the script to activate the venv in this session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
python -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Setup complete. To activate the venv later, run:`n& .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
