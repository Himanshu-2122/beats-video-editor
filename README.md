Beats Video Editor — Quick setup

Prerequisites
- Python 3.8+ installed and on PATH
- FFmpeg (ffmpeg and ffprobe) installed and on PATH

Setup (PowerShell, run from project root):

```powershell
# Create and activate venv, then install deps
.\setup_env.ps1
```

Or run the commands manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the GUI frontend (from project root):

```powershell
python -m app.frontend
```

Notes
- This project uses `tkinter` for the GUI and `ffmpeg` for video processing.
- On Windows, installing `librosa` can require compiling native wheels; using the script above should attempt to install compatible wheels from PyPI. If you encounter build errors, consider installing with conda or using prebuilt wheels.
