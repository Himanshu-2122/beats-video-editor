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

Run the Streamlit frontend (from project root):

```powershell
# Start the web UI
streamlit run app/frontend.py
```

Notes
- This project uses `streamlit` for the web UI and `ffmpeg` for video processing.
- On Windows, installing `librosa` can require compiling native wheels; using the script above should attempt to install compatible wheels from PyPI. If you encounter build errors, consider installing with conda or using prebuilt wheels.

Low-RAM encoding examples

Use these `ffmpeg` patterns to improve final quality while keeping RAM usage low.

- Single-pass CRF (good quality, low memory):

```bash
ffmpeg -i input.mp4 -c:v libx264 -preset slow -crf 20 -c:a aac -b:a 192k output.mp4
```

- Chunked encode (split → encode sequentially → concat) — reduces peak RAM:

```bash
# split into 60s segments
ffmpeg -i input.mp4 -c copy -map 0 -f segment -segment_time 60 -reset_timestamps 1 parts%03d.mp4

# encode each part one-by-one (use a shell loop on Windows via PowerShell)
ffmpeg -i parts000.mp4 -vf "scale=1920:1080,setsar=1,fps=30" -c:v libx264 -preset slow -crf 20 -an parts000_enc.mp4
# repeat for parts001.mp4, parts002.mp4, ...

# create concat list and join
printf "file 'parts000_enc.mp4'\nfile 'parts001_enc.mp4'\n" > mylist.txt
ffmpeg -f concat -safe 0 -i mylist.txt -c copy final_output.mp4
```

Notes:
- Increase CPU time (slower presets) for smaller files and better quality without extra RAM.
- Use `libx265` for better compression at the cost of CPU time (useful if RAM is critical).
- If your system supports hardware encoders (NVENC, QSV), prefer them to reduce CPU load; memory usage is typically lower but driver-dependent.
