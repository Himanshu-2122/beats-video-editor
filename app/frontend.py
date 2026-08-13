import os
import shutil
import threading
import traceback
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from app.beat import get_beats
from app.sync import sync_clips_with_beats
from app.video import (
    concatenate_videos,
    add_audio,
)

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Beat Video Editor"

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
)

AUDIO_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
)

# ============================================================
# GUI
# ============================================================

class BeatVideoEditor:

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("950x760")
        self.root.minsize(900, 700)

        self.music_path = tk.StringVar()
        self.video_folder = tk.StringVar()
        self.output_folder = tk.StringVar(value=os.path.abspath("output"))
        self.min_beats = tk.IntVar(value=4)
        self.max_beats = tk.IntVar(value=8)
        self.enable_transitions = tk.BooleanVar(value=True)
        self.transition_min = tk.IntVar(value=4)
        self.transition_max = tk.IntVar(value=8)
        self.transition_duration = tk.DoubleVar(value=0.35)
        self.resolution = tk.StringVar(value="1080p")
        self.quality = tk.StringVar(value="Very High")
        self.status = tk.StringVar(value="Ready")
        self.is_running = False

        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="🎬 Beat Video Editor",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            main,
            text=(
                "Random unique clips • "
                "Ascending flow • "
                "Beat synchronization • "
                "Selective transitions"
            ),
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 15))

        files = ttk.LabelFrame(main, text="📁 Input / Output", padding=15)
        files.pack(fill="x", pady=5)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="🎵 Music").grid(row=0, column=0, sticky="w", pady=7)
        ttk.Entry(files, textvariable=self.music_path).grid(row=0, column=1, padx=10, sticky="ew")
        ttk.Button(files, text="Browse", command=self.select_music).grid(row=0, column=2)

        ttk.Label(files, text="🎬 Video Folder").grid(row=1, column=0, sticky="w", pady=7)
        ttk.Entry(files, textvariable=self.video_folder).grid(row=1, column=1, padx=10, sticky="ew")
        ttk.Button(files, text="Select Folder", command=self.select_video_folder).grid(row=1, column=2)

        ttk.Label(files, text="📂 Output Folder").grid(row=2, column=0, sticky="w", pady=7)
        ttk.Entry(files, textvariable=self.output_folder).grid(row=2, column=1, padx=10, sticky="ew")
        ttk.Button(files, text="Select Folder", command=self.select_output_folder).grid(row=2, column=2)

        settings = ttk.LabelFrame(main, text="⚙️ Video Settings", padding=15)
        settings.pack(fill="x", pady=10)

        ttk.Label(settings, text="Minimum beats / clip").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Spinbox(settings, from_=2, to=20, textvariable=self.min_beats, width=10).grid(row=0, column=1, padx=10, sticky="w")
        ttk.Label(settings, text="Maximum beats / clip").grid(row=0, column=2, sticky="w", pady=6)
        ttk.Spinbox(settings, from_=3, to=30, textvariable=self.max_beats, width=10).grid(row=0, column=3, padx=10, sticky="w")

        ttk.Label(settings, text="Resolution").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            settings,
            textvariable=self.resolution,
            values=["720p", "1080p", "Original"],
            state="readonly",
            width=12,
        ).grid(row=1, column=1, padx=10, sticky="w")

        ttk.Label(settings, text="Quality").grid(row=1, column=2, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.quality,
            values=["Balanced", "High", "Very High"],
            state="readonly",
            width=12,
        ).grid(row=1, column=3, padx=10, sticky="w")

        transitions = ttk.LabelFrame(main, text="🎞️ Selective Random Transitions", padding=15)
        transitions.pack(fill="x", pady=5)

        self.transition_check = ttk.Checkbutton(
            transitions,
            text=(
                "Enable random transitions "
                "(not between every clip)"
            ),
            variable=self.enable_transitions,
            command=self.update_transition_state,
        )
        self.transition_check.grid(row=0, column=0, columnspan=4, sticky="w", pady=5)

        ttk.Label(transitions, text="Every minimum").grid(row=1, column=0, sticky="w")
        self.transition_min_box = ttk.Spinbox(
            transitions,
            from_=2,
            to=20,
            textvariable=self.transition_min,
            width=10,
        )
        self.transition_min_box.grid(row=1, column=1, padx=10, sticky="w")

        ttk.Label(transitions, text="Every maximum").grid(row=1, column=2, sticky="w")
        self.transition_max_box = ttk.Spinbox(
            transitions,
            from_=3,
            to=30,
            textvariable=self.transition_max,
            width=10,
        )
        self.transition_max_box.grid(row=1, column=3, padx=10, sticky="w")

        ttk.Label(transitions, text="Duration").grid(row=2, column=0, sticky="w", pady=8)
        self.transition_duration_box = ttk.Combobox(
            transitions,
            textvariable=self.transition_duration,
            values=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
            state="readonly",
            width=10,
        )
        self.transition_duration_box.grid(row=2, column=1, padx=10, sticky="w")

        ttk.Label(transitions, text="Recommended: 0.35 sec").grid(row=2, column=2, columnspan=2, sticky="w")

        self.generate_button = ttk.Button(main, text="🚀 Generate Video", command=self.start_generation)
        self.generate_button.pack(fill="x", pady=(15, 8), ipady=8)

        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=5)

        ttk.Label(main, textvariable=self.status, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=5)

        log_frame = ttk.LabelFrame(main, text="📋 Progress Log", padding=5)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.update_transition_state()

    def select_music(self):
        path = filedialog.askopenfilename(
            title="Select Music",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.aac"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.music_path.set(path)

    def select_video_folder(self):
        path = filedialog.askdirectory(title="Select Video Folder")
        if path:
            self.video_folder.set(path)
            videos = self.find_videos(path)
            self.log(f"Found {len(videos)} source videos.")

    def select_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_folder.set(path)

    def find_videos(self, folder):
        videos = []
        if not os.path.isdir(folder):
            return videos
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(path)
        return videos

    def update_transition_state(self):
        enabled = self.enable_transitions.get()
        state = "normal" if enabled else "disabled"
        self.transition_min_box.configure(state=state)
        self.transition_max_box.configure(state=state)
        self.transition_duration_box.configure(state="readonly" if enabled else "disabled")

    def log(self, message):
        def update():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", str(message) + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(0, update)

    def set_status(self, message):
        self.root.after(0, lambda: self.status.set(message))

    def set_progress(self, value):
        self.root.after(0, lambda: self.progress.configure(value=value))

    def start_generation(self):
        if self.is_running:
            return

        music = self.music_path.get()
        video_folder = self.video_folder.get()
        output_folder = self.output_folder.get()

        if not music:
            messagebox.showerror("Missing Music", "Please select a music file.")
            return

        if not os.path.isfile(music):
            messagebox.showerror("Invalid Music", "Music file does not exist.")
            return

        if not video_folder:
            messagebox.showerror("Missing Videos", "Please select a video folder.")
            return

        if not os.path.isdir(video_folder):
            messagebox.showerror("Invalid Folder", "Video folder does not exist.")
            return

        videos = self.find_videos(video_folder)
        if not videos:
            messagebox.showerror("No Videos", "No supported videos found.")
            return

        try:
            min_beats = int(self.min_beats.get())
            max_beats = int(self.max_beats.get())
            if min_beats > max_beats:
                messagebox.showerror(
                    "Invalid Settings",
                    "Minimum beats cannot be greater than maximum beats.",
                )
                return
        except Exception:
            messagebox.showerror("Invalid Settings", "Invalid beat settings.")
            return

        os.makedirs(output_folder, exist_ok=True)
        self.is_running = True
        self.generate_button.configure(state="disabled")
        self.progress["value"] = 0
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        thread = threading.Thread(
            target=self.generate_video,
            args=(music, videos, output_folder),
            daemon=True,
        )
        thread.start()

    def generate_video(self, music_path, video_paths, output_folder):
        temp_dir = None
        try:
            import app.video as video_module

            resolution = self.resolution.get()
            if resolution == "720p":
                video_module.TARGET_WIDTH = 1280
                video_module.TARGET_HEIGHT = 720
            elif resolution == "1080p":
                video_module.TARGET_WIDTH = 1920
                video_module.TARGET_HEIGHT = 1080
            else:
                video_module.TARGET_WIDTH = 1920
                video_module.TARGET_HEIGHT = 1080

            quality = self.quality.get()
            if quality == "Very High":
                video_module.QSV_QUALITY = 18
            elif quality == "High":
                video_module.QSV_QUALITY = 20
            else:
                video_module.QSV_QUALITY = 22

            video_module.TRANSITION_DURATION = float(self.transition_duration.get())
            video_module.TRANSITION_MIN_CLIPS = int(self.transition_min.get())
            video_module.TRANSITION_MAX_CLIPS = int(self.transition_max.get())

            self.set_status("Analyzing music...")
            self.log("[1/5] Analyzing music...")
            beats = get_beats(music_path)
            if len(beats) < 2:
                raise RuntimeError("Not enough beats detected.")
            self.log(f"Detected {len(beats)} beats.")
            self.set_progress(10)

            self.set_status("Preparing source videos...")
            self.log("\n[2/5] Preparing source videos...")
            self.log(f"Available videos: {len(video_paths)}")
            self.set_progress(15)

            self.set_status("Generating beat-synchronized clips...")
            self.log("\n[3/5] Generating clips...")

            def progress_callback(current, total):
                percentage = 15 + (current / max(total, 1)) * 45
                self.set_progress(percentage)
                self.set_status(f"Generating clip {current}/{total}...")
                temp_clip_paths, beat_groups, temp_dir = sync_clips_with_beats(
                    video_paths,
                    beats,
                    min_beats=int(self.min_beats.get()),
                    max_beats=int(self.max_beats.get()),
                    progress_callback=progress_callback,
                )

                if not temp_clip_paths:
                    raise RuntimeError("No clips were generated.")

                self.log(f"Created {len(temp_clip_paths)} clips.")
                self.set_progress(60)

                self.set_status("Creating final video...")
                self.log("\n[4/5] Creating final video...")
                os.makedirs(output_folder, exist_ok=True)
                video_without_audio = os.path.join(output_folder, "_video_without_audio.mp4")

                if self.enable_transitions.get():
                    self.log("Adding selective random transitions...")
                    concatenate_videos(temp_clip_paths, video_without_audio, beat_groups=beat_groups,
                                       transition_min=self.transition_min.get(),
                                       transition_max=self.transition_max.get(),
                                       transition_duration=self.transition_duration.get())
                else:
                    self.log("Combining clips without transitions...")
                    from app.video import concat_group
                    concat_group(temp_clip_paths, video_without_audio)

                self.set_progress(80)

                self.set_status("Adding music...")
                self.log("\n[5/5] Adding music...")
                final_path = os.path.join(output_folder, "final.mp4")
                add_audio(video_path=video_without_audio, audio_path=music_path, output_path=final_path)
                self.set_progress(100)

                if os.path.exists(video_without_audio):
                    os.remove(video_without_audio)

                self.set_status("✅ Video generated successfully!")
                self.log("\n======================================")
                self.log("       VIDEO GENERATION COMPLETE")
                self.log("======================================")
            self.log(f"\nOutput:\n{final_path}")

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    ("Video generated successfully!\n\n" f"{final_path}"),
                ),
            )

            try:
                os.startfile(output_folder)
            except Exception:
                pass

        except Exception as exc:
            self.set_status("❌ Generation failed")
            self.log("\nERROR:")
            self.log(str(exc))
            self.log("\n" + traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Generation Failed", str(exc)))
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        finally:
            self.is_running = False
            self.root.after(0, lambda: self.generate_button.configure(state="normal"))


def main():
    root = tk.Tk()
    BeatVideoEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
