import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Callable
import threading


class LogManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.log_dir = None
        self.frontend_logger: Optional[Callable[[str], None]] = None
        self._setup_loggers()

    def _setup_loggers(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.app_logger = logging.getLogger("app")
        self.app_logger.setLevel(logging.DEBUG)
        self.app_logger.propagate = False

        self.render_logger = logging.getLogger("render")
        self.render_logger.setLevel(logging.DEBUG)
        self.render_logger.propagate = False

        self.analysis_logger = logging.getLogger("analysis")
        self.analysis_logger.setLevel(logging.DEBUG)
        self.analysis_logger.propagate = False

        self.ffmpeg_logger = logging.getLogger("ffmpeg")
        self.ffmpeg_logger.setLevel(logging.DEBUG)
        self.ffmpeg_logger.propagate = False

        self._clear_handlers()

        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )

        # App log file
        app_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        app_handler.setFormatter(file_formatter)
        app_handler.setLevel(logging.DEBUG)
        self.app_logger.addHandler(app_handler)

        # Render log file
        render_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "render.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        render_handler.setFormatter(file_formatter)
        render_handler.setLevel(logging.DEBUG)
        self.render_logger.addHandler(render_handler)

        # Analysis log file
        analysis_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "analysis.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        analysis_handler.setFormatter(file_formatter)
        analysis_handler.setLevel(logging.DEBUG)
        self.analysis_logger.addHandler(analysis_handler)

        # FFmpeg log file
        ffmpeg_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "ffmpeg.log",
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        ffmpeg_handler.setFormatter(file_formatter)
        ffmpeg_handler.setLevel(logging.DEBUG)
        self.ffmpeg_logger.addHandler(ffmpeg_handler)

        # Console handler for ALL loggers (INFO and above for terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        self.app_logger.addHandler(console_handler)
        self.render_logger.addHandler(console_handler)
        self.analysis_logger.addHandler(console_handler)
        self.ffmpeg_logger.addHandler(console_handler)

    def _clear_handlers(self):
        for logger in [self.app_logger, self.render_logger, self.analysis_logger, self.ffmpeg_logger]:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    def set_frontend_logger(self, callback: Callable[[str], None]):
        """Set callback for frontend UI updates (clean messages only)."""
        self.frontend_logger = callback

    def app_info(self, msg: str):
        self.app_logger.info(msg)
        if self.frontend_logger:
            self.frontend_logger(f"ℹ️ {msg}")

    def app_debug(self, msg: str):
        self.app_logger.debug(msg)

    def app_warning(self, msg: str):
        self.app_logger.warning(msg)
        if self.frontend_logger:
            self.frontend_logger(f"⚠️ {msg}")

    def app_error(self, msg: str):
        self.app_logger.error(msg)
        if self.frontend_logger:
            self.frontend_logger(f"❌ {msg}")

    def render_info(self, msg: str):
        self.render_logger.info(msg)

    def render_debug(self, msg: str):
        self.render_logger.debug(msg)

    def analysis_info(self, msg: str):
        self.analysis_logger.info(msg)

    def analysis_debug(self, msg: str):
        self.analysis_logger.debug(msg)

    def ffmpeg_info(self, msg: str):
        self.ffmpeg_logger.info(msg)

    def ffmpeg_debug(self, msg: str):
        self.ffmpeg_logger.debug(msg)

    def ffmpeg_command(self, cmd: list):
        """Log FFmpeg command to ffmpeg log and terminal."""
        self.ffmpeg_logger.info(f"COMMAND: {' '.join(cmd)}")

    def ffmpeg_output(self, line: str):
        """Log FFmpeg output line to ffmpeg log only (DEBUG to avoid terminal spam)."""
        self.ffmpeg_logger.debug(line.strip())

    def progress(self, stage: str, pct: int, details: str = ""):
        """Log progress update - goes to app log and frontend."""
        msg = f"[{stage}] {pct}% {details}".strip()
        self.app_logger.info(msg)
        if self.frontend_logger:
            self.frontend_logger(msg)

    def stage_start(self, stage_num: int, total_stages: int, name: str):
        """Log stage start - clean message for frontend."""
        msg = f"Stage {stage_num}/{total_stages}: {name}"
        self.app_logger.info(msg)
        if self.frontend_logger:
            self.frontend_logger(msg)

    def stage_complete(self, stage_num: int, total_stages: int, name: str):
        """Log stage completion."""
        msg = f"✅ Stage {stage_num}/{total_stages}: {name} complete"
        self.app_logger.info(msg)
        if self.frontend_logger:
            self.frontend_logger(msg)


log_manager = LogManager()


def get_logger() -> LogManager:
    return log_manager


def init_logging(log_dir: str = "logs"):
    """Initialize logging system with custom log directory."""
    global log_manager
    log_manager = LogManager()
    log_manager.log_dir = Path(log_dir)
    log_manager.log_dir.mkdir(exist_ok=True)
    log_manager._setup_loggers()
    return log_manager