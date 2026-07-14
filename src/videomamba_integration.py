from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


@dataclass
class VideoMambaConfig:
    script: Path
    args: list[str]


def launch(config: VideoMambaConfig) -> None:
    if not config.script.exists():
        raise FileNotFoundError(f"VideoMamba script not found: {config.script}")
    subprocess.run([sys.executable, str(config.script), *config.args], check=True)

