from dataclasses import dataclass
import logging
import shutil
import subprocess
from collections.abc import Sequence


@dataclass(slots=True)
class CommandRunner:
    dry_run: bool = False
    logger: logging.Logger = logging.getLogger("reconstruction")

    def require_tool(self, executable: str) -> str:
        path = shutil.which(executable)
        if not path and not self.dry_run:
            raise FileNotFoundError(f"required executable not found on PATH: {executable}")
        return path or executable

    def run(self, command: Sequence[str], cwd=None) -> subprocess.CompletedProcess:
        self.logger.info("RUN: %s", " ".join(map(str, command)))
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
