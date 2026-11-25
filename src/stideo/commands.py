"""Run external media commands and preserve their diagnostics."""

import subprocess

FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]


def run(command: list[str], label: str) -> bytes:
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"{label}\n{result.stderr.decode(errors='replace').strip() or '外部命令执行失败'}")
    return result.stdout
