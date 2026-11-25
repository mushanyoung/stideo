"""Run external media commands and preserve their diagnostics."""

import subprocess

FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]

def run(command, label):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(f"{label}\n{result.stderr.decode(errors='replace').strip() or '外部命令执行失败'}")
    return result.stdout


