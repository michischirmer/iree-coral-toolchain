"""Helpers for invoking the proprietary Edge TPU compiler."""

import os
import subprocess

from .paths import require_compiler


class CoralCompilationError(RuntimeError):
    pass


def compile_model(input_tflite, output_dir, show_operations=True):
    compiler = require_compiler()
    os.makedirs(output_dir, exist_ok=True)
    cmd = [compiler]
    if show_operations:
        cmd.append("--show_operations")
    cmd += ["-o", output_dir, input_tflite]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise CoralCompilationError(completed.stderr or completed.stdout)
    output_name = os.path.splitext(os.path.basename(input_tflite))[0] + "_edgetpu.tflite"
    output_path = os.path.join(output_dir, output_name)
    return {
        "output_path": output_path if os.path.exists(output_path) else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": cmd,
    }
