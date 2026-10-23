"""``iree-coral doctor``: diagnose a toolchain installation.

Prints one line per check with OK / MISSING / NOTE, then a verdict.  This is the
Python counterpart of ``scripts/check_prerequisites.sh`` but checks *built*
artifacts as well as inputs, so it is the fastest way to see why a compile or a
hardware test fails.
"""

import os
import shutil
import subprocess

from . import paths


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None


def _tool_version(cmd, args=("--version",)):
    proc = _run(list(cmd) + list(args))
    if proc is None:
        return None
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return out[0] if out else "?"


def _workspace_root():
    return paths.WORKSPACE


def run(args):
    from .version import summary as version_summary

    problems = []
    notes = []

    def ok(msg):
        print(f"  OK       {msg}")

    def missing(msg):
        print(f"  MISSING  {msg}")
        problems.append(msg)

    def note(msg):
        print(f"  NOTE     {msg}")
        notes.append(msg)

    print(f"iree-coral doctor  (workspace: {_workspace_root()})")
    print(f"  version  {version_summary()}")
    print()

    print("== build tools ==")
    for tool in ("cmake", "ninja", "clang", "clang++", "gcc", "git", "python3"):
        path = shutil.which(tool)
        if path:
            ver = _tool_version((tool,)) or ""
            ok(f"{tool}: {ver}")
        else:
            missing(f"{tool} not on PATH")
    print()

    print("== toolchain artifacts ==")
    checks = [
        (paths.IREE_COMPILE, "compiler (iree-compile)"),
        (paths.IREE_RUN_MODULE, "runtime (iree-run-module)"),
        (paths.CORAL_NATIVE_SO, "native Coral shim (libcoral_native.so)"),
    ]
    for path, label in checks:
        if os.path.exists(path):
            ok(f"{label}: {path}")
        else:
            missing(f"{label}: {path} (run scripts/bootstrap_all.sh)")
    print()

    print("== IREE fork ==")
    git = os.path.join(paths.IREE_DIR, ".git")
    if os.path.isdir(git):
        ok(f"IREE fork: {paths.IREE_DIR}")
        for rel in ("compiler/plugins/target/Coral/CoralGenerationPass.cpp",
                    "runtime/src/iree/hal/local/loaders/coral_executable_loader.c"):
            if os.path.exists(os.path.join(paths.IREE_DIR, rel)):
                ok(f"  {rel}")
            else:
                missing(f"  {rel} (wrong branch/commit)")
    else:
        missing(f"IREE fork: {paths.IREE_DIR} (run scripts/bootstrap_iree.sh)")
    print()

    print("== external inputs ==")
    tf = os.path.join(_workspace_root(), "tensorflow-source")
    if os.path.isdir(tf):
        ok(f"TensorFlow source: {tf}")
    else:
        missing(f"TensorFlow source: {tf} (run scripts/fetch_third_party.sh)")
    if os.path.exists(paths.LIBEDGETPU):
        ok(f"libedgetpu: {paths.LIBEDGETPU}")
    else:
        note(f"libedgetpu: {paths.LIBEDGETPU} (needed for the runtime build)")
    model = os.path.join(paths.TEST_DATA, "mobilenet_v2_1.0_224_quant.tflite")
    if os.path.exists(model):
        ok(f"test model: {model}")
    else:
        note(f"test model: {model} (run scripts/fetch_test_data.sh)")
    print()

    print("== Edge TPU hardware ==")
    if os.path.exists(paths.APEX_DEVICE):
        ok(f"{paths.APEX_DEVICE} present")
    else:
        note(f"{paths.APEX_DEVICE} absent (no physical device; hardware tests skip)")
    if os.path.exists(paths.EDGETPU_COMPILER):
        ver = _tool_version((paths.EDGETPU_COMPILER,))
        ok(f"edgetpu_compiler: {ver}")
    else:
        note(f"edgetpu_compiler: {paths.EDGETPU_COMPILER} (needed for hardware tests)")
    lsmod = _run(["lsmod"])
    if lsmod and "gasket" in (lsmod.stdout or ""):
        ok("gasket kernel module loaded")
    else:
        note("gasket kernel module not loaded (required for the physical device)")
    print()

    if problems:
        print(f"DOCTOR: {len(problems)} problem(s) found; see MISSING above.")
        return 1
    print("DOCTOR: OK")
    if notes:
        print(f"  ({len(notes)} optional item(s) missing; documented above)")
    return 0
