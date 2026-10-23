"""Path resolution for the IREE Coral Edge TPU toolchain.

Every path is derived from an explicit, overridable root so the CLI works from
any checkout:

    IREE_CORAL_WORKSPACE   workspace holding the IREE fork and build trees
                           (default: the toolchain repository's parent)
    IREE_CORAL_SOURCE      the IREE fork checkout (default: a sibling fork)
    IREE_CORAL_BUILD_DIR   root for the build trees (default: <workspace>)
    IREE_CORAL_INSTALL_PREFIX  runtime install prefix
    ... or set the individual roots below

This mirrors scripts/lib.sh so the Python CLI and the shell scripts agree.
"""

import os

TOOLCHAIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
WORKSPACE = os.environ.get(
    "IREE_CORAL_WORKSPACE", os.path.dirname(TOOLCHAIN_ROOT))
BUILD_ROOT = os.environ.get("IREE_CORAL_BUILD_DIR", WORKSPACE)


def _default_source():
    """Pick a sibling IREE fork without assuming a particular directory name."""
    for name in ("iree-coral-fork", "iree"):
        candidate = os.path.join(WORKSPACE, name)
        if os.path.isdir(os.path.join(candidate, "compiler", "plugins",
                                      "target", "Coral")):
            return candidate
    return os.path.join(WORKSPACE, "iree-coral-fork")


IREE_DIR = os.environ.get(
    "IREE_CORAL_SOURCE", os.environ.get("IREE_CORAL_DIR", _default_source()))
COMPILER_BUILD = os.environ.get(
    "IREE_CORAL_COMPILER_BUILD", os.path.join(BUILD_ROOT, "iree-compiler-build"))
RUNTIME_INSTALL_BUILD = os.environ.get(
    "IREE_CORAL_RUNTIME_INSTALL_BUILD",
    os.path.join(BUILD_ROOT, "iree-runtime-install-build"))

IREE_COMPILE = os.path.join(COMPILER_BUILD, "tools", "iree-compile")
IREE_RUN_MODULE = os.path.join(RUNTIME_INSTALL_BUILD, "tools", "iree-run-module")

GENERATED = os.environ.get(
    "IREE_CORAL_GENERATED_ROOT", os.path.join(BUILD_ROOT, "generated"))
CORAL_NATIVE_SO = os.path.join(TOOLCHAIN_ROOT, "runtime", "libcoral_native.so")

EDGETPU_LIB = os.path.join(WORKSPACE, "third-party", "libedgetpu", "lib")
LIBEDGETPU = os.path.join(EDGETPU_LIB, "libedgetpu.so.1")
EDGETPU_COMPILER = os.environ.get(
    "IREE_CORAL_EDGETPU_COMPILER",
    os.path.join(WORKSPACE, "edgetpu", "compiler", "x86_64",
                 "edgetpu_compiler"))

IMPORTER = os.path.join(TOOLCHAIN_ROOT, "tools", "import_tflite.py")
TEST_DATA = os.path.join(TOOLCHAIN_ROOT, "tests", "test_data")
APEX_DEVICE = "/dev/apex_0"
VENV = os.environ.get("IREE_CORAL_VENV", os.path.join(BUILD_ROOT, ".venv"))
PYTHON = os.path.join(VENV, "bin", "python")
TOOLS_ROOT = os.path.join(TOOLCHAIN_ROOT, "tools")


def compiler_env(base=None):
    """Environment for ``iree-compile``.

    The fork's Coral generation pass shells out to this toolchain's generator
    scripts and the proprietary ``edgetpu_compiler``, and writes its
    intermediates under a generated root.  Those locations live in this
    repository, not beside the build tree, so they are passed explicitly rather
    than left to the pass's build-tree heuristic.
    """
    env = dict(os.environ if base is None else base)
    env["IREE_CORAL_TOOLS_ROOT"] = TOOLS_ROOT
    env["IREE_CORAL_PYTHON"] = PYTHON
    env["IREE_CORAL_GENERATED_ROOT"] = GENERATED
    env["IREE_CORAL_EDGETPU_COMPILER"] = EDGETPU_COMPILER
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return env


def require_compiler():
    if not os.path.exists(EDGETPU_COMPILER):
        raise FileNotFoundError(f"Edge TPU compiler not found: {EDGETPU_COMPILER}")
    return EDGETPU_COMPILER


def require_runtime():
    if not os.path.exists(LIBEDGETPU):
        raise FileNotFoundError(f"libedgetpu not found: {LIBEDGETPU}")
    return LIBEDGETPU
