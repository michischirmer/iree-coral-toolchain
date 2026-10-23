"""Path resolution shared by the toolchain's Python tests.

The roots are the ones the CLI and the shell scripts use (see
``python/iree_coral/paths.py`` and ``scripts/lib.sh``); this module only adds
the test-specific locations and aliases ``RUN_MODULE`` to the runtime binary.

The layout is:

    <workspace>/
      iree-coral-toolchain/     this repository (tools/, tests/, runtime/, ...)
      iree-coral-fork/          the IREE fork
      iree-compiler-build/
      iree-runtime-install-build/
      tensorflow-source/  tflite-build/  third-party/  edgetpu/
"""

import os
import sys

_TOOLCHAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_TOOLCHAIN_ROOT, "python"))

from iree_coral.paths import (  # noqa: E402
    APEX_DEVICE,
    COMPILER_BUILD,
    EDGETPU_COMPILER,
    EDGETPU_LIB,
    GENERATED,
    IMPORTER,
    IREE_COMPILE,
    IREE_RUN_MODULE,
    PYTHON,
    RUNTIME_INSTALL_BUILD,
    TEST_DATA,
    TOOLCHAIN_ROOT,
    TOOLS_ROOT,
    VENV,
    WORKSPACE,
    compiler_env,
)

# The hardware test refers to the runtime binary as RUN_MODULE.
RUN_MODULE = IREE_RUN_MODULE

COMPARE_LAYERS = os.path.join(TOOLCHAIN_ROOT, "tools", "compare_layers.py")
SOURCE_MODEL = os.path.join(TEST_DATA, "mobilenet_v2_1.0_224_quant.tflite")
