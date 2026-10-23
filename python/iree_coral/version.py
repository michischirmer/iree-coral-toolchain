"""Toolchain version reporting.

The pinned revisions live in versions.lock at the repository root; this reads
that file so there is exactly one source of truth.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCK = os.path.join(_ROOT, "versions.lock")
_TOOLCHAIN_VERSION = "0.1.0"


def _read_lock():
    values = {}
    try:
        with open(_LOCK) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip()
    except OSError:
        pass
    return values


def summary():
    lock = _read_lock()
    return (
        f"iree-coral-toolchain {_TOOLCHAIN_VERSION} "
        f"(IREE {lock.get('IREE_CORAL_COMMIT', '?')[:12]}, "
        f"edgetpu_compiler {lock.get('EDGETPU_COMPILER_VERSION', '?')})"
    )


def details():
    lock = _read_lock()
    order = [
        ("toolchain version", _TOOLCHAIN_VERSION),
        ("IREE branch", lock.get("IREE_CORAL_BRANCH")),
        ("IREE commit", lock.get("IREE_CORAL_COMMIT")),
        ("IREE upstream commit", lock.get("IREE_UPSTREAM_COMMIT")),
        ("edgetpu_compiler", lock.get("EDGETPU_COMPILER_VERSION")),
        ("edgetpu commit", lock.get("EDGETPU_COMMIT")),
        ("pycoral commit", lock.get("PYCORAL_COMMIT")),
        ("TensorFlow (build-time)", lock.get("TENSORFLOW_VERSION")),
        ("TensorFlow commit", lock.get("TENSORFLOW_COMMIT")),
        ("Eigen commit", lock.get("EIGEN_COMMIT")),
        ("python", lock.get("PYTHON_VERSION")),
        ("tensorflow (runtime)", lock.get("PY_TENSORFLOW")),
        ("numpy", lock.get("PY_NUMPY")),
    ]
    return [(k, v) for k, v in order if v]
