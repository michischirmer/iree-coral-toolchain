# Installation

The scripts reproduce the complete compiler and runtime on Ubuntu 24.04.
Everything is incremental; re-running any step is safe.

## 1. Prerequisites

Supported host: **Ubuntu 24.04 on x86-64**. The compiler and runtime are built
and tested there; the proprietary `edgetpu_compiler` is distributed for x86-64
only. Other Linux distributions may work but are not verified.

| Requirement | Version |
| --- | --- |
| Ubuntu | 24.04 (verified) |
| Architecture | x86-64 |
| cmake | >= 3.28 |
| ninja | >= 1.11 |
| clang / clang++ | >= 18 |
| gcc | >= 13 |
| git | any |
| curl, tar, sha256sum | any (fetching and verifying inputs) |
| Python | 3.12 with `venv` |
| Free disk space | about 15 GB |

On a minimal Ubuntu 24.04 installation these are typically:

```bash
sudo apt install build-essential cmake ninja-build clang git curl python3 python3-venv
```

`scripts/check_prerequisites.sh` is authoritative: it reports what is present
and what is missing and exits non-zero if a hard requirement is absent.

```bash
scripts/check_prerequisites.sh
```

The Edge TPU device, the `gasket` kernel module and `edgetpu_compiler` are
optional at this point; they are needed for inference and the hardware tests,
not for building.

## 2. Layout

The toolchain expects a workspace with the fork and the build trees beside it:

```
workspace/
  iree-coral-toolchain/         this repository
  iree-coral-fork/              the IREE fork (cloned by bootstrap_iree.sh)
  iree-compiler-build/          compiler build tree      (generated)
  iree-runtime-build/           runtime build tree       (generated)
  iree-runtime-install/         runtime install prefix   (generated)
  iree-runtime-install-build/   iree-run-module build    (generated)
  tensorflow-source/            TensorFlow 2.7 source    (fetched)
  eigen-<commit>.tar.gz         pinned Eigen archive     (fetched, hash-verified)
  tflite-build/                 libtensorflow-lite.a     (generated)
  edgetpu/                      Coral edgetpu (headers + edgetpu_compiler)  (fetched)
  pycoral/                      Coral pycoral (libedgetpu.so.1.0)          (fetched)
  third-party/libedgetpu/       staged libedgetpu.so.1   (fetched)
  tools -> iree-coral-toolchain/tools                 (link)
  .venv/                        python environment       (generated)
  generated/                    compiler intermediates   (generated)
```

Nothing is derived from the current working directory, the hostname, the user
name or the home directory: every script resolves the repository root from its
own location and takes the workspace from the environment, so the tree may live
anywhere and may contain spaces.

Canonical overrides (use these on a fresh machine):

| Variable | Meaning | Default |
| --- | --- | --- |
| `IREE_CORAL_SOURCE` | the IREE fork checkout | a sibling `iree-coral-fork/` (or `iree/`) |
| `IREE_CORAL_REPO` | URL the fork is cloned from when absent | the pin in `versions.lock` |
| `IREE_CORAL_BUILD_DIR` | root for every build tree | `<workspace>` |
| `IREE_CORAL_INSTALL_PREFIX` | runtime install prefix | `<build dir>/iree-runtime-install` |
| `IREE_CORAL_WORKSPACE` | umbrella root for fork + builds | the toolchain's parent |

The individual trees can be overridden on their own as well
(`IREE_CORAL_COMPILER_BUILD`, `IREE_CORAL_RUNTIME_INSTALL_BUILD`,
`IREE_CORAL_VENV`, `IREE_CORAL_GENERATED_ROOT`, `IREE_CORAL_DIR`). The defaults
match the tree above.

The scripts also export the paths the compiler's Coral generation pass needs at
run time (`IREE_CORAL_TOOLS_ROOT`, `IREE_CORAL_PYTHON`,
`IREE_CORAL_EDGETPU_COMPILER`, `IREE_CORAL_GENERATED_ROOT`), because those live
in this repository rather than beside the build tree. The Python CLI and the
tests set the same variables via `paths.compiler_env()`, so `iree-compile`
behaves identically however it is invoked.

Relocating the tree needs no configuration: see `docs/TESTING.md` for the
relocation test (`scripts/test_relocation.sh`).

### The IREE fork

IREE is not vendored. `versions.lock` pins the fork URL (`IREE_CORAL_REPO`), the
branch (`coral-edge-tpu`) and the exact commit. `scripts/bootstrap_iree.sh`
clones that URL when no checkout exists, initializes the IREE submodules and
checks out the pinned commit; a checkout at a different revision is moved to
the pin.

The pinned commit must be reachable from `IREE_CORAL_REPO`. If it is not (for
example the branch lives in a private mirror), pass a clone or mirror:

```bash
scripts/bootstrap_iree.sh --repo /path/to/iree-coral-fork
# or set IREE_CORAL_REPO in the environment
```

## 3. One-shot bootstrap

```bash
scripts/bootstrap_all.sh
```

This runs the nine steps below in order and needs no pre-existing IREE
checkout: the fork is cloned from the URL pinned in `versions.lock`
(`IREE_CORAL_REPO`). Pass `--repo <url>` to clone from a different mirror.

### Step by step

```bash
# 1. system + tool prerequisites
scripts/check_prerequisites.sh

# 2. python virtual environment with the pinned packages
scripts/setup_venv.sh

# 3. external inputs: TensorFlow 2.7 source, Eigen archive, Coral edgetpu,
#    libedgetpu
scripts/fetch_third_party.sh

# 4. the IREE fork at the pinned commit (cloned from IREE_CORAL_REPO)
scripts/bootstrap_iree.sh

# 5. the Coral-enabled compiler
scripts/build_compiler.sh

# 6. the native TFLite/libedgetpu shim (libcoral_native.so)
scripts/build_native_runtime.sh

# 7. the Coral-enabled IREE runtime (iree-run-module)
scripts/build_runtime.sh

# 8. test data (models + images)
scripts/fetch_test_data.sh

# 9. verification
scripts/verify_installation.sh
```

Build parallelism is `-j2` by default; override with `JOBS=<n>`.

## 4. Verify

```bash
scripts/verify_installation.sh
```

Always checks that `iree-compile`, `iree-run-module` and `libcoral_native.so`
exist and that the MobileNetV2 import is lossless (exact filters, exact
activation quantisation, bit-exact per-operator outputs). When a physical Edge
TPU and `edgetpu_compiler` are present it also compiles for the device and
confirms the hardware reproduces the original model's predictions.

## 5. Use

The smallest complete example imports, compiles and runs the reference model:

```bash
examples/minimal.sh
examples/mobilenet_v2.sh      # same pipeline with a real image
```

Or drive the tools directly:

```bash
export PYTHONPATH=$PWD/python

# full pipeline for a quantized TFLite model
python -m iree_coral.cli run mobilenet_v2_1.0_224_quant.tflite -o ./coral-out

# or the individual steps
python tools/import_tflite.py model.tflite -o model.mlir --tag model
$WORKSPACE/iree-compiler-build/tools/iree-compile \
    --iree-hal-target-backends=coral model.mlir -o model.vmfb
$WORKSPACE/iree-runtime-install-build/tools/iree-run-module \
    --device=local-sync --module=model.vmfb --function=run \
    --input=1x224x224x3xi8=@input.bin
```

## Notes

* The IREE fork is built but never modified. The toolchain applies no patches.
* The proprietary `edgetpu_compiler` and `libedgetpu` are fetched from the
  Google Coral `edgetpu` and `pycoral` repositories; they are not redistributed
  here. The header and compiler come from `edgetpu`; the TFLite-2.7-compatible
  `libedgetpu.so.1.0` comes from `pycoral`'s `libedgetpu_bin/`.
* Building TensorFlow 2.7 (only its `tensorflow-lite` target) is the longest
  step, a few minutes at `-j2`. TensorFlow 2.7 would clone Eigen from
  `gitlab.com`, which intermittently rejects requests; instead
  `scripts/fetch_third_party.sh` downloads the exact commit from the TensorFlow
  GCS mirror pinned in `versions.lock` (`EIGEN_URL`, `EIGEN_SHA256`) and
  cached in the workspace as `eigen-<commit>.tar.gz`;
  `scripts/build_native_runtime.sh` points CMake at that verified archive.
  Nothing is taken from a pre-existing host cache. `EIGEN_URL` may be pointed
  at a mirror; the sha256 pin is always enforced.
