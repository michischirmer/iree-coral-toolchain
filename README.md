# IREE Coral Edge TPU toolchain

Build scripts, tests and a small CLI for running quantized TFLite models on the
Google Coral Edge TPU through [IREE](https://iree.dev).

## Purpose

IREE compiles a model to a `.vmfb` that an IREE runtime executes. This project
adds a Coral Edge TPU target: a quantized TFLite network is imported into IREE,
compiled with the proprietary `edgetpu_compiler`, and the compiled Edge TPU
executable is embedded in the `.vmfb`. At runtime IREE's own HAL loads that
executable and runs it on the device — no Python, no TFLite interpreter and no
`edgetpu_compiler` involved in inference.

## Supported hardware

* Google Coral Edge TPU (verified on the Coral M.2 Accelerator).
* Requires the `gasket` kernel module and `/dev/apex_0` for inference.
* The compiler half runs on any machine; only the runtime dispatch needs a
  device. Tests that need hardware skip cleanly when it is absent.

## Supported operations and limitations

The `coral` compiler target accepts an MLIR `func.func` carrying a
`coral.graph` attribute — a directed acyclic graph of named nodes. Nodes may use:

`conv_2d`, `depthwise_conv_2d`, `fully_connected`, `relu`, `relu6`, `logistic`,
`tanh`, `max_pool_2d`, `average_pool_2d`, `reshape`, `add`, `mul`,
`concatenation`, `softmax`, `log_softmax`.

Limitations:

* **Fully integer.** Models must be fully quantized (INT8/UINT8). The pass
  preserves the source model's per-tensor quantization losslessly.
* **Single Edge TPU subgraph.** Every operation must map to the Edge TPU; a
  partial mapping is a compile error, not a silent CPU fallback.
* **Batch size 1**, statically shaped tensors.
* **Single input / single output** per graph.
* **x86-64 Linux host** for the compiler and runtime builds.

A pretrained `mobilenet_v2_1.0_224_quant.tflite` is the reference end-to-end
model: 66 operators, all mapped to one Edge TPU subgraph.

## Architecture overview

Two repositories:

| repository | contents |
| --- | --- |
| `iree-coral-fork` | a fork of upstream IREE adding the `coral` compiler target backend and the Coral HAL executable loader |
| `iree-coral-toolchain` | *this repository* — bootstrap scripts, the native TFLite/libedgetpu shim, Python tools, tests, examples and documentation |

The toolchain contains no IREE source; it checks the fork out at a pinned commit
and builds it (`versions.lock` records every pinned revision).

```
 mobilenet_v2_1.0_224_quant.tflite   pretrained, UINT8
        |  tools/import_tflite.py             (toolchain)
        v
   coral.graph  (MLIR)                graph + per-node quantization metadata
        |  iree-compile --iree-hal-target-backends=coral  (fork)
        v
   CoralGenerationPass
        |  spec.json -> tools/generate_net_model.py -> model.tflite
        |  edgetpu_compiler                         -> model_edgetpu.tflite
        |  verifies every operator mapped to the Edge TPU
        v
   model.vmfb                          one Edge TPU executable, embedded
        |  iree-run-module --device=local-sync            (fork + toolchain)
        v
   coral_executable_loader -> libcoral_native.so -> libedgetpu -> Edge TPU
```

IREE never links TensorFlow: `libcoral_native.so` is the only bridge, a small C
interface (`runtime/coral_native.h`) implemented against the TFLite C API and
`libedgetpu`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Prerequisites

Supported host: Ubuntu 24.04 on x86-64. Required: `cmake >= 3.28`,
`ninja >= 1.11`, `clang >= 18`, `gcc >= 13`, `git`, Python 3.12 with `venv`,
and about 15 GB of free disk space (the TensorFlow source tree and the IREE
build trees are large).

`scripts/check_prerequisites.sh` reports exactly what is present and what is
missing and exits non-zero if a hard requirement is absent.

## Installation

```bash
git clone <toolchain-url> iree-coral-toolchain
cd iree-coral-toolchain
scripts/bootstrap_all.sh
```

The bootstrap creates the complete workspace next to the repository; no IREE
checkout is needed beforehand. The layout is:

```
workspace/
  iree-coral-toolchain/    this repository
  iree-coral-fork/         the IREE fork (cloned by bootstrap_iree.sh)
  tensorflow-source/       TensorFlow 2.7 source    (fetched)
  eigen-<commit>.tar.gz    pinned Eigen archive     (fetched, hash-verified)
  edgetpu/                 Coral edgetpu (headers + edgetpu_compiler)  (fetched)
  pycoral/                 Coral pycoral (libedgetpu.so.1.0)          (fetched)
  tools -> iree-coral-toolchain/tools          (link; generator lookup)
  .venv/                   Python environment       (generated)
  iree-*-build/            build trees             (generated)
  generated/               compiler intermediates   (generated)
```

Nothing is derived from the current directory, the hostname or the home
directory: every script resolves its repository root from its own location and
takes the workspace from the environment, so the tree may live anywhere and may
contain spaces. See [docs/INSTALL.md](docs/INSTALL.md) for the environment
overrides (`IREE_CORAL_SOURCE`, `IREE_CORAL_BUILD_DIR`,
`IREE_CORAL_INSTALL_PREFIX`, `IREE_CORAL_WORKSPACE`).

### Obtaining the IREE fork

IREE is not vendored. `versions.lock` pins the fork URL (`IREE_CORAL_REPO`),
the branch (`coral-edge-tpu`) and the exact commit. `scripts/bootstrap_iree.sh`
clones the fork into `../iree-coral-fork`, initializes its submodules and checks
out that commit; a checkout at another revision is moved to the pin, so the
build always uses the verified revision.

The pinned commit must exist on `IREE_CORAL_REPO`. To use another clone or
mirror that carries the branch:

```bash
scripts/bootstrap_all.sh --repo /path/to/iree-coral-fork
# or, for every script that clones:
IREE_CORAL_REPO=https://host/iree.git scripts/bootstrap_all.sh
```

## Building

```bash
scripts/bootstrap_all.sh
```

This runs, in order: prerequisites, the Python environment, the external inputs
(TensorFlow 2.7 source + Coral `edgetpu` + `pycoral` + the pinned Eigen
archive), the fork at its pinned commit, the Coral-enabled compiler, the native
shim, the Coral-enabled IREE runtime, the test data, and verification. The fork
is cloned from the URL pinned in `versions.lock` (`IREE_CORAL_REPO`); override
it with `--repo <url>`. Each step is a separate, re-runnable script:

```bash
scripts/check_prerequisites.sh
scripts/setup_venv.sh
scripts/fetch_third_party.sh
scripts/bootstrap_iree.sh            # --repo <url> to override the pin
scripts/build_compiler.sh            # iree-compile with the coral target
scripts/build_native_runtime.sh      # libcoral_native.so
scripts/build_runtime.sh             # iree-run-module with the Coral loader
scripts/fetch_test_data.sh
scripts/verify_installation.sh
```

Build parallelism is `-j2` by default; override with `JOBS=<n>`.

## Compiling models

```bash
export PYTHONPATH=$PWD/python

# import + compile in one step
python -m iree_coral.cli run mobilenet_v2_1.0_224_quant.tflite -o ./coral-out
```

Or the individual steps:

```bash
# 1. import a quantized TFLite model into a coral.graph MLIR module
python tools/import_tflite.py model.tflite -o model.mlir --tag model

# 2. compile to a VMFB with the Coral backend
"$WORKSPACE/iree-compiler-build/tools/iree-compile" \
    --iree-hal-target-backends=coral model.mlir -o model.vmfb
```

Compilation needs `edgetpu_compiler` but not the Edge TPU itself, so a
build-only machine can produce `.vmfb` files.

The compiler finds the generator scripts, the Python environment and the
proprietary `edgetpu_compiler` on its own: it looks for them under the
workspace, which is why the workspace carries the `tools` link shown above. The
invocation above therefore needs no `IREE_CORAL_*` variables, no
`LD_LIBRARY_PATH` and no particular working directory.

## Running inference

```bash
"$WORKSPACE/iree-runtime-install-build/tools/iree-run-module" \
    --device=local-sync --module=model.vmfb --function=run \
    --input=1x224x224x3xi8=@input.bin
```

The VMFB embeds the compiled model bytes and carries no model-file path, so it
can be copied to another machine and run there. No `LD_LIBRARY_PATH` is needed:
the runtime binary and libraries carry a relative `$ORIGIN` RUNPATH and find
`libcoral_native.so` and `libedgetpu.so.1` beside them.

## Examples

```bash
examples/minimal.sh          # import -> compile -> run, zero input tensor
examples/mobilenet_v2.sh     # same pipeline, classifies one test image
```

Both use the reference model fetched by `scripts/fetch_test_data.sh`, write
their artifacts to `$WORKSPACE/coral-example/`, and stop with a clear note when
no Edge TPU is present. `examples/minimal.sh` is the shortest complete pipeline;
`examples/mobilenet_v2.sh` adds image preprocessing and runs real data.

## Testing

```bash
scripts/run_all_tests.sh                 # every suite
scripts/verify_installation.sh           # per-check OK/FAIL
```

| suite | hardware | checks |
| --- | --- | --- |
| `tests/test_mobilenet_v2_quant.py` | no | lossless quantization: byte-exact filters, exact activation quantization, bit-exact per-operator outputs |
| `tests/test_mobilenet_v2_hardware.py` | yes | end-to-end on the device: one subgraph, all operators mapped, agreement with the original model |

Hardware suites are skipped with a clear note when `/dev/apex_0` or
`edgetpu_compiler` is missing. See [docs/TESTING.md](docs/TESTING.md) for what
each suite asserts and the relocation test.

## Troubleshooting

* **`doctor`** — `python -m iree_coral.cli doctor` reports missing build tools,
  artifacts and hardware, and points at the step that produced each.
* **No device** — check `gasket` is loaded and `/dev/apex_0` exists
  (`python -m iree_coral.cli devices`).
* **A node failed to map** — the compile aborts if any operation stayed on the
  CPU; inspect the `edgetpu_compiler` report printed during compilation.
* **`edgetpu_compiler` not found** — set `IREE_CORAL_EDGETPU_COMPILER` to its
  path; `scripts/fetch_third_party.sh` fetches a pinned build.
* **Bootstrap cannot clone the fork** — the pinned commit must exist on
  `IREE_CORAL_REPO`. Pass `--repo <url-or-path>` (or set `IREE_CORAL_REPO`) to a
  mirror that contains the `coral-edge-tpu` branch.
* **The TFLite build stops while fetching Eigen** — Eigen is pinned in
  `versions.lock` and downloaded from the TensorFlow GCS mirror (with a
  hash-verified cache in the workspace); `scripts/fetch_third_party.sh` fetches
  it, so no GitLab clone is attempted. If downloads are blocked, put the pinned
  archive at `EIGEN_ARCHIVE` (or set `EIGEN_URL` to a mirror).
* **Relocating a build tree** — a CMake build tree is not relocatable; re-run
  the build script in the new location. `scripts/test_relocation.sh` proves the
  sources and installed binaries are portable.

## Repository layout

```
iree-coral-toolchain/
  versions.lock            pinned revisions (IREE fork, TensorFlow, Eigen,
                           edgetpu, pycoral)
  scripts/                 bootstrap and verification scripts
  runtime/                 native TFLite + libedgetpu shim, loader negative test
  tools/                   importer and generator the compiler shells out to
  python/iree_coral/       the iree-coral CLI (doctor/version/compile/run/test)
  tests/                   MobileNetV2 quantization + hardware tests
  examples/                minimal and MobileNetV2 end-to-end examples
  docs/                    installation, architecture, testing notes
```

## License

The toolchain scripts and sources are provided under the Apache License 2.0.
The proprietary `edgetpu_compiler` and `libedgetpu` are fetched from the Google
Coral `edgetpu` and `pycoral` repositories and are not redistributed here.
