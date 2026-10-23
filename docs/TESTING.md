# Testing

```bash
scripts/run_all_tests.sh                 # every suite
scripts/run_all_tests.sh test_mobilenet_v2_quant.py
```

Suites share the `generated/` artifact directory, so they run one at a time.
Hardware suites are skipped with a clear note when `/dev/apex_0` or the
proprietary `edgetpu_compiler` is missing, so the same driver works on a
build-only machine.

## Suites

### `test_mobilenet_v2_quant.py` — lossless quantisation

The regression that pins the quantisation behaviour. Compiles the real
`mobilenet_v2_1.0_224_quant.tflite` through the Coral pipeline and asserts,
against the source model:

* all convolution filters are byte-for-byte identical and dequantise to the
  same real values (max real error 0.0);
* all 66 operator output quantisations (scale and zero point) match the source;
* all 66 operators are bit-exact in real space on a real image
  (max absolute error 0.0, correlation 1.0);
* top-1 agrees and the logits correlate 1.0 end to end.

The per-operator check uses `tools/compare_layers.py`, which reads each
operator's output from a model truncated after that operator. Reading a live
interpreter's output tensor is unreliable: TFLite reuses intermediate buffers,
so earlier operators' buffers hold later data by the time `invoke()` returns.

No hardware is required.

### `test_mobilenet_v2_hardware.py` — physical Edge TPU

Imports, compiles and runs the model on the device and checks:

* exactly one Edge TPU subgraph, all 66 operators mapped (no CPU fallback);
* the hardware output is within a small documented tolerance
  (`HW_TOLERANCE = 8` codes; the measured maximum is 6) of the generated
  model's CPU reference — the Edge TPU uses its own fixed-point requantisation
  rounding, which moves only the two or three near-argmax logits;
* the predicted class agrees with the *original* pretrained model on every test
  image;
* latency and artifact sizes are reported.

Requires `/dev/apex_0`, the `gasket` driver and `edgetpu_compiler`.

### `runtime/coral_loader_negative_test.c` — loader rejection paths

This one is not part of `run_all_tests.sh`: it links against the runtime build
tree and needs a compiled Edge TPU model. Build and run it explicitly:

```bash
scripts/build_negative_test.sh
runtime/coral_loader_negative_test \
    "$WORKSPACE/generated/<tag>/model_edgetpu.tflite"
```

The program drives the Coral HAL loader through its public vtable and asserts
that malformed or foreign input is rejected with the documented status instead
of crashing or falling back to the CPU: it checks that `query_support` accepts
only the coral format, that an unknown format is `NOT_FOUND`, that empty and
garbage payloads are rejected, and that a valid model with a wrong binding count
or undersized input/output binding fails with `FAILED_PRECONDITION`. With
`expect-no-device` it also asserts that a valid model fails cleanly when no Edge
TPU is reachable.

## Data

`scripts/fetch_test_data.sh` copies the model and images from the Google Coral
`edgetpu` repository into `tests/test_data/` and verifies the model's sha256.

## Verifying an installation

```bash
scripts/verify_installation.sh              # everything available
scripts/verify_installation.sh --skip-hardware
```

Reports OK/FAIL per check and exits non-zero on failure.

## Path independence

Nothing in the toolchain depends on where it was checked out. Every script finds
the repository from its own path and takes every other root from the
environment (`IREE_CORAL_SOURCE`, `IREE_CORAL_BUILD_DIR`,
`IREE_CORAL_INSTALL_PREFIX`, `IREE_CORAL_WORKSPACE`, and the per-tree
overrides), never from the current directory, the host name or the home
directory. Two consequences are enforced by tests:

* built binaries and shared libraries carry a **relative** `$ORIGIN` RUNPATH, so
  they find `libcoral_native.so` and `libedgetpu.so.1` (copied beside them)
  without `LD_LIBRARY_PATH`;
* the generated `.vmfb` embeds the compiled model bytes and carries no
  model-file path, so it can be copied anywhere and still run.

The regression for both is the relocation test:

```bash
scripts/test_relocation.sh                  # native runtime + runtime
scripts/test_relocation.sh --with-compiler  # also rebuild iree-compile (slow)
```

It copies the sources to a fresh scratch directory, proves the copy contains no
reference to the original checkout, clean-rebuilds there (a CMake build tree is
never assumed relocatable — it is regenerated), checks that every RUNPATH is
relative, and finally runs a MobileNetV2 `.vmfb` from an unrelated directory
with a scrubbed environment (no `LD_LIBRARY_PATH`).
