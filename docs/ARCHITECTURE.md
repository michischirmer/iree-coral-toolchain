# Architecture

The toolchain builds and drives a Coral-enabled IREE compiler and runtime. The
IREE source itself lives in a separate fork; this document explains what each
repository contributes and how a model reaches the Edge TPU.

## Two repositories

### `iree-coral-fork` (IREE fork)

A fork of upstream IREE with one commit on the `coral-edge-tpu` branch adding:

* **`compiler/plugins/target/Coral/`** — the `coral` target backend. A
  generation pass converts an imported `coral.graph` module into a real TFLite
  model, runs the proprietary `edgetpu_compiler` on it, verifies that every
  operation mapped to the Edge TPU, and embeds the resulting Edge TPU
  executable in the VMFB.
* **`runtime/src/iree/hal/local/loaders/coral_executable_loader.c`** — a HAL
  executable loader that recognises the embedded `coral-edgetpu-tflite`
  executable and dispatches it through the native shim.

The toolchain passes the compiler's run-time roots explicitly
(`IREE_CORAL_TOOLS_ROOT`, `IREE_CORAL_PYTHON`, `IREE_CORAL_EDGETPU_COMPILER`,
`IREE_CORAL_GENERATED_ROOT`), because the generator scripts and the Edge TPU
compiler live in the toolchain repository rather than beside the build tree, so
a compiler built in the toolchain workspace finds them without any hard-coded
path. A bare `iree-compile` with none of them set falls back to a build-tree
heuristic; the workspace's `tools` link is what makes that heuristic resolve to
this repository.

### `iree-coral-toolchain` (this repository)

* **`scripts/`** — bootstrap and verification. `lib.sh` centralises path
  resolution so the Python CLI and the shell scripts agree.
* **`runtime/`** — the native `libcoral_native.so` shim (TFLite 2.7 C API +
  `libedgetpu` behind the small C interface in `coral_native.h`) and the loader
  negative-test source.
* **`tools/`** — the importer (`import_tflite.py`) that turns a quantized
  TFLite model into `coral.graph`, and the generator (`generate_net_model.py`)
  that turns the pass's generation spec back into a TFLite model.
* **`python/iree_coral/`** — the `iree-coral` CLI.
* **`tests/`** — the MobileNetV2 quantisation regression and hardware tests.
* **`examples/`** — the minimal end-to-end pipeline and a MobileNetV2 image
  classification demo.

## Data flow

```
 mobilenet_v2_1.0_224_quant.tflite      pretrained, UINT8, 1001 classes
        |
        |  tools/import_tflite.py         (toolchain)
        v
   coral.graph  (MLIR)                  graph + per-node quantisation metadata
        |
        |  iree-compile --iree-hal-target-backends=coral   (fork)
        v
   CoralGenerationPass
        |  writes spec.json
        |  runs tools/generate_net_model.py  -> model.tflite
        |  runs edgetpu_compiler             -> model_edgetpu.tflite
        |  verifies 66/66 operators mapped to the Edge TPU
        v
   model.vmfb                           one Edge TPU executable, embedded
        |
        |  iree-run-module --device=local-sync             (fork + toolchain)
        v
   coral_executable_loader  ->  libcoral_native.so  ->  libedgetpu  ->  Edge TPU
```

No Python and no TFLite run at inference time: the Edge TPU executable is
embedded in the VMFB and the loader hands its bytes to `libedgetpu`.

## Quantisation preservation

The generated model is required to preserve the source model's quantisation
**losslessly** — this is what the tests assert, and it is why the pipeline is
more than a shape conversion:

* each filter is installed byte-exactly with the source scale and zero point;
* every operator's output (activation) scale and zero point is installed from
  the source operator;
* the interface tensors carry the source model's exact input/output quantisation;
* the source's INT32 bias is installed verbatim, in units of
  `source_input_scale * weight_scale`.

MobileNetV2's filters have non-zero zero points, which an INT8 tensor cannot
represent, so the whole model is relabelled to the source's own UINT8 scheme (a
value-preserving byte shift) — the scheme TFLite's CPU kernels and
`edgetpu_compiler` both accept. The regression test
(`tests/test_mobilenet_v2_quant.py`) pins this behaviour.

## The native shim

IREE never links TensorFlow. `libcoral_native.so` is the only bridge:
`coral_native.h` declares a handful of functions (create a session, get
tensor sizes, invoke) implemented against the TFLite C API plus `libedgetpu`.
The HAL loader calls those functions. This keeps the IREE build free of a
TensorFlow dependency and confines the proprietary pieces to one shared object.
