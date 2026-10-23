// Native Coral Edge TPU runtime API.
//
// Provides the pieces the IREE Coral HAL executable loader needs to turn the
// embedded `coral-edgetpu-tflite` payload into a loadable, dispatchable
// executable:
//
//   * `coral_native_model_*` keeps a TFLite interpreter + Edge TPU delegate
//     alive across dispatches so the model is loaded once per executable.
//   * `coral_native_reference_*` runs the same model on the CPU (no Edge TPU
//     delegate) so tests can compare hardware results against a reference.
//
// These functions are implemented in `libcoral_native.so`, which links the
// TensorFlow Lite C API and `libedgetpu`. The IREE runtime side never links
// TensorFlow directly.

#ifndef IREE_CORAL_NATIVE_H_
#define IREE_CORAL_NATIVE_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a loaded model session.
typedef struct coral_native_model coral_native_model_t;

// Creates a model session that runs on the Edge TPU.
// `model_data`/`model_size` are the bytes of a compiled `_edgetpu.tflite`
// model. Returns NULL on failure.
coral_native_model_t* coral_native_model_create(const unsigned char* model_data,
                                                size_t model_size);

// Creates a model session that runs entirely on the CPU (no Edge TPU
// delegate). Used as the reference implementation in tests.
coral_native_model_t* coral_native_reference_create(
    const unsigned char* model_data, size_t model_size);

// Destroys a model session created by either create function.
void coral_native_model_destroy(coral_native_model_t* model);

// Returns true if the session is backed by an Edge TPU delegate.
int coral_native_model_uses_edgetpu(const coral_native_model_t* model);

// Number of bytes expected for the (single) input and output tensors.
size_t coral_native_model_input_size(const coral_native_model_t* model);
size_t coral_native_model_output_size(const coral_native_model_t* model);

// Runs one inference. `in_data`/`in_size` must match the input tensor size and
// `out_data`/`out_capacity` must be at least the output tensor size. On success
// returns 0 and stores the number of bytes written in `*out_written`.
int coral_native_model_invoke(coral_native_model_t* model,
                              const unsigned char* in_data, size_t in_size,
                              unsigned char* out_data, size_t out_capacity,
                              size_t* out_written);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_CORAL_NATIVE_H_
