// Native Coral Edge TPU runtime implementation.
//
// Links the TensorFlow Lite C API and `libedgetpu` and exposes a small, stable
// C API (see coral_native.h) used by the IREE Coral HAL executable loader.

#include "coral_native.h"

#include <stdlib.h>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

#include "edgetpu_c.h"

//===----------------------------------------------------------------------===//
// Model session
//===----------------------------------------------------------------------===//

struct coral_native_model {
  TfLiteModel* model;
  TfLiteInterpreter* interpreter;
  TfLiteDelegate* delegate;  // NULL for the CPU reference session.
  struct edgetpu_device* devices;
  size_t device_count;
  TfLiteTensor* input_tensor;
  const TfLiteTensor* output_tensor;
  int uses_edgetpu;
};

static coral_native_model_t* coral_native_model_create_impl(
    const unsigned char* model_data, size_t model_size, int use_edgetpu) {
  if (!model_data || !model_size) return NULL;

  coral_native_model_t* session =
      (coral_native_model_t*)calloc(1, sizeof(coral_native_model_t));
  if (!session) return NULL;

  if (use_edgetpu) {
    session->devices = edgetpu_list_devices(&session->device_count);
    if (!session->devices || session->device_count == 0) {
      free(session);
      return NULL;
    }
  }

  session->model = TfLiteModelCreate(model_data, model_size);
  if (!session->model) goto fail;

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  if (!options) goto fail;

  if (use_edgetpu) {
    session->delegate = edgetpu_create_delegate(
        session->devices[0].type, session->devices[0].path, NULL, 0);
    if (!session->delegate) {
      TfLiteInterpreterOptionsDelete(options);
      goto fail;
    }
    TfLiteInterpreterOptionsAddDelegate(options, session->delegate);
  }

  session->interpreter = TfLiteInterpreterCreate(session->model, options);
  TfLiteInterpreterOptionsDelete(options);
  if (!session->interpreter) goto fail;

  if (TfLiteInterpreterAllocateTensors(session->interpreter) != kTfLiteOk) {
    goto fail;
  }
  if (TfLiteInterpreterGetInputTensorCount(session->interpreter) < 1 ||
      TfLiteInterpreterGetOutputTensorCount(session->interpreter) < 1) {
    goto fail;
  }

  session->input_tensor = TfLiteInterpreterGetInputTensor(session->interpreter, 0);
  session->output_tensor =
      TfLiteInterpreterGetOutputTensor(session->interpreter, 0);
  session->uses_edgetpu = use_edgetpu;
  return session;

fail:
  coral_native_model_destroy(session);
  return NULL;
}

coral_native_model_t* coral_native_model_create(const unsigned char* model_data,
                                                size_t model_size) {
  return coral_native_model_create_impl(model_data, model_size,
                                        /*use_edgetpu=*/1);
}

coral_native_model_t* coral_native_reference_create(
    const unsigned char* model_data, size_t model_size) {
  return coral_native_model_create_impl(model_data, model_size,
                                        /*use_edgetpu=*/0);
}

void coral_native_model_destroy(coral_native_model_t* model) {
  if (!model) return;
  if (model->interpreter) TfLiteInterpreterDelete(model->interpreter);
  if (model->model) TfLiteModelDelete(model->model);
  if (model->delegate) edgetpu_free_delegate(model->delegate);
  if (model->devices) edgetpu_free_devices(model->devices);
  free(model);
}

int coral_native_model_uses_edgetpu(const coral_native_model_t* model) {
  return model ? model->uses_edgetpu : 0;
}

size_t coral_native_model_input_size(const coral_native_model_t* model) {
  if (!model || !model->input_tensor) return 0;
  return TfLiteTensorByteSize(model->input_tensor);
}

size_t coral_native_model_output_size(const coral_native_model_t* model) {
  if (!model || !model->output_tensor) return 0;
  return TfLiteTensorByteSize(model->output_tensor);
}

int coral_native_model_invoke(coral_native_model_t* model,
                              const unsigned char* in_data, size_t in_size,
                              unsigned char* out_data, size_t out_capacity,
                              size_t* out_written) {
  if (!model || !model->interpreter || !model->input_tensor ||
      !model->output_tensor || !in_data || !out_data || !out_written) {
    return 1;
  }
  *out_written = 0;

  const size_t input_size = TfLiteTensorByteSize(model->input_tensor);
  const size_t output_size = TfLiteTensorByteSize(model->output_tensor);
  if (in_size != input_size) return 2;
  if (out_capacity < output_size) return 3;

  if (TfLiteTensorCopyFromBuffer(model->input_tensor, in_data, input_size) !=
      kTfLiteOk) {
    return 4;
  }
  if (TfLiteInterpreterInvoke(model->interpreter) != kTfLiteOk) {
    return 5;
  }
  if (TfLiteTensorCopyToBuffer(model->output_tensor, out_data, output_size) !=
      kTfLiteOk) {
    return 6;
  }

  *out_written = output_size;
  return 0;
}
