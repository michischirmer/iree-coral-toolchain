// Native Coral loader tests: the loader is the supported runtime path, so these
// check that it rejects malformed or foreign executables.
//
// These drive the loader through its public vtable (the same code path the
// regular IREE runtime uses) and assert that every malformed input is rejected
// with the expected status. Nothing here falls back to the CPU/VMVX: each
// failure is the loader refusing to dispatch.
//
// Usage:
//   coral_loader_negative_test <model_edgetpu.tflite> [expect-no-device]
//
// With "expect-no-device" the process additionally asserts that loading a valid
// model fails because no Edge TPU is reachable (used with /dev/apex_0 hidden).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coral_native.h"
#include "iree/base/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/coral_executable_loader.h"
#include "iree/hal/local/local_executable.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                       \
  do {                                         \
    ++checks;                                  \
    if (cond) {                                \
      printf("ok   - ");                       \
      printf(__VA_ARGS__);                     \
      printf("\n");                            \
    } else {                                   \
      ++failures;                              \
      printf("FAIL - ");                       \
      printf(__VA_ARGS__);                     \
      printf("\n");                            \
    }                                          \
  } while (0)

static uint8_t* read_file(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long size = ftell(f);
  if (size <= 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  uint8_t* data = (uint8_t*)malloc((size_t)size);
  if (!data) {
    fclose(f);
    return NULL;
  }
  if (fread(data, 1, (size_t)size, f) != (size_t)size) {
    free(data);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *out_size = (size_t)size;
  return data;
}

// Attempts a load of the given coral-format payload and returns the status.
static iree_status_t try_load_coral(iree_hal_executable_loader_t* loader,
                                    const void* data, size_t size,
                                    iree_hal_executable_t** out_executable) {
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format =
      iree_make_cstring_view(IREE_HAL_CORAL_EXECUTABLE_FORMAT);
  params.executable_data = iree_make_const_byte_span(data, size);
  return iree_hal_executable_loader_try_load(loader, &params,
                                             /*worker_capacity=*/1,
                                             out_executable);
}

// Runs one issue_call with the given binding count/lengths and returns status.
static iree_status_t issue(iree_hal_executable_t* executable,
                           uint8_t binding_count, void* const* binding_ptrs,
                           const size_t* binding_lengths) {
  iree_hal_executable_dispatch_state_v0_t dispatch_state;
  memset(&dispatch_state, 0, sizeof(dispatch_state));
  dispatch_state.workgroup_size_x = 1;
  dispatch_state.workgroup_size_y = 1;
  dispatch_state.workgroup_size_z = 1;
  dispatch_state.workgroup_count_x = 1;
  dispatch_state.workgroup_count_y = 1;
  dispatch_state.workgroup_count_z = 1;
  dispatch_state.binding_count = binding_count;
  dispatch_state.binding_ptrs = binding_ptrs;
  dispatch_state.binding_lengths = binding_lengths;
  iree_hal_executable_workgroup_state_v0_t workgroup_state;
  memset(&workgroup_state, 0, sizeof(workgroup_state));
  return iree_hal_local_executable_issue_call(
      (iree_hal_local_executable_t*)executable, /*ordinal=*/0, &dispatch_state,
      &workgroup_state, /*worker_id=*/0);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model_edgetpu.tflite> [expect-no-device]\n",
            argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  int expect_no_device = (argc > 2 && strcmp(argv[2], "expect-no-device") == 0);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_executable_loader_t* loader = NULL;
  iree_status_t status =
      iree_hal_coral_executable_loader_create(host_allocator, &loader);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }

  //--------------------------------------------------------------------------
  // Invalid format.
  //--------------------------------------------------------------------------
  CHECK(!iree_hal_executable_loader_query_support(
            loader, /*caching_mode=*/0,
            iree_make_cstring_view("vmvx-bytecode-fb")),
        "query_support rejects the vmvx format");
  CHECK(!iree_hal_executable_loader_query_support(
            loader, /*caching_mode=*/0,
            iree_make_cstring_view("embedded-elf-x86_64")),
        "query_support rejects the embedded-elf format");
  CHECK(iree_hal_executable_loader_query_support(
            loader, /*caching_mode=*/0,
            iree_make_cstring_view(IREE_HAL_CORAL_EXECUTABLE_FORMAT)),
        "query_support accepts the coral format");

  {
    iree_hal_executable_params_t params;
    iree_hal_executable_params_initialize(&params);
    params.executable_format = iree_make_cstring_view("not-a-real-format");
    const uint8_t junk[1] = {0};
    params.executable_data = iree_make_const_byte_span(junk, sizeof(junk));
    iree_hal_executable_t* executable = NULL;
    status = iree_hal_executable_loader_try_load(loader, &params, 1, &executable);
    CHECK(iree_status_is_not_found(status) && executable == NULL,
          "try_load rejects an unknown format with NOT_FOUND (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
  }

  //--------------------------------------------------------------------------
  // Empty payload.
  //--------------------------------------------------------------------------
  {
    iree_hal_executable_t* executable = NULL;
    status = try_load_coral(loader, NULL, 0, &executable);
    CHECK(iree_status_is_invalid_argument(status) && executable == NULL,
          "try_load rejects an empty payload with INVALID_ARGUMENT (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
  }

  //--------------------------------------------------------------------------
  // Garbage payload (non-empty but not a valid Edge TPU model).
  //--------------------------------------------------------------------------
  {
    uint8_t junk[4096];
    memset(junk, 0xA5, sizeof(junk));
    iree_hal_executable_t* executable = NULL;
    status = try_load_coral(loader, junk, sizeof(junk), &executable);
    CHECK(!iree_status_is_ok(status) && executable == NULL,
          "try_load rejects a garbage payload (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
  }

  //--------------------------------------------------------------------------
  // Real model: this is where the Edge TPU is actually required.
  //--------------------------------------------------------------------------
  size_t model_size = 0;
  uint8_t* model_data = read_file(model_path, &model_size);
  CHECK(model_data != NULL && model_size > 0,
        "read reference model '%s' (%zu bytes)", model_path, model_size);

  iree_hal_executable_t* executable = NULL;
  if (model_data) {
    status = try_load_coral(loader, model_data, model_size, &executable);
  } else {
    status = iree_make_status(IREE_STATUS_NOT_FOUND, "missing model file");
  }

  if (expect_no_device) {
    CHECK(!iree_status_is_ok(status) && executable == NULL,
          "try_load fails when no Coral device is present (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
    free(model_data);
    iree_hal_executable_loader_release(loader);
    printf("\n%d checks, %d failures (no-device mode)\n", checks, failures);
    return failures == 0 ? 0 : 1;
  }

  CHECK(iree_status_is_ok(status) && executable != NULL,
        "try_load loads the real model from embedded bytes");
  if (!iree_status_is_ok(status) || executable == NULL) {
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
    }
    free(model_data);
    iree_hal_executable_loader_release(loader);
    printf("\n%d checks, %d failures\n", checks, failures);
    return 1;
  }

  //--------------------------------------------------------------------------
  // Wrong binding count.
  //--------------------------------------------------------------------------
  {
    uint8_t in[4096] = {0};
    uint8_t out[4096] = {0};
    void* ptrs[2] = {in, out};
    size_t lengths[2] = {sizeof(in), sizeof(out)};
    status = issue(executable, /*binding_count=*/1, ptrs, lengths);
    CHECK(iree_status_is_failed_precondition(status),
          "issue_call rejects binding_count=1 with FAILED_PRECONDITION (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
  }

  //--------------------------------------------------------------------------
  // Size error: input binding smaller than the model input.
  //--------------------------------------------------------------------------
  {
    uint8_t in[1] = {0};
    uint8_t out[4096] = {0};
    void* ptrs[2] = {in, out};
    size_t lengths[2] = {0, sizeof(out)};  // input length 0 < model input size.
    status = issue(executable, /*binding_count=*/2, ptrs, lengths);
    CHECK(iree_status_is_failed_precondition(status),
          "issue_call rejects an undersized input binding with "
          "FAILED_PRECONDITION (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
  }

  //--------------------------------------------------------------------------
  // Size error: output binding smaller than the model output.
  //--------------------------------------------------------------------------
  {
    uint8_t in[4096] = {0};
    uint8_t out[1] = {0};
    void* ptrs[2] = {in, out};
    size_t lengths[2] = {sizeof(in), 0};  // output length 0 < model output size.
    status = issue(executable, /*binding_count=*/2, ptrs, lengths);
    CHECK(iree_status_is_failed_precondition(status),
          "issue_call rejects an undersized output binding with "
          "FAILED_PRECONDITION (got %s)",
          iree_status_code_string(iree_status_code(status)));
    iree_status_ignore(status);
  }

  free(model_data);
  iree_hal_executable_t* tmp = executable;
  iree_hal_executable_release(tmp);
  iree_hal_executable_loader_release(loader);
  printf("\n%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
