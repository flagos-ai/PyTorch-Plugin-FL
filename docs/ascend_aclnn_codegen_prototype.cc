// Copyright (c) 2026, BAAI. All rights reserved.
//
// Standalone feasibility prototype for the "aclnn codegen" path (docs/ascend_npu_plan.md).
//
// Goal: prove that the *kernel body shape* an aclnn codegen would emit — a
// two-phase `aclnn<Op>GetWorkspaceSize` + `aclnn<Op>` call over aclTensors
// built from raw NPU storage — actually computes on real Ascend hardware,
// WITHOUT depending on the full torch_fl build (which is currently broken on
// Ascend, see plan doc §现状). It mirrors EXEC_ASCEND_CMD + AclTensorWrapper
// but uses raw aclrt allocations instead of at::Tensor storage.
//
// Build & run (on the Ascend 910 box, CANN at /usr/local/Ascend):
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   bash docs/build_ascend_prototype.sh
//
// Expected: "PASS: aclnnSqrt on NPU matches CPU reference".

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_sqrt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(expr)                                                       \
  do {                                                                    \
    aclError _ret = (expr);                                               \
    if (_ret != ACL_SUCCESS) {                                            \
      fprintf(stderr, "FAIL %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, _ret); \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Build an aclTensor over an NPU device buffer. This is exactly what
// AclTensorWrapper does in op_api_common.h, minus the at::Tensor plumbing.
static aclTensor* MakeAclTensor(void* dev_ptr,
                                const std::vector<int64_t>& shape,
                                std::vector<int64_t>& strides_out) {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  strides_out.assign(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides_out[i] = strides_out[i + 1] * shape[i + 1];
  }
  std::vector<int64_t> storage_dims = {n};
  return aclCreateTensor(
      shape.data(), shape.size(), ACL_FLOAT,
      strides_out.data(), /*offset=*/0, ACL_FORMAT_ND,
      storage_dims.data(), storage_dims.size(), dev_ptr);
}

int main() {
  CHECK(aclInit(nullptr));
  CHECK(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  CHECK(aclrtCreateStream(&stream));

  const std::vector<int64_t> shape = {4, 8};
  const int64_t n = 32;
  const size_t nbytes = n * sizeof(float);

  std::vector<float> h_in(n), h_out(n);
  for (int64_t i = 0; i < n; ++i) h_in[i] = static_cast<float>(i + 1);

  void* d_in = nullptr;
  void* d_out = nullptr;
  CHECK(aclrtMalloc(&d_in, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&d_out, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(d_in, nbytes, h_in.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

  std::vector<int64_t> st_in, st_out;
  aclTensor* t_in = MakeAclTensor(d_in, shape, st_in);
  aclTensor* t_out = MakeAclTensor(d_out, shape, st_out);

  // ---- This is the codegen-emitted body shape (cf. EXEC_ASCEND_CMD) ----
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK(aclnnSqrtGetWorkspaceSize(t_in, t_out, &workspace_size, &executor));

  void* workspace = nullptr;
  if (workspace_size > 0) {
    CHECK(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  CHECK(aclnnSqrt(workspace, workspace_size, executor, stream));
  CHECK(aclrtSynchronizeStream(stream));
  // ----------------------------------------------------------------------

  CHECK(aclrtMemcpy(h_out.data(), nbytes, d_out, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

  double max_err = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    double ref = std::sqrt(static_cast<double>(h_in[i]));
    max_err = std::max(max_err, std::fabs(ref - h_out[i]));
  }
  printf("aclnnSqrt sample: in[3]=%.1f out[3]=%.6f (ref=%.6f)  max_err=%.3e\n",
         h_in[3], h_out[3], std::sqrt(h_in[3]), max_err);

  if (workspace) aclrtFree(workspace);
  aclDestroyTensor(t_in);
  aclDestroyTensor(t_out);
  aclrtFree(d_in);
  aclrtFree(d_out);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();

  if (max_err < 1e-5) {
    printf("PASS: aclnnSqrt on NPU matches CPU reference\n");
    return 0;
  }
  printf("FAIL: max_err too large\n");
  return 1;
}
