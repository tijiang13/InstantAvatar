#include "ATen/Functions.h"
#include "ATen/core/TensorBody.h"
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

void launch_broyden_kernel(
    torch::Tensor &x, const torch::Tensor &xd_tgt, const torch::Tensor &grid,
    const torch::Tensor &grid_J_inv, const torch::Tensor &tfs,
    const torch::Tensor &bone_ids, bool align_corners, torch::Tensor &J_inv,
    torch::Tensor &is_valid, const torch::Tensor &offset,
    const torch::Tensor &scale, float cvg_threshold, float dvg_threshold);

void fuse_broyden(torch::Tensor &x, const torch::Tensor &xd_tgt,
                  const torch::Tensor &grid, const torch::Tensor &grid_J_inv,
                  const torch::Tensor &tfs, const torch::Tensor &bone_ids,
                  bool align_corners, torch::Tensor &J_inv,
                  torch::Tensor &is_valid, torch::Tensor &offset,
                  torch::Tensor &scale, float cvg_threshold,
                  float dvg_threshold) {
  launch_broyden_kernel(x, xd_tgt, grid, grid_J_inv, tfs, bone_ids,
                        align_corners, J_inv, is_valid, offset, scale,
                        cvg_threshold, dvg_threshold);
  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fuse_broyden", &fuse_broyden);
}