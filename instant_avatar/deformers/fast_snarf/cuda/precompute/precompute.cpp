#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

void launch_precompute(const torch::Tensor &voxel_w, const torch::Tensor &tfs, torch::Tensor &voxel_d, torch::Tensor &voxel_J, const torch::Tensor &offset, const torch::Tensor &scale);

void precompute(const torch::Tensor &voxel_w, const torch::Tensor &tfs,
                torch::Tensor &voxel_d, torch::Tensor &voxel_J,
                const torch::Tensor &offset, const torch::Tensor &scale) {
  launch_precompute(voxel_w, tfs, voxel_d, voxel_J, offset, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("precompute", &precompute); }
