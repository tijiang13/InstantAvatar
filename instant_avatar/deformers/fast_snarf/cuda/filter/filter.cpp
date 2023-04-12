#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>





void launch_filter(
  const torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &mask);

torch::Tensor filter(const torch::Tensor &x, const torch::Tensor &mask) {
  auto mask_size = mask.sizes();

  auto output = at::empty({mask_size[0], mask_size[1], mask_size[2]}, mask.options());
  launch_filter(output, x, mask);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("filter", &filter);
}
