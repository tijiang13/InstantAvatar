#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> raymarch_test_cuda(const torch::Tensor& rays_o,
                                              const torch::Tensor& rays_d,
                                              torch::Tensor& nears,
                                              const torch::Tensor& fars,
                                              const torch::Tensor& alives,
                                              const torch::Tensor& density_grid,
                                              const torch::Tensor& scale,
                                              const torch::Tensor& offset,
                                              const torch::Tensor& step_size,
                                              const int N_steps);

std::vector<torch::Tensor> raymarch_test(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    torch::Tensor& nears,
    const torch::Tensor& fars,
    const torch::Tensor& alives,
    const torch::Tensor& density_grid,
    const torch::Tensor& scale,
    const torch::Tensor& offset,
    const torch::Tensor& step_size,
    const int N_steps) {

    return raymarch_test_cuda(rays_o, rays_d, nears, fars, alives, density_grid, scale, offset, step_size, N_steps);
}

torch::Tensor raymarch_train_cuda(const torch::Tensor& rays_o,
                                  const torch::Tensor& rays_d,
                                  const torch::Tensor& nears,
                                  const torch::Tensor& fars,
                                  const torch::Tensor& density_grid,
                                  const torch::Tensor& scale,
                                  const torch::Tensor& offset,
                                  const torch::Tensor& step_size,
                                  const int N_steps);

torch::Tensor raymarch_train(
    const torch::Tensor& rays_o,
    const torch::Tensor& rays_d,
    const torch::Tensor& nears,
    const torch::Tensor& fars,
    const torch::Tensor& density_grid,
    const torch::Tensor& scale,
    const torch::Tensor& offset,
    const torch::Tensor& step_size,
    const int N_steps) {

    return raymarch_train_cuda(rays_o, rays_d, nears, fars, density_grid, scale, offset, step_size, N_steps);
}

void composite_test_cuda(const torch::Tensor& rgb_vals,
                         const torch::Tensor& sigma_vals,
                         const torch::Tensor& delta_vals,
                         const torch::Tensor& depth_vals,
                         const torch::Tensor& alive_indices,
                         torch::Tensor& color,
                         torch::Tensor& depth,
                         torch::Tensor& no_hit,
			 float thresh);

void composite_test(const torch::Tensor& rgb_vals,
                   const torch::Tensor& sigma_vals,
                   const torch::Tensor& delta_vals,
                   const torch::Tensor& depth_vals,
                   const torch::Tensor& alive_indices,
                   torch::Tensor& color,
                   torch::Tensor& depth,
                   torch::Tensor& no_hit,
		   float thresh) {
    composite_test_cuda(rgb_vals, sigma_vals, delta_vals, depth_vals, alive_indices, color, depth, no_hit, thresh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raymarch_test", raymarch_test);
    m.def("raymarch_train", raymarch_train);
    m.def("composite_test", composite_test);
}
