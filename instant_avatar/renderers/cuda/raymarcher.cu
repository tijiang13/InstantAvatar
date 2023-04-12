#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/macros/Macros.h>
#include <math.h>
using namespace at;

inline __device__ float signf(const float x) { return copysignf(1.0f, x); }
inline __device__ float clamp(float f, float a, float b) { return fmaxf(a, fminf(f, b)); }

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void raymarch_test_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_d,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> nears,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> fars,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> alive_indices,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> density_grid,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> scale,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> step_size,
    const int N_steps,
    const int grid_size,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> pts,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> deltas,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> depths
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= alive_indices.size(0)) return;

    const size_t n = alive_indices[i];
    const float ox = rays_o[n][0], oy = rays_o[n][1], oz = rays_o[n][2];
    const float dx = rays_d[n][0], dy = rays_d[n][1], dz = rays_d[n][2];
    const float dx_inv = 1 / dx, dy_inv = 1 / dy, dz_inv = 1 / dz;
    const float cx = offset[0], cy = offset[1], cz = offset[2];
    const float sx = grid_size / scale[0], sy = grid_size / scale[1], sz = grid_size / scale[2];

    const auto far = fars[n];
    const float dt = step_size[n];

    auto s = 0;
    auto t = nears[n];
    while (t < far && s < N_steps){
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;

        const int nx = clamp((x - cx) * sx, 0.0f, grid_size - 1.0f);
        const int ny = clamp((y - cy) * sy, 0.0f, grid_size - 1.0f);
        const int nz = clamp((z - cz) * sz, 0.0f, grid_size - 1.0f);
        const bool occupied = density_grid[nx][ny][nz];

        if (occupied) {
            pts[i][s][0] = x;
            pts[i][s][1] = y;
            pts[i][s][2] = z;

            deltas[i][s] = dt;
            depths[i][s] = t;

            t += dt;
            s++;
        } else {
            t += dt;
            // const float tx = ((nx + 0.5 * (1 + signf(dx))) / sx + cx - x) * dx_inv;
            // const float ty = ((ny + 0.5 * (1 + signf(dy))) / sy + cy - y) * dy_inv;
            // const float tz = ((nz + 0.5 * (1 + signf(dz))) / sz + cz - z) * dz_inv;
            // t = max(t + dt, fminf(tx, fminf(ty, tz)));
        }
    }
    nears[n] = t;
}


std::vector<torch::Tensor> raymarch_test_cuda(const torch::Tensor& rays_o,
                                              const torch::Tensor& rays_d,
                                              torch::Tensor& nears,
                                              const torch::Tensor& fars,
                                              const torch::Tensor& alive_indices,
                                              const torch::Tensor& density_grid,
                                              const torch::Tensor& scale,
                                              const torch::Tensor& offset,
                                              const torch::Tensor& step_size,
                                              const int N_steps) {
    const auto N_rays = alive_indices.size(0);
    const auto grid_size = density_grid.size(0);

    auto pts = at::zeros({N_rays, N_steps, 3}, rays_o.options());
    auto deltas = at::zeros({N_rays, N_steps}, rays_o.options());
    auto depths = at::zeros({N_rays, N_steps}, rays_o.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarch_test_cuda", [&] {
        raymarch_test_kernel<scalar_t><<<(N_rays + 512 - 1) / 512, 512>>>(
            rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            nears.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            fars.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            alive_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            density_grid.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            step_size.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            N_steps,
            grid_size,
            pts.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            deltas.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            depths.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    });
    return {pts, deltas, depths};
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void raymarch_train_kernel(
    const uint32_t n_rays,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> nears,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> fars,
    const torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> density_grid,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> scale,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> step_size,
    const int N_steps,
    const int grid_size,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> depths
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_rays) return;
    const float ox = rays_o[n][0], oy = rays_o[n][1], oz = rays_o[n][2];
    const float dx = rays_d[n][0], dy = rays_d[n][1], dz = rays_d[n][2];
    const float dx_inv = 1 / dx, dy_inv = 1 / dy, dz_inv = 1 / dz;
    const float cx = offset[0], cy = offset[1], cz = offset[2];
    const float sx = grid_size / scale[0], sy = grid_size / scale[1], sz = grid_size / scale[2];

    const auto far = fars[n];
    const float dt = step_size[n];

    auto s = 0;
    auto t = nears[n];
    while (t < far && s < N_steps){
        const float x = ox + t * dx;
        const float y = oy + t * dy;
        const float z = oz + t * dz;

        const int nx = clamp((x - cx) * sx, 0.0f, grid_size - 1.0f);
        const int ny = clamp((y - cy) * sy, 0.0f, grid_size - 1.0f);
        const int nz = clamp((z - cz) * sz, 0.0f, grid_size - 1.0f);
        const bool occupied = density_grid[nx][ny][nz];

        if (occupied) {
            depths[n][s] = t;
            t += dt;
            s++;
        } else {
            t += dt;
        }
    }
}


torch::Tensor raymarch_train_cuda(const torch::Tensor& rays_o,
                                  const torch::Tensor& rays_d,
                                  const torch::Tensor& nears,
                                  const torch::Tensor& fars,
                                  const torch::Tensor& density_grid,
                                  const torch::Tensor& scale,
                                  const torch::Tensor& offset,
                                  const torch::Tensor& step_size,
                                  const int N_steps) {
    const auto N_rays = rays_o.size(0);
    const auto grid_size = density_grid.size(0);

    auto depths = at::zeros({N_rays, N_steps}, rays_o.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rays_o.type(), "raymarch_train_cuda", [&] {
        raymarch_train_kernel<scalar_t><<<(N_rays + 512 - 1) / 512, 512>>>(
	    N_rays,
            rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            nears.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            fars.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            density_grid.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
            scale.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            offset.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            step_size.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            N_steps,
            grid_size,
            depths.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    });
    return depths;
}


template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void composite_test_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rgb_vals,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sigma_vals,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> delta_vals,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> depth_vals,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> alive_indices,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> color,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> depth,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> nohit,
    float thresh
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= alive_indices.size(0)) return;
    const size_t n = alive_indices[i];
    const int N_steps = rgb_vals.size(1);

    auto T = nohit[n];
    int s = 0;
    while (s < N_steps && T > 1e-4 && delta_vals[i][s] > 0) {
        const auto tau = __expf(-sigma_vals[i][s] * delta_vals[i][s]);
        const auto alpha = 1.0f - tau;
	if (alpha < thresh) {
	    s++;
	    continue;
	}
        const auto w = alpha * T;

        color[n][0] += w * rgb_vals[i][s][0];
        color[n][1] += w * rgb_vals[i][s][1];
        color[n][2] += w * rgb_vals[i][s][2];
        depth[n] += w * depth_vals[i][s];
        T *= tau;
        s++;
    }
    nohit[n] = T;
}


void composite_test_cuda(const torch::Tensor& rgb_vals,
                         const torch::Tensor& sigma_vals,
                         const torch::Tensor& delta_vals,
                         const torch::Tensor& depth_vals,
                         const torch::Tensor& alive_indices,
                         torch::Tensor& color,
                         torch::Tensor& depth,
                         torch::Tensor& no_hit,
			 float thresh) {
    const auto N = alive_indices.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigma_vals.type(), "composite_test_cuda", [&] {
        composite_test_kernel<scalar_t><<<(N + 512 - 1) / 512, 512>>>(
            rgb_vals.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sigma_vals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            delta_vals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            depth_vals.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            alive_indices.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            color.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            depth.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            no_hit.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
	    thresh
        );
    });
    return;
}
