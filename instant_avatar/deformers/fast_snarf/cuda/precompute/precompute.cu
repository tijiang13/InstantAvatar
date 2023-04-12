#include "ATen/Functions.h"
#include "ATen/core/TensorAccessor.h"
#include "c10/cuda/CUDAException.h"
#include "c10/cuda/CUDAStream.h"

#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/macros/Macros.h>
#include <ratio>
#include <vector>

#include <chrono>
using namespace std::chrono;

using namespace at;
using namespace at::cuda::detail;

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void precompute_kernel(
    const index_t npoints, const index_t d, const index_t h, const index_t w,
    PackedTensorAccessor32<scalar_t, 5> voxel_w, // shape=(N,200000, 9, 3)
    PackedTensorAccessor32<scalar_t, 4> tfs,     // shape=(N,200000, 3)
    PackedTensorAccessor32<scalar_t, 5> voxel_d, // shape=(N,3,8,32,32)
    PackedTensorAccessor32<scalar_t, 5> voxel_J, // shape=(N,9,8,32,32)
    PackedTensorAccessor32<scalar_t, 3> offset,  // shape=(N, 1, 3)
    PackedTensorAccessor32<scalar_t, 3> scale    // shape=(N, 1, 3)
) {
  index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= npoints)
    return;

  index_t idx_b = index / (d * h * w);
  index_t idx_d = index % (d * h * w) / (h * w);
  index_t idx_h = index % (d * h * w) % (h * w) / w;
  index_t idx_w = index % (d * h * w) % (h * w) % w;

  scalar_t coord_x =
      (((scalar_t)idx_w) / (w - 1) * 2 - 1) / scale[0][0][0] - offset[0][0][0];
  scalar_t coord_y =
      (((scalar_t)idx_h) / (h - 1) * 2 - 1) / scale[0][0][1] - offset[0][0][1];
  scalar_t coord_z =
      (((scalar_t)idx_d) / (d - 1) * 2 - 1) / scale[0][0][2] - offset[0][0][2];

  scalar_t J[12];

  for (index_t i0 = 0; i0 < 3; i0++) {
    for (index_t i1 = 0; i1 < 4; i1++) {
      J[i0 * 4 + i1] = 0;
      for (index_t j = 0; j < 24; j++) {
        J[i0 * 4 + i1] +=
            voxel_w[0][j][idx_d][idx_h][idx_w] * tfs[idx_b][j][i0][i1];
      }
    }
  }
  for (index_t i0 = 0; i0 < 3; i0++) {
    for (index_t i1 = 0; i1 < 4; i1++) {
      voxel_J[idx_b][i0 * 4 + i1][idx_d][idx_h][idx_w] = J[i0 * 4 + i1];
    }
  }

  for (index_t i0 = 0; i0 < 3; i0++) {
    scalar_t xi = J[i0 * 4 + 0] * coord_x + J[i0 * 4 + 1] * coord_y +
                  J[i0 * 4 + 2] * coord_z + J[i0 * 4 + 3];
    voxel_d[idx_b][i0][idx_d][idx_h][idx_w] = xi;
  }
}

void launch_precompute(const Tensor &voxel_w, const Tensor &tfs,
                       Tensor &voxel_d, Tensor &voxel_J, const Tensor &offset,
                       const Tensor &scale) {

  // calculate #threads required
  int64_t n_batch = voxel_d.size(0);
  int64_t d = voxel_d.size(2);
  int64_t h = voxel_d.size(3);
  int64_t w = voxel_d.size(4);

  int64_t count = n_batch * d * h * w;

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        voxel_w.scalar_type(), "precompute", [&] {
          precompute_kernel<<<GET_BLOCKS(count, 512), 512, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
              static_cast<int>(count), static_cast<int>(d), static_cast<int>(h),
              static_cast<int>(w), voxel_w.packed_accessor32<scalar_t, 5>(),
              tfs.packed_accessor32<scalar_t, 4>(),
              voxel_d.packed_accessor32<scalar_t, 5>(),
              voxel_J.packed_accessor32<scalar_t, 5>(),
              offset.packed_accessor32<scalar_t, 3>(),
              scale.packed_accessor32<scalar_t, 3>());

          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }

  cudaDeviceSynchronize();
}
