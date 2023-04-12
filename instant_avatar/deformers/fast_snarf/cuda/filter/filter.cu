#include <vector>
#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAContext.h>


template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void filter(
    const index_t nthreads,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> mask,
    torch::PackedTensorAccessor32<bool, 3, torch::RestrictPtrTraits> output) {

    index_t n_batch = mask.size(0);
    index_t n_point = mask.size(1);
    index_t n_init = mask.size(2);
    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {

        const index_t i_batch = index / (n_batch*n_point);
        const index_t i_point = index % (n_batch*n_point);


        for(index_t i = 0; i < n_init; i++) {
            if(!mask[i_batch][i_point][i]){
                output[i_batch][i_point][i] = false;
                continue;
            }
            scalar_t xi0 = x[i_batch][i_point][i][0];
            scalar_t xi1 = x[i_batch][i_point][i][1];
            scalar_t xi2 = x[i_batch][i_point][i][2];

            bool flag = true;
            for(index_t j = i+1; j < n_init; j++){
                if(!mask[i_batch][i_point][j]){
                    continue;
                }
                scalar_t d0 = xi0 - x[i_batch][i_point][j][0];
                scalar_t d1 = xi1 - x[i_batch][i_point][j][1];
                scalar_t d2 = xi2 - x[i_batch][i_point][j][2];

                scalar_t dist = d0*d0 + d1*d1 + d2*d2;
                if(dist<0.0001*0.0001){
                    flag=false;
                    break;
                }
            }

            output[i_batch][i_point][i] = flag;
        }

    }
}

void launch_filter(
    const torch::Tensor &output,
    const torch::Tensor &x,
    const torch::Tensor &mask) {

  // calculate #threads required
  auto B = output.size(0);
  auto N = output.size(1);

  int64_t count = B*N;
  if (count > 0) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "filter", [&] {
            filter<scalar_t>
            <<<at::cuda::detail::GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
              static_cast<int>(count),
              x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
              mask.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
              output.packed_accessor32<bool,3,torch::RestrictPtrTraits>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  }
}
