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
__device__ void fuse_J_inv_update(const index_t index, scalar_t *J_inv,
                                  scalar_t x0, scalar_t x1, scalar_t x2,
                                  scalar_t g0, scalar_t g1, scalar_t g2) {
  scalar_t J00 = J_inv[3 * 0 + 0];
  scalar_t J01 = J_inv[3 * 0 + 1];
  scalar_t J02 = J_inv[3 * 0 + 2];
  scalar_t J10 = J_inv[3 * 1 + 0];
  scalar_t J11 = J_inv[3 * 1 + 1];
  scalar_t J12 = J_inv[3 * 1 + 2];
  scalar_t J20 = J_inv[3 * 2 + 0];
  scalar_t J21 = J_inv[3 * 2 + 1];
  scalar_t J22 = J_inv[3 * 2 + 2];

  auto c0 = J00 * x0 + J10 * x1 + J20 * x2;
  auto c1 = J01 * x0 + J11 * x1 + J21 * x2;
  auto c2 = J02 * x0 + J12 * x1 + J22 * x2;

  auto s = c0 * g0 + c1 * g1 + c2 * g2;

  auto r0 = -J00 * g0 - J01 * g1 - J02 * g2;
  auto r1 = -J10 * g0 - J11 * g1 - J12 * g2;
  auto r2 = -J20 * g0 - J21 * g1 - J22 * g2;

  J_inv[3 * 0 + 0] += c0 * (r0 + x0) / s;
  J_inv[3 * 0 + 1] += c1 * (r0 + x0) / s;
  J_inv[3 * 0 + 2] += c2 * (r0 + x0) / s;
  J_inv[3 * 1 + 0] += c0 * (r1 + x1) / s;
  J_inv[3 * 1 + 1] += c1 * (r1 + x1) / s;
  J_inv[3 * 1 + 2] += c2 * (r1 + x1) / s;
  J_inv[3 * 2 + 0] += c0 * (r2 + x2) / s;
  J_inv[3 * 2 + 1] += c1 * (r2 + x2) / s;
  J_inv[3 * 2 + 2] += c2 * (r2 + x2) / s;
}

static __forceinline__ __device__ bool within_bounds_3d(int d, int h, int w,
                                                        int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__ scalar_t clip_coordinates(scalar_t in,
                                                            int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1),
               ::max(in, static_cast<scalar_t>(0)));
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
compute_coordinates(scalar_t coord, int size, bool align_corners) {
  // clip coordinates to image borders
  // coord = clip_coordinates(coord, size);
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord, int size, bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, align_corners);
  return coord;
}

template <typename scalar_t, typename index_t>
__device__ void grid_sampler_3d(
    index_t i_batch, TensorInfo<scalar_t, index_t> input, scalar_t grid_x,
    scalar_t grid_y, scalar_t grid_z,
    // TensorInfo<scalar_t, index_t> output,
    PackedTensorAccessor32<scalar_t, 5> input_p, // [1, 3, 8, 32, 32]
    scalar_t *output,
    // PackedTensorAccessor32<scalar_t, 3> output_p, // [1800000, 3, 1]
    bool align_corners, bool nearest) {

  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];

  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];

  index_t out_sC = 1; // output size is same as grid size...

  // get the corresponding input x, y, z co-ordinates from grid
  scalar_t ix = grid_x;
  scalar_t iy = grid_y;
  scalar_t iz = grid_z;

  // c0 ix,iy,iz=-0.848051,0.592726,0.259927
  // c1 ix,iy,iz=2.355216,24.687256,4.409743

  ix = grid_sampler_compute_source_index(ix, inp_W, align_corners);
  iy = grid_sampler_compute_source_index(iy, inp_H, align_corners);
  iz = grid_sampler_compute_source_index(iz, inp_D, align_corners);
  if (!nearest) {
    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    index_t ix_tnw = static_cast<index_t>(::floor(ix));
    index_t iy_tnw = static_cast<index_t>(::floor(iy));
    index_t iz_tnw = static_cast<index_t>(::floor(iz));

    index_t ix_tne = ix_tnw + 1;
    index_t iy_tne = iy_tnw;
    index_t iz_tne = iz_tnw;

    index_t ix_tsw = ix_tnw;
    index_t iy_tsw = iy_tnw + 1;
    index_t iz_tsw = iz_tnw;

    index_t ix_tse = ix_tnw + 1;
    index_t iy_tse = iy_tnw + 1;
    index_t iz_tse = iz_tnw;

    index_t ix_bnw = ix_tnw;
    index_t iy_bnw = iy_tnw;
    index_t iz_bnw = iz_tnw + 1;

    index_t ix_bne = ix_tnw + 1;
    index_t iy_bne = iy_tnw;
    index_t iz_bne = iz_tnw + 1;

    index_t ix_bsw = ix_tnw;
    index_t iy_bsw = iy_tnw + 1;
    index_t iz_bsw = iz_tnw + 1;

    index_t ix_bse = ix_tnw + 1;
    index_t iy_bse = iy_tnw + 1;
    index_t iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    for (index_t xyz = 0; xyz < C; xyz++) {
      output[xyz] = 0;

      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_tnw][iy_tnw][ix_tnw] * tnw;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH +
        // ix_tnw * inp_sW] * tnw;
      }
      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_tne][iy_tne][ix_tne] * tne;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH +
        // ix_tne * inp_sW] * tne;
      }
      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_tsw][iy_tsw][ix_tsw] * tsw;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH +
        // ix_tsw * inp_sW] * tsw;
      }
      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_tse][iy_tse][ix_tse] * tse;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH +
        // ix_tse * inp_sW] * tse;
      }
      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_bnw][iy_bnw][ix_bnw] * bnw;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH +
        // ix_bnw * inp_sW] * bnw;
      }
      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_bne][iy_bne][ix_bne] * bne;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH +
        // ix_bne * inp_sW] * bne;
      }
      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_bsw][iy_bsw][ix_bsw] * bsw;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH +
        // ix_bsw * inp_sW] * bsw;
      }
      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
        output[xyz] += input_p[i_batch][xyz][iz_bse][iy_bse][ix_bse] * bse;
        // *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH +
        // ix_bse * inp_sW] * bse;
      }
    }
  } else {
    index_t ix_nearest = static_cast<index_t>(::round(ix));
    index_t iy_nearest = static_cast<index_t>(::round(iy));
    index_t iz_nearest = static_cast<index_t>(::round(iz));

    for (index_t xyz = 0; xyz < C; xyz++) {
      if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H,
                           inp_W)) {
        output[xyz] = input_p[i_batch][xyz][iz_nearest][iy_nearest][ix_nearest];
      } else {
        output[xyz] = static_cast<scalar_t>(0);
      }
    }
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void broyden_kernel(
    const index_t npoints, const index_t n_batch, const index_t n_point,
    const index_t n_init, TensorInfo<scalar_t, index_t> voxel_ti,
    TensorInfo<scalar_t, index_t> voxel_J_ti,
    PackedTensorAccessor32<scalar_t, 4> x,          // shape=(N,200000, 9, 3)
    PackedTensorAccessor32<scalar_t, 3> xd_tgt,     // shape=(N,200000, 3)
    PackedTensorAccessor32<scalar_t, 5> voxel,      // shape=(N,3,8,32,32)
    PackedTensorAccessor32<scalar_t, 5> grid_J_inv, // shape=(N,9,8,32,32)
    PackedTensorAccessor32<scalar_t, 4> tfs,        // shape=(N,24,4,4)
    PackedTensorAccessor32<int, 1> bone_ids,        // shape=(9)
    PackedTensorAccessor32<scalar_t, 5> J_inv,      // shape=(N,200000, 9, 9)
    PackedTensorAccessor32<bool, 3> is_valid,       // shape=(N,200000, 9)
    PackedTensorAccessor32<scalar_t, 3> offset,     // shape=(N, 1, 3)
    PackedTensorAccessor32<scalar_t, 3> scale,      // shape=(N, 1, 3)
    float cvg_threshold, float dvg_threshold, int N) {

  index_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= npoints)
    return;

  const index_t i_batch = index / (n_point * n_init);
  const index_t i_point = (index % (n_point * n_init)) / n_init;
  const index_t i_init = (index % (n_point * n_init)) % n_init;

  scalar_t gx[3];
  scalar_t gx_new[3];

  scalar_t xd_tgt_index[3];

  xd_tgt_index[0] = xd_tgt[i_batch][i_point][0];
  xd_tgt_index[1] = xd_tgt[i_batch][i_point][1];
  xd_tgt_index[2] = xd_tgt[i_batch][i_point][2];

  scalar_t x_l[3];

  int i_bone = bone_ids[i_init];
  scalar_t ixd = xd_tgt_index[0] - tfs[i_batch][i_bone][0][3];
  scalar_t iyd = xd_tgt_index[1] - tfs[i_batch][i_bone][1][3];
  scalar_t izd = xd_tgt_index[2] - tfs[i_batch][i_bone][2][3];
  x_l[0] = ixd * tfs[i_batch][i_bone][0][0] + iyd * tfs[i_batch][i_bone][1][0] + izd * tfs[i_batch][i_bone][2][0];
  x_l[1] = ixd * tfs[i_batch][i_bone][0][1] + iyd * tfs[i_batch][i_bone][1][1] + izd * tfs[i_batch][i_bone][2][1];
  x_l[2] = ixd * tfs[i_batch][i_bone][0][2] + iyd * tfs[i_batch][i_bone][1][2] + izd * tfs[i_batch][i_bone][2][2];

  scalar_t J_local[12];
  grid_sampler_3d(i_batch, voxel_J_ti,
                  scale[0][0][0] * (x_l[0] + offset[0][0][0]),
                  scale[0][0][1] * (x_l[1] + offset[0][0][1]),
                  scale[0][0][2] * (x_l[2] + offset[0][0][2]),
                  grid_J_inv, J_local, true, false);

  scalar_t J_inv_local[9];
  J_inv_local[3 * 0 + 0] = J_local[4 * 0 + 0];
  J_inv_local[3 * 1 + 0] = J_local[4 * 0 + 1];
  J_inv_local[3 * 2 + 0] = J_local[4 * 0 + 2];
  J_inv_local[3 * 0 + 1] = J_local[4 * 1 + 0];
  J_inv_local[3 * 1 + 1] = J_local[4 * 1 + 1];
  J_inv_local[3 * 2 + 1] = J_local[4 * 1 + 2];
  J_inv_local[3 * 0 + 2] = J_local[4 * 2 + 0];
  J_inv_local[3 * 1 + 2] = J_local[4 * 2 + 1];
  J_inv_local[3 * 2 + 2] = J_local[4 * 2 + 2];

  for (int i = 0; i < 10; i++) {
    scalar_t J00 = J_inv_local[3 * 0 + 0];
    scalar_t J01 = J_inv_local[3 * 0 + 1];
    scalar_t J02 = J_inv_local[3 * 0 + 2];
    scalar_t J10 = J_inv_local[3 * 1 + 0];
    scalar_t J11 = J_inv_local[3 * 1 + 1];
    scalar_t J12 = J_inv_local[3 * 1 + 2];
    scalar_t J20 = J_inv_local[3 * 2 + 0];
    scalar_t J21 = J_inv_local[3 * 2 + 1];
    scalar_t J22 = J_inv_local[3 * 2 + 2];

    if (i == 0) {
      gx[0] = J_local[4 * 0 + 0] * x_l[0] + J_local[4 * 0 + 1] * x_l[1] + J_local[4 * 0 + 2] * x_l[2] + J_local[4 * 0 + 3];
      gx[1] = J_local[4 * 1 + 0] * x_l[0] + J_local[4 * 1 + 1] * x_l[1] + J_local[4 * 1 + 2] * x_l[2] + J_local[4 * 1 + 3];
      gx[2] = J_local[4 * 2 + 0] * x_l[0] + J_local[4 * 2 + 1] * x_l[1] + J_local[4 * 2 + 2] * x_l[2] + J_local[4 * 2 + 3];

      gx[0] = gx[0] - xd_tgt_index[0];
      gx[1] = gx[1] - xd_tgt_index[1];
      gx[2] = gx[2] - xd_tgt_index[2];
    } else {
      gx[0] = gx_new[0];
      gx[1] = gx_new[1];
      gx[2] = gx_new[2];
    }

    // update = -J_inv @ gx
    scalar_t u0 = -J00 * gx[0] + -J01 * gx[1] + -J02 * gx[2];
    scalar_t u1 = -J10 * gx[0] + -J11 * gx[1] + -J12 * gx[2];
    scalar_t u2 = -J20 * gx[0] + -J21 * gx[1] + -J22 * gx[2];

    // x += update
    x_l[0] += u0;
    x_l[1] += u1;
    x_l[2] += u2;

    scalar_t ix = scale[0][0][0] * (x_l[0] + offset[0][0][0]);
    scalar_t iy = scale[0][0][1] * (x_l[1] + offset[0][0][1]);
    scalar_t iz = scale[0][0][2] * (x_l[2] + offset[0][0][2]);

    // gx_new = g(x)
    grid_sampler_3d(i_batch, voxel_J_ti, ix, iy, iz, grid_J_inv, J_local, true,
                    false);

    gx_new[0] = J_local[4 * 0 + 0] * x_l[0] + J_local[4 * 0 + 1] * x_l[1] +
                J_local[4 * 0 + 2] * x_l[2] + J_local[4 * 0 + 3] -
                xd_tgt_index[0];
    gx_new[1] = J_local[4 * 1 + 0] * x_l[0] + J_local[4 * 1 + 1] * x_l[1] +
                J_local[4 * 1 + 2] * x_l[2] + J_local[4 * 1 + 3] -
                xd_tgt_index[1];
    gx_new[2] = J_local[4 * 2 + 0] * x_l[0] + J_local[4 * 2 + 1] * x_l[1] +
                J_local[4 * 2 + 2] * x_l[2] + J_local[4 * 2 + 3] -
                xd_tgt_index[2];

    // convergence checking
    scalar_t norm_gx = gx_new[0] * gx_new[0] + gx_new[1] * gx_new[1] + gx_new[2] * gx_new[2];

    // convergence/divergence criterion
    if (norm_gx < cvg_threshold * cvg_threshold) {

      auto b = 1;
      bool is_valid_ =
          ix >= -b && ix <= b && iy >= -b && iy <= b && iz >= -b && iz <= b;

      is_valid[i_batch][i_point][i_init] = is_valid_;

      if (is_valid_) {
        x[i_batch][i_point][i_init][0] = x_l[0];
        x[i_batch][i_point][i_init][1] = x_l[1];
        x[i_batch][i_point][i_init][2] = x_l[2];

        J_inv[i_batch][i_point][i_init][0][0] = J00;
        J_inv[i_batch][i_point][i_init][0][1] = J01;
        J_inv[i_batch][i_point][i_init][0][2] = J02;
        J_inv[i_batch][i_point][i_init][1][0] = J10;
        J_inv[i_batch][i_point][i_init][1][1] = J11;
        J_inv[i_batch][i_point][i_init][1][2] = J12;
        J_inv[i_batch][i_point][i_init][2][0] = J20;
        J_inv[i_batch][i_point][i_init][2][1] = J21;
        J_inv[i_batch][i_point][i_init][2][2] = J22;
      }
      return;

    } else if (norm_gx > dvg_threshold * dvg_threshold) {
      is_valid[i_batch][i_point][i_init] = false;
      return;
    }

    // delta_x = update
    scalar_t delta_x_0 = u0;
    scalar_t delta_x_1 = u1;
    scalar_t delta_x_2 = u2;

    // delta_gx = gx_new - gx
    scalar_t delta_gx_0 = gx_new[0] - gx[0];
    scalar_t delta_gx_1 = gx_new[1] - gx[1];
    scalar_t delta_gx_2 = gx_new[2] - gx[2];

    fuse_J_inv_update(index, J_inv_local, delta_x_0, delta_x_1, delta_x_2,
                      delta_gx_0, delta_gx_1, delta_gx_2);
  }
}

void launch_broyden_kernel(Tensor &x, const Tensor &xd_tgt, const Tensor &voxel,
                           const Tensor &grid_J_inv, const Tensor &tfs,
                           const Tensor &bone_ids, bool align_corners,
                           Tensor &J_inv, Tensor &is_valid,
                           const Tensor &offset, const Tensor &scale,
                           float cvg_threshold, float dvg_threshold) {

  // calculate #threads required
  int64_t n_batch = xd_tgt.size(0);
  int64_t n_point = xd_tgt.size(1);
  int64_t n_init = bone_ids.size(0);
  int64_t count = n_batch * n_point * n_init;

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "fuse_kernel_cuda", [&] {
          broyden_kernel<<<GET_BLOCKS(count, 512), 512, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
              static_cast<int>(count), static_cast<int>(n_batch),
              static_cast<int>(n_point), static_cast<int>(n_init),
              getTensorInfo<scalar_t, int>(voxel),
              getTensorInfo<scalar_t, int>(grid_J_inv),
              x.packed_accessor32<scalar_t, 4>(),
              xd_tgt.packed_accessor32<scalar_t, 3>(),
              voxel.packed_accessor32<scalar_t, 5>(),
              grid_J_inv.packed_accessor32<scalar_t, 5>(),
              tfs.packed_accessor32<scalar_t, 4>(),
              bone_ids.packed_accessor32<int, 1>(),
              J_inv.packed_accessor32<scalar_t, 5>(),
              is_valid.packed_accessor32<bool, 3>(),
              offset.packed_accessor32<scalar_t, 3>(),
              scale.packed_accessor32<scalar_t, 3>(), cvg_threshold,
              dvg_threshold, 0);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  cudaDeviceSynchronize();
}
