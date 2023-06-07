from third_parties.pytorch3d import ops
import torch
from torch import einsum
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import numpy as np
import os

cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
fuse_kernel = load(name='fuse_cuda',
                   extra_cuda_cflags=[],
                   sources=[f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
                            f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel_fast.cu'])
filter_cuda = load(name='filter',
                   sources=[f'{cuda_dir}/filter/filter.cpp',
                            f'{cuda_dir}/filter/filter.cu'])
precompute_cuda = load(name='precompute',
                       sources=[f'{cuda_dir}/precompute/precompute.cpp',
                                f'{cuda_dir}/precompute/precompute.cu'])


class ForwardDeformer(torch.nn.Module):
    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt
        self.soft_blend = 20

        self.init_bones = [0, 1, 2, 4, 5, 10, 11, 12, 15, 16, 17, 18, 19]
        # self.init_bones = [i for i in range(24)]
        self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()

        # the bounding box should be slighter larger than the actual mesh
        self.global_scale = 1.2
        self.version = opt.get("version", 1)

    def forward(self, xd, cond, tfs, eval_mode=False):
        """Given deformed point return its caonical correspondence
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        xc_opt, others = self.search(xd, cond, tfs, eval_mode=True)
        if eval_mode:
            return xc_opt, others

        if self.version == 1:
            xc_opt = xc_opt.detach()
            xc_opt[~others['valid_ids']] = 0
            n_batch, n_point, n_init, n_dim = xc_opt.shape

            mask = others['valid_ids']
            xd_opt = self.forward_skinning(xc_opt, cond, tfs, mask=mask)

            grad_inv = others['J_inv'][others['valid_ids']]
            correction = xd_opt - xd_opt.detach()
            correction = bmv(-grad_inv, correction.unsqueeze(-1)).squeeze(-1)

            # trick for implicit diff with autodiff:
            # xc = xc_opt + 0 and xc' = correction'
            xc = xc_opt
            xc[others['valid_ids']] += correction
            xc = xc.reshape(n_batch, n_point, n_init, n_dim)
            return xc, others
        else:
            mask = others['valid_ids']
            weights = self.query_weights(xc_opt, cond, mask=mask)
            T = einsum("pn,nij->pij", weights[mask], tfs[0])
            pts = xd[..., None, :].expand(1, -1, len(self.init_bones), 3)[mask]
            xc = torch.zeros_like(xc_opt)
            xc[mask] = ((pts - T[:, :3, 3]).unsqueeze(-2) @ T[:, :3, :3]).squeeze(1)
            return xc, others

    def precompute(self, tfs):
        b, c, d, h, w = tfs.shape[0], 3, self.resolution // 4, self.resolution, self.resolution
        voxel_d = torch.zeros((b, 3, d, h, w), device=tfs.device)
        voxel_J = torch.zeros((b, 12, d, h, w), device=tfs.device)
        precompute_cuda.precompute(self.lbs_voxel_final, tfs, voxel_d, voxel_J, self.offset_kernel, self.scale_kernel)
        self.voxel_d = voxel_d
        self.voxel_J = voxel_J

    def search(self, xd, cond, tfs, eval_mode=False):
        """Search correspondences.
        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """
        with torch.no_grad():
            result = self.broyden_cuda(xd, self.voxel_d, self.voxel_J, tfs)
        return result['result'], result

    def broyden_cuda(self, xd_tgt, voxel, voxel_J_inv, tfs, cvg_thresh=1e-5, dvg_thresh=1e-1):
        b, n, _ = xd_tgt.shape
        n_init = self.init_bones_cuda.shape[0]

        xc_init_IN = torch.zeros((b, n, n_init, 3), device=xd_tgt.device, dtype=torch.float32)
        J_inv_init_IN = torch.zeros((b, n, n_init, 3, 3), device=xd_tgt.device, dtype=torch.float32)
        is_valid = torch.zeros((b, n, n_init), device=xd_tgt.device, dtype=torch.bool)
        fuse_kernel.fuse_broyden(xc_init_IN, xd_tgt, voxel, voxel_J_inv, tfs,
                                 self.init_bones_cuda, True, J_inv_init_IN,
                                 is_valid, self.offset_kernel,
                                 self.scale_kernel, cvg_thresh, dvg_thresh)
        mask = filter_cuda.filter(xc_init_IN, is_valid)
        return {
            "result": xc_init_IN,
            'valid_ids': mask,
            'J_inv': J_inv_init_IN
        }

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point
        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        weights = self.query_weights(xc, cond, mask=mask)
        return skinning_mask(xc[mask], weights[mask], tfs, inverse=False)

    def switch_to_explicit(self, resolution=32, smpl_verts=None, smpl_weights=None, use_smpl=False):
        self.resolution = resolution
        # convert to voxel grid
        device = self.device
        b, c, d, h, w = 1, 24, resolution // 4, resolution, resolution
        self.ratio = h / d
        x_range = (torch.linspace(-1, 1, steps=w, device=device)).view(1, 1, 1, w).expand(1, d, h, w)
        y_range = (torch.linspace(-1, 1, steps=h, device=device)).view(1, 1, h, 1).expand(1, d, h, w)
        z_range = (torch.linspace(-1, 1, steps=d, device=device)).view(1, d, 1, 1).expand(1, d, h, w)
        grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(b, 3, -1).permute(0, 2, 1)

        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, smpl_verts.max(dim=1).values], dim=0).to(device)
        offset = (gt_bbox[0] + gt_bbox[1])[None, None, :] * 0.5
        scale = (gt_bbox[1] - gt_bbox[0]).max() / 2 * self.global_scale

        corner = torch.ones_like(offset[0]) * scale
        corner[0, 2] /= self.ratio
        min_vert = (offset - corner).reshape(1, 3)
        max_vert = (offset + corner).reshape(1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=0)

        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

        self.register_buffer('offset_kernel', -self.offset)
        scale_kernel = torch.zeros_like(self.offset)
        scale_kernel[...] = 1. / self.scale
        scale_kernel[:, :, -1] = scale_kernel[:, :, -1] * self.ratio
        self.register_buffer('scale_kernel', scale_kernel)

        def normalize(x):
            x_normalized = x.clone()
            x_normalized -= self.offset
            x_normalized /= self.scale
            x_normalized[..., -1] *= self.ratio
            return x_normalized

        def denormalize(x):
            x_denormalized = x.clone()
            x_denormalized[..., -1] /= self.ratio
            x_denormalized *= self.scale
            x_denormalized += self.offset
            return x_denormalized

        self.normalize = normalize
        self.denormalize = denormalize

        grid_denorm = self.denormalize(grid)

        if use_smpl:
            weights = query_weights_smpl(grid_denorm,
                                         smpl_verts=smpl_verts.detach().clone(),
                                         smpl_weights=smpl_weights.detach().clone(),
                                         resolution=resolution).detach().clone()
        else:
            weights = self.query_weights(grid_denorm, {}, None)

        self.register_buffer('lbs_voxel_final', weights.detach())
        self.register_buffer('grid_denorm', grid_denorm)

        def query_weights(xc, cond=None, mask=None, mode='bilinear'):
            shape = xc.shape
            N = 1
            xc = xc.view(1, -1, 3)
            w = F.grid_sample(self.lbs_voxel_final.expand(N, -1, -1, -1, -1),
                              self.normalize(xc)[:, :, None, None],
                              align_corners=True,
                              mode=mode,
                              padding_mode='border')
            w = w.squeeze(-1).squeeze(-1).permute(0, 2, 1)
            w = w.view(*shape[:-1], -1)
            return w
        self.query_weights = query_weights

def skinning_mask(x, w, tfs, inverse=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    p, n = w.shape
    w_tf = einsum("pn,nij->pij", w, tfs.squeeze(0))
    x_h = x_h.view(p, 1, 4).expand(p, 4, 4)
    x_h = (w_tf * x_h).sum(-1)
    return x_h[:, :3]


def bmv(m, v):
    return (m * v.transpose(-1, -2).expand(-1, 3, -1)).sum(-1, keepdim=True)


def query_weights_smpl(x, smpl_verts, smpl_weights, resolution=128):
    # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
    dist, idx, _ = ops.knn_points(x, smpl_verts.detach(), K=30)
    dist = dist.sqrt().clamp_(0.0001, 1.)
    weights = smpl_weights[0, idx]

    ws = 1. / dist
    ws = ws / ws.sum(-1, keepdim=True)
    weights = (ws[..., None] * weights).sum(-2)

    b, c, d, h, w = 1, 24, resolution // 4, resolution, resolution
    weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
    for _ in range(30):
        mean=(weights[:,:,2:,1:-1,1:-1]+weights[:,:,:-2,1:-1,1:-1]+\
              weights[:,:,1:-1,2:,1:-1]+weights[:,:,1:-1,:-2,1:-1]+\
              weights[:,:,1:-1,1:-1,2:]+weights[:,:,1:-1,1:-1,:-2])/6.0
        weights[:, :, 1:-1, 1:-1, 1:-1] = (weights[:, :, 1:-1, 1:-1, 1:-1] - mean) * 0.7 + mean
        sums = weights.sum(1, keepdim=True)
        weights = weights / sums
    return weights.detach()
