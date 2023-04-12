import torch
from ..models.structures.utils import Rays


def stratified_sampling(near, far, N):
    """stratified sampling
    partition [near, far] into N evenly-spaced bins, and perform uniform sampling
    within each bin.
    Args:
        near: (batch_size, sample_size)
        far : (batch_size, sample_size)
    Returns
        samples (batch_size, sample_size, N)
    """
    splits = torch.linspace(0, 1, N+1, device=near.device)
    end_pts = (far - near)[..., None] * splits + near[..., None]
    width = end_pts[..., 1:] - end_pts[..., :-1]
    samples = torch.rand_like(width) * width + end_pts[..., :-1]
    return samples


def importance_sampling(bins, weights, N):
    """importance sampling
    importance sampling using the bins divided in stratified sampling
    Args:
        bins: (batch_size, sample_size, #bins + 1) end points of bins
        weights: (batch_size, sample_size, #bins)
    Return:
        samples (batch_size, sample_size, N)
    """
    # to avoid "divided by zero" error
    weights = weights + 1e-5
    w = weights / weights.sum(dim=-1, keepdims=True)
    cdf = torch.cumsum(w, dim=-1)
    cdf[..., -1] = 1 + 1e-3

    # generate indices of bins to be sampled from
    u = torch.rand(*w.shape[:-1], N, device=bins.device)
    idx = torch.searchsorted(cdf, u)

    # sample within each bins
    bins_l = torch.gather(bins, dim=-1, index=idx)
    bins_r = torch.gather(bins, dim=-1, index=idx+1)
    samples = bins_l + (bins_r - bins_l) * torch.rand_like(u)
    samples, _ = samples.sort(dim=-1)
    return samples


def volume_render(network_fn, rays_o, rays_d, z_vals):
    """volume render
    
    Args:
        rays_o: (batch_size, ray_num, 3)
        rays_d: (batch_size, ray_num, 3)
        z_vals: (batch_size, ray_num, z_num)
    """
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # run network
    rgb_vals, sigma_vals = network_fn(pts.reshape(-1, 3),
                                      rays_d[..., None, :].expand(pts.shape).reshape(-1, 3))
    rgb_vals = rgb_vals.reshape(pts.shape)
    sigma_vals = sigma_vals.reshape(z_vals.shape)

    # volume render
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.ones_like(z_vals[..., :1]) * 1e10], dim=-1)

    # 0 (transparent) <= alpha <= 1 (opaque)
    alpha = 1 - torch.exp(-sigma_vals * dists)
    cumprod = torch.cat([torch.ones_like(alpha[..., 0:1]),
                        torch.cumprod(1 - alpha[..., :-1] + 1e-10, dim=-1)], dim=-1)

    # sum(weights) = 1 - torch.prod(1 - alpha) <= 1
    weights = alpha * cumprod

    # assuming background is (1, 1, 1)
    rgb_vals = (weights[..., None] * rgb_vals).sum(dim=-2)
    rgb_vals = rgb_vals + (1 - weights.sum(-1, keepdim=True))

    # depth_vals = compute_depth_map(weights, z_vals)
    depth_vals = (weights * z_vals).sum(dim=-1)
    return rgb_vals, depth_vals, weights


class VolumeRenderer:
    def __init__(self, N_coarse: int, N_fine: int, **kwargs) -> None:
        self.N_coarse = N_coarse
        self.N_fine = N_fine

    def render(self, rays, model):
        outputs = {}
        z_coarse = stratified_sampling(rays.near, rays.far, self.N_coarse)
        rgb_coarse, depth_coarse, weight_coarse = volume_render(model, rays.o, rays.d, z_coarse)
        outputs["rgb_coarse"] = rgb_coarse
        outputs["depth_coarse"] = depth_coarse
        outputs["alpha_coarse"] = weight_coarse.sum(dim=-1)
        if self.N_fine > 0:
            bins = 0.5 * (z_coarse[..., :-1] + z_coarse[..., 1:])
            z_fine = importance_sampling(bins, weight_coarse[..., 1:-1], self.N_fine)
            rgb_fine, depth_fine, weight_fine = volume_render(model, rays.o, rays.d, z_fine)
            outputs["rgb_fine"] = rgb_fine
            outputs["depth_fine"] = depth_fine
            outputs["alpha_fine"] = weight_fine.sum(dim=-1)
        return outputs

    def render_train(self, rays, model):
        raise NotImplementedError
        return self.render(rays, model)

    @torch.no_grad()
    def render_test(self, rays, model):
        num_rays = rays.o.shape[1]
        output = {} 

        K = 1024
        for i in range(0, num_rays, K):
            l = i
            r = l + K
            o = self.render(Rays(o=rays.o[:, l:r],
                                 d=rays.d[:, l:r],
                                 near=rays.near[:, l:r],
                                 far=rays.far[:, l:r]),
                            model)
            if i == 0:
                for k in o: output[k] = []
            for k in o: output[k].append(o[k])
        for k in output:
            output[k] = torch.cat(output[k], dim=1)
        return output
        # return self.render(rays, model)

    def __call__(self, rays, model, eval_mode=True, **kwargs):
        if eval_mode:
            return self.render_test(rays, model)
        else:
            return self.render_train(rays, model)
            
