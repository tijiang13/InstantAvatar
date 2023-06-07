import torch
import torch.nn.functional as F


class DensityGrid(torch.nn.Module):
    """Multi-resolution grid"""
    def __init__(self, grid_size=64, aabb=None) -> None:
        super().__init__()
        # self.register_buffer("density_grid", torch.zeros(grid_size ** 3))

        idx = torch.arange(0, grid_size)
        coords = torch.meshgrid((idx, idx, idx), indexing="ij")
        coords = torch.stack(coords, dim=-1)
        coords = coords.reshape(-1, 3) / grid_size
        self.coords = coords.cuda()
        self.grid_size = grid_size

        # density_cached = torch.zeros_like(self.coords[:, 0])
        self.register_buffer("density_cached", torch.zeros_like(self.coords[:, 0]))
        self.register_buffer("density_field", torch.zeros(grid_size, grid_size, grid_size, dtype=torch.bool))
        self.aabb = aabb

    @property
    def min_corner(self):
        return self.aabb[0]

    @property
    def max_corner(self):
        return self.aabb[1]

    # @torch.no_grad()
    def update(self, deformer, net, step):
        if self.aabb is None:
            bbox = deformer.get_bbox_deformed()
            self.aabb = bbox

        coords = (self.coords + torch.rand_like(self.coords) / self.grid_size) * (self.aabb[1] - self.aabb[0]) + self.aabb[0]
        with torch.enable_grad():
            _, density = deformer(coords, net, eval_mode=False)
        density = density.clip(min=0)

        old = self.density_field.reshape(-1)
        self.density_cached = torch.maximum(self.density_cached * 0.8, density.detach())
        self.density_field = 1 - torch.exp(0.01 * -self.density_cached)
        self.density_field = self.density_field.reshape(self.grid_size, self.grid_size, self.grid_size)
        self.density_field = F.max_pool3d(self.density_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        self.density_field = self.density_field > torch.clamp(self.density_field.mean(), max=0.01)

        if step > 1000:
            valid = old
        else:
            valid = self.density_field.reshape(-1)
        return coords, density, valid
    
    @torch.no_grad()
    def initialize(self, deformer, net, iters=5):
        self.aabb = deformer.get_bbox_deformed()

        density = torch.zeros_like(self.coords[:, 0])
        for _ in range(iters):
            delta = torch.rand_like(self.coords)
            coords = (self.coords + delta / self.grid_size) * (self.aabb[1] - self.aabb[0]) + self.aabb[0]
            _, d = deformer(coords, net)
            density = torch.maximum(density, d)

        self.density_field = 1 - torch.exp(0.01 * -density)
        self.density_field = self.density_field.reshape(self.grid_size, self.grid_size, self.grid_size)
        self.density_field = F.max_pool3d(self.density_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        # self.density_field = self.density_field > torch.clamp(self.density_field.mean(), max=0.01)
        self.density_field = self.density_field > 0.00
