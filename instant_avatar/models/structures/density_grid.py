import torch
import torch.nn.functional as F
from third_parties.pytorch3d import ops
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces


def get_aabb(vs, scale=1.2):
    gt_bbox = torch.stack([vs.min(dim=0).values, vs.max(dim=0).values], dim=0)
    c = (gt_bbox[0] + gt_bbox[1]) * 0.5
    s = (gt_bbox[1] - gt_bbox[0]) * 0.5
    return torch.stack([c - s * scale, c + s * scale], dim=0)

def denormalize(coords, aabb):
    return coords * (aabb[1] - aabb[0]) + aabb[0]

class DensityGrid(torch.nn.Module):
    """Multi-resolution grid"""
    def __init__(self, grid_size=64, aabb=None, smpl_init=False) -> None:
        super().__init__()
        # self.register_buffer("density_grid", torch.zeros(grid_size ** 3))

        idx = torch.arange(0, grid_size)
        coords = torch.meshgrid((idx, idx, idx), indexing="ij")
        coords = torch.stack(coords, dim=-1)
        coords = coords.reshape(grid_size, grid_size, grid_size, 3) / grid_size
        self.coords = coords.cuda()
        self.grid_size = grid_size

        # density_cached = torch.zeros_like(self.coords[:, 0])
        self.register_buffer("density_cached", torch.zeros_like(self.coords[..., 0]))
        self.register_buffer("density_field", torch.zeros_like(self.coords[..., 0], dtype=torch.bool))
        self.aabb = aabb
        self.initialized = False
        self.smpl_init = smpl_init

    @property
    def min_corner(self):
        return self.aabb[0]

    @property
    def max_corner(self):
        return self.aabb[1]

    # @torch.no_grad()
    def update(self, deformer, net, step):
        coords = denormalize(self.coords + torch.rand_like(self.coords) / self.grid_size, self.aabb)
        with torch.enable_grad():
            _, density = deformer(coords.reshape(-1, 3), net, eval_mode=False)
        density = density.clip(min=0).reshape(coords.shape[:-1])
        old = self.density_field

        if step < 500 and self.smpl_init:
            # initialize the grid upon the first call
            if not self.initialized:
                coords = denormalize(self.coords + 0.5 / self.grid_size, self.aabb)

                self.mesh_v_cano = deformer.vertices
                self.mesh_f_cano = deformer.body_model.faces_tensor
                self.mesh_face_vertices = index_vertices_by_faces(self.mesh_v_cano, self.mesh_f_cano)

                distance = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                    coords.reshape(1, -1, 3).contiguous(), self.mesh_face_vertices
                )[0].reshape(coords.shape[:-1]).sqrt()

                sign = kaolin.ops.mesh.check_sign(
                    self.mesh_v_cano, self.mesh_f_cano, coords.reshape(1, -1, 3)
                ).reshape(coords.shape[:-1]).float()
                sign = 1 - 2 * sign
                signed_distance = sign * distance
                self.density_field = (signed_distance < 0.01)

                opacity = -torch.log(1 - self.density_field.float()) * 100
                self.density_cached = torch.maximum(self.density_cached * 0.8, opacity)
                self.initialized = True
        else:
            self.density_cached = torch.maximum(self.density_cached * 0.8, density.detach())
            self.density_field = 1 - torch.exp(0.01 * -self.density_cached)
            self.density_field = F.max_pool3d(self.density_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            self.density_field = self.density_field > torch.clamp(self.density_field.mean(), max=0.01)

            # get maximum connected component 
            mcc = max_connected_component(self.density_field)
            label = torch.mode(mcc[self.density_field], 0).values
            self.density_field = (mcc == label)

        density = 1 - torch.exp(0.01 * -F.relu(density))
        if step < 500:
            valid = self.density_field
        else:
            valid = old
        return density, valid
    
    @torch.no_grad()
    def initialize(self, deformer, net, iters=5):
        self.aabb = deformer.get_bbox_deformed()

        density = torch.zeros_like(self.coords[..., 0])
        for _ in range(iters):
            coords = denormalize(self.coords + torch.rand_like(self.coords) / self.grid_size, self.aabb)
            _, d = deformer(coords.reshape(-1, 3), net)
            density = torch.maximum(density, d.reshape(density.shape))

        self.density_field = 1 - torch.exp(0.01 * -density)
        self.density_field = F.max_pool3d(self.density_field[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        self.density_field = self.density_field > torch.clamp(self.density_field.mean(), max=0.01)

        mcc = max_connected_component(self.density_field)
        label = torch.mode(mcc[self.density_field], 0).values
        self.density_field = (mcc == label)

    def export_mesh(self):
        import trimesh
        density = self.density_field.cpu().numpy()
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(density, pitch=1.0)
        return mesh

def max_connected_component(grid):
    grid = grid.unsqueeze(0).unsqueeze(0)
    comp = torch.arange(1, grid.numel() + 1, device=grid.device).reshape(grid.shape).float()
    comp[~grid] = 0
    for _ in range(grid.shape[-1] * 3):
        comp = F.max_pool3d(comp, kernel_size=3, stride=1, padding=1)
        comp *= grid
    return comp.squeeze(0).squeeze(0)

