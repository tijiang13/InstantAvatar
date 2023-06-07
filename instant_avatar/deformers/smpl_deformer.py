from third_parties.pytorch3d import ops
from .smplx import SMPL
import torch
import hydra


def get_bbox_from_smpl(vs, factor=1.2):
    assert vs.shape[0] == 1
    min_vert = vs.min(dim=1).values
    max_vert = vs.max(dim=1).values

    c = (max_vert + min_vert) / 2
    s = (max_vert - min_vert) / 2
    s = s.max(dim=-1).values * factor

    min_vert = c - s[:, None]
    max_vert = c + s[:, None]
    return torch.cat([min_vert, max_vert], dim=0)

class SMPLDeformer():
    def __init__(self, model_path, gender, threshold=0.05, k=1) -> None:
        model_path = hydra.utils.to_absolute_path(model_path)
        self.body_model = SMPL(model_path, gender=gender)

        self.k = k
        self.threshold = threshold
        self.strategy = "nearest_neighbor"

        # define template canonical pose
        #   T-pose does't work very well for some cases (legs are too close) 
        self.initialized = False

    def initialize(self, betas, device):
        # convert to canonical space: deformed => T pose => Template pose
        batch_size = betas.shape[0]
        body_pose_t = torch.zeros((batch_size, 69), device=device)
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
        smpl_outputs = self.body_model(betas=betas, body_pose=body_pose_t)
        self.bbox = get_bbox_from_smpl(smpl_outputs.vertices[0:1].detach())

        self.T_template = smpl_outputs.T
        self.vs_template = smpl_outputs.vertices
        self.pose_offset_t = smpl_outputs.pose_offsets
        self.shape_offset_t = smpl_outputs.shape_offsets

    def get_bbox_deformed(self):
        return get_bbox_from_smpl(self.vertices[0:1].detach())

    def prepare_deformer(self, smpl_params):
        device = smpl_params["betas"].device
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
            # self.body_model.eval()

        if not self.initialized:
            self.initialize(smpl_params["betas"], device)

            # TODO: we have to intialized it every time because betas might change
            # self.initialized = True

        smpl_outputs = self.body_model(betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"])

        # remove & reapply the blendshape
        s2w = smpl_outputs.A[:, 0]
        w2s = torch.inverse(s2w)
        T_inv = torch.inverse(smpl_outputs.T.float()).clone() @ s2w[:, None]
        T_inv[..., :3, 3] += self.pose_offset_t - smpl_outputs.pose_offsets
        T_inv[..., :3, 3] += self.shape_offset_t - smpl_outputs.shape_offsets
        T_inv = self.T_template @ T_inv
        self.T_inv = T_inv
        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        self.w2s = w2s

    def transform_rays_w2s(self, rays):
        """transform rays from world to smpl coordinate system"""
        w2s = self.w2s
        rays.o = (rays.o @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        rays.d = (rays.d @ w2s[:, :3, :3].permute(0, 2, 1)).to(rays.d)
        d = torch.norm(rays.o, dim=-1)
        rays.near = d - 1
        rays.far = d + 1

    def deform(self, pts):
        """transform pts to canonical space"""
        # construct template pose
        batch_size = self.vertices.shape[0]

        # find nearest neighbors
        pts = pts.reshape(batch_size, -1, 3)
        with torch.no_grad():
            dist_sq, idx = ops.knn_points(pts.float(), self.vertices.float(), K=self.k)

        # if valid => return the transformed points
        valid = dist_sq < self.threshold ** 2

        # squeeze because we use the nearest neighbor only
        idx = idx.squeeze(-1)
        valid = valid.squeeze(-1)

        # T ~ (batch_size, #pts, #neighbors, 4, 4)
        pts_cano = torch.zeros_like(pts, dtype=torch.float32)
        for i in range(batch_size):
            # mask = valid[i]
            Tv_inv = self.T_inv[i][idx[i]]
            pts_cano[i] = (Tv_inv[..., :3, :3] @ pts[i][..., None]).squeeze(-1) + Tv_inv[..., :3, 3]
        return pts_cano.reshape(-1, 3), valid.reshape(-1)

    def deform_train(self, pts, model):
        pts_cano, valid = self.deform(pts)
        rgb_cano = torch.zeros_like(pts, dtype=torch.float32)
        sigma_cano = torch.ones_like(pts[..., 0]) * -1e5
        if valid.any():
            with torch.cuda.amp.autocast():
                rgb_cano[valid], sigma_cano[valid] = model(pts_cano[valid], None)
            valid = torch.isfinite(rgb_cano).all(-1) & torch.isfinite(sigma_cano)
            rgb_cano[~valid] = 0
            sigma_cano[~valid] = -1e5
        return rgb_cano, sigma_cano

    def deform_test(self, pts, model):
        pts_cano, valid = self.deform(pts)
        rgb_cano = torch.zeros_like(pts, dtype=torch.float32)
        sigma_cano = torch.zeros_like(pts[..., 0])
        if valid.any():
            with torch.cuda.amp.autocast():
                rgb_cano[valid], sigma_cano[valid] = model(pts_cano[valid], None)
        return rgb_cano, sigma_cano

    def __call__(self, pts, model, eval_mode=True):
        if eval_mode:
            return self.deform_test(pts, model)
        else:
            return self.deform_train(pts, model)
