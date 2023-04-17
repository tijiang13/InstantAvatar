from skimage import measure
import trimesh
from tqdm import tqdm
import torch


@torch.no_grad()
def marching_cubes(func,
                   bbox,
                   resolution=256,
                   level_set=0,
                   gradient_direction="ascent",
                   extract_max_component=False,
                   device="cuda"):

    idx = torch.arange(0, resolution)
    coords = torch.meshgrid((idx, idx, idx), indexing="ij")
    coords = torch.stack(coords, dim=-1).to(device)
    coords = coords.reshape(-1, 3) / resolution
    coords = coords * (bbox[1] - bbox[0]) + bbox[0]

    val = []
    for b in tqdm(coords.split(2**20)):
        val.append(func(b))
    val = torch.cat(val, dim=0)
    val = val.reshape(resolution, resolution, resolution)
    val = val.cpu().numpy()

    verts, faces, _, _ = measure.marching_cubes(
        val.transpose(1, 0, 2), level_set, gradient_direction=gradient_direction)

    bbox = bbox.cpu().numpy()
    verts = verts / resolution * (bbox[1] - bbox[0]) + bbox[0]

    mesh = trimesh.Trimesh(verts, faces)
    if extract_max_component:
        # remove disconnect part
        connected_comp = mesh.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        mesh = max_comp
        return mesh
    else:
        return mesh