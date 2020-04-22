import trimesh
import trimesh.transformations as TF
import torch
import numpy as np
import random
import Render
import cv2
Render.extIOR, Render.intIOR = 1.15, 1.0
# Render.extIOR, Render.intIOR = 1.5, 1.0
res=512
Float = torch.float64
device='cuda'
Render.res = res
Render.device = device
Render.Float = Float

from trimesh.proximity import closest_point

def mean_hausd(Target, vertices):
    closest, distance, triangle_id = closest_point(Target.mesh, vertices)
    distance = distance*9.7*2
    return distance.mean()   

def area_var(vertices, faces):
    v0=vertices[faces[:,0]]
    v1=vertices[faces[:,1]]
    v2=vertices[faces[:,2]]
    edge1 = v1-v0
    edge2 = v2-v0
    area =torch.cross(edge1,edge2).norm(dim=1)
    area_ave = area.mean().detach()
    area_var = ((area-area_ave)/area_ave).pow(2).mean()
    return area_var

def opt_kitten(config):
    def render_target(scene:Render.Scene, deg):
        #俯视
        M=TF.rotation_matrix(-np.pi/180*0, [1,0,0])
        angle = TF.euler_from_matrix(TF.rotation_matrix(np.pi/180*deg, [0,1,0]) @ M)
        # scene.set_camera(fov=(60,60), distance = 1.0, center=(0,0.0,0), angles=angle)
        # scene.set_camera(fov=(60,60), distance = 1.1, center=(0,0.0,0), angles=angle)
        scene.set_camera(fov=(60,60), distance = 1.3, center=(0,0.0,0), angles=angle)
        R, K = scene.camera_RK()
        origin, ray_dir = scene.generate_ray()
        target, twice_mask = scene.render_transparent(origin, ray_dir)
        mask = scene.mask(origin, ray_dir)

        M = mask.view((res,res,1)).cpu().numpy()
        M = M.astype(np.uint8)
        bound = 1
        dist= (cv2.distanceTransform(M, cv2.DIST_L2, 0)-1).clip(0,bound)\
         - (cv2.distanceTransform(1-M, cv2.DIST_L2, 0)-1).clip(0,bound)
        soft_mask = (dist + bound) / (2*bound)
        soft_mask = torch.tensor(soft_mask, dtype=Float, device=device)

        return target, twice_mask, soft_mask, origin, ray_dir, R, K

    Target = Render.Scene("/root/workspace/data/kitten.obj")
    # Target = Render.Scene("/root/workspace/data/bunny.ply")
    Views = []
    for deg in range(-90,90,15):
        Views.append(render_target(Target, deg))
    # # for deg in [-30,-15,0,15,30]:
    for deg in range(-90,90,15):
        Views.append(render_target(Target, deg+180))

    kitten_vh = Render.Scene("/root/workspace/data/kitten_vh_sub.ply")
    vh_faces = torch.tensor(kitten_vh.mesh.faces, device=device)
    vh_vertices = kitten_vh.vertices
    parameter = torch.zeros(vh_vertices.shape, dtype=Float, requires_grad=True, device=device)
    opt = torch.optim.Adam([parameter], lr=config["lr"])

    mu = 1/ (config["kpb"] - 1/config["lambda"])
    assert mu < 0

    for it in range(6999):
        V_index = random.randint(0, len(Views)-1)
        target, twice_mask, mask, origin, ray_dir, R, K = Views[V_index]
        # Zero out gradients before each iteration
        opt.zero_grad()
        # vertices = vh_vertices + parameter * vh_normals
        vertices = vh_vertices + parameter
        kitten_vh.update_verticex(vertices)
        render_img, render_twice_mask = kitten_vh.render_transparent(origin, ray_dir)
        twice_mask = (twice_mask * render_twice_mask)
        loss = config["ray"]*(render_img[twice_mask]-target[twice_mask]).pow(2).mean()

        silhouette_edge = kitten_vh.silhouette_edge(origin[0])
        index, output = kitten_vh.primary_visibility(silhouette_edge, R, K, origin[0], detach_depth=True)
        vh_loss = config["vh"]*(mask.view((res,res))[index[:,0],index[:,1]] - output).abs().sum()
        rough = config["rough"]*parameter.abs().mean()
        loss_area = config["area"]*area_var(vertices, vh_faces)
        (loss+rough+vh_loss+loss_area).backward()
        # (loss+rough).backward()
        # loss.backward()

        if it % 200 == 0:
            error = mean_hausd(Target, vertices.detach().cpu().numpy())
            track.log(mean_hausd=error)
        # Optimizer: take a gradient step
        opt.step()

        vertices_detach = vh_vertices + parameter.detach()
        laplac = vertices_detach - kitten_vh.weightM.mm(vertices_detach) 
        vh_vertices -= config["lambda"]*laplac
        vertices_detach = vh_vertices + parameter.detach()
        laplac = vertices_detach - kitten_vh.weightM.mm(vertices_detach)

        vh_vertices -= mu*laplac

import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets

from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler

search_space = {
    "lr": tune.loguniform(5e-5,5e-3),
    "ray": tune.loguniform(1e-1,10),
    "vh": tune.loguniform(1e-3, 10),
    "rough": tune.loguniform(1e-6, 1e-2),
    "area": tune.loguniform(1e-4, 1e-1),
    "lambda": tune.uniform(0.1, 0.5),
    "kpb": tune.uniform(0, 0.2),
}
analysis = tune.run(
    opt_kitten,
    num_samples=3000,
    scheduler=ASHAScheduler(metric="mean_hausd", mode="min"),
    config=search_space,
    resources_per_trial={
         "cpu": 3,
         "gpu": 1,
     }
    )

# Obtain a trial dataframe from all run trials of this `tune.run` call.
dfs = analysis.trial_dataframes