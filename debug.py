import trimesh
import trimesh.transformations as TF
import torch
import numpy as np
import random
import Render
Render.extIOR, Render.intIOR = 1.15, 1.0
# Render.extIOR, Render.intIOR = 1.5, 1.0
res=512
Float = torch.float64
device='cuda'
Render.res = res
Render.device = device
Render.Float = Float
kitten = Render.Scene("/root/workspace/data/kitten_vh_sim.ply")
kitten.set_camera((60,60), 1.3, center=(0,0,0), angles=None)
R,K = kitten.camera_RK()
origin, ray_dir = kitten.generate_ray()
target_mask = kitten.mask(origin, ray_dir)
target_mask = target_mask
# Render.PILimage(target_mask)

new_vert = kitten.vertices + torch.tensor([0.1,0.1,-0.1], device=device)
kitten.update_verticex(new_vert)
# Render.PILimage(kitten.mask(origin, ray_dir))

vertices = kitten.vertices
parameter = torch.zeros(3,  device=device, requires_grad=True)
opt = torch.optim.Adam([parameter], lr=.02)
for i in range(99):
    opt.zero_grad()
    new_vert = vertices + parameter
    kitten.update_verticex(new_vert)
    silhouette_edge = kitten.silhouette_edge(origin[0])
    index, output = kitten.primary_visibility(silhouette_edge, R, K, origin[0])
    loss = (target_mask.view((res,res))[index[:,0],index[:,1]] - output).abs().mean()
    loss.backward()
    opt.step()
    image = torch.zeros((res,res))
    image[index[:,0],index[:,1]]=1
    Render.save_image(str(i)+'.png', image)
    mask = kitten.mask(origin, ray_dir)
    error = (target_mask - mask).abs().mean()
    print('Iteration %03i: error=%g' % (i, error))