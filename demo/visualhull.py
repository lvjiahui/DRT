import numpy as np
import torch
import space_carving
import time
import tqdm

res=1024
name = 'hailuo'

Masks = []
R_invs = []
Ks = []
capture_data = torch.load(f'data/{name}_capture.pt')
for i in range(len(capture_data)):
    out_ray_dir, valid, mask, camera_M = capture_data[i]
    R, K, R_inverse, K_inverse = camera_M
    Masks.append(mask.cpu().numpy().astype(dtype=np.uint8))
    R_invs.append(R_inverse.cpu().numpy())
    Ks.append(K.cpu().numpy())

Ks = np.array(Ks)
R_invs = np.array(R_invs)
Masks = np.array(Masks)

vol_bnds = np.zeros((3,2))
for K,R_inv,M in zip(Ks,R_invs,Masks):
    view_frust_pts = space_carving.get_view_frustum(M, K, R_inv, 2.5)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

tsdf_vol = space_carving.TSDFVolume(vol_bnds, voxel_size=0.05)
# Loop through RGB-D images and fuse them together
t0_elapse = time.time()
for K,R_inv,M in tqdm.tqdm(zip(Ks,R_invs,Masks)):
    cam_pose = R_inv
    cam_intr = K
    color_image = 128*np.ones((res,res,3))
    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, M, cam_intr, cam_pose, obs_weight=1.)


verts, faces, norms, colors = tsdf_vol.get_mesh()

vol_bnds2 = np.vstack( (verts.min(axis=0),verts.max(axis=0)) ).T

tsdf_vol = space_carving.TSDFVolume(vol_bnds2*1.1, voxel_size=0.005)
# Loop through RGB-D images and fuse them together
t0_elapse = time.time()
for K,R_inv,M in tqdm.tqdm(zip(Ks,R_invs,Masks)):
    cam_pose = R_inv
    cam_intr = K
    color_image = 128*np.ones((res,res,3))
    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, M, cam_intr, cam_pose, obs_weight=1.)


# Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
print("Saving mesh to mesh.ply...")
verts, faces, norms, colors = tsdf_vol.get_mesh()
space_carving.meshwrite(f"data/{name}_vh.ply", verts, faces, norms, colors)
print("done")
