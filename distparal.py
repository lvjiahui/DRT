import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import numpy as np
import random
import cv2
import h5py
from tqdm import trange

resy=960
resx=1280
Float = torch.float64

name = 'mouse'
pid = os.getpid()



def run(rank, size):
    device='cuda:{}'.format(rank)
    import Render_opencv as Render
    Render.extIOR, Render.intIOR = 1.0, 1.4723
    Render.resy = resy
    Render.resx = resx
    Render.Float = Float
    Render.device = device

    def cal_valid_mask(out_ray):
        src = out_ray.reshape((resy,resx,3))
        # src = cv2.GaussianBlur(src, (3, 3), 0)
        lap = cv2.Laplacian(src, cv2.CV_64F)
        lap = np.linalg.norm(lap,axis=2)
        # mask = (lap>1e-2)*(lap<0.5)
        mask = (lap>1e-3)
        kernel = np.ones((5,5),np.uint8) 
        mask = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE, kernel)
        mask = (mask>0)
        return mask
    
    def limit_hook(grad):
        grad[grad>1]=1
        grad[grad<-1]=-1
        return grad
    
    def setup_opt(scene):
        init_vertices = scene.vertices
        parameter = torch.zeros(init_vertices.shape, dtype=Float, requires_grad=True, device=device)    
        # parameter = torch.zeros([init_vertices.shape[0], 1], dtype=Float, requires_grad=True, device=device)    
        parameter.register_hook(limit_hook)
        # opt = torch.optim.SGD([parameter], lr=.004, momentum = 0.9, nesterov =True)
        opt = torch.optim.SGD([parameter], lr=0.1, momentum = 0.9, nesterov =True)
        # opt = torch.optim.Adam([parameter], lr=0.05)
        return init_vertices, parameter, opt
    
    def remesh():
        tmpply = "/dev/shm/DR/temp_{}.ply".format(pid)
        remeshply = "/dev/shm/DR/remesh_{}.ply".format(pid)
        # script = "DR/DR/remesh_fine.mlx"
        script = "DR/DR/remesh.mlx"
        if rank == 0:
            os.system('rm '+tmpply)
            os.system('rm '+remeshply)
            scene.mesh.export(tmpply)
            ssh = "ssh jiahui@172.31.224.138 "
            cmd = "DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage -i " + tmpply + " -o "  + remeshply + " -s " + script
            assert(os.system(ssh + cmd) == 0)
        
        # syn
        dist.barrier()

        scene.update_mesh(remeshply)
        
        

    h5data = h5py.File('/root/workspace/data/{}.h5'.format(name),'r')
    Views = []
    for i in range(rank*9, (rank+1)*9):
        out_ray = h5data['ray'][i,:,-3:]
        mask = h5data['mask'][i][:,:,0]
        origin = h5data['ray'][i,:,:3]
        ray_dir = h5data['ray'][i,:,3:6]
        R_inverse = h5data['cam_proj'][i]
        K = h5data['cam_k'][:]
        R = np.linalg.inv(R_inverse)
        K_inverse = np.linalg.inv(K)
        valid = (cal_valid_mask(out_ray) * mask).reshape(-1)
        M = mask
        bound = 2
        distance= (cv2.distanceTransform(M, cv2.DIST_L2, 0)-1).clip(0,bound)\
         - (cv2.distanceTransform(255-M, cv2.DIST_L2, 0)-1).clip(0,bound)
        mask = (distance + bound) / (2*bound)

        camera_M = (R, K, R_inverse, K_inverse)
        Views.append((out_ray, valid, mask, origin, ray_dir, camera_M))

    scene = Render.Scene("/root/workspace/data/{}_vh.ply".format(name), cuda_device=rank)
    init_vertices, parameter, opt = setup_opt(scene)

    for it in range(3999):
        ray_loss = vh_loss = sm_loss = bound_loss = dihedral_loss = var_loss = ray_lap_loss= 0

        V_index = random.randint(0, len(Views)-1)
        out_ray, valid, mask, origin, ray_dir, camera_M = Views[V_index]
        R, K, R_inverse, K_inverse = camera_M
        out_ray = torch.tensor(out_ray, dtype=Float, device=device)
        valid = torch.tensor(valid, dtype=bool, device=device)
        mask = torch.tensor(mask, dtype=Float, device=device)
        origin = torch.tensor(origin, dtype=Float, device=device)
        ray_dir = torch.tensor(ray_dir, dtype=Float, device=device)
        R_inverse = torch.tensor(R_inverse, dtype=Float, device=device)
        K_inverse = torch.tensor(K_inverse, dtype=Float, device=device)
        R = torch.tensor(R, dtype=Float, device=device)
        K = torch.tensor(K, dtype=Float, device=device)
        camera_M = (R, K, R_inverse, K_inverse)
        target = out_ray

        if it%500==0:
            remesh()
            del init_vertices, parameter, opt
            init_vertices, parameter, opt = setup_opt(scene)
            
        # Zero out gradients before each iteration
        opt.zero_grad()
        vertices = init_vertices + parameter
        # vertices = init_vertices + parameter * scene.normals
        scene.update_verticex(vertices)

        render_img, render_twice_mask = scene.render_transparent(origin, ray_dir)
        diff = (render_img - target)
        valid_mask = (diff.norm(dim=1)<1) * valid * render_twice_mask[:,0]
        ray_loss = 1e2*(diff[valid_mask]).pow(2).mean()
        # ray_loss = 4e1*(diff[valid_mask]).pow(2).mean()

        silhouette_edge = scene.silhouette_edge(origin[0])
        index, output = scene.primary_visibility(silhouette_edge, camera_M, origin[0], detach_depth=True)
        vh_loss = 1 * (mask.view((resy,resx))[index[:,1],index[:,0]] - output).abs().sum()

        var_loss = 2*scene.edge_var()

        dihedral_angle = scene.dihedral_angle() # cosine of angle [-1,1]
        dihedral_angle = -torch.log(1+dihedral_angle)
        dihedral_loss = 4e1 * dihedral_angle.mean()

        # bound_loss = 1e0 * torch.nn.functional.relu(Render.dot(parameter/scene.mean_len, scene.normals)).pow(2).mean()

        (10*(ray_loss+ ray_lap_loss + vh_loss + var_loss + dihedral_loss + sm_loss+ bound_loss)).backward()
        dist.all_reduce(parameter.grad.data, op=dist.ReduceOp.SUM)
        parameter.grad.data /= size

        if rank == 0 and it%100==0:
            print('Iteration %03i: ray=%g ray_lap=%g  bound=%g vh=%g var=%g dihedral=%g sm=%g  grad=%g' % \
                (it, ray_loss, ray_lap_loss, bound_loss, vh_loss, var_loss, dihedral_loss, sm_loss, parameter.grad.data.abs().max()))
        opt.step()

    if rank == 0: _=scene.mesh.export("/root/workspace/DR/result/{}.ply".format(name))


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()