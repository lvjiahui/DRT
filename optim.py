import os
import torch
import numpy as np
import random
import cv2
import h5py
from tqdm import trange
import meshlabxml as mlx

resy=960
resx=1280
Float = torch.float64
device='cuda'


def limit_hook(grad):
    if torch.isnan(grad).any():
        print("nan in grad")
    grad[torch.isnan(grad)] = 0
    grad[grad>1]=1
    grad[grad<-1]=-1
    # N_part = Render.dot(grad, scene.normals, keepdim=True) * scene.normals
    # grad = N_part + 0.3*(grad-N_part)
    return grad

def interp_L(start, end, it, Pass):
    # 0 <= it <= Pass-1
    assert it <= Pass-1
    step = (end - start)/(Pass-1)
    return it*step + start

def interp_R(start, end, it, Pass):
    return 1/interp_L(1/start, 1/end, it, Pass)



def optimize(HyperParams, cuda_num = 0, output=True, track=None):
    name = HyperParams['name']
    def setup_opt(scene, lr, HyperParams):

        init_vertices = scene.vertices
        parameter = torch.zeros(init_vertices.shape, dtype=Float, requires_grad=True, device=device)    
        # parameter = torch.zeros([init_vertices.shape[0], 1], dtype=Float, requires_grad=True, device=device)    
        parameter.register_hook(limit_hook)
        if HyperParams['optimizer'] == 'sgd':
            opt = torch.optim.SGD([parameter], lr=lr, momentum = HyperParams['momentum'] , nesterov =True)
        if HyperParams['optimizer'] == 'adam':
            opt = torch.optim.Adam([parameter], lr=lr, )
        return init_vertices, parameter, opt

    def remesh(scene, meshlab_remesh_srcipt):
        pid = str(os.getpid())
        tmpply = f"/dev/shm/DR/temp_{pid}.ply"
        remeshply = f"/dev/shm/DR/remesh_{pid}.ply"
        # script = "DR/DR/remesh.mlx"
        script = f"/dev/shm/DR/script_{pid}.mlx"
        with open(script, 'w') as script_file:
            script_file.write(meshlab_remesh_srcipt)
        scene.mesh.export(tmpply)
        ssh = "ssh jiahui@172.31.224.138 "
        cmd = "DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage -i " + tmpply + " -o "  + remeshply + " -s " + script
        assert(os.system(ssh + cmd + " 1>/dev/null 2>&1") == 0)
        scene.update_mesh(remeshply)

        # assert(os.system(f"cp {tmpply} /root/workspace/show/mouse/{i_pass}_tmp.ply") == 0)
        # assert(os.system(f"cp {remeshply} /root/workspace/show/mouse/{i_pass}_remesh.ply") == 0)

        # os.system('rm '+tmpply)
        # os.system('rm '+remeshply)
        os.system('rm '+script)

    def mean_hausd(scene):
        pid = str(os.getpid())
        tmpply = f"/dev/shm/DR/temp_{pid}.ply"
        scene.mesh.export(tmpply)
        logpath = f"/dev/shm/DR/hausd_log_{pid}"
        os.system('rm '+logpath)
        ssh = "ssh jiahui@172.31.224.138 "
        cmd = "DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage -i DR/data/hand_gt_align.ply " + tmpply + " -s DR/data/cut_hausd.mlx -l " + logpath
        assert (os.system(ssh+cmd + " 1>/dev/null")==0)
        dist = mlx.compute.parse_hausdorff(logpath)['mean_distance']
        os.system('rm '+tmpply)
        os.system('rm '+logpath)
        return dist

    def cal_vh_loss(Views, V_index, scene):
        # silhouette_edge = scene.silhouette_edge(origin[0])
        # index, output = scene.primary_visibility(silhouette_edge, camera_M, origin[0], detach_depth=True)
        # vh_loss = 1e2 * (mask.view((resy,resx))[index[:,1],index[:,0]] - output).abs().mean()

        vh_loss = 0
        for v in np.arange(0,72,9):
            index =  (V_index+v)%72
            out_dir, out_origin, valid, mask, origin, ray_dir, camera_M = Views[index]
        # for view in Views:
        #     out_dir, valid, mask, origin, ray_dir, camera_M = view
            R, K, R_inverse, K_inverse = camera_M
            origin = torch.tensor(origin[0], dtype=Float, device=device)
            mask = torch.tensor(mask, dtype=Float, device=device)
            R_inverse = torch.tensor(R_inverse, dtype=Float, device=device)
            K_inverse = torch.tensor(K_inverse, dtype=Float, device=device)
            R = torch.tensor(R, dtype=Float, device=device)
            K = torch.tensor(K, dtype=Float, device=device)
            camera_M = (R, K, R_inverse, K_inverse)
            silhouette_edge = scene.silhouette_edge(origin)
            index, output = scene.primary_visibility(silhouette_edge, camera_M, origin, detach_depth=True)
            vh_loss += (mask.view((resy,resx))[index[:,1],index[:,0]] - output).abs().sum()

        return vh_loss

    def cal_var_loss(scene):
        # var_loss = 2*scene.area_var()
        # var_loss = 2*scene.edge_var()
        laplac = scene.vertices - scene.weightM.mm(scene.vertices) 
        normals = scene.normals.detach()
        N_laplac = Render.dot(laplac, normals, keepdim=True) * normals
        vertical_laplac = laplac - N_laplac
        var_loss = torch.norm(vertical_laplac/scene.mean_len, dim=1).pow(2).mean()    
        return var_loss

    def cal_sm_loss(scene):
        if HyperParams['sm_method'] == 'laplac' :
            laplac = scene.vertices - scene.weightM.mm(scene.vertices) 
            sm_loss = torch.norm(laplac/scene.mean_len, dim=1).pow(2).mean()  
        if HyperParams['sm_method'] == 'fairing' :
            laplac = scene.vertices - scene.weightM.mm(scene.vertices) 
            laplac = laplac - scene.weightM.mm(laplac) 
            sm_loss = torch.norm(laplac/scene.mean_len, dim=1).pow(2).mean()  
        if HyperParams['sm_method'] == 'dihedral' :
            dihedral_angle = scene.dihedral_angle() # cosine of angle [-1,1]
            dihedral_angle = -torch.log(1+dihedral_angle)
            # sm_loss = dihedral_angle.mean()
            sm_loss = 10*dihedral_angle.mean()/(scene.mean_len)
            # sm_loss = 20*dihedral_angle.mean()/(scene.mean_len*scene.mean_len)

            # laplac = scene.vertices - scene.weightM.mm(scene.vertices) 
            # laplac = laplac - scene.weightM.mm(laplac) 
            # laplac = laplac - scene.weightM.mm(laplac) 
            # sm_loss += 100*torch.norm(laplac/scene.mean_len, dim=1).pow(2).mean()  

        return sm_loss

    def cal_ray_loss(scene, out_origin, out_dir, origin, ray_dir, target, valid):
        # with torch.no_grad():
        #     _, reverse_mask = scene.render_transparent(out_origin, -out_dir)
        render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir)
        target = target  - render_out_ori.detach()
        target = target/target.norm(dim=1, keepdim=True)
        diff = (render_out_dir - target)
        # valid_mask = (diff.norm(dim=1)<0.5) * valid * render_mask[:,0]
        valid_mask = valid * render_mask[:,0]

        # diff = 1-Render.dot(render_out_dir, target)
        # valid_mask = (diff<HyperParams['ray_thred']) * valid * render_mask[:,0]
        
        # ray_loss = (diff[valid_mask]).pow(2).mean()
        ray_loss = (diff[valid_mask]).pow(2).sum() / 1e5
        # ray_loss = 7 * ray_loss / (scene.mean_len)

        # dot = Render.dot(render_out_dir, target)[valid_mask]
        # ray_loss = (1-torch.nn.functional.relu(dot)).mean()
        return ray_loss

    def save_illustration():
        out_dir, out_origin, valid, mask, origin, ray_dir, camera_M = get_view_torch(36)
        target = out_origin
        render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir)
        target = target  - render_out_ori.detach()
        target = target/target.norm(dim=1, keepdim=True)
        diff = (render_out_dir - target)
        # valid_mask = (diff.norm(dim=1)<0.5) * valid * render_mask[:,0]
        valid_mask = valid * render_mask[:,0]       
        image = torch.zeros_like(diff)
        image[valid_mask] = diff[valid_mask]
        image = image.pow(2).sum(dim=1,keepdim=True).sqrt()
        image[0] = 1.5
        # assert image.max() <= 1.51
        image[image>1.5]=1.5
        Render.save_torch(f"/root/workspace/show/mouse/{i_pass}_{it}.png", image)

    pid = os.getpid()
    # os.environ['CUDA_VISIBLE_DEVICES']=str(cuda_num)
    import Render_opencv as Render
    # Render.extIOR, Render.intIOR = 1.0, 1.4723
    Render.intIOR = HyperParams['IOR']

    # Render.intIOR = torch.tensor(Render.intIOR, device = device, requires_grad=True)
    # IORopt = torch.optim.SGD([Render.intIOR], lr=1e-5, momentum = 0.9 , nesterov =True)
    # IORopt = torch.optim.SGD([Render.intIOR], lr=1e-4)
    # IORopt = torch.optim.Adam([Render.intIOR], lr=1e-3)

    Render.resy = resy
    Render.resx = resx
    Render.device = device
    Render.Float = Float

    h5data = h5py.File(f'/root/workspace/data/{name}.h5','r')
    # h5data = h5py.File(f'/dev/shm/DR/{name}.h5','r')
    Views = []

    if output : viewnum = trange(72)
    else : viewnum = range(72)
    for i in viewnum:
    # for i in trange(0,72,9):
        out_dir = h5data['ray'][i,:,-3:]
        # out_origin = h5data['ray'][i,:,-6:-3]
        out_origin = h5data['cleaned_position'][i,:]
        mask = h5data['mask'][i][:,:,0]
        origin = h5data['ray'][i,:,:3]
        ray_dir = h5data['ray'][i,:,3:6]
        R_inverse = h5data['cam_proj'][i]
        K = h5data['cam_k'][:]
        R = np.linalg.inv(R_inverse)
        K_inverse = np.linalg.inv(K)
        mask = mask//255
        # valid = (cal_valid_mask(out_origin.reshape((resy,resx,3))) * mask).reshape(-1)
        valid = out_origin[:,0] != 0
        M = mask
        if M.max() == 255: M //= 255
        assert M.max() == 1
        bound = 2
        dist= (cv2.distanceTransform(M, cv2.DIST_L2, 0)-0).clip(0,bound)\
         - (cv2.distanceTransform(1-M, cv2.DIST_L2, 0)-1).clip(0,bound) #[-bound,+bound]
        mask = (dist + bound) / (2*bound) #[0,1]

        camera_M = (R, K, R_inverse, K_inverse)
        Views.append((out_dir, out_origin, valid, mask, origin, ray_dir, camera_M))

    def get_view_torch(V_index):
        out_dir, out_origin, valid, mask, origin, ray_dir, camera_M = Views[V_index]
        R, K, R_inverse, K_inverse = camera_M
        out_dir = torch.tensor(out_dir, dtype=Float, device=device)
        out_origin = torch.tensor(out_origin, dtype=Float, device=device)
        valid = torch.tensor(valid, dtype=bool, device=device)
        mask = torch.tensor(mask, dtype=Float, device=device)
        origin = torch.tensor(origin, dtype=Float, device=device)
        ray_dir = torch.tensor(ray_dir, dtype=Float, device=device)
        R_inverse = torch.tensor(R_inverse, dtype=Float, device=device)
        K_inverse = torch.tensor(K_inverse, dtype=Float, device=device)
        R = torch.tensor(R, dtype=Float, device=device)
        K = torch.tensor(K, dtype=Float, device=device)
        camera_M = (R, K, R_inverse, K_inverse)
        return out_dir, out_origin, valid, mask, origin, ray_dir, camera_M

    scene = Render.Scene(f"/root/workspace/data/{name}_vh.ply")
    # scene = Render.Scene(f"/root/workspace/DR/result/{name}_sm.ply")
    # scene = Render.Scene(f"/root/workspace/DR/result/{name}_pixel.ply")
    # torch.autograd.set_detect_anomaly(True)

    def rand_generator(num=72):
        head_view = {
            'mouse': 16,
            'rabbit': 18,
            'hand': 18,
            'dog': 17,
            'monkey': 19,
        }
        # if name in head_view.keys():
        #     view_range = HyperParams['view_range']
        #     head_num = head_view[name]
        #     index = list(np.arange(head_num-18-view_range, head_num+1-18+view_range))
        #     index = index + list(np.arange(head_num+18-view_range, head_num+1+18+view_range))  
        # else:
        #     index = list(np.arange(num))

        #mouse debug
        index = list(np.arange(-5, 10))
        index = index + list(np.arange(22,40))
        # index = list(np.arange(72))

        while True:
            # index = list(np.arange(-5, 10))#mouse
            # index = index + list(np.arange(22,40))
            np.random.shuffle(index)
            for i in index: yield i % 72

    view  = rand_generator(len(Views))

    for i_pass in range(HyperParams['Pass']):
        remesh_len = interp_R(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        # remesh_len = interp_L(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        # lr = interp_L(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])
        lr = interp_R(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])
        
        remesh(scene, Render.meshlab_remesh_srcipt.format(remesh_len))
        init_vertices, parameter, opt = setup_opt(scene, lr, HyperParams)

        print(f'remesh_len {remesh_len} lr {lr}')


        for it in range(HyperParams['Iters']):
            # if it % 100 == 0:
            #     save_illustration()
            # if track and it%200 == 0:
            #     track.log(mean_hausd=mean_hausd(scene))
            V_index = next(view)
            out_dir, out_origin, valid, mask, origin, ray_dir, camera_M = get_view_torch(V_index)
            # target = out_dir
            target = out_origin


            # Zero out gradients before each iteration
            opt.zero_grad()
            # IORopt.zero_grad()
            vertices = init_vertices + parameter
            # vertices = init_vertices + parameter * scene.normals
            scene.update_verticex(vertices)

            if torch.isnan(vertices).any():
                print("nan in vertices")

            zeroloss = torch.tensor(0, device=device)
            ray_loss = HyperParams['ray_w'] * cal_ray_loss(scene, out_origin, out_dir, origin, ray_dir, target, valid)\
                if HyperParams['ray_w'] !=0 else zeroloss
            vh_loss = HyperParams['vh_w'] * cal_vh_loss(Views, V_index, scene)\
                if HyperParams['vh_w'] !=0 else zeroloss
            var_loss = HyperParams['var_w'] * cal_var_loss(scene)\
                if HyperParams['var_w'] !=0 else zeroloss
            sm_loss = HyperParams['sm_w'] * cal_sm_loss(scene)\
                if HyperParams['sm_w'] !=0 else zeroloss
            bound_loss = zeroloss
            # bound_loss = 1e0 * torch.nn.functional.relu(Render.dot(parameter/scene.mean_len, scene.normals)).pow(2).mean()

            LOSS = 10*(ray_loss + vh_loss + var_loss + sm_loss+ bound_loss)
            if torch.isnan(LOSS).any():
                print("nan in LOSS")
            LOSS.backward()

            if track:
                track.log(ray_loss=ray_loss.item(), sm_loss=sm_loss.item(), LOSS=LOSS.item()/10)

            # if i_pass == 0:
            #     if it % 10 == 0:
            #         scene.mesh.export(f'/root/workspace/show/mouse/tmp_{i_pass}_{it}.obj')
            # else:
            #     if it % 100 == 0:
            #         scene.mesh.export(f'/root/workspace/show/mouse/tmp_{i_pass}_{it}.obj')

            if it%100==0 and output:
                print('Iteration %03i: ray=%g bound=%g vh=%g var=%g sm=%g  grad=%g' % \
                    (it, ray_loss, bound_loss, vh_loss, var_loss, sm_loss, parameter.grad.abs().max()))
                # print('IOR %g IORgrad %g' % (Render.intIOR, Render.intIOR.grad.item()))
            opt.step()
            # IORopt.step()

            if HyperParams['taubin'] > 0:
                vertices = init_vertices + parameter
                laplac = vertices.detach() - scene.weightM.mm(vertices.detach()) 
                init_vertices -= HyperParams['taubin'] * laplac
                vertices = init_vertices + parameter
                laplac = vertices.detach() - scene.weightM.mm(vertices.detach()) 
                init_vertices += HyperParams['taubin'] * laplac

    return scene

if __name__ == "__main__":
    name = 'mouse'


    HyperParams = { 
        'name' :  name,
        'IOR' : 1.4723,
        'Pass' : 8,
        # 'Pass' : 2,
        'Iters' : 500,
        # 'Iters' : 2000,
        "ray_w" : 1e2,
        # "ray_w" : 0, #fairing
        # "var_w" : 4e0,
        "var_w" : 0,
        # "sm_w": 2e0,
        "sm_w": 10e0,
        # "sm_w": 2e1, #mouse
        # "sm_w": 2e2, #fairing
        "vh_w": 0.1,
        # "sm_method": "laplac",
        # "sm_method": "fairing",
        "sm_method": "dihedral",
        "optimizer": "sgd",
        # "optimizer": "adam",
        "momentum": 0.95,
        "start_lr": 0.1,
        "lr_decay": 0.1,
        # "lr_decay": 1,
        "taubin" : 0,
        "start_len": 10,
        'view_range' : 12,
        # 'view_range' : 17,

        "end_len": 1,
                    }

 
    scene = optimize(HyperParams)
    # scene = optimize(sm_Params)
    # _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_sm.ply")
    _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_pixel.ply")
    # _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_IOR.ply")
    # _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_fair.ply")
