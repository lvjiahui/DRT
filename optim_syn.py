import os
import torch
import numpy as np
import random
import cv2
import h5py
from tqdm import trange
import meshlabxml as mlx


resy=512
resx=512
Float = torch.float64
device='cuda'

meshlab_remesh_srcipt = """
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Remeshing: Isotropic Explicit Remeshing">
  <Param value="3" isxmlparam="0" name="Iterations" type="RichInt" description="Iterations" tooltip="Number of iterations of the remeshing operations to repeat on the mesh."/>
  <Param value="false" isxmlparam="0" name="Adaptive" type="RichBool" description="Adaptive remeshing" tooltip="Toggles adaptive isotropic remeshing."/>
  <Param value="false" isxmlparam="0" name="SelectedOnly" type="RichBool" description="Remesh only selected faces" tooltip="If checked the remeshing operations will be applied only to the selected faces."/>
  <Param value="{}" isxmlparam="0" name="TargetLen" type="RichAbsPerc" description="Target Length" min="0" max="214.384" tooltip="Sets the target length for the remeshed mesh edges."/>
  <Param value="180" isxmlparam="0" name="FeatureDeg" type="RichFloat" description="Crease Angle" tooltip="Minimum angle between faces of the original to consider the shared edge as a feature to be preserved."/>
  <Param value="true" isxmlparam="0" name="CheckSurfDist" type="RichBool" description="Check Surface Distance" tooltip="If toggled each local operation must deviate from original mesh by [Max. surface distance]"/>
  <Param value="0.008" isxmlparam="0" name="MaxSurfDist" type="RichAbsPerc" description="Max. Surface Distance" min="0" max="214.384" tooltip="Maximal surface deviation allowed for each local operation"/>
  <Param value="true" isxmlparam="0" name="SplitFlag" type="RichBool" description="Refine Step" tooltip="If checked the remeshing operations will include a refine step."/>
  <Param value="true" isxmlparam="0" name="CollapseFlag" type="RichBool" description="Collapse Step" tooltip="If checked the remeshing operations will include a collapse step."/>
  <Param value="true" isxmlparam="0" name="SwapFlag" type="RichBool" description="Edge-Swap Step" tooltip="If checked the remeshing operations will include a edge-swap step, aimed at improving the vertex valence of the resulting mesh."/>
  <Param value="true" isxmlparam="0" name="SmoothFlag" type="RichBool" description="Smooth Step" tooltip="If checked the remeshing operations will include a smoothing step, aimed at relaxing the vertex positions in a Laplacian sense."/>
  <Param value="true" isxmlparam="0" name="ReprojectFlag" type="RichBool" description="Reproject Step" tooltip="If checked the remeshing operations will include a step to reproject the mesh vertices on the original surface."/>
 </filter>
</FilterScript>
"""


def intersectPlane(n, p0, ray_origin, ray_dir):
    denom = n[0] * ray_dir[:,0] + n[1] * ray_dir[:,1] +n[2] * ray_dir[:,2]
    on_screen = denom>0.3
    p0_ro = p0 - ray_origin
    t = (n[0] * p0_ro[:,0] + n[1] * p0_ro[:,1] +n[2] * p0_ro[:,2])/denom
    return on_screen, t




def generate_ray(K_inverse, R_inverse):
    y_range = torch.arange(0, resy, device=device, dtype=Float)
    x_range = torch.arange(0, resx, device=device, dtype=Float)
    pixely, pixelx = torch.meshgrid(y_range, x_range)
    pixelz = torch.ones_like(pixely)
    pixel = torch.stack([pixelx, pixely, pixelz], dim=2).view([-1,3])
    pixel_p = K_inverse @ pixel.T
    
    pixel_world_p  = R_inverse[:3,:3] @ pixel_p + R_inverse[:3, 3:4]
    ray_origin = R_inverse[:3, 3:4] #[3x1]
    ray_dir = pixel_world_p - ray_origin
    ray_dir = ray_dir.T #[nx3]
    ray_dir = ray_dir/ray_dir.norm(dim=1,keepdim=True)
    ray_origin = ray_origin.T.expand_as(ray_dir)
    return ray_origin, ray_dir



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
        GT_path = 'DR/data/mouse_scan_fixed_tri.ply'
        pid = str(os.getpid())
        tmpply = f"/dev/shm/DR/temp_{pid}.ply"
        scene.mesh.export(tmpply)
        logpath = f"/dev/shm/DR/hausd_log_{pid}"
        os.system('rm '+logpath)
        ssh = "ssh jiahui@172.31.224.138 "
        cmd = f"DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage -i  {tmpply} {GT_path}   -s DR/data/hausd.mlx -l " + logpath
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
            origin = get_origin_torch(index) #[3]
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
            sm_loss = dihedral_angle.sum()

        return sm_loss

    def cal_ray_loss(scene, origin, ray_dir, out_ray_dir, valid):
        render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir)

        # target = pixel_coor  - render_out_ori.detach()
        # # target = pixel_coor  - render_out_ori
        # target = target/target.norm(dim=1, keepdim=True)

        diff = (render_out_dir - out_ray_dir)
        valid_mask = valid * render_mask[:,0]


        ray_loss = (diff[valid_mask]).pow(2).sum()
        # ray_loss = 7 * ray_loss / (scene.mean_len)

        return ray_loss

    def save_illustration():
        out_origin, valid, mask, origin, ray_dir, camera_M = get_view_torch(36)
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
    IOR = HyperParams['IOR']
    # Render.extIOR, Render.intIOR = 1.0, 1.4723
    Render.intIOR = IOR

    # Render.intIOR = torch.tensor(Render.intIOR, device = device, requires_grad=True)
    # IORopt = torch.optim.SGD([Render.intIOR], lr=1e-5, momentum = 0.9 , nesterov =True)
    # IORopt = torch.optim.SGD([Render.intIOR], lr=1e-4)
    # IORopt = torch.optim.Adam([Render.intIOR], lr=1e-3)

    Render.resy = resy
    Render.resx = resx
    Render.device = device
    Render.Float = Float

    Target_scene = Render.Scene(f"/root/workspace/data/{name}.ply")
    Views = []

    if output : viewnum = trange(72)
    else : viewnum = range(72)
    for i in viewnum:
        rad = np.pi * i * 5 / 180
        R = torch.tensor([[np.cos(rad), 0, np.sin(rad),0],
                        [0,-1,0,0],
                        [-np.sin(rad), 0, np.cos(rad),1.3],
                        [0,0,0,1]],dtype=Float,device=device)
        K = torch.tensor([[443.40500674,   0.        , 256.        ],
                            [  0.        , 443.40500674, 256.        ],
                            [  0.        ,   0.        ,   1.        ]],dtype=Float,device=device)
        R_inverse = torch.inverse(R)
        K_inverse = torch.inverse(K)
        ray_origin, ray_dir = generate_ray(K_inverse, R_inverse)
        out_origin, out_ray_dir, valid = Target_scene.render_transparent(ray_origin, ray_dir)

        # ray_center = ray_dir.view([resy,resx,3])[resy//2,resx//2]
        # n = ray_center #screen normal
        # p0 = ray_origin[0] + 3*n #screen position
        # on_screen, t = intersectPlane(n, p0, out_origin, out_ray_dir)
        # pixel_coor = out_origin + t.view([-1,1]) * out_ray_dir

        # valid = valid[:,0] * on_screen
        valid = valid[:,0]
        mask = Target_scene.render_mask(ray_origin, ray_dir)
        M = mask.reshape([resy,resx]).cpu().numpy().astype(np.uint8)
        assert M.max() == 1
        bound = 2
        dist= (cv2.distanceTransform(M, cv2.DIST_L2, 0)-0).clip(0,bound)\
         - (cv2.distanceTransform(1-M, cv2.DIST_L2, 0)-1).clip(0,bound) #[-bound,+bound]
        mask = (dist + bound) / (2*bound) #[0,1]
        mask = torch.tensor(mask, dtype=Float,device=device)

        camera_M = (R, K, R_inverse, K_inverse)
        Views.append((out_ray_dir, valid, mask, camera_M))

    def get_view_torch(V_index):
        out_ray_dir, valid, mask, camera_M = Views[V_index]
        R, K, R_inverse, K_inverse = camera_M

        origin, ray_dir = generate_ray(K_inverse, R_inverse)

        camera_M = (R, K, R_inverse, K_inverse)
        
        return out_ray_dir, valid, mask, origin, ray_dir, camera_M

    def get_origin_torch(V_index):
        out_ray_dir, valid, mask, camera_M = Views[V_index]
        R, K, R_inverse, K_inverse = camera_M
        R_inverse = R_inverse.to(device)
        ray_origin = R_inverse[:3,3] #[3]
        return ray_origin
        
    scene = Render.Scene(f"/root/workspace/data/{name}_vh_sub.ply")


    def rand_generator(num=72):
        index = list(np.arange(72))

        while True:
            np.random.shuffle(index)
            for i in index: yield i % 72

    view  = rand_generator(len(Views))

    paper_path = f'/root/workspace/show/paper/{name}'
    paper = HyperParams['is_paper']
    # paper = True
    if paper:
        All_ray_loss = []
        All_vh_loss = []
        All_sm_loss = []
        os.makedirs(f'{paper_path}/{IOR}', exist_ok=True)
        h5_record = h5py.File(f'{paper_path}/{IOR}/data.h5','w')

    for i_pass in range(HyperParams['Pass']):
        remesh_len = interp_R(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        # remesh_len = interp_L(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        # lr = interp_L(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])
        lr = interp_R(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])
        
        remesh(scene, meshlab_remesh_srcipt.format(remesh_len))
        init_vertices, parameter, opt = setup_opt(scene, lr, HyperParams)

        print(f'remesh_len {remesh_len} lr {lr}')



        for it in range(HyperParams['Iters']):
            # if it % 100 == 0:
            #     save_illustration()
            # if track and it%200 == 0:
            #     track.log(mean_hausd=mean_hausd(scene))
            V_index = next(view)
            out_ray_dir, valid, mask, origin, ray_dir, camera_M = get_view_torch(V_index)
            target = out_ray_dir



            # Zero out gradients before each iteration
            opt.zero_grad()
            # IORopt.zero_grad()
            vertices = init_vertices + parameter
            # vertices = init_vertices + parameter * scene.normals
            scene.update_verticex(vertices)

            if torch.isnan(vertices).any():
                print("nan in vertices")

            zeroloss = torch.tensor(0, device=device)
            ray_loss = cal_ray_loss(scene,  origin, ray_dir, target, valid)\
                if HyperParams['ray_w'] !=0 else zeroloss
            vh_loss = cal_vh_loss(Views, V_index, scene)\
                if HyperParams['vh_w'] !=0 else zeroloss
            sm_loss = cal_sm_loss(scene)\
                if HyperParams['sm_w'] !=0 else zeroloss
            bound_loss = zeroloss
            var_loss = zeroloss
            # bound_loss = 1e0 * torch.nn.functional.relu(Render.dot(parameter/scene.mean_len, scene.normals)).pow(2).mean()
            if paper:
                All_ray_loss.append(ray_loss.item())
                All_vh_loss.append(vh_loss.item())
                All_sm_loss.append(sm_loss.item())

            LOSS = HyperParams['ray_w'] * 1.65/resy/resy * (1.47/IOR)* (1.47/IOR)* ray_loss\
                + HyperParams['vh_w'] * 1.65/resy * vh_loss\
                + HyperParams['sm_w'] *1.65/217.5*1.65/217.5 * sm_loss
                # + HyperParams['sm_w'] *1.65/217.5*1.65/217.5* scene.mean_len/0.08 * sm_loss
            if torch.isnan(LOSS).any():
                print("nan in LOSS")
            LOSS.backward()

            # opt.zero_grad()
            # (HyperParams['ray_w'] * 1.65/resy/resy * ray_loss).backward()
            # print(parameter.grad.abs().mean())
            # opt.zero_grad()
            # (HyperParams['vh_w'] *1.65/resy * vh_loss).backward()
            # print(parameter.grad.abs().mean())            
            # opt.zero_grad()
            # (HyperParams['sm_w'] *1.65/217.5*1.65/217.5 *scene.mean_len/0.08 * sm_loss).backward()
            # print(parameter.grad.abs().mean())

            # if track:
            #     track.log(ray_loss=ray_loss.item(), sm_loss=sm_loss.item(), LOSS=LOSS.item()/10)

            if paper and it % 50 == 0:
                scene.mesh.export(f'{paper_path}/{IOR}/tmp_{i_pass}_{it}.obj')


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

    if paper:
        _ = scene.mesh.export(f'{paper_path}/{IOR}/result.obj')
        # save in h5py
        All_ray_loss = np.array(All_ray_loss)
        All_vh_loss = np.array(All_vh_loss)
        All_sm_loss = np.array(All_sm_loss)
        h5_record.create_dataset('ray_loss', data=All_ray_loss)
        h5_record.create_dataset('vh_loss', data=All_vh_loss)
        h5_record.create_dataset('sm_loss', data=All_sm_loss)


    return scene

if __name__ == "__main__":
    name = 'bunny'


    HyperParams = {
        'name' :  name,
        'IOR' : 1.4723,
        # 'Pass' : 8,
        # 'Iters' : 500,
        'Pass' : 20,
        'Iters' : 200,
        "ray_w" : 40,
        "var_w" : 0,
        "sm_w": 0.08,
        "vh_w": 2e-3,
        "sm_method": "dihedral",
        "optimizer": "sgd",
        "momentum": 0.95,
        "start_lr": 0.0008,
        "lr_decay": 0.1,
        # "lr_decay": 0.1,
        "taubin" : 0,
        "start_len": 0.08,
        "end_len": 0.008,
                    }

 
    scene = optimize(HyperParams)
    _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_pixel.ply")

 
 
