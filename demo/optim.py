import os
from h5py import h5p
import torch
import numpy as np
import cv2
from tqdm import trange
# import DiffRender as Render
import Render_opencv as Render
import sys
import h5py
import time

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
    return grad

def interp_L(start, end, it, Pass):
    assert it <= Pass-1
    step = (end - start)/(Pass-1)
    return it*step + start

def interp_R(start, end, it, Pass):
    return 1/interp_L(1/start, 1/end, it, Pass)

def torch2img(img:torch.Tensor):
    image = (255 * (img-img.min()) / (img.max()-img.min())).to(torch.uint8)
    image = image.view(resy,resx,-1).cpu().numpy()
    if image.shape[2] == 1: image = image[:,:,0]
    # return Image.fromarray(image)
    return image

class Data_syn:
    def __init__(self,HyperParams):
        self.name = HyperParams['name']
        self.HyperParams = HyperParams
        IOR = HyperParams['IOR']
        Render.intIOR = IOR
        Render.resy = resy
        Render.resx = resx
        Render.device = device
        Render.Float = Float
        self.Target_scene = Render.Scene(f"data/{self.name}.ply")

    def generate_syn_data(self):
        Views = []
        for i in trange(72):
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
            out_origin, out_ray_dir, valid = self.Target_scene.render_transparent(ray_origin, ray_dir)

            valid = valid[:,0]
            mask = self.Target_scene.render_mask(ray_origin, ray_dir)
            M = mask.reshape([resy,resx]).cpu().numpy().astype(np.uint8)
            assert M.max() == 1
            dist= (cv2.distanceTransform(M, cv2.DIST_L2, 0)-0).clip(0,1)\
             - (cv2.distanceTransform(1-M, cv2.DIST_L2, 0)-1).clip(0,1) 
            mask = (dist + 1) / 2 #[0,1]
            mask = torch.tensor(mask, dtype=Float,device=device)

            camera_M = (R, K, R_inverse, K_inverse)
            Views.append((out_ray_dir, valid, mask, camera_M))

        torch.save(Views, f'data/{self.name}_capture.pt')

    def load_data(self):
        self.visual_hull_scene = Render.Scene(f"data/{self.name}_vh.ply")
        self.Views = torch.load(f'data/{self.name}_capture.pt')

    def __len__(self):
        return len(self.Views)

    def get_view_torch(self, V_index):
        out_ray_dir, valid, mask, camera_M = self.Views[V_index]
        R, K, R_inverse, K_inverse = camera_M
        origin, ray_dir = generate_ray(K_inverse, R_inverse)
        ####################################################
        # exchange  R_inverse and R name 
        # camera_M = (R, K, R_inverse, K_inverse)
        camera_M = (R_inverse, K, R, K_inverse)
        ######################################################
        return out_ray_dir, valid, mask, origin, ray_dir, camera_M

    def get_origin_torch(self, V_index):
        out_ray_dir, valid, mask, camera_M = self.Views[V_index]
        R, K, R_inverse, K_inverse = camera_M
        R_inverse = R_inverse.to(device)
        ray_origin = R_inverse[:3,3] #[3]
        ####################################################
        # exchange  R_inverse and R name 
        # camera_M = (R, K, R_inverse, K_inverse)
        camera_M = (R_inverse, K, R, K_inverse)
        ######################################################
        return ray_origin, camera_M, mask

class DRT_Optimizer():
    def __init__(self,HyperParams, Data = 0, output=True, track=None):
        self.name = HyperParams['name']
        self.output = output
        self.HyperParams = HyperParams
        pid = os.getpid()
        IOR = HyperParams['IOR']
        Render.intIOR = IOR
        Render.resy = resy
        Render.resx = resx
        Render.device = device
        Render.Float = Float
        self.Data = Data
        self.visual_hull_scene = Data.visual_hull_scene

    def start_optim(self):
        def setup_opt(scene, lr, HyperParams):

            init_vertices = scene.vertices
            parameter = torch.zeros(init_vertices.shape, dtype=Float, requires_grad=True, device=device)      
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

            os.system('rm '+script)

        def cal_vh_loss(V_index, scene):

            vh_loss = 0
            for v in np.arange(0,72,9):
                index =  (V_index+v)%72
                origin, camera_M, mask = self.Data.get_origin_torch(index) #[3]
                silhouette_edge = scene.silhouette_edge(origin)
                index, output = scene.primary_visibility(silhouette_edge, camera_M, origin, detach_depth=True)
                vh_loss += (mask.view((resy,resx))[index[:,1],index[:,0]] - output).abs().sum()

            return vh_loss


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
                sm_loss = dihedral_angle.sum()

            return sm_loss

        def cal_ray_loss(scene, origin, ray_dir, out_ray_dir, valid):
            render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir)

            diff = (render_out_dir - out_ray_dir)
            valid_mask = valid * render_mask[:,0]
            ray_loss = (diff[valid_mask]).pow(2).sum()

            return ray_loss


        def rand_generator(self, num=72):
            index = list(np.arange(72))

            while True:
                np.random.shuffle(index)
                for i in index: yield i % 72

        HyperParams = self.HyperParams
        IOR     = Render.intIOR   
        resy    = Render.resy     
        resx    = Render.resx     
        device  = Render.device   
        Float   = Render.Float    
        scene = self.visual_hull_scene
        view = rand_generator(len(self.Data))

        for i_pass in range(HyperParams['Pass']):
            remesh_len = interp_L(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
            # remesh_len = interp_R(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
            lr = interp_R(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])

            # if i_pass != 0:
            remesh(scene, meshlab_remesh_srcipt.format(remesh_len))
            init_vertices, parameter, opt = setup_opt(scene, lr, HyperParams)

            print(f'remesh_len {remesh_len} lr {lr}')



            for it in range(HyperParams['Iters']):
                V_index = next(view)
                out_ray_dir, valid, mask, origin, ray_dir, camera_M = self.Data.get_view_torch(V_index)
                target = out_ray_dir

                # Zero out gradients before each iteration
                opt.zero_grad()
                vertices = init_vertices + parameter
                scene.update_verticex(vertices)

                if torch.isnan(vertices).any():
                    print("nan in vertices")

                zeroloss = torch.tensor(0, device=device)
                ray_loss = cal_ray_loss(scene,  origin, ray_dir, target, valid)\
                    if HyperParams['ray_w'] !=0 else zeroloss
                vh_loss = cal_vh_loss(V_index, scene)\
                    if HyperParams['vh_w'] !=0 else zeroloss
                sm_loss = cal_sm_loss(scene)\
                    if HyperParams['sm_w'] !=0 else zeroloss

                LOSS = HyperParams['ray_w'] * 1.65/resy/resy * (1.47/IOR)* (1.47/IOR)* ray_loss\
                    + HyperParams['vh_w'] * 1.65/resy * vh_loss\
                    + HyperParams['sm_w'] *1.65/217.5*1.65/217.5 * sm_loss
                if torch.isnan(LOSS).any():
                    print("nan in LOSS")
                LOSS.backward()


                if it%100==0 and self.output:
                    print('Iteration %03i: ray=%g  vh=%g  sm=%g  grad=%g' % \
                        (it, ray_loss, vh_loss, sm_loss, parameter.grad.abs().max()))
                opt.step()
        return scene

if __name__ == "__main__":
    name = 'hailuo'
    HyperParams = {
        'name' :  name,
        'IOR' : 1.4723,
        'Pass' : 20,
        'Iters' : 200,
        "ray_w" : 40,
        "sm_w": 0.08,
        # "sm_w": 0.02,
        # "vh_w": 2e-3,
        "vh_w": 3e-3,
        "sm_method": "dihedral",
        "optimizer": "sgd",
        "momentum": 0.95,
        "start_lr": 0.0008,
        # "lr_decay": 0.1,
        "lr_decay": 0.5,
        # "start_len": 0.08,
        "start_len": 0.04,
        # "end_len": 0.008,
        "end_len": 0.004,
                    }

    data = Data_syn(HyperParams)
    if sys.argv[1] == 'capture':
        data.generate_syn_data()
    elif sys.argv[1] == 'optimize':
        data.load_data()
        optimizer = DRT_Optimizer(HyperParams, data)
        t = time.time()
        result_scene = optimizer.start_optim()
        _ = result_scene.mesh.export(f"data/{name}_rec.ply")
        print("optmize time:", time.time() - t, "s")
    else: 
        print("capture or optimize?")


 
 
