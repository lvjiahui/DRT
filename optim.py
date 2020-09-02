import os
import torch
import time
import captured_data
import numpy as np
import random
import cv2
import h5py
from tqdm import trange
import meshlabxml as mlx


# os.environ['CUDA_VISIBLE_DEVICES']=str(cuda_num)
import Render_opencv as Render

Float = captured_data.Float
device= captured_data.device


def limit_hook(grad):
    # max = 0.5
    max = 1
    if torch.isnan(grad).any():
        print("nan in grad")
    grad[torch.isnan(grad)] = 0
    grad[grad>max]=max
    grad[grad<-max]=-max
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

    # For Refraction loss visualization
    # def save_visualization():
    #     screen_pixel, valid, mask, origin, ray_dir, camera_M = data.get_view(36)
    #     target = screen_pixel
    #     render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir)
    #     target = target  - render_out_ori.detach()
    #     target = target/target.norm(dim=1, keepdim=True)
    #     diff = (render_out_dir - target)
    #     # valid_mask = (diff.norm(dim=1)<0.5) * valid * render_mask[:,0]
    #     valid_mask = valid * render_mask[:,0]       
    #     image = torch.zeros_like(diff)
    #     image[valid_mask] = diff[valid_mask]
    #     image = image.pow(2).sum(dim=1,keepdim=True).sqrt()
    #     image[0] = 1.5
    #     # assert image.max() <= 1.51
    #     image[image>1.5]=1.5
    #     Render.save_torch(f"/root/workspace/show/mouse/{i_pass}_{it}.png", image)

class Meshlabserver:
    def __init__(self):
        self.meshlab_remesh_srcipt = """
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Remeshing: Isotropic Explicit Remeshing">
          <Param value="3" isxmlparam="0" name="Iterations" type="RichInt" description="Iterations" tooltip="Number of iterations of the remeshing operations to repeat on the mesh."/>
          <Param value="false" isxmlparam="0" name="Adaptive" type="RichBool" description="Adaptive remeshing" tooltip="Toggles adaptive isotropic remeshing."/>
          <Param value="false" isxmlparam="0" name="SelectedOnly" type="RichBool" description="Remesh only selected faces" tooltip="If checked the remeshing operations will be applied only to the selected faces."/>
          <Param value="{}" isxmlparam="0" name="TargetLen" type="RichAbsPerc" description="Target Length" min="0" max="214.384" tooltip="Sets the target length for the remeshed mesh edges."/>
          <Param value="180" isxmlparam="0" name="FeatureDeg" type="RichFloat" description="Crease Angle" tooltip="Minimum angle between faces of the original to consider the shared edge as a feature to be preserved."/>
          <Param value="true" isxmlparam="0" name="CheckSurfDist" type="RichBool" description="Check Surface Distance" tooltip="If toggled each local operation must deviate from original mesh by [Max. surface distance]"/>
          <Param value="1" isxmlparam="0" name="MaxSurfDist" type="RichAbsPerc" description="Max. Surface Distance" min="0" max="214.384" tooltip="Maximal surface deviation allowed for each local operation"/>
          <Param value="true" isxmlparam="0" name="SplitFlag" type="RichBool" description="Refine Step" tooltip="If checked the remeshing operations will include a refine step."/>
          <Param value="true" isxmlparam="0" name="CollapseFlag" type="RichBool" description="Collapse Step" tooltip="If checked the remeshing operations will include a collapse step."/>
          <Param value="true" isxmlparam="0" name="SwapFlag" type="RichBool" description="Edge-Swap Step" tooltip="If checked the remeshing operations will include a edge-swap step, aimed at improving the vertex valence of the resulting mesh."/>
          <Param value="true" isxmlparam="0" name="SmoothFlag" type="RichBool" description="Smooth Step" tooltip="If checked the remeshing operations will include a smoothing step, aimed at relaxing the vertex positions in a Laplacian sense."/>
          <Param value="true" isxmlparam="0" name="ReprojectFlag" type="RichBool" description="Reproject Step" tooltip="If checked the remeshing operations will include a step to reproject the mesh vertices on the original surface."/>
         </filter>
        </FilterScript>
        """
        pid = str(os.getpid())
        self.tmpply_path = f"/dev/shm/DR/temp_{pid}.ply"
        self.remeshply_path = f"/dev/shm/DR/remesh_{pid}.ply"
        self.script_path = f"/dev/shm/DR/script_{pid}.mlx"
        self.ssh = "ssh jiahui@172.31.224.138 "
        self.cmd = "DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage -i "\
            + self.tmpply_path + " -o "\
            + self.remeshply_path\
            + " -s " + self.script_path
        self.cmd = self.ssh + self.cmd
        self.cmd = self.cmd + " 1>/dev/null 2>&1"
        # print(self.cmd)
        
    def remesh(self, scene, remesh_len):
        meshlab_remesh_srcipt = self.meshlab_remesh_srcipt.format(remesh_len)
        with open(self.script_path, 'w') as script_file:
            script_file.write(meshlab_remesh_srcipt)
        scene.mesh.export(self.tmpply_path)
        assert(os.system(self.cmd) == 0)
        scene.update_mesh(self.remeshply_path) 

        # os.system('rm '+tmpply)
        # os.system('rm '+remeshply)     
        os.system('rm '+self.script_path)

    # TODO: convert to class method
    # def mean_hausd(scene):
    #     # GT_path = 'DR/data/dog_scan_fixed.OBJ'
    #     GT_path = '/root/workspace/data/dog_scan_fixed.OBJ'
    #     pid = str(os.getpid())
    #     tmpply = f"/dev/shm/DR/temp_{pid}.ply"
    #     scene.mesh.export(tmpply)
    #     logpath = f"/dev/shm/DR/hausd_log_{pid}"
    #     os.system('rm '+logpath)
    #     ssh = "ssh jiahui@172.31.224.138 "
    #     # cmd = f"DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage -i  {tmpply} {GT_path}   -s DR/data/hausd.mlx -l " + logpath
    #     # assert (os.system(ssh+cmd + " 1>/dev/null")==0)
    #     cmd = f"DISPLAY=:1 meshlabserver -i  {tmpply} {GT_path}   -s /root/workspace/data/hausd.mlx -l " + logpath
    #     assert (os.system(cmd + " 1>/dev/null")==0)
    #     dist = mlx.compute.parse_hausdorff(logpath)['mean_distance']
    #     os.system('rm '+tmpply)
    #     os.system('rm '+logpath)
    #     return dist

class Loss_calculator:
    def __init__(self, scene, data, HyperParams):
        self.scene = scene
        self.data = data
        self.HyperParams = HyperParams
        self.ray_view  = data.ray_view_generator()
        self.silh_view = data.silh_view_generator()

    def vh_loss(self):
        scene = self.scene
        data = self.data
        # silhouette_edge = scene.silhouette_edge(origin[0])
        # index, output = scene.primary_visibility(silhouette_edge, camera_M, origin[0], detach_depth=True)
        # vh_loss = 1e2 * (mask.view((resy,resx))[index[:,1],index[:,0]] - output).abs().mean()

        vh_loss = 0
        for v in np.arange(0,72,9):
            # index =  (V_index+v)%72
            index =  next(self.silh_view)
            screen_pixel, valid, mask, origin, ray_dir, camera_M = data.get_view(index)
            silhouette_edge = scene.silhouette_edge(origin[0])
            index, output = scene.primary_visibility(silhouette_edge, camera_M, origin[0], detach_depth=True)
            vh_loss += (mask.view((data.resy,data.resx))[index[:,1],index[:,0]] - output).abs().sum()

        return vh_loss

    def var_loss(self):
        scene = self.scene
        # var_loss = 2*scene.area_var()
        # var_loss = 2*scene.edge_var()
        laplac = scene.vertices - scene.weightM.mm(scene.vertices) 
        normals = scene.normals.detach()
        N_laplac = Render.dot(laplac, normals, keepdim=True) * normals
        vertical_laplac = laplac - N_laplac
        var_loss = torch.norm(vertical_laplac/scene.mean_len, dim=1).pow(2).mean()    
        return var_loss

    def sm_loss(self):
        scene = self.scene
        HyperParams = self.HyperParams
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

            # sm_loss = 20*dihedral_angle.mean()/(scene.mean_len*scene.mean_len)

            # laplac = scene.vertices - scene.weightM.mm(scene.vertices) 
            # laplac = laplac - scene.weightM.mm(laplac) 
            # laplac = laplac - scene.weightM.mm(laplac) 
            # sm_loss += 100*torch.norm(laplac/scene.mean_len, dim=1).pow(2).mean()  

        return sm_loss

    def ray_loss(self):
        scene = self.scene
        data = self.data

        V_index = next(self.ray_view)
        target, valid, mask, origin, ray_dir, camera_M = data.get_view(V_index)
        render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir)

        if not self.HyperParams['synthetic']:
            screen_pixel = target
            target = screen_pixel  - render_out_ori.detach()
            # target = screen_pixel  - render_out_ori
            target = target/target.norm(dim=1, keepdim=True)

        diff = (render_out_dir - target)
        valid_mask = valid * render_mask[:,0]
        ray_loss = (diff[valid_mask]).pow(2).sum()

        return ray_loss    

    def all_loss(self):
        scene = self.scene
        data = self.data
        HyperParams = self.HyperParams

        zeroloss = torch.tensor(0, device=device)
        ray_loss = self.ray_loss()\
            if HyperParams['ray_w'] !=0 else zeroloss
        vh_loss = self.vh_loss()\
            if HyperParams['vh_w'] !=0 else zeroloss
        sm_loss = self.sm_loss()\
            if HyperParams['sm_w'] !=0 else zeroloss
        var_loss = zeroloss
        # bound_loss = 1e0 * torch.nn.functional.relu(Render.dot(parameter/scene.mean_len, scene.normals)).pow(2).mean()

        # LOSS = HyperParams['ray_w'] /resy/resy * ray_loss\
        #     + HyperParams['vh_w'] / resy * vh_loss\
        #     + HyperParams['sm_w'] / scene.mean_len * sm_loss

        LOSS = HyperParams['ray_w'] * 217.5 /data.resy/data.resy * ray_loss\
            + HyperParams['vh_w'] * 217.5 / data.resy * vh_loss\
            + HyperParams['sm_w'] * scene.mean_len/10 * sm_loss
        return LOSS, f'ray={ray_loss:g} vh={vh_loss:g} sm={sm_loss:g}'

def get_data(HyperParams):
    Redmi_cam = ['tiger','pig','horse','rabbit_new', 'cup']
    graypoint_cam = ['hand', 'mouse', 'dog', 'monkey']
    syn_cam = ['bunny', 'kitten']
    name = HyperParams['name']
    syn_scene = None
    if HyperParams['synthetic']: syn_scene = Render.Scene(f"/root/workspace/data/{name}_scan_simplify.ply")
    if name in graypoint_cam:
        data = captured_data.Data_Graypoint(HyperParams, syn_scene)
    elif name in Redmi_cam:
        data = captured_data.Data_Redmi(HyperParams, syn_scene)
    else: 
        assert False
    return data

def optimize(HyperParams, output=True, track=None):
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


    name = HyperParams['name']
    scene = Render.Scene(f"/root/workspace/data/{name}_vh.ply")
    meshlabserver = Meshlabserver()
    data = get_data(HyperParams)
    Render.intIOR = HyperParams['IOR']
    Render.resy = data.resy
    Render.resx = data.resx
    Render.device = device
    Render.Float = Float

    loss_calculator = Loss_calculator(scene, data, HyperParams)

    # torch.autograd.set_detect_anomaly(True)

    start_time = time.time()

    for i_pass in range(HyperParams['Pass']):
        remesh_len = interp_R(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        lr = interp_R(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])
        # remesh_len = interp_L(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        # lr = interp_L(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])

        print(f'remesh_len {remesh_len:g} lr {lr:g}')
        meshlabserver.remesh(scene, remesh_len)
        init_vertices, parameter, opt = setup_opt(scene, lr, HyperParams)

        for it in range(HyperParams['Iters']):
            # if track and it%200 == 0:
            #     track.log(mean_hausd=mean_hausd(scene))

            # Zero out gradients before each iteration
            opt.zero_grad()
            # IORopt.zero_grad()
            vertices = init_vertices + parameter
            # vertices = init_vertices + parameter * scene.normals
            scene.update_verticex(vertices)

            if torch.isnan(vertices).any():
                print("nan in vertices")

            loss, loss_str = loss_calculator.all_loss()
            if torch.isnan(loss).any():
                print("nan in LOSS")
            loss.backward()

            # if track:
            #     track.log(ray_loss=ray_loss.item(), sm_loss=sm_loss.item(), LOSS=LOSS.item()/10)

            if it%100==0 and output:
                print(f'Iteration {it}: {loss_str} maxgrad={parameter.grad.abs().max():g}')
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

    print(f"optimize time : {time.time() - start_time}")

    return scene

if __name__ == "__main__":
    name = 'david'

    num_view = 72
    HyperParams = {
        'synthetic' :  True,
        # 'synthetic' :  False,
        'name' :  name,
        'IOR' : 1.4723,
        'Pass' : 20,
        'Iters' : 200,
        "ray_w" : 40,
        # "var_w" : 4e0,
        "var_w" : 0,
        # "sm_w": 0.08,
        "sm_w": 0.02,
        "vh_w": 2e-3,
        # "sm_method": "laplac",
        # "sm_method": "fairing",
        "sm_method": "dihedral",
        "optimizer": "sgd",
        # "optimizer": "adam",
        "momentum": 0.95,
        "start_lr": 0.1,
        "lr_decay": 0.5,
        # "lr_decay": 1,
        "taubin" : 0,
        "start_len": 10,
        "end_len": 1,
        'num_view': num_view,
                    }

    scene = optimize(HyperParams)
    # _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_dilate.ply")
    _ = scene.mesh.export(f"/root/workspace/DR/result/{name}.ply")
