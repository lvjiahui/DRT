
import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, UInt32, UInt64, Vector2f, Vector3f, Ray3f
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock

from mitsuba.core.xml import load_file, load_string
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time
import imageio
import numpy as np
import kornia
import trimesh

# Load example scene
# Thread.thread().file_resolver().append('data')
# target_scene = load_file('/home/jiahui/DR/DR/data/hand_target.xml')

extIOR, intIOR = 1.0, 1.5
res = 512
para_res = 200

import torch
class MyScene:
    def __init__(self, scene_path):
        scene = load_file(scene_path)
        # Find differentiable scene parameters
        params = traverse(scene)
        self.M_scene = scene
        self.M_params = params
        # Discard all parameters except for one we want to differentiate
        self.P_vertices = params['PLYMesh.vertex_positions'].torch()
        self.P_ns = params['PLYMesh.vertex_normals'].torch()
        self.P_uvs = params['PLYMesh.vertex_texcoords'].torch()
        self.P_faces = params['PLYMesh.faces'].torch().to(torch.long)
        self.P_bak_vertices = self.P_vertices

class MyRay:
    @staticmethod
    def torch( origin:torch.tensor, dir:torch.tensor):
        # construct by torch mean it can have gradient
        ray = MyRay()
        ray.P_dir = dir
        ray.P_origin = origin
        ray.M_Ray = Ray3f(Vector3f(origin), Vector3f(dir), 0, [])
        return ray
    @staticmethod
    def mitsuba(Rays:Ray3f):
        ray = MyRay()
        ray.P_dir = Rays.d.torch()
        ray.P_origin = Rays.o.torch().expand_as(ray.P_dir)
        ray.M_Ray = Rays
        return ray

def optix_intersect(scene:MyScene, rays:MyRay):
    surface_interaction = scene.M_scene.ray_intersect(rays.M_Ray)
    prim_index = surface_interaction.prim_index.torch().to(torch.long)
    mask = surface_interaction.is_valid().torch()
    mask = mask>0
    hitted = torch.nonzero(mask).squeeze()
    prim_index = prim_index[mask]
    return prim_index, hitted

def D_intersect(scene:MyScene, prim_index, origin, ray_dir):
    faces = scene.P_faces
    vertices = scene.P_vertices
    face = faces[prim_index]
    v0 = vertices[face[:,0]]
    v1 = vertices[face[:,1]]
    v2 = vertices[face[:,2]]

    edge1 = v1-v0
    edge2 = v2-v0
    n = torch.cross(edge1, edge2)
    n = n / n.norm(dim=1, p=2).view(-1,1).detach()
    pvec = torch.cross(ray_dir, edge2)
    det = dot(edge1, pvec)
    inv_det = 1/det
    tvec = origin - v0
    u = dot(tvec, pvec) * inv_det
    qvec = torch.cross(tvec, edge1)
    v = dot(ray_dir, qvec) * inv_det
    t = dot(edge2, qvec) * inv_det


    # A = torch.stack( (-edge1, -edge2, ray_dir), dim=2)
    # B = -tvec.reshape((-1,3,1))
    # X, LU = torch.solve(B, A)
    # u = X[:,0,0]
    # v = X[:,1,0]
    # t = X[:,2,0]
    # assert( v.max()<=1.00001 and v.min()>=-0.00001)
    # assert( u.max()<=1.00001 and u.min()>=-0.00001)
    # assert( (v+u).max()<=1.00001 and (v+u).min()>=-0.00001)
    assert(t.min()>0)

    return u, v, t, n

def dot(v1, v2, keepdim = False):
    ''' v1, v2: [n,3]'''
    result = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1] + v1[:,2]*v2[:,2]
    if keepdim:
        return result.view(-1,1)
    return result

def Reflect(wo, n):
    return -wo + 2 * dot(wo, n, True) * n
def Refract(wo: torch.tensor, n, eta):
    eta = eta.view(-1,1)
    cosThetaI = dot(n, wo, True)
    sin2ThetaI = (1 - cosThetaI * cosThetaI).clamp(min = 0)
    sin2ThetaT = eta * eta * sin2ThetaI
    totalInerR = (sin2ThetaT >= 1).view(-1)
    cosThetaT = torch.sqrt(1 - sin2ThetaI.clamp(max = 1))
    wt = eta * -wo + (eta * cosThetaI - cosThetaT) * n
    #  Numerical error or something wrong in the code ?????????
    wt = wt / wt.norm(dim=1).view(-1,1).detach()

    return totalInerR, wt

def FrDielectric(cosThetaI: torch.tensor, etaI, etaT):

    sinThetaI = torch.sqrt( (1-cosThetaI*cosThetaI).clamp(0, 1))
    sinThetaT = sinThetaI * etaI / etaT
    totalInerR = sinThetaT >= 1
    cosThetaT = torch.sqrt( (1-sinThetaT*sinThetaT).clamp(min = 0))
    Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT))
    Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT))
    R = (Rparl * Rparl + Rperp * Rperp) / 2
    return totalInerR, R

def trace2(scene:MyScene, rays:MyRay, depth=1, santy_check=False):
    vertices = scene.P_vertices
    ray_dir = rays.P_dir
    origin = rays.P_origin
    def debug_cos():
        index = cosThetaI.argmax()
        return wo[index].item() , n[index].item() 
    if (depth <= 2):
        prim_index, hitted = optix_intersect(scene, rays)
        ray_dir = ray_dir[hitted]
        origin = origin[hitted]
        # etaI, etaT = extIOR, intIOR
        u, v, t, n = D_intersect(scene, prim_index, origin, ray_dir)
        wo = -ray_dir
        cosThetaI = dot(wo, n)
        if not (cosThetaI.max()<=1.00001 and cosThetaI.min()>=-1.00001):
            print("max={},min={}".format(cosThetaI.max(), cosThetaI.min()))
        # assert cosThetaI.max()<=1.00001 and cosThetaI.min()>=-1.00001, "wo={},n={}".format(*debug_cos())
        cosThetaI = cosThetaI.clamp(-1, 1)
        entering = cosThetaI >= 0

        # assert(entering.all() or torch.logical_not(entering).all())
        # etaI, etaT = extIOR*torch.ones_like(hitted), intIOR*torch.ones_like(hitted)
        # if not entering.all(): 
        #     etaI, etaT = etaT, etaI
        #     n = -n
        #     cosThetaI = -cosThetaI
        exc = torch.logical_not(entering)
        etaI, etaT = extIOR*torch.ones_like(hitted), intIOR*torch.ones_like(hitted)
        etaI[exc], etaT[exc] = etaT[exc], etaI[exc]
        n[exc] = -n[exc]
        cosThetaI[exc] = -cosThetaI[exc]  

        totalInerR1, R = FrDielectric(cosThetaI, etaI, etaT)
        wr = Reflect(wo, n)
        totalInerR2, wt = Refract(wo, n, etaI/etaT)
        # print(totalInerR1.shape, totalInerR2.shape)
        assert (totalInerR1 == totalInerR2).all(), (totalInerR1 != totalInerR2).sum()/(256*256*3)
        refracted = torch.logical_not(totalInerR1)
        # refracted = torch.ones(totalInerR1.shape[0])>0

        # print(t.shape, ray_dir[hitted].shape)
        new_origin = origin[refracted] + t[refracted].view(-1,1) * ray_dir[refracted]
        new_dir = wt[refracted]
        # new_dir = wr[refracted]
        new_Rays = MyRay.torch(new_origin, new_dir)

        index, color = trace2(scene, new_Rays, depth+1, santy_check)
        return hitted[refracted][index], color
    else:
        if santy_check:
            return torch.ones(ray_dir.shape[0])>0, origin
        return torch.ones(ray_dir.shape[0])>0, ray_dir

def save_image(name, img:torch.tensor, res=res):
    image = torch.empty(img.shape, dtype=torch.uint8)
    image = (255 * (img-img.min()) / (img.max()-img.min())).to(torch.uint8)
    imageio.imsave(name, image.view(res,res,3).permute(1,0,2).cpu())


def render(scene:MyScene, deg=0, santy_check=False):
    origin = np.array([0,0.2,0.8])
    deg = np.array(np.math.pi * deg/180)
    rotate_Y = np.array([
                        [np.cos(deg), 0, np.cos(deg + np.math.pi/2)],
                        [0, 1, 0],
                        [np.sin(deg), 0, np.sin(deg + np.math.pi/2)]])
    sensor = load_string("""
    <sensor  version="2.0.0" type="perspective">
        <transform name="to_world">
            <lookat origin="0,0.2,0.8"
                    target="0,0.2,0"
                    up="0, 1, 0"/>
        </transform>
        <float name="fov" value="60"/>
        <film type="hdrfilm">
            <string name="pixel_format" value="rgb"/>
            <integer name="width" value="{}"/>
            <integer name="height" value="{}"/>
        </film>
        <sampler type="independent">
            <integer name="sample_count" value="1"/>
        </sampler>
    </sensor>""".format(res,res)
    )
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    spp = 1
    # Seed the sampler
    total_sample_count = ek.hprod(film_size) * spp
    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(ek.arange(UInt64, total_sample_count))
    # Enumerate discrete sample & pixel indices, and uniformly sample
    # positions within each pixel.
    pos = ek.arange(UInt32, total_sample_count)
    pos //= spp
    scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
    pos = Vector2f(Float(pos  % int(film_size[0])),
                   Float(pos // int(film_size[0])))
    # pos += sampler.next_2d()
    pos += Vector2f(0.5,0.5)
    # Sample rays starting from the camera sensor
    M_rays, weights = sensor.sample_ray(
        time=0,
        sample1=sampler.next_1d(),
        sample2=pos * scale,
        sample3=0
    )
    rays = MyRay().mitsuba(M_rays)
    img = torch.zeros([res*res,3], dtype=torch.float32, device='cuda')
    ind, color = trace2(scene, rays, santy_check=santy_check)
    img[ind]=color
    img_mask = torch.zeros([res*res,3], dtype=torch.bool, device='cuda')
    img_mask[ind] = True
    return img, img_mask

target_scene = MyScene('/root/mnt/DR/data/hand_target.xml')

check, _ = render(target_scene, santy_check=True)
save_image("result/santy_check.png", check)
target_img, target_mask = render(target_scene)
save_image("result/target.png", target_img)

del target_scene
ek.cuda_malloc_trim()

scene = MyScene('/root/mnt/DR/data/hand.xml')

parameter = torch.zeros((para_res,para_res), dtype=torch.float32, requires_grad=True, device='cuda')
opt = torch.optim.Adam([parameter], lr=.0002)
displacement_map = parameter.view((1,1,para_res,para_res))
uv = scene.P_uvs.expand([1,1, scene.P_uvs.shape[0], 2])
laplac = kornia.filters.Laplacian(3)

for it in range(99):
    # Zero out gradients before each iteration
    opt.zero_grad()
    displacement = torch.nn.functional.grid_sample(displacement_map, uv)
    scene.P_vertices = scene.P_bak_vertices + displacement.view(-1,1) * scene.P_ns
    # update vertices for optix
    scene.M_params['PLYMesh.vertex_positions'] = scene.P_vertices
    scene.M_params.update()

    render_img, render_mask = render(scene)
    mask = (target_mask * render_mask)
    loss = (render_img[mask]-target_img[mask]).abs().mean()
    # curvat = 10*laplac(displacement_map).abs().mean()
    curvat = 1000*laplac(displacement_map).pow(2).mean()
    shrink = 0.01*torch.nn.functional.gelu(1*displacement_map-0.7517).mean()
    # gif.append(img.detach().numpy())


    (loss + curvat + shrink).backward()
    # (loss).backward()

    # Optimizer: take a gradient step
    opt.step()

    save_image("result/img_{}.png".format(it), render_img.detach())
    print('Iteration %03i: error=%g curvat=%g shrink=%g' % (it, loss, curvat, shrink))
    # save_image("result/img_{}.png".format(it), parameter.detach().cpu(),res=para_res)
    imageio.imsave("result/disp_{}.png".format(it), parameter.detach().cpu())
    
mesh = trimesh.Trimesh(scene.P_vertices.detach().cpu(), scene.P_faces.detach().cpu())
mesh.export("result/hand_optim.ply")

imageio.imsave("displacment.png", parameter.detach().cpu())
# save_image("result/optim.png", render.detach())