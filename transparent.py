import trimesh
import trimesh.transformations as TF
import torch
import kornia
import numpy as np
import imageio
assert(trimesh.ray.has_embree)
import PIL.Image
import random
# trimesh.util.attach_to_log()
res=256

extIOR, intIOR = 1.0, 1.5
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

def intersect(vertices: torch.tensor, mesh: trimesh.Trimesh, origin, ray_dir,):
    ind_tri = mesh.ray.intersects_first(origin.detach(), ray_dir.detach())
    hitted = (ind_tri != -1)
    hitted = torch.nonzero(torch.from_numpy(hitted)).squeeze()
    ind_vert = mesh.faces[ind_tri[hitted]]
    # ind_tri, hitted = mesh.ray.intersects_id(origin, ray_dir,False, 1, False)
    # ind_vert = mesh.faces[ind_tri]

    # <<Fast, Minimum Storage Ray/Triangle Intersection>> 
    # https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
    
    v0 = vertices[ind_vert[:,0]]
    v1 = vertices[ind_vert[:,1]]
    v2 = vertices[ind_vert[:,2]]

    # Find vectors for two edges sharing v[0]
    edge1 = v1-v0
    edge2 = v2-v0
    n = torch.cross(edge1, edge2)
    n = n / n.norm(dim=1).view(-1,1)
    pvec = torch.cross(ray_dir[hitted], edge2)
    # If determinant is near zero, ray lies in plane of triangle
    det = dot(edge1, pvec)
    inv_det = 1/det
    # Calculate distance from v[0] to ray origin
    tvec = origin[hitted] - v0
    # Calculate U parameter
    u = dot(tvec, pvec) * inv_det
    qvec = torch.cross(tvec, edge1)
    # Calculate V parameter
    v = dot(ray_dir[hitted], qvec) * inv_det
    # Calculate T
    t = dot(edge2, qvec) * inv_det
    return u, v, t, n, hitted

def trace2(vertices, mesh, origin, ray_dir, depth=1, santy_check=False):
    if (depth <= 2):
        # etaI, etaT = extIOR, intIOR
        u, v, t, n, hitted = intersect(vertices, mesh, origin, ray_dir)
        wo = -ray_dir[hitted]
        cosThetaI = dot(wo, n)
        # print("max={},min={}".format(cosThetaI.max(), cosThetaI.min()))
        assert cosThetaI.max()<=1.1 and cosThetaI.min()>=-1.1, "max={},min={}".format(cosThetaI.max(), cosThetaI.min())
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
        new_origin = origin[hitted][refracted] + t[refracted].view(-1,1) * ray_dir[hitted][refracted]
        new_dir = wt[refracted]
        # new_dir = wr[refracted]

        # embree seems to miss epsilon check to avoid self intersection?
        # TODO: a better way to determine epsilon(1e-6)
        new_origin += 1e-6 * new_dir

        index, color = trace2(vertices, mesh, new_origin, new_dir, depth+1, santy_check)
        # index, color = trace2(vertices.detach(), mesh, new_origin, new_dir, depth+1, santy_check)
        return hitted[refracted][index], color
    else:
        if santy_check:
            return torch.ones(ray_dir.shape[0])>0, origin
        # u, v, t, n, hitted = intersect(vertices, mesh, origin, ray_dir)
        # new_origin = origin[hitted] + t.view(-1,1) * ray_dir[hitted]
        # return hitted, new_origin
        return torch.ones(ray_dir.shape[0])>0, ray_dir
        # return torch.ones(ray_dir.shape[0])>0, origin

def save_image(name, img:torch.tensor):
    image = torch.empty(img.shape, dtype=torch.uint8)
    image = (255 * (img-img.min()) / (img.max()-img.min())).to(torch.uint8)
    imageio.imsave(name, image.view(res,res,3).permute(1,0,2))

# #simple camera
# x = torch.linspace(-0.2, 0.2, 256)
# z = torch.linspace(-0.2, 0.2, 256)
# zv,xv=torch.meshgrid(z,x)
# zv = -zv
# yv = torch.ones_like(xv)

# ray_dir = torch.stack((xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)), dim=1)
# ray_dir = ray_dir / ray_dir.norm(dim=1).view(-1,1)
# origin = torch.zeros_like(ray_dir) + torch.tensor([0,-0.5,0])


target_mesh = trimesh.load("/home/jiahui/DR/DR/data/hand.ply")
target_mesh.apply_transform(TF.rotation_matrix(-np.pi/2, [1,0,0]))
target_mesh.export("result/hand_target.ply")
vertices = torch.tensor(target_mesh.vertices, dtype=torch.float64)

scene = target_mesh.scene()

##################### render from front side
angle = None
scene.set_camera(resolution=(res,res), fov=(60,60), distance = 0.8, center=(0,0.2,0), angles=angle)
origin, ray_dir, _ = scene.camera_rays()
origin = torch.from_numpy(origin)
ray_dir = torch.from_numpy(ray_dir)
check = torch.zeros(ray_dir.shape, dtype=torch.float64)
ind, color = trace2(vertices, target_mesh, origin, ray_dir, santy_check=True)
check[ind]=color
save_image("santy_check.png", check)


def render_target(deg):
    angle = TF.euler_from_matrix(TF.rotation_matrix(np.pi/180*deg, [0,1,0]))
    scene.set_camera(resolution=(res,res), fov=(60,60), distance = 0.8, center=(0,0.2,0), angles=angle)
    origin, ray_dir, _ = scene.camera_rays()
    origin = torch.from_numpy(origin)
    ray_dir = torch.from_numpy(ray_dir)

    target = torch.zeros(ray_dir.shape, dtype=torch.float64)
    ind, color = trace2(vertices, target_mesh, origin, ray_dir)
    target[ind]=color
    # save_image("back_target.png", back_target)

    target_mask = torch.zeros(ray_dir.shape, dtype=torch.bool)
    target_mask[ind] = True
    return target, target_mask, origin, ray_dir

Views = []
for i in [-30,-15,0,15,30]:
    Views.append(render_target(i))
for i in [-30,-15,0,15,30]:
    Views.append(render_target(i+180))


# mesh_sm = trimesh.load("/home/jiahui/DR/DR/data/untitled.obj")
# mesh_sm = trimesh.load("/home/jiahui/DR/DR/data/hand_sm.obj")
mesh_sm = trimesh.load("/home/jiahui/DR/tsdf-fusion-python/uv_vh.obj")
mesh_sm.apply_transform(TF.rotation_matrix(-np.pi/2, [1,0,0]))

mesh_sm.export("result/hand_init.ply")
sm_vertices = torch.tensor(mesh_sm.vertices, dtype=torch.float64)
sm_normals = torch.tensor(mesh_sm.vertex_normals, dtype=torch.float64)

# parameter = torch.zeros((sm_vertices.shape[0],1), requires_grad=True)
# parameter = torch.zeros(bunny_sm.vertices.shape, requires_grad=True)
parameter = torch.zeros((200,200), dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([parameter], lr=.0002)
# gif = []
displacement_map = parameter.view((1,1,200,200))
uv = torch.tensor(mesh_sm.visual.uv * 2 - 1, dtype=torch.float64)
uv = uv.view((1,1,mesh_sm.visual.uv.shape[0],2))

laplac = kornia.filters.Laplacian(3)

for it in range(150):
    V_index = random.randint(0, len(Views)-1)
    target, target_mask, origin, ray_dir = Views[V_index]
    # Zero out gradients before each iteration
    opt.zero_grad()

    # vertices = sm_vertices + parameter * sm_normals
    # vertices = sm_vertices + parameter
    displacement = torch.nn.functional.grid_sample(displacement_map, uv)
    vertices = sm_vertices + displacement.view(-1,1) * sm_normals
    mesh_sm.vertices = vertices.detach().numpy()
    render = torch.zeros(ray_dir.shape, dtype=torch.float64)
    ind, color = trace2(vertices, mesh_sm, origin, ray_dir)
    render[ind] = color
    render_mask = torch.zeros(ray_dir.shape, dtype=torch.bool)
    render_mask[ind] = True
    mask = (target_mask * render_mask)
    loss = (render[mask]-target[mask]).pow(2).mean()
    curvat = 10*laplac(displacement_map).abs().mean()
    shrink = 0.01*torch.nn.functional.gelu(1000*displacement_map-0.7517).mean()
    # gif.append(img.detach().numpy())

    # loss = (render-target).pow(2).sum()/(256*256)
    # (loss + 0.01*parameter.pow(2).sum()).backward()
    (loss + curvat + shrink).backward()

    # Optimizer: take a gradient step
    opt.step()

    save_image("result/img_{}.png".format(it), render.detach())
    print('Iteration %03i: error=%g curvat=%g shrink=%g' % (it, loss, curvat, shrink))
    

imageio.imsave("displacment.png", parameter.detach())
save_image("result/optim.png", render.detach())
mesh_sm.export("result/hand_optim.ply")