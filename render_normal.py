import torch
import imageio

# conda install -c conda-forge trimesh
# https://trimsh.org/index.html
import trimesh
assert trimesh.ray.has_embree


#simple camera
x = torch.linspace(-0.2, 0.2, 256)
z = torch.linspace(-0.2, 0.2, 256)
xv,zv=torch.meshgrid(x,z)
yv = torch.ones_like(xv)

ray_dir = torch.stack((xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)), dim=1)
ray_dir = ray_dir / ray_dir.norm(dim=1).reshape(-1,1)
origin = torch.zeros_like(ray_dir) + torch.tensor([0,-0.5,0])


def render_norm(vertices:torch.tensor, mesh:trimesh.Trimesh):
    index_tri = mesh.ray.intersects_first(origin, ray_dir)
    hit = (index_tri != -1)
    index_vert = mesh.faces[index_tri[hit]]

    # <<Fast, Minimum Storage Ray/Triangle Intersection>> 
    # https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
    def dot(v1, v2):
        return v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1] + v1[:,2]*v2[:,2]

    v0 = vertices[index_vert[:,0]]
    v1 = vertices[index_vert[:,1]]
    v2 = vertices[index_vert[:,2]]
    n0 = torch.tensor(mesh.vertex_normals[index_vert[:,0]], dtype=torch.float)
    n1 = torch.tensor(mesh.vertex_normals[index_vert[:,1]], dtype=torch.float)
    n2 = torch.tensor(mesh.vertex_normals[index_vert[:,2]], dtype=torch.float)

    # Find vectors for two edges sharing v[0]
    edge1 = v1-v0
    edge2 = v2-v0
    pvec = torch.cross(ray_dir[hit], edge2)

    # If determinant is near zero, ray lies in plane of triangle
    det = dot(edge1, pvec)
    inv_det = 1/det
    # Calculate distance from v[0] to ray origin
    tvec = origin[hit] - v0
    # Calculate U parameter
    u = dot(tvec, pvec) * inv_det
    qvec = torch.cross(tvec, edge1)
    # Calculate V parameter
    v = dot(ray_dir[hit], qvec) * inv_det
    # Calculate T
    t = dot(edge2, qvec) * inv_det
    # print( v.max(), v.min())
    assert( v.max()<=1.00001 and v.min()>=-0.00001)
    assert( u.max()<=1.00001 and u.min()>=-0.00001)
    assert( (v+u).max()<=1.00001 and (v+u).min()>=-0.00001)
    assert(t.min()>0)
    # interpolate normal
    n = (1-u-v).reshape((-1,1)) * n0 + u.reshape((-1,1)) * n1 + v.reshape((-1, 1)) * n2
    n = n / n.norm(dim=1).reshape(-1,1)

    img = torch.zeros(ray_dir.shape)
    img[hit] = n
    img = img.reshape(256,256,3)

    return img


bunny = trimesh.load("data/bunny.obj")
bunny.vertices -= bunny.center_mass
bunny.apply_transform([[1,0,0,0],
                        [0,0,1,0],
                        [0,-1,0,0],
                        [0,0,0,1]])

target = render_norm(torch.tensor(bunny.vertices, dtype=torch.float), bunny)
imageio.imsave("data/target.png", target)

#initial position
bunny.vertices += 0.1*bunny.scale
vertices = torch.tensor(bunny.vertices, dtype=torch.float)

parameter = torch.zeros(3, requires_grad=True)
opt = torch.optim.Adam([parameter], lr=.002)

for it in range(50):
    # Zero out gradients before each iteration
    opt.zero_grad()

    new_vertices = vertices + parameter
    bunny.vertices = new_vertices.detach().numpy()
    img = render_norm(new_vertices, bunny)

    loss = (img-target).pow(2).sum()/(256*256)
    loss.backward()

    # Optimizer: take a gradient step
    opt.step()

    imageio.imsave("result/step_{}.png".format(it), img.detach())
    print('Iteration %03i: error=%g' % (it, loss))