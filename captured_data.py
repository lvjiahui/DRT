import torch
import numpy as np
import cv2
import h5py
from tqdm import trange
import imageio

Float = torch.float64
device='cuda'

def process_mask(M):

    # imageio.imsave("origin_mask.png", M)
    # kernel = np.ones((5,5),np.uint8)
    # M = cv2.dilate(M,kernel,iterations = 1)
    # imageio.imsave("dialate_mask.png", M)

    if M.max() == 255: M //= 255
    assert M.max() == 1
    dist= (cv2.distanceTransform(M, cv2.DIST_L2, 0)-0).clip(0,1)\
     - (cv2.distanceTransform(1-M, cv2.DIST_L2, 0)-1).clip(0,1) #[-1,+1]
    mask = (dist + 1) / 2 #[0,1]
    mask[-1] = 0.5
    return mask


def generate_ray(resy, resx, K_inverse, R_inverse):
    K_inverse = torch.tensor(K_inverse, device=device, dtype=Float)
    R_inverse = torch.tensor(R_inverse, device=device, dtype=Float)

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

def synthetic_data(Target_scene, resy, resx, K_inverse, R_inverse):

    ray_origin, ray_dir = generate_ray(resy, resx, K_inverse, R_inverse)
    _, out_ray_dir, valid = Target_scene.render_transparent(ray_origin, ray_dir)
    target = out_ray_dir
    valid = valid[:,0]
    mask = Target_scene.render_mask(ray_origin, ray_dir)
    mask = mask.reshape([resy, resx]).cpu().numpy().astype(np.uint8)
    return ray_origin, ray_dir, target, valid, mask

class Data:
    def get_view(self, V_index):
        screen_pixel, valid, mask, origin, ray_dir, camera_M = self.Views[V_index]
        R, K, R_inverse, K_inverse = camera_M

        screen_pixel  = screen_pixel.to(device)
        valid       = valid     .to(device)
        mask        = mask      .to(device)
        origin      = origin    .to(device)
        ray_dir     = ray_dir   .to(device)
        R_inverse   = R_inverse .to(device)
        K_inverse   = K_inverse .to(device)
        R           = R         .to(device)
        K           = K         .to(device)

        camera_M = (R, K, R_inverse, K_inverse)
        return screen_pixel, valid, mask, origin, ray_dir, camera_M

    def ray_view_generator(self):
        # head_view = {
        #     # 'mouse': 16,
        #     'rabbit': 18,
        #     'hand': 18,
        #     'dog': 17,
        #     'monkey': 19,
        # }
        # if name in head_view.keys():
        #     view_range = HyperParams['view_range']
        #     head_num = head_view[name]
        #     index = list(np.arange(head_num-18-view_range, head_num+1-18+view_range))
        #     index = index + list(np.arange(head_num+18-view_range, head_num+1+18+view_range))  
        # else:
        #     index = list(np.arange(num))


        index = list(np.arange(0, 72, 72//self.num_view))
        # mouse debug
        # if self.name == 'mouse':
        #     index = list(np.arange(-5, 10))
        #     index = index + list(np.arange(22,40))

        print('num_view ray', len(index))

        while True:
            np.random.shuffle(index)
            for i in index: yield i % 72

    def silh_view_generator(self):
        index = list(np.arange(72))

        # index = list(np.arange(0,72, 72//self.num_view))
        print('num_view silh', len(index))
        while True:
            np.random.shuffle(index)
            for i in index: yield i % 72


class Data_Graypoint(Data):
    def __init__(self, HyperParams, syn_scene=None):
        self.resy=960
        self.resx=1280
        self.num_view = HyperParams['num_view']
        self.name = HyperParams['name']
        h5data = h5py.File(f'/root/workspace/data/{self.name}.h5','r')

        self.Views = []
        for i in trange(72):
            # out_dir = h5data['ray'][i,:,-3:]
            # out_origin = h5data['ray'][i,:,-6:-3]
            # out_origin = h5data['cleaned_position'][i,:]
            K = h5data['cam_k'][:]
            R = h5data['cam_proj'][i]
            R_inverse = np.linalg.inv(R)
            K_inverse = np.linalg.inv(K)
            if not HyperParams['synthetic']:
                screen_pixel = h5data['cleaned_position'][i,:]
                target = screen_pixel
                mask = h5data['mask'][i][:,:,0]
                ray_origin = h5data['ray'][i,:,:3]
                ray_dir = h5data['ray'][i,:,3:6]
                valid = screen_pixel[:,0] != 0
            else:
                ray_origin, ray_dir, target, valid, mask = synthetic_data(syn_scene, self.resy, self.resx, K_inverse, R_inverse)
            mask = process_mask(mask)

            target          = torch.tensor(target       , dtype = Float).pin_memory()
            valid           = torch.tensor(valid        , dtype = bool ).pin_memory()
            mask            = torch.tensor(mask         , dtype = Float).pin_memory()
            ray_origin      = torch.tensor(ray_origin   , dtype = Float).pin_memory()
            ray_dir         = torch.tensor(ray_dir      , dtype = Float).pin_memory()
            R_inverse       = torch.tensor(R_inverse    , dtype = Float).pin_memory()
            K_inverse       = torch.tensor(K_inverse    , dtype = Float).pin_memory()
            R               = torch.tensor(R            , dtype = Float).pin_memory()
            K               = torch.tensor(K            , dtype = Float).pin_memory()

            camera_M = (R, K, R_inverse, K_inverse)
            self.Views.append((target, valid, mask, ray_origin, ray_dir, camera_M))


class Data_Redmi(Data):
    def __init__(self, HyperParams, syn_scene=None):
        self.resy=1080
        self.resx=1920
        self.num_view = HyperParams['num_view']
        self.name = HyperParams['name']
        if self.name == 'cup': 
            self.resy = 1200
            print('cup')
        h5data = h5py.File(f'/root/workspace/data/{self.name}.h5','r')
        self.Views = []

        for i in trange(72):
            R = h5data['cam_proj'][i]
            K = h5data['cam_k'][:]
            R_inverse = np.linalg.inv(R)
            K_inverse = np.linalg.inv(K)

            if not HyperParams['synthetic']:
                screen_pixel = h5data['cleaned_position'][i,:].reshape([-1,3])
                target = screen_pixel
                mask = h5data['mask'][i]
                valid = screen_pixel[:,0] != 0
                ray_origin, ray_dir = generate_ray(self.resy, self.resx, K_inverse, R_inverse)
            else:
                ray_origin, ray_dir, target, valid, mask = synthetic_data(syn_scene, self.resy, self.resx, K_inverse, R_inverse)

            mask = process_mask(mask)

            target          = torch.tensor(target       , dtype = Float).pin_memory()
            valid           = torch.tensor(valid        , dtype = bool ).pin_memory()
            mask            = torch.tensor(mask         , dtype = Float).pin_memory()
            ray_origin      = torch.tensor(ray_origin   , dtype = Float).pin_memory()
            ray_dir         = torch.tensor(ray_dir      , dtype = Float).pin_memory()
            R_inverse       = torch.tensor(R_inverse    , dtype = Float).pin_memory()
            K_inverse       = torch.tensor(K_inverse    , dtype = Float).pin_memory()
            R               = torch.tensor(R            , dtype = Float).pin_memory()
            K               = torch.tensor(K            , dtype = Float).pin_memory()

            camera_M = (R, K, R_inverse, K_inverse)
            self.Views.append((target, valid, mask, ray_origin, ray_dir, camera_M))


