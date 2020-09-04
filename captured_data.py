import torch
import numpy as np
import cv2
import h5py
from tqdm import trange
import imageio
import config

Float = torch.float64
device='cuda'

def process_mask(M):

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


        index = list(np.arange(0, 72, 72//self.num_view))

        # mouse debug
        if self.name == 'mouse':
            index = list(np.arange(-5, 10))
            index = index + list(np.arange(22,40))

        print('num_view ray', len(index))

        while True:
            np.random.shuffle(index)
            for i in index: yield i % 72

    def silh_view_generator(self):
        index = list(np.arange(72))
        print('num_view silh', len(index))
        while True:
            np.random.shuffle(index)
            for i in index: yield i % 72


class Data_Pointgray(Data):
    '''
    data captured by camera pointgray
    '''
    def __init__(self, HyperParams):
        self.resy=960
        self.resx=1280
        self.num_view = HyperParams['num_view']
        self.name = HyperParams['name']
        h5data = h5py.File(f'{config.data_path}{self.name}.h5','r')

        self.Views = []
        print('loading data..............')
        for i in trange(72):
            R = h5data['cam_proj'][i]
            K = h5data['cam_k'][:]
            R_inverse = np.linalg.inv(R)
            K_inverse = np.linalg.inv(K)
            screen_pixel = h5data['screen_position'][i]
            target = screen_pixel
            mask = h5data['mask'][i]
            valid = screen_pixel[:,0] != 0
            ray_origin = h5data['ray_origin'][i]
            ray_dir = h5data['ray_dir'][i]

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
        h5data.close()

class Data_Redmi(Data):
    '''
    data captured by my cellphone Redmi
    '''
    def __init__(self, HyperParams):
        self.resy=1080
        self.resx=1920
        self.num_view = HyperParams['num_view']
        self.name = HyperParams['name']

        h5data = h5py.File(f'{config.data_path}{self.name}.h5','r')
        self.Views = []

        print('loading data..............')
        for i in trange(72):
            R = h5data['cam_proj'][i]
            K = h5data['cam_k'][:]
            R_inverse = np.linalg.inv(R)
            K_inverse = np.linalg.inv(K)
            screen_pixel = h5data['screen_position'][i].reshape([-1,3])
            target = screen_pixel
            mask = h5data['mask'][i]
            valid = screen_pixel[:,0] != 0
            ray_origin, ray_dir = generate_ray(self.resy, self.resx, K_inverse, R_inverse)

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
        h5data.close()


