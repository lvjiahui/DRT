import trimesh
import trimesh.transformations as TF
from trimesh.proximity import closest_point
import torch
import numpy as np
import random
from PIL import Image
import cv2
import Render
Render.extIOR, Render.intIOR = 1.15, 1.0
Target = Render.Scene("/root/workspace/data/kitten.obj")
Target.set_camera(fov=(60,60), distance = 1.3, center=(0,0.0,0), angles=None)
origin, ray_dir = Target.generate_ray()
mask = Target.mask(origin, ray_dir)
Render.save_torch("kitten_mask.png",mask)