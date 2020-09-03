# DRT
Implementation of [Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects.](https://vcc.tech/research.html)(TODO : update link)

![overview](./doc/lod.png)

We introduce a reconstruction method for solid transparent objects. Using a static background with a coded pattern, we establish a mapping between the camera view rays and locations on the background ([Environment Matting](http://grail.cs.washington.edu/projects/digital-matting/envmatte/)). Differentiable tracing of refractive ray paths is then used to directly optimize a 3D mesh approximation(visual hull) of the object, while simultaneously ensuring silhouette consistency and smoothness.

## Dependencies
Requirements:
- [NVIDIA-OptiX-SDK-6.5.0](https://developer.nvidia.com/designworks/optix/download) : Used to find the triangles intersected by each ray path.
- [Meshlab](https://www.meshlab.net/): We use meshlabserver to remesh and calculate average per-vertex distance (Hausdorff Distance). Note that meshlab older than 2020.03 may not support remesh operation.
- Python with pytorch, numpy, trimesh, opencv-python, h5py, tqdm and imageio.

## Captured data
Our example data is captured by graypoint camera or cellphone Redmi and released [here]().

<img src="./doc/setup.jpg" width="500">

## Usage
- Download our captured data, Optix-SDK and meshlabserver.
- Set corresponding paths in `config.py`.
- run `python optim.py`

### (To be updated)
