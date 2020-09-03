# DRT
Implementation of [Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects.](https://vcc.tech/research.html)(TODO : update link)

![overview](./doc/lod.png)

## Dependencies
Requirements:
- [NVIDIA-OptiX-SDK-6.5.0](https://developer.nvidia.com/designworks/optix/download) : Used to find the triangles intersected by each ray path.
- [Meshlab](https://www.meshlab.net/): We use meshlabserver to remesh and calculate average per-vertex distance (Hausdorff Distance). Note that meshlab older than 2020.03 may not support remesh operation.
- Python with pytorch, numpy, trimesh, pycuda and imageio.

## Captured data
Our example data is captured by graypoint camera or cellphone Redmi and released [here]().

<img src="./doc/setup.jpg" width="500">

## Usage
- Download our captured data, Optix-SDK and meshlabserver.
- Set corresponding paths in `config.py`.
- run `python optim.py`

### (To be updated)
