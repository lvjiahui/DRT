# DRT
Implementation of [Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects.](https://vcc.tech/research/2020/DRT)

### [Project page](https://vcc.tech/research/2020/DRT)  |   [paper](https://vcc.tech/file/upload_file//image/research/att202009041252/DRT.pdf)  |   [video](https://vcc.tech/file/upload_file//image/research/att202009051343/%E7%8E%BB%E7%92%83%E9%87%8D%E5%BB%BA%E8%A7%86%E9%A2%9110.mp4)

![overview](./doc/lod.png)

We introduce a reconstruction method for solid transparent objects. Using a static background with a coded pattern, we establish a mapping between the camera view rays and locations on the background ([Environment Matting](http://grail.cs.washington.edu/projects/digital-matting/envmatte/)). Differentiable tracing of refractive ray paths is then used to directly optimize a 3D mesh approximation(visual hull) of the object, while simultaneously ensuring silhouette consistency and smoothness.

## Dependencies
Requirements: (tested on ubuntu 16.04)
- [NVIDIA-OptiX-SDK-6.5.0](https://developer.nvidia.com/designworks/optix/download) : Used to find the triangles intersected by each ray path.
- [Meshlab](https://github.com/cnr-isti-vclab/meshlab/releases): We use meshlabserver to remesh and calculate average per-vertex distance (Hausdorff Distance). Note that meshlab older than 2020.04 may not support explicit remesh operation.(Tested on MeshLabServer2020.04-linux.AppImage)
- Python with pytorch, numpy, trimesh, opencv-python, h5py, tqdm and imageio.

## Captured data
Our example data is captured by camera *point grey* or cellphone *Redmi* and released [here](https://vcc.tech/research/2020/DRT)

Scanned mesh(GT) and visual hull already contained in *./data/*

<img src="./doc/setup.jpg" width="500">

## Usage
- Download Optix-SDK, meshlabserver and our captured data.
- Set corresponding paths in `config.py`.
- Run `python optim.py`
- Reconstructed mesh will be saved in *./result/* by default.

## Citation
Please cite the paper in your publications if it helps your research:
```
@article{DRT,
title = {Differentiable Refraction-Tracing for Mesh Reconstruction of Transparent Objects},
author = {Jiahui Lyu and Bojian Wu and Dani Lischinski and Daniel Cohen-Or and Hui Huang},
journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH ASIA 2020)},
volume = {39},
number = {6},
pages = {},
year = {2020},
}
```
