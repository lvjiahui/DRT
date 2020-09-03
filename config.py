
# path to your optix libary
optix_include = "/root/workspace/docker/build/DR/NVIDIA-OptiX-SDK-6.5.0-linux64/include"
optix_ld = "/root/workspace/docker/build/DR/NVIDIA-OptiX-SDK-6.5.0-linux64/lib64"

# cmd to call meshlabserver
meshlabserver_cmd = "ssh jiahui@172.31.224.138 DISPLAY=:1 DR/MeshLabServer2020.04-linux.AppImage"
# exchange temporary mesh file with meshlabserver
tmp_path = "/dev/shm/DR/"

# path to hdf5 file and visual hull mesh
data_path = "./data/"
result_path = "./result/"

HyperParams = {
    'name' :  'hand',
    'IOR' : 1.4723,
    'Pass' : 20,
    'Iters' : 200,
    "ray_w" : 40,
    "sm_w": 0.08,
    # "sm_w": 0.02,
    "vh_w": 2e-3,

    "momentum": 0.95,
    "start_lr": 0.1,
    "start_len": 10,
    "end_len": 1,
    "lr_decay": 0.5,
    'num_view': 72,
                }