import meshlabxml as mlx
import os
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler

def mean_hausd(mesh_path):
    pid = os.getpid()
    logpath = f"/dev/shm/DR/hausd_log_{pid}"
    os.system('rm '+logpath)
    cmd = "DISPLAY=:1 meshlabserver -i /root/workspace/data/hand_gt_align.ply " + mesh_path + " -s /root/workspace/data/real_hand_hausd.mlx -l " + logpath
    assert (os.system(cmd)==0)
    dist = mlx.compute.parse_hausdorff(logpath)['mean_distance']
    return dist

def optimize(config):
    import coarse
    name = 'hand'
    scene = coarse.optimize(name, config, output=False, track=track)
    pid = os.getpid()
    _ = scene.mesh.export(f"/root/workspace/DR/result/{name}/{str(pid)}.ply")    


search_space = {
    "optimizer": tune.grid_search(["sgd"]),
    "momentum": tune.grid_search([0.5, 0.8, 0.9, 0.95]),
    "start_lr": tune.grid_search([0.1,0.2, 0.4]),
    "lr_decay": tune.grid_search([0.1, 0.5, 1]),
    "taubin": tune.grid_search([0]),
    # "IOR": tune.grid_search([1.3, 1.4, 1.4723]),
    "IOR": tune.grid_search([1.45]),
    "Pass": tune.grid_search([8]),
    "ray_w" : tune.grid_search([50, 100, 200]),
    "var_w" : tune.grid_search([0, 4, 40]),
    "sm_method": tune.grid_search(["laplac", "dihedral"]),
    "sm_w": tune.grid_search([2, 20]),
    "vh_w": tune.grid_search([0.03, 0.1]),
    "start_len": tune.grid_search([3, 5]),
    "end_len": tune.grid_search([1]),
}
analysis = tune.run(
    optimize,
    config=search_space,
    scheduler=ASHAScheduler(metric="mean_hausd", mode="min"),
    resources_per_trial={
         "cpu": 1,
         "gpu": 0.5,
     }
    )
