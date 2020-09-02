import meshlabxml as mlx
import os
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from ray.tune.schedulers import FIFOScheduler
import optim


# logpath = '/root/workspace/DR/result/ablation_sm'
# logpath = '/root/workspace/show/paper'
# os.makedirs(logpath, exist_ok=True)

def optimize(config):
    name = config['name']
    view = config['num_view']

    # path = f"/root/workspace/show/paper/validate/view/{name}/{view}"
    # os.makedirs(path, exist_ok=True)

    # track_name = track.trial_name()
    scene = optim.optimize(config, output=False, track=track)
    
    _ = scene.mesh.export(f"/root/workspace/DR/result/synthetic/{name}_{view}.ply")
    # _ = scene.mesh.export(path+"/result.obj")



# models = ['hand', 'mouse', 'dog', 'monkey','tiger','pig','horse','rabbit_new']
models = ['mouse', 'dog', 'monkey', 'hand']


search_space = { 
    'synthetic' :  True,
    'name': tune.grid_search(models),
    # 'IOR' : tune.grid_search([1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
    'IOR' : 1.4723,
    # 'Pass' : 8,
    # 'Iters' : tune.grid_search([500]),
    'Pass' : 20,
    'Iters' : tune.grid_search([200]),
    "ray_w" : tune.grid_search([40]),
    "var_w" : 0,
    "sm_w": tune.grid_search([0.08]),
    "vh_w": tune.grid_search([2e-3]),
    "sm_method": "dihedral",
    "optimizer": "sgd",
    "momentum": tune.grid_search([0.95]),

    "start_lr": tune.grid_search([0.1]),
    # "start_lr": tune.grid_search([0.0008]),
    "lr_decay":  tune.grid_search([0.5]),
    "taubin" : 0,
    "start_len": tune.grid_search([10]),
    "end_len": tune.grid_search([1]),
    # "start_len": tune.grid_search([5]),
    # "end_len": tune.grid_search([0.5]),
    # "is_paper": True,
    # "num_view": tune.grid_search([9,18,36,72])
    "num_view": tune.grid_search([4, 9, 18, 36, 72]),
    # "start_len": tune.grid_search([0.08]),
    # "end_len": tune.grid_search([0.008]),
                }


# assert(os.system("cp /root/workspace/DR/hyper.py "+logpath) == 0)
# assert(os.system("cp /root/workspace/DR/optim.py "+logpath) == 0)
# assert(os.system("cp /root/workspace/DR/Render_opencv.py "+logpath) == 0)

analysis = tune.run(
    optimize,
    config=search_space,
    # scheduler=ASHAScheduler(metric="mean_hausd", mode="min"),
    scheduler=FIFOScheduler(),
    resources_per_trial={
         "cpu": 1,
         "gpu": 1,
     },
     num_samples=1,
    #  verbose=1
    )

# dataframe = analysis.dataframe()

# with open(logpath+'/log','w') as log:
#     log.write(dataframe.to_string())

# with open(logpath+'/log.csv','w') as log:
#     log.write(dataframe.to_csv())