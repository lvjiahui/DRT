import meshlabxml as mlx
import os
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from ray.tune.schedulers import FIFOScheduler

Redmi_models = ['tiger','pig','horse','rabbit_new']
cam_models = ['hand', 'mouse', 'dog', 'monkey']
syn_models = ['bunny', 'kitten']



# logpath = '/root/workspace/DR/result/ablation_sm'
logpath = '/root/workspace/show/paper'
os.makedirs(logpath, exist_ok=True)

def optimize(config):
    name = config['name']
    if name in Redmi_models:
        import optim_Redmi as optim
    elif name in cam_models:
        import optim
    elif name in syn_models:
        import optim_syn as optim
    else: assert False

    # view_range = config['view_range']
    # dilate = config['dilate']
    # sm_w = config['sm_w']
    track_name = track.trial_name()
    scene = optim.optimize(config, output=False, track=track)
    # _ = scene.mesh.export(f"/root/workspace/DR/result/hyper/{name}_mask_{dilate}.ply")    
    # _ = scene.mesh.export(f"{logpath}/{track_name}.ply")    
    # _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_ave/{sm_w}_{track_name}.ply")    


# models = ['hand', 'mouse', 'dog', 'monkey','tiger','pig','horse','rabbit_new']
# models = ['mouse']
models = ['kitten','bunny']
# models = ['horse']
# models = []

search_space = { 
    'name': tune.grid_search(models),
    'IOR' : tune.grid_search([1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
    # 'IOR' : 1.4723,
    'Pass' : 8,
    'Iters' : tune.grid_search([500]),
    # 'Pass' : 20,
    # 'Iters' : tune.grid_search([200]),
    "ray_w" : tune.grid_search([40]),
    "var_w" : 0,
    "sm_w": tune.grid_search([0.08]),
    # "sm_w": tune.grid_search([200, 1000,  2000, 5000]), #fairing
    "vh_w": tune.grid_search([2e-3]),
    # "vh_w": tune.grid_search([2e-5]),
    "sm_method": "dihedral",
    "optimizer": "sgd",
    "momentum": tune.grid_search([0.95]),

    # "start_lr": tune.grid_search([0.1]),
    "start_lr": tune.grid_search([0.0008]),
    "lr_decay":  tune.grid_search([0.1]),
    "taubin" : 0,
    # "start_len": tune.grid_search([10]),
    # "end_len": tune.grid_search([1]),
    # "start_len": tune.grid_search([5]),
    # "end_len": tune.grid_search([0.5]),
    "is_paper": True,
    # "num_view": tune.grid_search([9,18,36,72])
    # "num_view": tune.grid_search([72]),
    "start_len": tune.grid_search([0.08]),
    "end_len": tune.grid_search([0.008]),
    # 'view_range' : tune.grid_search([17]),
    # 'dilate': tune.grid_search([1,2,3,4]),
                }


assert(os.system("cp /root/workspace/DR/hyper.py "+logpath) == 0)
assert(os.system("cp /root/workspace/DR/optim.py "+logpath) == 0)
assert(os.system("cp /root/workspace/DR/Render_opencv.py "+logpath) == 0)

analysis = tune.run(
    optimize,
    config=search_space,
    # scheduler=ASHAScheduler(metric="mean_hausd", mode="min"),
    scheduler=FIFOScheduler(),
    resources_per_trial={
         "cpu": 1,
        #  "gpu": 0.5,
         "gpu": 1,
     },
     num_samples=1,
    #  verbose=1
    )

dataframe = analysis.dataframe()

with open(logpath+'/log','w') as log:
    log.write(dataframe.to_string())

with open(logpath+'/log.csv','w') as log:
    log.write(dataframe.to_csv())