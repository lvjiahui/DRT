import meshlabxml as mlx
import os
from ray import tune
from ray.tune import track
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from ray.tune.schedulers import FIFOScheduler

logpath = '/root/workspace/DR/result/hyper/rabbit_new/'
os.makedirs(logpath, exist_ok=True)

def optimize(config):
    # import optim
    import optim_Redmi as optim
    name = config['name']
    view_range = config['view_range']
    sm_w = config['sm_w']
    track_name = track.trial_name()
    scene = optim.optimize(config, output=False, track=track)
    # _ = scene.mesh.export(f"/root/workspace/DR/result/hyper/{name}_{view_range}.ply")    
    _ = scene.mesh.export(f"{logpath}{track_name}.ply")    
    # _ = scene.mesh.export(f"/root/workspace/DR/result/{name}_ave/{sm_w}_{track_name}.ply")    


# models = ['rabbit', 'hand', 'mouse', 'dog', 'monkey']
models = ['rabbit_new']

search_space = { 
    'name': tune.grid_search(models),
    # 'IOR' : tune.grid_search([1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
    'IOR' : 1.4723,
    'Pass' : 8,
    'Iters' : tune.grid_search([500]),
    "ray_w" : tune.grid_search([100]),
    "var_w" : 0,
    "sm_w": tune.grid_search([10]),
    # "sm_w": tune.grid_search([200, 1000,  2000, 5000]), #fairing
    "vh_w": tune.grid_search([0.1]),
    "sm_method": "dihedral",
    "optimizer": "sgd",
    "momentum": tune.grid_search([0.95]),
    "start_lr": tune.grid_search([0.1]),
    "lr_decay":  tune.grid_search([0.5]),
    "taubin" : 0,
    "start_len": tune.grid_search([5, 10]),
    "end_len": tune.grid_search([1]),
    'view_range' : tune.grid_search([17,12,9,6]),
    # 'view_range' : tune.grid_search([12]),
                }


sm_space = { 
    'name': tune.grid_search(models),
    # 'IOR' : tune.grid_search([1.3, 1.4, 1.5, 1.6, 1.7, 1.8]),
    'IOR' : 1.4723,
    'Pass' : 2,
    'Iters' : tune.grid_search([500]),
    "ray_w" : tune.grid_search([0]),
    "var_w" : 0,
    "sm_w": tune.grid_search([10]),
    # "sm_w": tune.grid_search([200, 1000,  2000, 5000]), #fairing
    "vh_w": tune.grid_search([0.1]),
    "sm_method": "dihedral",
    "optimizer": "sgd",
    "momentum": tune.grid_search([0.95]),
    "start_lr": 0.1,
    "lr_decay":  tune.grid_search([1]),
    "taubin" : 0,
    "start_len": tune.grid_search([3,5,10]),
    "end_len": tune.sample_from(lambda spec: spec.config.start_len),
    # 'view_range' : tune.grid_search([17,12,9,6,3]),
    'view_range' : tune.grid_search([12]),
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
     verbose=1
    )

dataframe = analysis.dataframe()

with open(logpath+'log','w') as log:
    log.write(dataframe.to_string())

with open(logpath+'log.csv','w') as log:
    log.write(dataframe.to_csv())