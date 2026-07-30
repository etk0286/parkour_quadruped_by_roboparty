# Isaac_Lab_Roboparty
conda env
``` 
conda activate env_isaaclab
``` 


train 
``` 
python robolab/scripts/rsl_rl/train.py   --task BigReddog-Parkour   --headless   --num_envs=4096 --logger=tensorboard

``` 


play example 
``` 
python robolab/scripts/rsl_rl/play_parkour.py   --task=BigReddog-Parkour-Play   --num_envs=1   --checkpoint=/home/csl/Isaac_Lab_Roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour/2026-07-28_17-04-27/model_29999.pt
```


tensorboard 
``` 
tensorboard --logdir logs/rsl_rl/bigreddog_parkour --port 6006

``` 

sim2sim parkour_quadruped
``` 
 python robolab/scripts/mujoco/sim2sim_bigreddog_parkour.py --depth_encoder logs/rsl_rl/bigreddog_parkour/2026-07-28_17-04-27/exported/0-depth_encoder.onnx --actor /home/csl/Isaac_Lab_Roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour/2026-07-28_17-04-27/exported/actor.onnx --scene stairs
``` 
![Robot demonstration](sim2sim.png)
