# Isaac_Lab_Roboparty
conda env
``` 
conda activate env_isaaclab_course
``` 


train 
``` 
python robolab/scripts/rsl_rl/train.py   --task BigReddog-Parkour   --headless   --num_envs=4096 --logger=tensorboard

``` 


play example 
``` 
python robolab/scripts/rsl_rl/play_parkour.py   --task=BigReddog-Parkour-Play   --num_envs=1   --checkpoint=/home/csl/Isaac_Lab_Roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour/2026-07-28_17-04-27/model_29999.pt
```cd 


tensorboard 
``` 
tensorboard --logdir logs/rsl_rl/bigreddog_parkour_amp --port 6006

``` 

sim2sim parkour_quadruped
``` 
 python robolab/scripts/mujoco/sim2sim_bigreddog_parkour.py \
 --depth_encoder /home/csl/parkour_quadruped_by_roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour_amp/2026-10-01_17-20-28_amp_flat/exported/0-depth_encoder.onnx \
 --actor /home/csl/parkour_quadruped_by_roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour_amp/2026-10-01_17-20-28_amp_flat/exported/actor.onnx \
 --scene stairs
``` 
![Robot demonstration](sim2sim.png)

/home/csl/Isaac_Lab_Roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour_amp/2026-08-17_10-17-18_amp_parkour_acc/exported/actor.onnx

/home/csl/Isaac_Lab_Roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour_amp/2026-08-17_10-17-18_amp_parkour_acc/exported/0-depth_encoder.onnx



amp+parkour


train
``` 
python robolab/scripts/rsl_rl/train.py \
  --task BigReddog-Parkour-AMP0 \
  --num_envs 1024 \
  --max_iterations 60000 \
  --run_name amp_flat\
  --logger tensorboard \
  --headless
``` 

test
```
python robolab/scripts/rsl_rl/train.py \
  --task BigReddog-Parkour-AMP0 \
  --num_envs 1 \
  --max_iterations 30000 \
  --run_name amp_flat \
  --logger tensorboard 
```



play

``` 

PYTHONPATH=/home/csl/parkour_quadruped_by_roboparty/roboparty_train/robolab 

python robolab/scripts/rsl_rl/play_parkour.py \
  --task BigReddog-Parkour-AMP-Play0 \
  --num_envs 1 \
  --checkpoint /home/csl/parkour_quadruped_by_roboparty/roboparty_train/logs/rsl_rl/bigreddog_parkour_amp/2026-10-01_17-20-28_amp_flat/model_59999.pt \
  --exportonnx
  ``` 

