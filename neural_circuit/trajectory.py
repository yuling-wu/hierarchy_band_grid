import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import *
from funcs import * 
import pandas as pd 
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt 

ratinabox.stylize_plots()
ratinabox.autosave_plots = False
ratinabox.figure_directory = 'figures'
#A circular environment made from many small walls
width = 5 #0.75
height = 5 #0.75
aspect = width/height
Env = Environment(params={"aspect": aspect, "scale": height})

# loc_land=bm.array([[1/4, 1/4],[1/4, 3/4],[3/4, 1/4],[3/4, 3/4]])
# fig,ax = Env.plot_environment()
# for i in range(4):
#     ax.plot(loc_land[i,0],loc_land[i,1],'ro')

# plt.savefig('Square_env.png')
# np.random.seed(0)

# 3 Add Agent.
agent = Agent(Env,params = {
            "speed_mean":0.04,
            "speed_std": 0.016,
            })
agent.pos = np.array([2.5, 2.5])
dt = 0.05
agent.dt = dt
T = 1000 * dt

for i in tqdm(range(int(T / dt))):
    agent.update(dt=dt)

positions, times, velocities = agent.history['pos'], np.arange(0, T, dt), agent.history['vel']
positions = np.array(positions)
Position = np.array(positions)

# Velocity = np.array(agent.history['vel']) * agent.dt
Velocity = np.diff(Position,axis=0)

# 构造零向量并插入第一行
zero_row = Velocity[0] * 1e-5
Velocity = np.vstack((zero_row, Velocity))
Velocity = Velocity/dt


speed = np.linalg.norm(Velocity, axis=1)
HD_angle = np.where(speed==0, 0, np.angle(Velocity[:,0]+ Velocity[:,1]*1j))
rot_vel = np.zeros_like(HD_angle)
rot_vel[1:] = map2pi(np.diff(HD_angle))

# filename = 'trajectory_train.npz'
filename = 'trajectory_int.npz'
# filename = 'trajectory_PC_int.npz'

np.savez(filename, position=Position, velocity=Velocity, speed=speed, hd_angle=HD_angle, rot_vel=rot_vel)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
fig, ax = agent.plot_trajectory(t_start=0, t_end=T, fig=fig, ax=ax,color="changing")
plt.savefig('trajectory_int.png')
print(np.max(Position, axis=0), np.min(Position, axis=0))