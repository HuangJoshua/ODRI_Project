#%%
#import mujoco
import mujoco
#%% Import Packages for Plotting and Creating Graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

#Graphics and Plotting 
print('Installing mediapy:')
!command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
!pip install -q mediapy
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#%% Load A Simple Model
xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)

# %% mJData
data = mujoco.MjData(model)

# %% make a Renderer and show pixels
renderer = mujoco.Renderer(model)
media.show_image(renderer.render())
#this render makes a black square since it has not entered the pipeline to render

#%% Render for real
mujoco.mj_forward(model, data)
renderer.update_scene(data)
media.show_image(renderer.render())

# %%
