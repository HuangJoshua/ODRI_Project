#%% Imports

# general
import mujoco as mj
from mujoco import viewer
import numpy as np
from dm_control import mjcf

# %%
# launching mujoco from python script
viewer.launch()
viewer.launch(model)
viewer.launch(model, data)

#%%
# example

xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mj.MjModel.from_xml_string(xml)
viewer.launch(model)

# %% PYMJCF TUTORIAL
# Class describes an abstracted articulated leg
class Leg(object):
    "2 dof leg with position actuators"

    def __init__(self, length, rgba):
        self.model = mjcf.RootElement()

        # default inits
        self.model.default.joint.damping = 2
        self.model.default.joint.type = "hinge"
        self.model.default.geom.type = "capsule"
        self.model.default.geom.rgba = rgba

        # thigh inits
        self.thigh = self.model.worldbody.add("body")
        self.hip = self.thigh.add("joint", axis=[0, 0, 1])
        self.thigh.add("geom", fromto=[0, 0, 0, length, 0, 0], size=[length / 4])

        # hip inits
        self.shin = self.thigh.add("body", pos=[length, 0, 0])
        self.knee = self.shin.add("joint", axis=[0, 1, 0])
        self.shin.add("geom", fromto=[0, 0, 0, 0, 0, -length], size=[length / 5])

        # position actuators init
        self.model.actuator.add("position", joint=self.hip, kp=10)
        self.model.actuator.add("position", joint=self.knee, kp=10)


# Build Creature Procedurally using MJCF Syntax
BODY_RADIUS = 0.1
BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)
random_state = np.random.RandomState(42)


def make_creature(num_legs):
    """Constructs a creature with 'num legs' legs."""
    rgba = random_state.uniform([0, 0, 0, 1], [1, 1, 1, 1])
    model = mjcf.RootElement()
    model.compiler.angle = "radian"  # use radians

    # Make the torso geometry
    model.worldbody.add(
        "geom", name="torso", type="ellipsoid", size=BODY_SIZE, rgba=rgba
    )

    # attach legs to equidistant sites on the circumference.
    for i in range(num_legs):
        theta = 2 * i * np.pi / num_legs
        hip_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
        hip_site = model.worldbody.add("site", pos=hip_pos, euler=[0, 0, theta])
        leg = Leg(length=BODY_RADIUS, rgba=rgba)
        hip_site.attach(leg.model)

    return model


# Spawn 6 creatures on floor with two lights and place on grid
arena = mjcf.RootElement()
chequered = arena.asset.add(
    "texture",
    type="2d",
    builtin="checker",
    width=300,
    height=300,
    rgb1=[0.2, 0.3, 0.4],
    rgb2=[0.3, 0.4, 0.5],
)
grid = arena.asset.add(
    "material", name="grid", texture=chequered, texrepeat=[5, 5], reflectance=0.2
)
arena.worldbody.add("geom", type="plane", size=[2, 2, 0.1], material=grid)

for x in [-2, 2]:
    arena.worldbody.add("light", pos=[x, -1, 3], dir=[-x, 1, -2])

# instantiate 6 creatures with 3 to 8 legs
creatures = [make_creature(num_legs=num_legs) for num_legs in range(3, 9)]

# place them on a grid in the arena.
height = 0.15
grid = 5 * BODY_RADIUS
xpos, ypos, zpos = np.meshgrid([-grid, 0, grid], [0, grid], [height])
for i, model in enumerate(creatures):
    # place spawn sites on a grid.
    spawn_pos = (xpos.flat[i], ypos.flat[i], zpos.flat[i])
    spawn_site = arena.worldbody.add("site", pos=spawn_pos, group=3)
    # attach to the arena at the spawn sites, with a free joint.
    spawn_site.attach(model).add("freejoint")

# instantiate the physics and render.
physics = mjcf.Physics.from_mjcf_model(arena)
viewer.launch(physics)

# %%
