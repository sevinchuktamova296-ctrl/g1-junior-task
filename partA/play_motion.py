import numpy as np
import mujoco
import imageio

# Use the scene file so there is a floor in the video
XML = "/home/sevinch/internship/unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml"
CSV = "motion.csv"
OUT = "motion.mp4"
FPS = 30

print("Loading MuJoCo model...")
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

# Optional: Print joint names to understand order
for i in range(model.njnt):
    print(f"Joint {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")

print("Reading motion CSV file...")
motion = np.loadtxt(CSV, delimiter=",")
print("Motion shape:", motion.shape, "| qpos size:", model.nq)
assert motion.shape[1] == model.nq, "CSV column count must equal model.nq"

# Setup renderer and camera framing
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 150

frames = []
print(f"Rendering {len(motion)} frames of motion...")

for row in motion:
    qpos = row.copy()
    # Reorder quaternion from (x, y, z, w) to MuJoCo's (w, x, y, z) format
    qpos[3:7] = [row[6], row[3], row[4], row[5]]
    
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    # Center camera on the robot's pelvis
    cam.lookat[:] = data.qpos[:3]
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render())

print(f"Saving video to {OUT}...")
imageio.mimsave(OUT, frames, fps=FPS)
print("Done! Video successfully created.")