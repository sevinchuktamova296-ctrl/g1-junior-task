import os
import numpy as np
import mujoco
import torch
import yaml
import imageio

GYM = os.path.expanduser("~/internship/unitree_rl_gym")
CONFIG = os.path.join(GYM, "deploy/deploy_mujoco/configs/g1.yaml")
OUT = "circle.mp4"
DURATION = 20.0          # simulated seconds


def get_gravity_orientation(q):
    qw, qx, qy, qz = q
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


with open(CONFIG) as f:
    c = yaml.load(f, Loader=yaml.FullLoader)
policy_path = c["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", GYM)
xml_path = c["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", GYM)
dt = c["simulation_dt"]
decim = c["control_decimation"]
kps = np.array(c["kps"], dtype=np.float32)
kds = np.array(c["kds"], dtype=np.float32)
default_angles = np.array(c["default_angles"], dtype=np.float32)
ang_vel_scale, dof_pos_scale = c["ang_vel_scale"], c["dof_pos_scale"]
dof_vel_scale, action_scale = c["dof_vel_scale"], c["action_scale"]
cmd_scale = np.array(c["cmd_scale"], dtype=np.float32)
num_actions, num_obs = c["num_actions"], c["num_obs"]
cmd = np.array(c["cmd_init"], dtype=np.float32)
print("command (vx, vy, wz):", cmd)

action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()
obs = np.zeros(num_obs, dtype=np.float32)

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
m.opt.timestep = dt
policy = torch.jit.load(policy_path)

renderer = mujoco.Renderer(m, height=360, width=480)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 1.0, 0.6]      # circle center is about 1 m left of the start
cam.distance, cam.elevation, cam.azimuth = 5.0, -55, 90

frames, path = [], []
counter = 0
for _ in range(int(DURATION / dt)):
    tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    counter += 1

    if counter % decim == 0:
        qj = (d.qpos[7:] - default_angles) * dof_pos_scale
        dqj = d.qvel[6:] * dof_vel_scale
        gravity = get_gravity_orientation(d.qpos[3:7])
        omega = d.qvel[3:6] * ang_vel_scale

        phase = (counter * dt) % 0.8 / 0.8
        obs[:3] = omega
        obs[3:6] = gravity
        obs[6:9] = cmd * cmd_scale
        obs[9:9 + num_actions] = qj
        obs[9 + num_actions:9 + 2 * num_actions] = dqj
        obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
        obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        target_dof_pos = action * action_scale + default_angles

        path.append(d.qpos[:2].copy())
        if (counter // decim) % 4 == 0:          # 25 video frames per simulated second
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())

imageio.mimsave(OUT, frames, fps=12)
path = np.array(path)
tail = path[len(path) // 3:]
cx, cy = tail.mean(axis=0)
r = np.linalg.norm(tail - [cx, cy], axis=1)
print("saved", OUT, "frames:", len(frames))
print("final position:", path[-1])
print("estimated circle center: (%.2f, %.2f), mean radius: %.2f m (min %.2f, max %.2f)" % (cx, cy, r.mean(), r.min(), r.max()))