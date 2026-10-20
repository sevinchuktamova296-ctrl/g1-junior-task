# Part B — Walking in a circle

Pretrained policy: `deploy/pre_train/g1/motion.pt` from Unitree RL Gym, run in MuJoCo.
`record_circle.py` is a copy of `deploy/deploy_mujoco/deploy_mujoco.py` where the live viewer is replaced by offscreen rendering (so it runs inside a VM) and the `legged_gym` import is replaced by a path. The policy and control code is unchanged.

## What I changed
In `configs/g1.yaml`: `cmd_init: [0.5, 0, 0]` changed to `cmd_init: [0.4, 0.0, 0.4]` (forward speed vx = 0.4 m/s, sideways speed vy = 0, turning rate wz = 0.4 rad/s).

## Why this gives a circle
The policy was trained to follow a velocity command. If the robot moves forward at a constant speed v while turning at a constant rate ω, its path is a circle of radius r = v / ω = 0.4 / 0.4 = 1 m. Measured in my simulation: <mean radius: 1.00 m (min 0.74, max 1.31)>.

## 1. What inputs does the policy receive, and where do they come from?
A 47-number observation, rebuilt every control step (50 Hz):
- base angular velocity × 0.25 (3 values), from MuJoCo `qvel[3:6]`
- gravity direction in the body frame (3), computed from the base quaternion in `qpos[3:7]`
- velocity command × cmd_scale (3), from `cmd_init` in the YAML (the value I changed)
- joint positions minus default angles (12), from `qpos[7:]`
- joint velocities × 0.05 (12), from `qvel[6:]`
- previous action (12), stored from the last policy call
- gait phase as sin and cos (2), computed from simulation time with a 0.8 s period

## 2. What does the policy output, and how does it become joint torque?
It outputs 12 numbers, one per leg joint. Each is converted to a target joint angle: `target = action × 0.25 + default_angle`. A PD controller turns that into torque: `tau = kp × (target − q) + kd × (0 − dq)`, using the `kps` and `kds` from the YAML. The torques are applied to the joints every physics step (500 Hz), and the policy updates the targets every 10 steps (50 Hz).