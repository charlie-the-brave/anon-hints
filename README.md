
# HINTs RL Framework

[![Docker](https://img.shields.io/badge/Docker-Setup-blue?logo=docker)](https://hub.docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-Ready-green?logo=nvidia)](https://developer.nvidia.com/cuda-downloads)

This project implements HINTs RL framework with hint-based policies for continuous control environments. 

To ensure fast and stable setup, we highly **recommend running in a docker container**. We've provided a Dockerfile that will setup an environment that satisfies dependencies including: cuda, hydra, mujoco-py, gym, etc. There are also tight restrictions on certain python package versions (e.g., numpy).

> **Known Issue:** Discrete environments have an outstanding bug (fix pending).


## Guide

- [Quick Start](#quick-start)
- [Configurations](#supported-configurations)
- [Generator Training](#generator-training)
- [Agent Training](#agent-training)
- [Agent Evaluation](#agent-evaluation)
- [Experiments](#experiments)
- [Troubleshooting](#troubleshooting)


## Quick Start

Setup environment in Docker container and run test experiments

```bash
# First, update the base image to ensure the CUDA version 
# matches your GPU. Verify with `nvcc --version` or `nvidia-smi`.
vim Dockerfile
# Build an image
docker build --no-cache -t hints:latest ./
# Create a container
docker create -it --name train hints:latest
# Spin up the container (on GPU)
docker run --rm --runtime=nvidia -it --gpus all hints:latest
```

> **Note:** If this step fails, there is likely a CUDA version mismatch between the base image and your host machine.

```bash
# Configure experiment parameters (Hydra)
# Adjust parameters (e.g., GPU memory constraints)
vim config/config.yaml
```

> See the [Hydra Submitit Launcher docs](https://hydra.cc/docs/plugins/submitit_launcher/#usage) for details.

Do a quick test sweep 

```bash
# (default policy)
python3 train.py -m \
  envs=pendulum,double,acrobot,cartpole,ant,cheetah,humanoid \
  run_name=anon-hints-v0 \
  state_type=observation \
  policy_type=default \
  max_trials=1 \
  z_info=none \
  horizon=10 \
  num_steps=10

# Copy results from container
docker cp <container_name>:<container_results_dir> <local_results_dir>
```


## Supported Configurations 

In train/eval scripts, environments are specified as `envs=<env_handle>` and cue types as `z_info=<cue_type>`

| Environment handle | Cue type | Note |
|---|---|---|
| `pendulum` | height ; angle ; heigt-angle | |
| `double` | act ; height ; angle-first ; angle-joint ; angle | `num_steps` |
| `acrobot` | act ; height ; angle-first ; angle-joint ; angle | Bug pending |
| `cartpole` | act ; height ; angle ; heigt-angle | Bug pending |
| `ant` | act ; height ; speed ; distance | increase `num_steps` |
| `cheetah` | rew ; height ; speed ; max ; dist | increase `num_steps` |
| `humanoid` | speed ; height ; distance | increase `num_steps` |


## Generator Training

Trains generator to predict cue type. Note the absolute path to the checkpoint for later use. Outputs are saved to `results/`. The size of the random-agent ensemble is inversely proportional to `generator.policy_change_interval`.

`z_info` takes the cue type (see options in [configurations](#supported-configurations))

```bash
python3 agent/hint_generator.py \
  envs=pendulum \
  z_info=angle \
  z_source=predicted-resnet \
  policy_type=random \
  generator.policy_change_interval=64 \
  generator.horizon=1024 &

# Plot predictions
python3 plot_predicted.py \
  --results_dir=results \
  --trials=1 \
  --run_name=predicted-resnet_angle \
  --single_run \
  --seek=1000 \
  --window=300
```


## Agent Training


Trains agents on tasks using only image observations or conditioned on cue types. (optionally) Requires `generator checkpoint` from  [gen training](#generator-training). Results are saved to `res/`; includes annotated **raw** policy inputs

```bash
# Default policy
python3 train.py \
  envs=pendulum \
  num_steps=10 \
  run_name=test-run \
  policy_type=default &

# Conditional policy (with generator checkpoint)
python3 train.py \
  envs=pendulum \
  num_steps=10 \
  test_interval=2 \
  run_name=test-run \
  policy_type=conditional \
  generator_checkpoint_path=<absolute_path>/anon-hints_best.pth \
  z_info=height-angle \
  z_source=predicted-resnet &

## Ground-truth conditional (multi-run sweep)
python3 train.py -m \
  envs=pendulum \
  num_steps=10 \
  run_name=test-hints \
  policy_type=conditional \
  z_info=none,height,angle,height-angle &
```

### Plot results

Ways of viewing results from different types of results (single exp, multi-run exp, generator). Use `--single_run` for results generated **without** the `-m` flag (multi-run)

```bash
# Single run
python3 plot.py --run_name test-run --trials 5 --single_run

# Multi-run
python3 plot.py --run_name test-hints --trials 5

# Plot predictions
python3 plot_predicted.py \
  --results_dir=./res \
  --run_name=test-hints \
  --trials=5 \
  --outdir=./res/plots/test \
  --seek=1000 \
  --window=200
```


## Agent Evaluation

Evaluates agents in 50 randomly initialised environments. Requires `agent checkpoint` from  [agent training](#agent-training). Results are saved to `eval_<run_name>/`; includes annotated **enlarged** policy inputs

```bash
python3 evaluate.py \
  run_name=test-run1 \
  max_trials=50 \
  checkpoint_path=<absolute_path_to_agent_checkpoint>/
```


## Experiments

> **Tip:** Use `envs=env_config.yaml` to load the correct parameters instead of command-line overrides.

```bash
# Running a single experiment
python3 train.py envs=cartpole num_steps=10 &

# Sweep over parameters (envs, z_info, etc.)
python3 train.py -m envs=cartpole,pendulum,ant num_steps=100 &
```

### Plotting

Use `--single_run` for results generated **without** the `-m` flag (multi-run)

```bash
# Evaluation only (multiple runs)
python3 plot.py --run_name test-run1,test-run2,test-runk --trials 50 --eval_only

# Generator predictions 
python3 plot_predicted.py \
  --results_dir=./res \
  --run_name=test-hints \
  --trials=5 \
  --outdir=./res/plots/test \
  --seek=1000 \
  --window=200
```


## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `docker run` fails on GPU | CUDA version mismatch | Verify base image CUDA matches host (`nvidia-smi`) |
| Import / dependency errors | Incorrect package versions | Use the provided Docker image |
| Out-of-memory errors | GPU memory constraints | Adjust settings in `config/config.yaml` |


