# Getting Started

This guide assumes the environment, CARLA, the pretrained weights and the dataset are set up as in [install.md](install.md). Run every command from the repository root.

SafeDriveVLA is built in three steps:

1. **World-model pre-training**: a latent world model (frozen V-JEPA 2 encoder and an action-conditioned predictor) on the PDM-Lite driving frames.
2. **VLA training**: InternVL3-1B with LoRA learns to emit the driving mode and the action tokens, reading the world tokens that the frozen world model dreams under the action anchor of the instructed meta-command.
3. **Closed-loop evaluation** in CARLA on Bench2Drive, CARLA-F, B2D-C and B2D-Adv.

To evaluate the released checkpoints, skip to [Evaluation](#evaluation).

### Overview of Scripts

| Script | Purpose |
| :-- | :-- |
| `scripts/train_world_model.sh` | world-model pre-training on one node |
| `scripts/train_vla.sh` | VLA training on one node |
| `scripts/eval.sh` | closed-loop evaluation on one node, one CARLA server per GPU, followed by the metrics |
| `scripts/slurm/*.sbatch` | Slurm wrappers of the three scripts above |

## Evaluation

### Released Checkpoints

The checkpoints on [Hugging Face](https://huggingface.co/Danielxsy/safedrive-vla) are downloaded to `ckpts/safedrive-vla` in [install.md](install.md#pretrained-weights). All three models are trained on the full PDM-Lite data.

| Checkpoint | Mode token | World model | Training recipe |
| :-- | :-: | :-: | :-- |
| `ckpts/safedrive-vla/safedrive_vla/model.ckpt` | ✓ | ✓ | `safedrive_vla` |
| `ckpts/safedrive-vla/baseline_mode/model.ckpt` | ✓ | | `baseline_mode` |
| `ckpts/safedrive-vla/baseline/model.ckpt` | | | `baseline` |

`safedrive_vla` also reads the world model `ckpts/safedrive-vla/world_model/latest.pt` and the V-JEPA 2 encoder `ckpts/vjepa2/vitl.pt`, as set in its `config.yaml`.

### Run a Benchmark

```bash
export CARLA_ROOT=/path/to/carla0915
CKPT=ckpts/safedrive-vla/safedrive_vla/model.ckpt

bash scripts/eval.sh bench2drive ${CKPT}                         # target waypoints
bash scripts/eval.sh bench2drive ${CKPT} --nav-signal command    # route-planner commands
bash scripts/eval.sh carla_f ${CKPT}
bash scripts/eval.sh b2d_c ${CKPT}
bash scripts/eval.sh b2d_adv ${CKPT}
```

`scripts/eval.sh` starts one CARLA server and one leaderboard process per GPU and runs one route at a time on each, then prints the metrics of the benchmark. It uses all visible GPUs unless `--gpu-ids` is given.

| Option | Description |
| :-- | :-- |
| `--gpu-ids 0,1,...` | GPUs to use (default: all visible GPUs) |
| `--nav-signal NAME` | Bench2Drive only: `target_point` (default) or `command` |
| `--routes-dir DIR` | route files to evaluate (default: `benchmark/data/<benchmark>`), e.g. a subset for a quick check |
| `--output-dir DIR` | default: `work_dirs/eval/<benchmark>/<model>`, or `<model>_command` with `--nav-signal command` |
| `--save-viz` | save front-camera images with the prompt, the predicted trajectory and the decoded answer |

Relative paths are resolved against the repository root. CARLA uses the RPC ports 2000, 2100, ... (one block of 100 per GPU), so run one evaluation per node at a time.

The navigation signal depends on the benchmark:

| Benchmark | Agent | Navigation signal in the prompt | Meta-command of the world-model action anchor |
| :-- | :-- | :-- | :-- |
| Bench2Drive | `team_code/agent_safedrive.py` | two target waypoints, or the route-planner command | route-planner command |
| CARLA-F | `team_code/agent_safedrive_carla_f.py` | active language instruction | parsed from the instruction text |
| B2D-C, B2D-Adv | `team_code/agent_safedrive_b2d_c.py` | active language instruction | parsed from the instruction text |

On the language benchmarks, no route-planner command or target waypoint reaches the model or the world model.

### Outputs and Metrics

```
work_dirs/eval/<benchmark>/<model>
├── res/<route>_res.json     # leaderboard record of each route
├── viz/<route>/<time>/      # metric_info.json: per-tick ego state; images/ with --save-viz
├── out/, err/               # stdout / stderr of each route
└── metrics.json             # written by the metric script
```

Rerunning the same command resumes an interrupted evaluation: routes with a complete result are skipped, and routes that failed for infrastructure reasons (e.g., a CARLA crash) are retried. To recompute the metrics of an output directory:

```bash
python benchmark/metrics/bench2drive_metrics.py work_dirs/eval/b2d_c/safedrive_vla --routes-dir benchmark/data/b2d_c
python benchmark/metrics/carla_f_metrics.py work_dirs/eval/carla_f/safedrive_vla --routes-dir benchmark/data/carla_f
```

| Benchmark | Metrics |
| :-- | :-- |
| Bench2Drive, B2D-C, B2D-Adv | Driving Score, Success Rate, collisions, traffic violations, out-of-route, Efficiency, Comfortness |
| CARLA-F | Navigation Compliance Rate per meta-command and its average (NCR Avg), Speed Error (m/s) |

Driving Score and Success Rate are averaged over all route files, so a missing or crashed route counts as zero; the CARLA-F metrics skip routes without a complete result and print a warning. `bench2drive_metrics.py` reports Comfortness in percent.

### Slurm

```bash
sbatch -p <partition> scripts/slurm/eval.sbatch carla_f ckpts/safedrive-vla/safedrive_vla/model.ckpt
```

Activate the environment and export `CARLA_ROOT` before submitting; the options of `scripts/eval.sh` are passed through, and resubmitting resumes the evaluation.

### Evaluation Time

A route takes 10-40 minutes, so a full benchmark takes about half a day on 8 GPUs (RTX PRO 6000). Closed-loop results vary between runs of the same checkpoint, by up to tens of Driving-Score points on a single route; compare models on a full benchmark.

## Training

The training recipes are the configs in [`world_model/configs`](../world_model/configs) and [`safedrive_vla/configs/experiment`](../safedrive_vla/configs/experiment). The released models were trained on one node with 8 GPUs of 96 GB.

### 1. World-Model Pre-Training

```bash
bash scripts/train_world_model.sh world_model/configs/world_model.yaml
```

The checkpoint is written to `work_dirs/world_model/latest.pt` (plus `epoch_XXXX.pt` every 5 epochs), and rerunning the command resumes from `latest.pt`. The batch size in the config is per GPU; set `NUM_GPUS` to change the number of GPUs (default 8). The 20 epochs take about 12 hours on 8 GPUs.

| Config | Data | Used by |
| :-- | :-- | :-- |
| `world_model.yaml` | all routes | `safedrive_vla` (the released world model) |
| `world_model_x0.5.yaml` | half of the routes | `safedrive_vla_x0.2` |

To train only the VLA, use the released world model instead: add `model.world_model.predictor_ckpt=ckpts/safedrive-vla/world_model/latest.pt` to the command of the next step.

### 2. VLA Training

```bash
bash scripts/train_vla.sh safedrive_vla
```

| Experiment | Mode token | World model | Data |
| :-- | :-: | :-: | :-- |
| `safedrive_vla` | ✓ | ✓ (`work_dirs/world_model/latest.pt`) | full |
| `baseline_mode` | ✓ | | full |
| `baseline` | | | full |
| `safedrive_vla_x0.2` | ✓ | ✓ (`work_dirs/world_model_x0.5/latest.pt`) | 0.2 of the frames |

Each run writes to `work_dirs/safedrive_vla/<experiment>/<timestamp>/`: the resolved config in `.hydra/config.yaml`, one DeepSpeed checkpoint per epoch in `checkpoints/epoch=XXX.ckpt/` and CSV logs in `logs/`. Training runs for 14 epochs, which take roughly one day on 8 GPUs for the full data.

Any config value can be overridden on the command line (Hydra syntax):

```bash
# 4 GPUs, the same global batch size via gradient accumulation
NUM_GPUS=4 bash scripts/train_vla.sh safedrive_vla accumulate_grad_batches=2

# smaller per-GPU batch for GPUs with less memory, logging to Weights & Biases
bash scripts/train_vla.sh safedrive_vla data.batch_size=6 accumulate_grad_batches=2 wandb_project=safedrive-vla
```

With `RESUME=1`, training continues from the newest complete epoch checkpoint of the experiment:

```bash
RESUME=1 bash scripts/train_vla.sh safedrive_vla
```

The data module indexes the SimLingo scenario buckets of `buckets_paths.pkl` at start-up, which takes a few minutes on the full dataset. The driving-mode labels are computed on the fly from the driving frames and the action-dreaming instructions.

On Slurm, activate the environment and submit (resubmit with `RESUME=1` to continue):

```bash
sbatch -p <partition> scripts/slurm/train_world_model.sbatch world_model/configs/world_model.yaml
sbatch -p <partition> scripts/slurm/train_vla.sbatch safedrive_vla
```

### 3. Evaluate the Trained Model

Pass the epoch checkpoint directory to `scripts/eval.sh`; the agent reads the config of the run from its `.hydra/config.yaml`:

```bash
bash scripts/eval.sh carla_f work_dirs/safedrive_vla/safedrive_vla/<timestamp>/checkpoints/epoch=013.ckpt
```

The results go to `work_dirs/eval/carla_f/safedrive_vla`.

## Assets and Benchmark Routes

The action codebook (`safedrive_vla/assets/action_codebook.pkl`) and the meta-command action anchors (`safedrive_vla/assets/action_anchors.npz`) used in the paper are included. They can be rebuilt from the dataset:

```bash
python -m safedrive_vla.tools.build_codebook --data-root database/simlingo/data/simlingo \
    --output safedrive_vla/assets/action_codebook.pkl
python -m safedrive_vla.tools.build_action_anchors --data-root database/simlingo --data-scale 0.2 \
    --output safedrive_vla/assets/action_anchors.npz
```

The scripts that generate CARLA-F and B2D-C from the Bench2Drive routes are documented in [`benchmark/generation`](../benchmark/generation/README.md).
