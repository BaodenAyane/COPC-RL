# Experiment configuration

First set up the model, data, and task environment using the [ReTool guide](../examples/retool/README.md) or [Search-R1 guide](../examples/search-r1/README.md). Run commands from the repository root. Defaults and available options are defined in [scripts/train.sh](../scripts/train.sh); use `--help` to inspect them or `DRY_RUN=1` to preview a launch.

## Algorithms and ablations

Both launchers accept `copc`, `ppo`, `ppo-sync`, `grpo`, `dapo`, `sao`, `vtrace`, and `areal`, plus:

| Ablation | Critic target | Actor |
| --- | --- | --- |
| `copc-wo-rc` | Standard PPO GAE | COPC two-sided masking |
| `copc-vtrace` | V-trace | COPC two-sided masking |

Both V-trace entries enable advantage normalization. `VTRACE_CLIP_ONLY=1` with `vtrace` also selects `copc-vtrace`. GRPO and DAPO do not require critic pretraining. DAPO enables Clip-Higher, token-level loss, and dynamic sampling.

## Main settings

Defaults for COPC:

| Setting | ReTool | Search-R1 |
| --- | --- | --- |
| Optimizer updates | 120 | 450 |
| `FULLY_ASYNC_MAX_STALENESS` | 8 | 14 |
| COPC `ROLLOUT_BATCH_SIZE` | 64 | 128 |
| `N_SAMPLES_PER_PROMPT` | 8 | 4 |
| `GLOBAL_BATCH_SIZE` | 512 | 512 |

`NUM_STEPS` counts rollouts. AReaL uses two updates per rollout, with defaults of 60/225 rollouts and 128/256 prompts for ReTool/Search-R1. Other algorithms use one update per rollout by default. GRPO and Search-R1 SAO use their own sampling configurations in the launcher.

All RL algorithms save every **10 optimizer updates** under the default batch settings. `SAVE_INTERVAL` counts rollouts: its default is 10, or 5 for AReaL. `UPDATE_WEIGHTS_INTERVAL` also counts rollouts.

## COPC parameters

| Correction | Lower margin | Upper margin |
| --- | --- | --- |
| Actor masking | `EPS_CLIP=0.2` | `EPS_CLIP_HIGH=0.28` |
| Return correction | `RHO_CLIP=0.2` | `RHO_CLIP_HIGH=0.28` |

Each interval is `[1 - lower_margin, 1 + upper_margin]`. The paper's coefficients map to margins `0.4 * eta` and `0.56 * eta`; defaults use `eta_P = eta_A = 0.5`.

```bash
EPS_CLIP=0.1 EPS_CLIP_HIGH=0.14 RHO_CLIP=0.2 RHO_CLIP_HIGH=0.28 \
OUTPUT_DIR=/path/to/copc-narrow-policy bash scripts/retool/train.sh copc
```

## Checkpoint saving and resume

Both RL launchers default to `ASYNC_SAVE=1`, which adds `--async-save`. After capturing a consistent snapshot, model checkpoint writes continue in the background while training resumes. Set `ASYNC_SAVE=0` for synchronous writes.

In asynchronous modes, the launchers also enable complete training checkpoints (`--async-checkpoint`) automatically. Snapshot capture briefly synchronizes the training and rollout pipeline. These checkpoints contain actor and critic state (when a critic is used), optimizer and learning-rate scheduler state, RNG state, training progress, and rollout/data-source state. Complete checkpoints are saved with either value of `ASYNC_SAVE`.

A complete checkpoint becomes available for recovery at `OUTPUT_DIR/resume/step_XXXXXXX/manifest.json` after all required files have been written successfully. Keep the whole source run directory: the manifest references its model checkpoints. Model weights alone are insufficient for full-state recovery.

To resume an asynchronous run, explicitly set `ASYNC_RESUME` (passed as `--async-resume`) to the previous run's `resume/` directory. This selects the latest valid complete checkpoint. Choose a new, nonexistent `OUTPUT_DIR` and keep the original code, model, data, and training configuration, including `ASYNC_SAVE`.

```bash
ASYNC_RESUME=/path/to/previous-run/resume \
OUTPUT_DIR=/path/to/resumed-run bash scripts/retool/train.sh copc
```

To select a specific checkpoint, pass its step directory or `manifest.json` instead. Recovery continues from the saved training progress and learning-rate schedule. Search-R1 uses the same interface.
